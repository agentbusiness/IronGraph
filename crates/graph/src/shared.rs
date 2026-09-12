//! Narrow checked bridge from externally owned shared allocations to immutable graph slices.

#![allow(unsafe_code)]

use std::{fmt, marker::PhantomData, ptr::NonNull, sync::Arc};

use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeSeq};

/// Lifetime owner for a CPU-readable allocation that may also be addressable by an accelerator.
///
/// # Safety
///
/// `base_ptr` must remain valid and immovable for exactly `byte_len` initialized bytes until the
/// final owner is dropped. Implementations must not permit those bytes to be freed or remapped
/// while any cloned owner exists.
pub unsafe trait SharedAllocation: Send + Sync + 'static {
    fn base_ptr(&self) -> *const u8;
    fn byte_len(&self) -> usize;
}

/// Typed immutable view into one externally owned flat allocation. Clones retain the allocation;
/// slicing changes only the checked byte window and never copies values.
pub struct SharedFlat<T> {
    allocation: Arc<dyn SharedAllocation>,
    pointer: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}

struct EmptyAllocation;

// SAFETY: zero-length views never dereference the reported pointer.
unsafe impl SharedAllocation for EmptyAllocation {
    fn base_ptr(&self) -> *const u8 {
        NonNull::<u8>::dangling().as_ptr()
    }

    fn byte_len(&self) -> usize {
        0
    }
}

// The owner contract freezes the allocation for the lifetime of every view. Access through this
// type is immutable; mutation first detaches the containing paged-vector page.
unsafe impl<T: Send + Sync> Send for SharedFlat<T> {}
unsafe impl<T: Send + Sync> Sync for SharedFlat<T> {}

impl<T> Clone for SharedFlat<T> {
    fn clone(&self) -> Self {
        Self {
            allocation: Arc::clone(&self.allocation),
            pointer: self.pointer,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for SharedFlat<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T> SharedFlat<T> {
    pub fn empty() -> Self {
        Self {
            allocation: Arc::new(EmptyAllocation),
            pointer: NonNull::dangling(),
            len: 0,
            _marker: PhantomData,
        }
    }

    /// Creates a typed view over a checked range of one shared allocation.
    pub fn new(
        allocation: Arc<dyn SharedAllocation>,
        byte_offset: usize,
        len: usize,
    ) -> Result<Self, &'static str> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or("shared flat allocation size overflow")?;
        let end = byte_offset
            .checked_add(bytes)
            .ok_or("shared flat allocation range overflow")?;
        if size_of::<T>() == 0 {
            return Err("zero-sized shared flat values are unsupported");
        }
        if len == 0 {
            return Ok(Self::empty());
        }
        if end > allocation.byte_len() {
            return Err("shared flat allocation range is out of bounds");
        }
        let base = allocation.base_ptr();
        if base.is_null() {
            return Err("shared flat allocation has a null base pointer");
        }
        // SAFETY: bounds were checked above and the unsafe allocation contract keeps the base
        // address stable for the lifetime of the retained owner.
        let pointer = unsafe { base.add(byte_offset) }.cast_mut().cast::<T>();
        if !pointer.is_aligned() {
            return Err("shared flat allocation is not aligned for its value type");
        }
        let pointer = NonNull::new(pointer).ok_or("shared flat allocation pointer is null")?;
        Ok(Self {
            allocation,
            pointer,
            len,
            _marker: PhantomData,
        })
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Proves that `extended` starts at the same address in the same still-live allocation and
    /// exposes at least this view's initialized prefix. Separate owner wrappers are permitted for
    /// one Metal buffer; retaining both views prevents allocator address reuse during this check.
    #[must_use]
    pub(crate) fn is_prefix_of(&self, extended: &Self) -> bool {
        self.allocation.base_ptr() == extended.allocation.base_ptr()
            && self.allocation.byte_len() == extended.allocation.byte_len()
            && self.pointer == extended.pointer
            && self.len <= extended.len
    }

    /// Extends this logical prefix while retaining the exact allocation owner that created it.
    ///
    /// This matters for pooled accelerator allocations: a separately cloned native buffer may
    /// keep the bytes alive without incrementing the pool's own owner count, allowing the pool to
    /// recycle storage that the graph still addresses. Extending the existing view preserves that
    /// original owner and changes only the checked logical length.
    pub fn extended_prefix(&self, len: usize) -> Result<Self, &'static str> {
        if len < self.len {
            return Err("shared flat prefix extension cannot shrink");
        }
        let base = self.allocation.base_ptr() as usize;
        let pointer = self.pointer.as_ptr() as usize;
        let byte_offset = pointer
            .checked_sub(base)
            .ok_or("shared flat prefix starts before its allocation")?;
        Self::new(Arc::clone(&self.allocation), byte_offset, len)
    }

    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: construction validated the initialized range, and the retained owner keeps it
        // allocated and immovable. This type never exposes mutable access.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }

    pub fn slice(&self, start: usize, len: usize) -> Result<Self, &'static str> {
        let end = start
            .checked_add(len)
            .ok_or("shared flat slice range overflow")?;
        if end > self.len {
            return Err("shared flat slice is out of bounds");
        }
        let base = self.allocation.base_ptr() as usize;
        let current = self.pointer.as_ptr() as usize;
        let current_offset = current
            .checked_sub(base)
            .ok_or("shared flat pointer precedes its allocation")?;
        let byte_offset = start
            .checked_mul(size_of::<T>())
            .and_then(|offset| current_offset.checked_add(offset))
            .ok_or("shared flat slice byte range overflow")?;
        Self::new(Arc::clone(&self.allocation), byte_offset, len)
    }
}

impl SharedFlat<u8> {
    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn into_bools(self) -> Result<SharedFlat<bool>, &'static str> {
        if self.as_slice().iter().any(|value| *value > 1) {
            return Err("shared boolean column contains a non-boolean byte");
        }
        let base = self.allocation.base_ptr() as usize;
        let current = self.pointer.as_ptr() as usize;
        let offset = current
            .checked_sub(base)
            .ok_or("shared boolean pointer precedes its allocation")?;
        SharedFlat::new(self.allocation, offset, self.len)
    }
}

/// Contiguous immutable column that is either ordinary host-owned storage (CPU/decode mode) or
/// an accelerator-owned shared allocation (Apple production mode). Both serialize identically as
/// one flat sequence, so checkpoints never contain device handles.
pub enum FlatColumn<T> {
    Owned(Arc<Vec<T>>),
    Shared(SharedFlat<T>),
}

impl<T> Clone for FlatColumn<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Owned(values) => Self::Owned(Arc::clone(values)),
            Self::Shared(values) => Self::Shared(values.clone()),
        }
    }
}

impl<T> Default for FlatColumn<T> {
    fn default() -> Self {
        Self::Owned(Arc::new(Vec::new()))
    }
}

impl<T: fmt::Debug> fmt::Debug for FlatColumn<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T> FlatColumn<T> {
    pub fn from_vec(values: Vec<T>) -> Self {
        Self::Owned(Arc::new(values))
    }

    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn from_shared(values: SharedFlat<T>) -> Self {
        Self::Shared(values)
    }

    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Owned(values) => values,
            Self::Shared(values) => values.as_slice(),
        }
    }
}

impl<T: PartialEq> PartialEq for FlatColumn<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for FlatColumn<T> {}

impl<T: Serialize> Serialize for FlatColumn<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.as_slice().len()))?;
        for value in self.as_slice() {
            sequence.serialize_element(value)?;
        }
        sequence.end()
    }
}

impl<'de, T> Deserialize<'de> for FlatColumn<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<T>::deserialize(deserializer).map(Self::from_vec)
    }
}

#[cfg(test)]
struct ArcSliceAllocation<T>(Arc<[T]>);

#[cfg(test)]
unsafe impl<T: Send + Sync + 'static> SharedAllocation for ArcSliceAllocation<T> {
    fn base_ptr(&self) -> *const u8 {
        self.0.as_ptr().cast()
    }

    fn byte_len(&self) -> usize {
        self.0.len().saturating_mul(size_of::<T>())
    }
}

#[cfg(test)]
impl<T: Send + Sync + 'static> SharedFlat<T> {
    pub fn from_arc_slice(values: Arc<[T]>) -> Result<Self, &'static str> {
        let len = values.len();
        let allocation: Arc<dyn SharedAllocation> = Arc::new(ArcSliceAllocation(values));
        Self::new(allocation, 0, len)
    }
}
