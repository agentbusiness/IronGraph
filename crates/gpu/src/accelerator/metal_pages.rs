//! Owned Metal allocations with exclusive cell edits and retained-generation copy-on-write.
//!
//! A writer excludes host views and synchronizes device readers before editing current cells.
//! Retained views use private CPU-written pages. GPU kernels only read canonical allocations:
//! device stores do not implement CPU VM faults.
#![allow(unsafe_code)]

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    ffi::c_void,
    ptr::NonNull,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use block2::RcBlock;
use candle_core::{Device, MetalStorage, Storage, Tensor, op::BackpropOp};
use candle_metal_kernels::metal::Buffer;
use objc2_metal::MTLDevice;

use crate::graph::PersistentMap;
use crate::{Error, ErrorCode, Result};

unsafe extern "C" {
    static mach_task_self_: u32;
    static vm_page_size: usize;
    fn mach_vm_allocate(task: u32, address: *mut u64, size: u64, flags: i32) -> i32;
    fn mach_vm_deallocate(task: u32, address: u64, size: u64) -> i32;
    fn mach_vm_remap(
        task: u32,
        address: *mut u64,
        size: u64,
        mask: u64,
        flags: i32,
        source_task: u32,
        source_address: u64,
        copy: i32,
        current: *mut i32,
        maximum: *mut i32,
        inheritance: u32,
    ) -> i32;
}

pub(super) fn page_bytes() -> usize {
    // SAFETY: libSystem initializes this process-wide constant before Rust startup.
    unsafe { vm_page_size }
}

struct Mapping {
    address: u64,
    bytes: usize,
    root: u64,
    exposed_bytes: usize,
}

#[derive(Clone)]
struct MappingInfo {
    reserved_bytes: usize,
    exposed_bytes: usize,
    root: u64,
    owner: Weak<Mapping>,
}

#[derive(Default)]
struct RootWrite {
    initial_bytes: usize,
    exposed_bytes: usize,
    new_root: bool,
    dirty_pages: BTreeSet<usize>,
    retired_bytes: usize,
}

/// One operation's native writes. Only affected roots/pages are visited, never the resident.
#[derive(Default)]
pub(crate) struct WritePages {
    roots: BTreeMap<u64, RootWrite>,
    host_pages: BTreeMap<(u64, usize), HostPageWrite>,
    in_place: bool,
    undo: Vec<UndoWrite>,
}

struct UndoWrite {
    buffer: Arc<Buffer>,
    offsets: Vec<usize>,
    bytes: Vec<u8>,
    width: usize,
    previous_exposed: Option<usize>,
}

impl Drop for WritePages {
    fn drop(&mut self) {
        for write in self.undo.iter().rev() {
            let pointer = write.buffer.contents().cast::<u8>();
            for (index, offset) in write.offsets.iter().enumerate().rev() {
                // SAFETY: the undo owner retains each checked writable cell. Exclusive
                // publication keeps CPU views and GPU readers out until rollback finishes.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        write.bytes.as_ptr().add(index * write.width),
                        pointer.add(*offset),
                        write.width,
                    );
                }
            }
            if let Some(previous) = write.previous_exposed {
                let mut registry = mappings().lock().unwrap_or_else(|e| e.into_inner());
                if let Some(info) = registry.get_mut(&(write.buffer.contents() as usize)) {
                    info.exposed_bytes = previous;
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct HostPageWrite {
    bytes: usize,
    retired: bool,
}

impl WritePages {
    pub(crate) fn commit(&mut self) {
        self.undo.clear();
    }
    /// Stable allocation/page identities let admission retain only versions visible to a pin.
    /// A suffix birth records native capacity growth without visiting its unchanged prefix.
    pub(crate) fn retirement_events(&self) -> (Vec<([u64; 5], u64)>, Vec<([u64; 6], usize)>) {
        let mut births = Vec::new();
        let mut changes = Vec::new();
        for (root, write) in &self.roots {
            let group = [0, *root, 0, 0, 0];
            if write.new_root {
                births.push((group, 0));
            } else if write.exposed_bytes > write.initial_bytes {
                births.push((group, (write.initial_bytes / page_bytes()) as u64));
            }
            if write.retired_bytes != 0 {
                changes.push(([0, *root, 0, 0, 0, u64::MAX], write.retired_bytes));
            } else {
                changes.extend(
                    write
                        .dirty_pages
                        .iter()
                        .map(|page| ([0, *root, 0, 0, 0, *page as u64], page_bytes())),
                );
            }
        }
        changes.extend(self.host_pages.iter().map(|((root, page), write)| {
            (
                [1, *root, 0, 0, 0, *page as u64],
                if write.retired { write.bytes } else { 0 },
            )
        }));
        (births, changes)
    }

    #[cfg(test)]
    pub(crate) fn retired_bytes(&self) -> usize {
        self.host_pages
            .values()
            .filter(|page| page.retired)
            .fold(self.native_retired_bytes(), |bytes, page| {
                bytes.saturating_add(page.bytes)
            })
    }

    #[cfg(test)]
    pub(crate) fn native_retired_bytes(&self) -> usize {
        self.roots.values().fold(0_usize, |total, root| {
            total.saturating_add(
                root.retired_bytes
                    .max(root.dirty_pages.len().saturating_mul(page_bytes())),
            )
        })
    }
}

#[derive(Clone, Copy)]
struct BranchRoot {
    base_bytes: usize,
    exposed_bytes: usize,
}

/// Complete branch divergence; an allocation/page identity is charged once across writes.
#[derive(Clone, Default)]
pub(crate) struct BranchPages {
    roots: PersistentMap<BranchRoot>,
    pages: PersistentMap<()>,
    bytes: usize,
    host_pages: PersistentMap<usize>,
}

impl BranchPages {
    pub(crate) fn apply(&mut self, writes: &WritePages) {
        // Allocation IDs vary in their low bits. Branch on those bits first; retirement
        // events retain the original IDs while this private accounting uses reordered keys.
        for ((root, page), write) in &writes.host_pages {
            let key =
                (u128::from(root.reverse_bits()) << 64) | u128::from((*page as u64).reverse_bits());
            let previous = self.host_pages.get(key).copied().unwrap_or(0);
            let next = previous.max(write.bytes);
            self.host_pages.insert_cow(key, next);
            self.bytes = self.bytes.saturating_add(next.saturating_sub(previous));
        }
        for (id, write) in &writes.roots {
            let mut root = self
                .roots
                .get(u128::from(*id).reverse_bits())
                .copied()
                .unwrap_or(BranchRoot {
                    base_bytes: if write.new_root {
                        0
                    } else {
                        write.initial_bytes
                    },
                    exposed_bytes: if write.new_root {
                        0
                    } else {
                        write.initial_bytes
                    },
                });
            self.bytes = self
                .bytes
                .saturating_add(write.exposed_bytes.saturating_sub(root.exposed_bytes));
            root.exposed_bytes = root.exposed_bytes.max(write.exposed_bytes);
            for page in &write.dirty_pages {
                if page.saturating_mul(page_bytes()) < root.base_bytes {
                    let key = (u128::from(id.reverse_bits()) << 64)
                        | u128::from((*page as u64).reverse_bits());
                    if !self.pages.contains_key(key) {
                        self.pages.insert_cow(key, ());
                        self.bytes = self.bytes.saturating_add(page_bytes());
                    }
                }
            }
            self.roots.insert_cow(u128::from(*id).reverse_bits(), root);
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        // Persistent identity metadata is private too; bounds radix nodes amortized per key.
        self.bytes.saturating_add(
            (self.roots.len() + self.pages.len() + self.host_pages.len()).saturating_mul(4096),
        )
    }
}

thread_local! {
    static WRITE_CAPTURE: RefCell<Option<WritePages>> = const { RefCell::new(None) };
    static EXCLUSIVE_WRITE: Cell<bool> = const { Cell::new(false) };
}

pub(crate) struct ExclusiveWrite(bool);

impl ExclusiveWrite {
    /// The backend's exclusive-publication entrypoint establishes the safety precondition.
    pub(crate) fn begin() -> Self {
        Self(EXCLUSIVE_WRITE.with(|exclusive| exclusive.replace(true)))
    }
}

impl Drop for ExclusiveWrite {
    fn drop(&mut self) {
        EXCLUSIVE_WRITE.with(|exclusive| exclusive.set(self.0));
    }
}

pub(crate) struct Capture;

impl Capture {
    pub(crate) fn begin() -> Result<Self> {
        WRITE_CAPTURE.with(|capture| {
            let mut capture = capture.borrow_mut();
            if capture.is_some() {
                return Err(Error::internal("nested Metal native write capture"));
            }
            let pages = WritePages {
                in_place: EXCLUSIVE_WRITE.with(Cell::get),
                roots: BTreeMap::new(),
                host_pages: BTreeMap::new(),
                undo: Vec::new(),
            };
            *capture = Some(pages);
            Ok(Self)
        })
    }

    pub(crate) fn finish(self) -> Result<WritePages> {
        WRITE_CAPTURE
            .with(|capture| capture.borrow_mut().take())
            .ok_or_else(|| Error::internal("Metal native write capture was lost"))
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        WRITE_CAPTURE.with(|capture| {
            capture.borrow_mut().take();
        });
    }
}

fn record_write(
    info: MappingInfo,
    new_root: bool,
    exposed_bytes: usize,
    pages: impl Iterator<Item = usize>,
) {
    WRITE_CAPTURE.with(|capture| {
        let mut capture = capture.borrow_mut();
        let Some(capture) = capture.as_mut() else {
            return;
        };
        let root = capture.roots.entry(info.root).or_insert_with(|| RootWrite {
            initial_bytes: if new_root { 0 } else { info.exposed_bytes },
            exposed_bytes: info.exposed_bytes,
            new_root,
            dirty_pages: BTreeSet::new(),
            retired_bytes: 0,
        });
        root.exposed_bytes = root.exposed_bytes.max(exposed_bytes);
        for page in pages {
            if page.saturating_mul(page_bytes()) < root.initial_bytes {
                root.dirty_pages.insert(page);
            }
        }
    });
}

/// Records private CPU index leaves and their bounded directory paths alongside native pages.
/// The identity is stable across generations and belongs to a separate host-page domain.
pub(crate) fn new_host_root() -> Result<u64> {
    static NEXT_HOST_ROOT: AtomicU64 = AtomicU64::new(1);
    NEXT_HOST_ROOT
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |root| {
            root.checked_add(1)
        })
        .map_err(|_| Error::internal("Metal host page identity overflow"))
}

pub(crate) fn record_host_pages(
    root: u64,
    previous_rows: usize,
    page_width: usize,
    pages: impl Iterator<Item = usize>,
    bytes_per_page: usize,
) {
    WRITE_CAPTURE.with(|capture| {
        let mut capture = capture.borrow_mut();
        let Some(capture) = capture.as_mut() else {
            return;
        };
        for page in pages {
            let entry = capture
                .host_pages
                .entry((root, page))
                .or_insert(HostPageWrite {
                    bytes: 0,
                    retired: false,
                });
            entry.bytes = entry.bytes.max(bytes_per_page);
            entry.retired |= page.saturating_mul(page_width) < previous_rows;
        }
    });
}

/// Records an old owned allocation removed from the replacement resident. COW-patched
/// sources need no call: their changed pages are already recorded by `extend`.
pub(crate) fn retire(tensor: &Tensor) -> Result<()> {
    let (storage, _) = tensor.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        return Err(Error::internal("retired Metal allocation is not resident"));
    };
    let info = mappings()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&(storage.buffer().contents() as usize))
        .cloned()
        .ok_or_else(|| Error::internal("retired Metal allocation has no native lineage"))?;
    WRITE_CAPTURE.with(|capture| {
        let mut capture = capture.borrow_mut();
        if let Some(capture) = capture.as_mut() {
            let root = capture.roots.entry(info.root).or_insert_with(|| RootWrite {
                initial_bytes: info.exposed_bytes,
                exposed_bytes: info.exposed_bytes,
                ..RootWrite::default()
            });
            if !root.new_root {
                root.retired_bytes = root.retired_bytes.max(info.exposed_bytes);
            }
        }
    });
    Ok(())
}

// Only allocations created here may be forked: pooled Candle buffers can be recycled while
// a VM alias still owns their pages. The registry also records the uncommitted growth range.
fn mappings() -> &'static Mutex<BTreeMap<usize, MappingInfo>> {
    static MAPPINGS: OnceLock<Mutex<BTreeMap<usize, MappingInfo>>> = OnceLock::new();
    MAPPINGS.get_or_init(Mutex::default)
}

impl Drop for Mapping {
    fn drop(&mut self) {
        mappings()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&(self.address as usize));
        // SAFETY: this owner releases only its own complete virtual mapping. Mach keeps shared
        // physical pages alive for other mappings independently of this address's lifetime.
        let status =
            unsafe { mach_vm_deallocate(mach_task_self_, self.address, self.bytes as u64) };
        debug_assert_eq!(status, 0, "owned Metal VM mapping failed to deallocate");
    }
}

fn allocation_error(operation: &str, status: i32) -> Error {
    Error::new(
        ErrorCode::GpuAdmissionFailure,
        format!("Metal {operation} failed: Mach status {status}"),
    )
}

impl Mapping {
    fn allocate(device: &candle_core::MetalDevice) -> Result<Self> {
        let bytes = device.metal_device().as_ref().maxBufferLength() / page_bytes() * page_bytes();
        if bytes == 0 {
            return Err(Error::internal("Metal VM allocation size is invalid"));
        }
        static NEXT_ROOT: AtomicU64 = AtomicU64::new(1);
        let root = NEXT_ROOT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| Error::internal("Metal allocation identity overflow"))?;
        let mut address = 0;
        // Anonymous demand-zero memory retains the kernel's COW strategy. Named objects
        // force COPY_NONE, making a retained-cell edit traverse the whole exposed prefix.
        // VM_FLAGS_4GB_CHUNK avoids the default 128 MiB chunk boundary during buffer growth;
        // the selected device's complete maximum range remains reserved without physical pages.
        // SAFETY: Mach creates a new page-aligned allocation owned exclusively by this Mapping.
        let status =
            unsafe { mach_vm_allocate(mach_task_self_, &mut address, bytes as u64, 1 | 4) };
        if status != 0 {
            return Err(allocation_error("VM allocation", status));
        }
        Ok(Self {
            address,
            bytes,
            root,
            exposed_bytes: 0,
        })
    }

    fn copy(
        buffer: &Buffer,
        device: &candle_core::MetalDevice,
        dirty_pages: &BTreeSet<usize>,
    ) -> Result<Self> {
        let info = mappings()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(buffer.contents() as usize))
            .cloned()
            .ok_or_else(|| Error::internal("Metal COW source is not an owned immutable mapping"))?;
        let bytes = info.exposed_bytes;
        if bytes == 0
            || bytes > info.reserved_bytes
            || bytes % page_bytes() != 0
            || buffer.contents() as usize % page_bytes() != 0
        {
            return Err(Error::internal(
                "Metal COW source is not a complete page-aligned mapping",
            ));
        }
        let mut mapping = Self::allocate(device)?;
        let mut address = mapping.address;
        let mut current = 0;
        let mut maximum = 0;
        // SAFETY: unchanged pages stay shared with immutable retained generations. Every
        // page written below is separately detached before the first CPU store. Exclusive
        // publication excludes retained generations before it can mutate these shared pages.
        // The unpublished tail remains demand-zero capacity.
        let status = unsafe {
            mach_vm_remap(
                mach_task_self_,
                &mut address,
                bytes as u64,
                0,
                0x4000,
                mach_task_self_,
                buffer.contents() as u64,
                0,
                &mut current,
                &mut maximum,
                2,
            )
        };
        if status != 0 {
            return Err(allocation_error("shared prefix remap", status));
        }
        for &page in dirty_pages {
            let offset = page
                .checked_mul(page_bytes())
                .ok_or_else(|| Error::internal("Metal dirty page offset overflow"))?;
            if offset >= bytes {
                continue;
            }
            let mut address = mapping.address + offset as u64;
            // SAFETY: replace only this changed page in the unpublished target. The source
            // remains retained and read-only; copy=true preserves its CPU and GPU contents.
            let status = unsafe {
                mach_vm_remap(
                    mach_task_self_,
                    &mut address,
                    page_bytes() as u64,
                    0,
                    0x4000,
                    mach_task_self_,
                    buffer.contents() as u64 + offset as u64,
                    1,
                    &mut current,
                    &mut maximum,
                    2,
                )
            };
            if status != 0 {
                return Err(allocation_error("changed page COW remap", status));
            }
        }
        mapping.root = info.root;
        mapping.exposed_bytes = bytes;
        Ok(mapping)
    }

    fn buffer(
        mut self,
        device: &candle_core::MetalDevice,
        exposed_bytes: usize,
        label: &str,
    ) -> Result<Arc<Buffer>> {
        // A narrowed tensor still owns the entire registered prefix. Preserve that highwater
        // when it is forked, including demand-zero pages already exposed to Metal.
        let bytes = rounded_bytes(exposed_bytes)?.max(self.exposed_bytes);
        if bytes > self.bytes {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "Metal column exceeds the selected device buffer limit",
            ));
        }
        self.exposed_bytes = bytes;
        Self::owned_buffer(Arc::new(self), device, bytes, label)
    }

    fn owned_buffer(
        owner: Arc<Self>,
        device: &candle_core::MetalDevice,
        bytes: usize,
        label: &str,
    ) -> Result<Arc<Buffer>> {
        let pointer = NonNull::new(owner.address as *mut c_void)
            .ok_or_else(|| Error::internal("Metal VM mapping has a null address"))?;
        let address = owner.address as usize;
        let previous = mappings().lock().unwrap_or_else(|e| e.into_inner()).insert(
            address,
            MappingInfo {
                reserved_bytes: owner.bytes,
                exposed_bytes: bytes,
                root: owner.root,
                owner: Arc::downgrade(&owner),
            },
        );
        // Capturing the owner also releases the mapping on buffer-creation failure. Metal
        // copies this sendable block and drops its capture only after the last buffer owner.
        let release = RcBlock::new(move |_: NonNull<c_void>, _: usize| {
            let _ = &owner;
        });
        // SAFETY: the complete buffer lies in one aligned owned VM allocation. The deallocator block owns
        // it, is sendable (integer fields only), and outlives all GPU and host buffer users.
        let raw = unsafe {
            device
                .metal_device()
                .as_ref()
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    pointer,
                    bytes,
                    candle_metal_kernels::RESOURCE_OPTIONS,
                    Some(&release),
                )
        };
        let Some(raw) = raw else {
            let mut registry = mappings().lock().unwrap_or_else(|e| e.into_inner());
            if let Some(previous) = previous {
                registry.insert(address, previous);
            } else {
                registry.remove(&address);
            }
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "Metal rejected a VM-backed shared buffer",
            ));
        };
        let buffer = Arc::new(Buffer::new(raw));
        buffer.set_label(label);
        Ok(buffer)
    }
}

pub(super) fn allocate(
    device: &candle_core::MetalDevice,
    bytes: usize,
    label: &str,
) -> Result<Arc<Buffer>> {
    let mapping = Mapping::allocate(device)?;
    let info = MappingInfo {
        reserved_bytes: mapping.bytes,
        exposed_bytes: 0,
        root: mapping.root,
        owner: Weak::new(),
    };
    let buffer = mapping.buffer(device, bytes, label)?;
    record_write(info, true, buffer.length(), std::iter::empty());
    Ok(buffer)
}

pub(super) fn rounded_bytes(bytes: usize) -> Result<usize> {
    bytes
        .max(1)
        .checked_add(page_bytes() - 1)
        .map(|bytes| bytes / page_bytes() * page_bytes())
        .ok_or_else(|| Error::internal("Metal VM allocation size overflow"))
}

/// A logical zero column without a host zero-vector or a loop over unrelated rows.
pub(super) fn zeros(dtype: candle_core::DType, len: usize, device: &Device) -> Result<Tensor> {
    let Device::Metal(device) = device else {
        return Err(Error::internal("Metal sparse allocation requires Metal"));
    };
    let bytes = len
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::internal("Metal column size overflow"))?;
    let buffer = allocate(device, bytes, "irongraph immutable sparse column")?;
    Ok(Tensor::from_storage(
        Storage::Metal(MetalStorage::new(buffer, device.clone(), len, dtype)),
        len,
        BackpropOp::none(),
        false,
    ))
}

/// Changes addressed rows in a private mapping or under the exclusive publication guard.
/// Retained views remain unchanged; exclusive edits keep the current allocation address.
pub(super) fn patch<T: Copy>(
    tensor: &Tensor,
    updates: &[(usize, T)],
    device: &Device,
) -> Result<Tensor> {
    extend(tensor, updates, tensor.elem_count(), device)
}

pub(super) fn buffer_bytes(tensor: &Tensor) -> usize {
    let (storage, _) = tensor.storage_and_layout();
    match &*storage {
        Storage::Metal(storage) => storage.buffer().length(),
        _ => tensor
            .elem_count()
            .saturating_mul(tensor.dtype().size_in_bytes()),
    }
}

pub(super) fn extend<T: Copy>(
    tensor: &Tensor,
    updates: &[(usize, T)],
    len: usize,
    device: &Device,
) -> Result<Tensor> {
    let Device::Metal(device) = device else {
        return Err(Error::internal("Metal COW patch requires Metal"));
    };
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        return Err(Error::internal("Metal COW source is not resident"));
    };
    if !layout.is_contiguous()
        || layout.start_offset() != 0
        || size_of::<T>() != tensor.dtype().size_in_bytes()
        || len < tensor.elem_count()
        || updates.iter().any(|(row, _)| *row >= len)
    {
        return Err(Error::internal("Metal COW patch has an invalid shape"));
    }
    let bytes = len
        .checked_mul(tensor.dtype().size_in_bytes())
        .ok_or_else(|| Error::internal("Metal column size overflow"))?;
    if updates.is_empty()
        && bytes <= storage.buffer().length()
        && mappings()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&(storage.buffer().contents() as usize))
    {
        // This changes only a read-only view's length, never its bytes. Retain the
        // owned native buffer; actual edits still detach from every retained view.
        return Ok(Tensor::from_storage(
            Storage::Metal(MetalStorage::new(
                Arc::new(storage.buffer().clone()),
                device.clone(),
                len,
                tensor.dtype(),
            )),
            len,
            BackpropOp::none(),
            false,
        ));
    }
    let in_place = WRITE_CAPTURE.with(|capture| {
        capture
            .borrow()
            .as_ref()
            .is_some_and(|capture| capture.in_place)
    });
    if in_place {
        let mut previous_exposed = None;
        let buffer = if bytes <= storage.buffer().length() {
            Arc::new(storage.buffer().clone())
        } else {
            let info = mappings()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&(storage.buffer().contents() as usize))
                .cloned()
                .ok_or_else(|| Error::internal("exclusive buffer has no mapping owner"))?;
            let owner = info
                .owner
                .upgrade()
                .ok_or_else(|| Error::internal("exclusive mapping owner was released"))?;
            let exposed = rounded_bytes(bytes)?;
            if exposed > info.reserved_bytes {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "Metal column exceeds the selected device buffer limit",
                ));
            }
            let buffer =
                Mapping::owned_buffer(owner, device, exposed, "irongraph exclusive growth")?;
            previous_exposed = Some(info.exposed_bytes);
            record_write(info, false, exposed, std::iter::empty());
            buffer
        };
        let mut undo = UndoWrite {
            buffer: Arc::clone(&buffer),
            offsets: Vec::with_capacity(updates.len()),
            bytes: Vec::with_capacity(updates.len() * size_of::<T>()),
            width: size_of::<T>(),
            previous_exposed,
        };
        let pointer = buffer.contents().cast::<u8>();
        for (row, _) in updates {
            let offset = row * size_of::<T>();
            undo.offsets.push(offset);
            // SAFETY: the checked row and width lie inside this exclusively owned buffer.
            let previous =
                unsafe { std::slice::from_raw_parts(pointer.add(offset), size_of::<T>()) };
            undo.bytes.extend_from_slice(previous);
        }
        WRITE_CAPTURE.with(|capture| {
            if let Some(capture) = capture.borrow_mut().as_mut() {
                capture.undo.push(undo);
            }
        });
        for (row, value) in updates {
            // SAFETY: the caller excludes readers, and the rollback guard owns these cells.
            unsafe {
                pointer.cast::<T>().add(*row).write_unaligned(*value);
            }
        }
        return Ok(Tensor::from_storage(
            Storage::Metal(MetalStorage::new(
                buffer,
                device.clone(),
                len,
                tensor.dtype(),
            )),
            len,
            BackpropOp::none(),
            false,
        ));
    }
    let dirty_pages = updates
        .iter()
        .map(|(row, _)| row * size_of::<T>() / page_bytes())
        .collect::<BTreeSet<_>>();
    let mapping = Mapping::copy(storage.buffer(), device, &dirty_pages)?;
    let info = MappingInfo {
        reserved_bytes: mapping.bytes,
        exposed_bytes: mapping.exposed_bytes,
        root: mapping.root,
        owner: Weak::new(),
    };
    if bytes > mapping.bytes {
        return Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            "Metal column exceeds the selected device buffer limit",
        ));
    }
    for (row, value) in updates {
        // SAFETY: the mapping is private and unpublished, row bounds and width were checked.
        // Writing through the CPU triggers COW for this page before Metal registers the buffer.
        unsafe {
            (mapping.address as *mut T)
                .add(*row)
                .write_unaligned(*value);
        }
    }
    let buffer = mapping.buffer(device, bytes, "irongraph immutable property delta")?;
    record_write(info, false, buffer.length(), dirty_pages.into_iter());
    let storage = Storage::Metal(MetalStorage::new(
        buffer,
        device.clone(),
        len,
        tensor.dtype(),
    ));
    Ok(Tensor::from_storage(
        storage,
        len,
        BackpropOp::none(),
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::{TensorUpload, candle_error};
    use super::*;

    unsafe extern "C" {
        fn task_info(task: u32, flavor: u32, info: *mut i32, count: *mut u32) -> i32;
    }

    fn resident_bytes() -> Result<u64> {
        // mach_task_basic_info: three u64 byte counts, two time_value_t pairs, two i32s.
        let mut info = [0_u64; 6];
        let mut count = 12;
        // SAFETY: the buffer has the ABI's complete 48-byte size and alignment.
        let status =
            unsafe { task_info(mach_task_self_, 20, info.as_mut_ptr().cast(), &mut count) };
        if status != 0 {
            return Err(allocation_error("resident-memory query", status));
        }
        Ok(info[1])
    }

    #[test]
    fn exclusive_growth_keeps_dirty_prefix_address_and_rolls_back_extent() -> Result<()> {
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        let values = (0_i64..8 * 1024 * 1024)
            .map(|value| value.wrapping_mul(7919))
            .collect::<Vec<_>>();
        let mut upload = TensorUpload::new(&device);
        upload.immutable_properties = true;
        let source = upload
            .optional(&values)?
            .ok_or_else(|| Error::internal("missing growth fixture"))?;
        let address = |tensor: &Tensor| {
            let (storage, _) = tensor.storage_and_layout();
            let Storage::Metal(storage) = &*storage else {
                panic!("Metal fixture")
            };
            storage.buffer().contents() as usize
        };
        let original_address = address(&source);
        let old_extent = buffer_bytes(&source);
        let before = resident_bytes()?;
        let _exclusive = ExclusiveWrite::begin();
        let capture = Capture::begin()?;
        let grown = extend(
            &source,
            &[(3, -7_i64), (values.len(), 37)],
            values.len() + 1,
            &device,
        )?;
        assert_eq!(address(&grown), original_address);
        assert_eq!(buffer_bytes(&grown), old_extent + page_bytes());
        assert!(resident_bytes()?.saturating_sub(before) < 1024 * 1024);
        drop(capture);
        drop(grown);
        assert_eq!(
            mappings().lock().unwrap_or_else(|e| e.into_inner())[&original_address].exposed_bytes,
            old_extent
        );
        let capture = Capture::begin()?;
        let mut current = extend(
            &source,
            &[(values.len(), 41_i64)],
            values.len() + 1,
            &device,
        )?;
        let mut writes = capture.finish()?;
        writes.commit();
        drop(source);
        drop(upload);
        for step in 1..=16 {
            let len = values.len() + step * (page_bytes() / 8);
            let capture = Capture::begin()?;
            current = extend(&current, &[(len, step as i64)], len + 1, &device)?;
            let mut writes = capture.finish()?;
            writes.commit();
            assert_eq!(address(&current), original_address);
        }
        let selected =
            Tensor::from_slice(&[3_u32, values.len() as u32], 2, &device).map_err(candle_error)?;
        assert_eq!(
            current
                .index_select(&selected, 0)
                .and_then(|result| result.to_vec1::<i64>())
                .map_err(candle_error)?,
            [values[3], 41]
        );
        drop(current);
        assert!(
            !mappings()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains_key(&original_address)
        );
        Ok(())
    }

    #[test]
    fn exclusive_cells_reuse_storage_and_rollback_until_commit() -> Result<()> {
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        let mut upload = TensorUpload::new(&device);
        upload.immutable_properties = true;
        let source = upload
            .optional(&[7_i64, 11, 13])?
            .ok_or_else(|| Error::internal("missing in-place fixture"))?;
        let address = |tensor: &Tensor| {
            let (storage, _) = tensor.storage_and_layout();
            let Storage::Metal(storage) = &*storage else {
                panic!("Metal fixture")
            };
            storage.buffer().contents() as usize
        };
        let original_address = address(&source);
        device.synchronize().map_err(candle_error)?;
        {
            let _exclusive = ExclusiveWrite::begin();
            let capture = Capture::begin()?;
            let staged = patch(&source, &[(0, 19_i64), (2, 23)], &device)?;
            assert_eq!(address(&staged), original_address);
            assert_eq!(staged.to_vec1::<i64>().map_err(candle_error)?, [19, 11, 23]);
            // Rejecting a later column must undo previously staged cells too.
            assert!(patch(&staged, &[(3, 29_i64)], &device).is_err());
            drop(capture);
        }
        assert_eq!(source.to_vec1::<i64>().map_err(candle_error)?, [7, 11, 13]);
        {
            let _exclusive = ExclusiveWrite::begin();
            let capture = Capture::begin()?;
            let first = patch(&source, &[(0, 19_i64)], &device)?;
            let second = patch(&first, &[(0, 31_i64), (2, 23)], &device)?;
            assert_eq!(address(&second), original_address);
            let writes = capture.finish()?;
            assert_eq!(
                writes
                    .undo
                    .iter()
                    .map(|write| write.bytes.len())
                    .sum::<usize>(),
                24
            );
            assert_eq!(writes.native_retired_bytes(), 0);
            // Admission can fail after staging; the finished guard still restores cells.
            drop(writes);
        }
        assert_eq!(source.to_vec1::<i64>().map_err(candle_error)?, [7, 11, 13]);
        {
            let _exclusive = ExclusiveWrite::begin();
            let capture = Capture::begin()?;
            let staged = patch(&source, &[(1, 37_i64)], &device)?;
            assert_eq!(address(&staged), original_address);
            let mut writes = capture.finish()?;
            writes.commit();
        }
        assert_eq!(source.to_vec1::<i64>().map_err(candle_error)?, [7, 37, 13]);
        Ok(())
    }

    #[test]
    fn metal_cow_dirty_buffers_copy_pages_and_survive_source_release() -> Result<()> {
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        for bytes in [4 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024] {
            let mut random = 123_456_789_u64;
            let values = (0..bytes / 8)
                .map(|_| {
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    random as i64
                })
                .collect::<Vec<_>>();
            let expected = values[10];
            let mut upload = TensorUpload::new(&device);
            upload.immutable_properties = true;
            let source = upload
                .optional(&values)?
                .ok_or_else(|| Error::internal("missing source"))?;
            // Native buffer registration has first-use driver costs. Warm the
            // exact operation before bounding additional physical page copies.
            drop(patch(&source, &[(2_048, -899_i64)], &device)?);
            let before = resident_bytes()?;
            let target = patch(&source, &[(2_048, -900_i64)], &device)?;
            let added = resident_bytes()?.saturating_sub(before);
            assert!(
                added < 1024 * 1024,
                "a one-row patch duplicated {added} bytes from a {bytes}-byte buffer"
            );
            let last = values.len() - 1;
            let selected = Tensor::from_slice(&[10_u32, 2_048, last as u32], 3, &device)
                .map_err(candle_error)?;
            let gather = |tensor: &Tensor| {
                tensor
                    .index_select(&selected, 0)
                    .and_then(|v| v.to_vec1::<i64>())
                    .map_err(candle_error)
            };
            assert_eq!(
                gather(&source)?,
                vec![expected, values[2_048], values[last]]
            );
            assert_eq!(gather(&target)?, vec![expected, -900, values[last]]);
            // A later generation writes a page that its parent previously shared, while
            // a sibling writes both pages. Every older view must keep its own GPU values.
            let descendants_before = resident_bytes()?;
            let child = patch(&target, &[(last, -901_i64)], &device)?;
            let sibling = patch(&source, &[(2_048, -902_i64), (last, -903)], &device)?;
            let descendants_added = resident_bytes()?.saturating_sub(descendants_before);
            assert!(
                descendants_added < 1024 * 1024,
                "child/sibling patches duplicated {descendants_added} bytes from a {bytes}-byte buffer"
            );
            assert_eq!(gather(&child)?, vec![expected, -900, -901]);
            assert_eq!(gather(&sibling)?, vec![expected, -902, -903]);
            assert_eq!(
                gather(&source)?,
                vec![expected, values[2_048], values[last]]
            );
            assert_eq!(gather(&target)?, vec![expected, -900, values[last]]);
            let last_value = values[last];
            drop(source);
            drop(upload);
            drop(values);
            // The new mapping owns its pages independently; freeing/replacing the source must
            // neither retain a chain of source buffers nor expose recycled source storage.
            let replacement = TensorUpload::new(&device).optional(&vec![42_i64; bytes / 8])?;
            assert_eq!(gather(&target)?, vec![expected, -900, last_value]);
            drop(target);
            assert_eq!(gather(&child)?, vec![expected, -900, -901]);
            assert_eq!(gather(&sibling)?, vec![expected, -902, -903]);
            drop(replacement);
            eprintln!("Metal COW: source_bytes={bytes}, resident_growth_bytes={added}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "measures process RSS; run alone on a physical Metal device"]
    fn metal_cow_replacement_churn_keeps_pinned_pages_bounded() -> Result<()> {
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        for bytes in [4 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024] {
            let values = (0..bytes / 8)
                .map(|row| (row as i64).wrapping_mul(7919))
                .collect::<Vec<_>>();
            let mut upload = TensorUpload::new(&device);
            upload.immutable_properties = true;
            let source = upload
                .optional(&values)?
                .ok_or_else(|| Error::internal("missing churn fixture"))?;
            let pinned = patch(&source, &[(2_048, -900_i64)], &device)?;
            drop(patch(&pinned, &[(2_048, -899_i64)], &device)?);
            let before = resident_bytes()?;
            let mut head = pinned.clone();
            for generation in 0..96 {
                head = patch(&head, &[(2_048, -1000_i64 - generation)], &device)?;
            }
            let added = resident_bytes()?.saturating_sub(before);
            assert!(
                added < 1024 * 1024,
                "replacement churn retained {added} bytes from a {bytes}-byte buffer"
            );
            let last = values.len() - 1;
            let selected = Tensor::from_slice(&[10_u32, 2_048, last as u32], 3, &device)
                .map_err(candle_error)?;
            let gather = |tensor: &Tensor| {
                tensor
                    .index_select(&selected, 0)
                    .and_then(|value| value.to_vec1::<i64>())
                    .map_err(candle_error)
            };
            assert_eq!(gather(&source)?, [values[10], values[2_048], values[last]]);
            assert_eq!(gather(&pinned)?, [values[10], -900, values[last]]);
            assert_eq!(gather(&head)?, [values[10], -1095, values[last]]);
            eprintln!("Metal COW churn: source_bytes={bytes}, resident_growth_bytes={added}");
        }
        Ok(())
    }

    #[test]
    fn metal_sparse_growth_crosses_vm_chunks_and_keeps_siblings_isolated() -> Result<()> {
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        let initial = zeros(candle_core::DType::I64, 4, &device)?;
        let source = patch(&initial, &[(0, 11_i64), (3, 22_i64)], &device)?;
        let maximum = device
            .as_metal_device()
            .map_err(candle_error)?
            .metal_device()
            .as_ref()
            .maxBufferLength();
        for bytes in [128_usize * 1024 * 1024, 4_usize * 1024 * 1024 * 1024] {
            let bytes = bytes + page_bytes();
            if bytes > maximum {
                continue;
            }
            let last = bytes / 8 - 1;
            let before = resident_bytes()?;
            let first = extend(&source, &[(last, 33_i64)], last + 1, &device)?;
            let second = extend(&source, &[(last, 44_i64)], last + 1, &device)?;
            let added = resident_bytes()?.saturating_sub(before);
            assert!(added < 1024 * 1024, "sparse growth allocated {added} bytes");
            let selected =
                Tensor::from_slice(&[0_u32, last as u32], 2, &device).map_err(candle_error)?;
            let gather = |tensor: &Tensor| {
                tensor
                    .index_select(&selected, 0)
                    .and_then(|value| value.to_vec1::<i64>())
                    .map_err(candle_error)
            };
            assert_eq!(gather(&first)?, vec![11, 33]);
            assert_eq!(gather(&second)?, vec![11, 44]);
            assert_eq!(
                source.to_vec1::<i64>().map_err(candle_error)?,
                [11, 0, 0, 22]
            );
            eprintln!("Metal sparse growth: bytes={bytes}, resident_growth_bytes={added}");
        }
        Ok(())
    }

    #[test]
    fn lineage_updates_detach_bounded_paths_and_preserve_pinned_charges() {
        let mut writes = WritePages::default();
        for root in 1..=256 {
            writes.roots.insert(
                root,
                RootWrite {
                    initial_bytes: 16 * 1024 * 1024,
                    exposed_bytes: 16 * 1024 * 1024,
                    dirty_pages: BTreeSet::from([0, 1023]),
                    ..RootWrite::default()
                },
            );
            for page in [0, 1023] {
                writes.host_pages.insert(
                    (root, page),
                    HostPageWrite {
                        bytes: 256 * 1024,
                        retired: true,
                    },
                );
            }
        }
        let mut branch = BranchPages::default();
        branch.apply(&writes);
        assert_eq!(branch.roots.len(), 256);
        assert_eq!(branch.pages.len(), 512);
        assert_eq!(branch.host_pages.len(), 512);
        let pinned = branch.clone();
        let before = pinned.bytes();
        let mut changed = WritePages::default();
        changed.roots.insert(
            129,
            RootWrite {
                initial_bytes: 16 * 1024 * 1024,
                exposed_bytes: 16 * 1024 * 1024 + page_bytes(),
                dirty_pages: BTreeSet::from([0]),
                ..RootWrite::default()
            },
        );
        changed.host_pages.insert(
            (129, 0),
            HostPageWrite {
                bytes: 512 * 1024,
                retired: true,
            },
        );
        branch.apply(&changed);
        assert_eq!(pinned.bytes(), before);
        assert_eq!(branch.bytes(), before + page_bytes() + 256 * 1024);
        assert!(branch.roots.changed_path_nodes(&pinned.roots).len() <= 4);
        assert!(
            branch
                .host_pages
                .changed_path_nodes(&pinned.host_pages)
                .len()
                <= 4
        );
        assert!(branch.pages.changed_path_nodes(&pinned.pages).is_empty());
        branch.apply(&changed);
        assert_eq!(branch.bytes(), before + page_bytes() + 256 * 1024);
    }

    #[test]
    fn surgical_native_lineage_unions_pages_growth_and_new_roots() -> Result<()> {
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        let width = page_bytes() / 8;
        let values = (0..512 * width)
            .map(|row| (row as i64).wrapping_mul(7919))
            .collect::<Vec<_>>();
        let mut upload = TensorUpload::new(&device);
        upload.immutable_properties = true;
        let original = upload
            .optional(&values)?
            .ok_or_else(|| Error::internal("missing lineage fixture"))?;
        let capture = Capture::begin()?;
        let mut current = patch(&original, &[(3, -1_i64), (4, -2), (width + 1, -3)], &device)?;
        let writes = capture.finish()?;
        assert_eq!(writes.retired_bytes(), 2 * page_bytes());
        let (births, changes) = writes.retirement_events();
        assert!(births.is_empty());
        let root = changes[0].0[1];
        assert_eq!(
            changes,
            vec![
                ([0, root, 0, 0, 0, 0], page_bytes()),
                ([0, root, 0, 0, 0, 1], page_bytes())
            ]
        );
        let mut branch = BranchPages::default();
        branch.apply(&writes);
        let retained = branch.bytes();
        for revision in 0_i64..64 {
            let capture = Capture::begin()?;
            current = patch(&current, &[(3, revision)], &device)?;
            branch.apply(&capture.finish()?);
            assert_eq!(
                branch.bytes(),
                retained,
                "repeated private page charge must plateau"
            );
        }
        let len = current.elem_count();
        let capture = Capture::begin()?;
        current = extend(&current, &[(len, 93_i64)], len + 1, &device)?;
        let writes = capture.finish()?;
        assert_eq!(
            writes.retired_bytes(),
            0,
            "new growth is not an old retained page"
        );
        assert_eq!(
            writes.retirement_events(),
            (vec![([0, root, 0, 0, 0], 512)], Vec::new())
        );
        branch.apply(&writes);
        assert_eq!(branch.bytes(), retained + page_bytes());
        let exposed = buffer_bytes(&current);
        let narrow = current.narrow(0, 0, 5).map_err(candle_error)?;
        let capture = Capture::begin()?;
        let narrow = patch(&narrow, &[(3, 94_i64)], &device)?;
        let writes = capture.finish()?;
        assert_eq!(
            buffer_bytes(&narrow),
            exposed,
            "narrowing cannot hide owned prefix pages"
        );
        branch.apply(&writes);
        assert_eq!(branch.bytes(), retained + page_bytes());
        let capture = Capture::begin()?;
        let fresh = zeros(candle_core::DType::I64, width * 64, &device)?;
        let fresh = patch(&fresh, &[(0, 95_i64)], &device)?;
        let writes = capture.finish()?;
        assert_eq!(writes.retired_bytes(), 0);
        let (fresh_births, fresh_changes) = writes.retirement_events();
        assert_eq!(fresh_births.len(), 1);
        assert_ne!(fresh_births[0].0[1], root);
        assert_eq!(fresh_births[0].1, 0);
        assert!(fresh_changes.is_empty());
        branch.apply(&writes);
        let with_fresh = branch.bytes();
        assert_eq!(
            with_fresh,
            retained + page_bytes() + buffer_bytes(&fresh) + 4096
        );
        let capture = Capture::begin()?;
        let _updated = patch(&fresh, &[(0, 96_i64)], &device)?;
        branch.apply(&capture.finish()?);
        assert_eq!(branch.bytes(), with_fresh);
        let capture = Capture::begin()?;
        retire(&original)?;
        let retired = capture.finish()?;
        assert_eq!(retired.retired_bytes(), buffer_bytes(&original));
        assert_eq!(
            retired.retirement_events(),
            (
                Vec::new(),
                vec![([0, root, 0, 0, 0, u64::MAX], buffer_bytes(&original))]
            )
        );
        let capture = Capture::begin()?;
        assert!(patch(&original, &[(original.elem_count(), 1_i64)], &device).is_err());
        drop(capture);
        assert_eq!(Capture::begin()?.finish()?.retired_bytes(), 0);
        assert_eq!(
            original
                .narrow(0, 0, 5)
                .and_then(|v| v.to_vec1::<i64>())
                .map_err(candle_error)?,
            values[..5]
        );
        Ok(())
    }

    #[test]
    fn surgical_host_page_lineage_charges_distinct_roots_and_rolls_back() -> Result<()> {
        let first = new_host_root()?;
        let second = new_host_root()?;
        assert_ne!(first, second);
        let capture = Capture::begin()?;
        record_host_pages(first, 100, 100, [0, 0, 1].into_iter(), 64 * 1024);
        record_host_pages(second, 100, 100, [0].into_iter(), 64 * 1024);
        let writes = capture.finish()?;
        assert_eq!(writes.native_retired_bytes(), 0);
        assert_eq!(writes.retired_bytes(), 128 * 1024);
        assert_eq!(
            writes.retirement_events(),
            (
                Vec::new(),
                vec![
                    ([1, first, 0, 0, 0, 0], 64 * 1024),
                    ([1, first, 0, 0, 0, 1], 0),
                    ([1, second, 0, 0, 0, 0], 64 * 1024),
                ]
            )
        );
        let mut branch = BranchPages::default();
        branch.apply(&writes);
        assert_eq!(branch.bytes(), 3 * (64 * 1024 + 4096));
        let pinned = branch.clone();
        branch.apply(&writes);
        assert_eq!(branch.bytes(), pinned.bytes());
        let capture = Capture::begin()?;
        record_host_pages(first, 300, 100, [2].into_iter(), 64 * 1024);
        branch.apply(&capture.finish()?);
        assert_eq!(branch.bytes(), pinned.bytes() + 64 * 1024 + 4096);
        let abandoned = Capture::begin()?;
        record_host_pages(first, 300, 100, [0].into_iter(), 64 * 1024);
        drop(abandoned);
        assert_eq!(Capture::begin()?.finish()?.retired_bytes(), 0);
        Ok(())
    }

    #[test]
    fn surgical_metal_growth_keeps_dirty_prefixes_and_pinned_generations() -> Result<()> {
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        for bytes in [4 * 1024 * 1024, 256 * 1024 * 1024] {
            let values = (0..bytes / 8)
                .map(|row| (row as i64).wrapping_mul(0x517c_c1b7_2722_0a95))
                .collect::<Vec<_>>();
            let mut upload = TensorUpload::new(&device);
            upload.immutable_properties = true;
            let original = upload
                .optional(&values)?
                .ok_or_else(|| Error::internal("missing growth fixture"))?;
            let len = original.elem_count();
            let narrowed = original.narrow(0, 0, 1_024).map_err(candle_error)?;
            let capture = Capture::begin()?;
            let unchanged = extend::<i64>(&narrowed, &[], len, &device)?;
            assert!(
                capture.finish()?.roots.is_empty(),
                "a view extension with no byte edits must not allocate or fork pages"
            );
            let before = resident_bytes()?;
            let first = extend(&unchanged, &[(len, 91_i64)], len + 1, &device)?;
            let sibling = extend(&unchanged, &[(len, -91_i64)], len + 1, &device)?;
            let second = extend(
                &first,
                &[(len, 92_i64), (len + 1, 93_i64), (3, -42_i64)],
                len + 2,
                &device,
            )?;
            let added = resident_bytes()?.saturating_sub(before);
            assert!(
                added < 2 * 1024 * 1024,
                "growth copied {added} bytes of a {bytes} byte column"
            );
            let read = |tensor: &Tensor, rows: &[u32]| -> Result<Vec<i64>> {
                let indices =
                    Tensor::from_slice(rows, rows.len(), &device).map_err(candle_error)?;
                tensor
                    .index_select(&indices, 0)
                    .and_then(|values| values.to_vec1::<i64>())
                    .map_err(candle_error)
            };
            assert_eq!(
                read(&original, &[3, (len - 1) as u32])?,
                vec![values[3], values[len - 1]]
            );
            assert_eq!(read(&first, &[3, len as u32])?, vec![values[3], 91]);
            assert_eq!(read(&sibling, &[3, len as u32])?, vec![values[3], -91]);
            assert_eq!(read(&unchanged, &[3])?, vec![values[3]]);
            assert_eq!(
                read(&second, &[3, len as u32, (len + 1) as u32])?,
                vec![-42, 92, 93]
            );
            drop(first);
            drop(original);
            drop(narrowed);
            drop(upload);
            assert_eq!(read(&unchanged, &[3])?, vec![values[3]]);
            assert_eq!(read(&sibling, &[len as u32])?, vec![-91]);
            assert_eq!(
                read(&second, &[3, (len - 1) as u32, (len + 1) as u32])?,
                vec![-42, values[len - 1], 93]
            );
            eprintln!("surgical growth: dirty_bytes={bytes}, private_resident_bytes={added}");
        }
        Ok(())
    }
}
