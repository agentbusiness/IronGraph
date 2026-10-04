//! Persistent reusable payload slots; descriptor publication touches only addressed rows.

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
use super::*;

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
#[derive(Clone, Copy, Default)]
struct Span {
    begin: u32,
    capacity: u32,
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
#[derive(Clone)]
struct FreeSpan {
    begin: u32,
    next: Option<Arc<Self>>,
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
impl Drop for FreeSpan {
    fn drop(&mut self) {
        // A large generation can own many obsolete spans. Release an unshared suffix
        // iteratively so eviction cannot recurse once per previously changed row.
        let mut tail = self.next.take();
        while let Some(next) = tail {
            match Arc::try_unwrap(next) {
                Ok(mut unique) => tail = unique.next.take(),
                Err(_) => break,
            }
        }
    }
}

/// Cold row spans are discovered only when first changed. Subsequent generations retain a
/// bounded radix path per changed descriptor and share unchanged free-list suffixes.
#[derive(Clone, Default)]
pub(super) struct Allocator {
    #[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
    spans: PersistentMap<Span>,
    #[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
    free: [Option<Arc<FreeSpan>>; 32],
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
impl Allocator {
    fn release(&mut self, span: Span) {
        let mut begin = span.begin;
        let mut remaining = span.capacity;
        // Cold spans need not be powers of two. Split one obsolete range into at most 32
        // reusable pieces rather than leaking the remainder or repacking neighboring rows.
        while remaining != 0 {
            let class = 31 - remaining.leading_zeros() as usize;
            let capacity = 1_u32 << class;
            self.free[class] = Some(Arc::new(FreeSpan {
                begin,
                next: self.free[class].take(),
            }));
            begin += capacity;
            remaining -= capacity;
        }
    }

    fn allocate(&mut self, length: u32, end: &mut usize) -> Result<Span> {
        if length == 0 {
            return Ok(Span::default());
        }
        let capacity = length.checked_next_power_of_two().ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "property payload slot exceeds u32",
            )
        })?;
        let class = capacity.trailing_zeros() as usize;
        // Preserve free larger slots for later larger replacements. Exact-class reuse makes
        // alternating grow/shrink cycles retain a bounded geometric set of reusable slots.
        if let Some(free) = self.free[class].take() {
            self.free[class] = free.next.clone();
            return Ok(Span {
                begin: free.begin,
                capacity,
            });
        }
        let begin = checked_u32(*end, "property payload start")?;
        *end = end
            .checked_add(capacity as usize)
            .filter(|end| *end <= u32::MAX as usize)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "property payload arena exceeds u32",
                )
            })?;
        Ok(Span { begin, capacity })
    }

    /// Updates independent `[start,end]` pairs and reusable payload slots in a private native
    /// generation. Callers publish the returned allocator only after every tensor patch succeeds.
    pub(super) fn patch(
        &self,
        offsets: &Tensor,
        bytes: &Option<Tensor>,
        row_count: usize,
        replacements: &[(usize, Vec<u8>)],
        device: &Device,
    ) -> Result<(Self, Tensor, Option<Tensor>)> {
        let descriptor_len = row_count.checked_mul(2).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "property span count overflow",
            )
        })?;
        if !offsets.elem_count().is_multiple_of(2) || offsets.elem_count() > descriptor_len {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "property span domain changed incompatibly",
            ));
        }
        let old = metal_shared_tensor_prefix::<u32>(offsets, offsets.elem_count())?;
        let old = old.as_slice();
        let mut next = self.clone();
        let mut end = bytes.as_ref().map_or(0, Tensor::elem_count);
        let mut descriptor_updates = Vec::with_capacity(replacements.len().saturating_mul(2));
        let mut value_updates = Vec::new();
        for (row, payload) in replacements {
            if *row >= row_count {
                return Err(Error::invalid_data("property payload row is out of bounds"));
            }
            let length = checked_u32(payload.len(), "property payload width")?;
            let previous = match next.spans.get(*row as u128).copied() {
                Some(span) => span,
                None => {
                    let begin = old.get(row * 2).copied().unwrap_or(0);
                    let finish = old.get(row * 2 + 1).copied().unwrap_or(0);
                    if begin > finish || finish as usize > end {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "property payload descriptor is outside the arena",
                        ));
                    }
                    Span {
                        begin,
                        capacity: finish - begin,
                    }
                }
            };
            let slot =
                if length != 0 && length <= previous.capacity && length > previous.capacity / 4 {
                    previous
                } else {
                    next.release(previous);
                    next.allocate(length, &mut end)?
                };
            next.spans.insert_cow(*row as u128, slot);
            descriptor_updates.push((row * 2, slot.begin));
            descriptor_updates.push((
                row * 2 + 1,
                slot.begin
                    .checked_add(length)
                    .ok_or_else(|| Error::internal("property payload span overflow"))?,
            ));
            value_updates.extend(
                payload
                    .iter()
                    .enumerate()
                    .map(|(offset, value)| (slot.begin as usize + offset, *value)),
            );
        }
        let offsets = if descriptor_updates.is_empty() && offsets.elem_count() == descriptor_len {
            offsets.clone()
        } else {
            metal_pages::extend(offsets, &descriptor_updates, descriptor_len, device)?
        };
        let bytes = if value_updates.is_empty() {
            bytes.clone()
        } else {
            let source = match bytes {
                Some(tensor) => tensor.clone(),
                None => metal_pages::zeros(DType::U8, 0, device)?,
            };
            Some(metal_pages::extend(&source, &value_updates, end, device)?)
        };
        Ok((next, offsets, bytes))
    }
}

#[cfg(all(test, feature = "accelerator", target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn surgical_variable_free_list_drop_preserves_shared_suffix_without_recursion() {
        let pinned = Arc::new(FreeSpan {
            begin: 7,
            next: None,
        });
        let mut tail = Some(Arc::clone(&pinned));
        for begin in 0..100_000 {
            tail = Some(Arc::new(FreeSpan { begin, next: tail }));
        }
        drop(tail);
        assert_eq!(pinned.begin, 7);
        assert_eq!(Arc::strong_count(&pinned), 1);
    }

    fn row(offsets: &Tensor, bytes: &Option<Tensor>, row: usize) -> Result<Vec<u8>> {
        let bounds = offsets
            .narrow(0, row * 2, 2)
            .and_then(|tensor| tensor.to_vec1::<u32>())
            .map_err(candle_error)?;
        let length = (bounds[1] - bounds[0]) as usize;
        if length == 0 {
            return Ok(Vec::new());
        }
        bytes
            .as_ref()
            .ok_or_else(|| Error::internal("payload bytes missing"))?
            .narrow(0, bounds[0] as usize, length)
            .and_then(|tensor| tensor.to_vec1::<u8>())
            .map_err(candle_error)
    }

    #[test]
    fn surgical_variable_arena_reuses_slots_and_preserves_native_pinned_rows() -> Result<()> {
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        for count in [4_096, 32_768] {
            let offsets = (0..count)
                .flat_map(|row| [(row * 257) as u32, ((row + 1) * 257) as u32])
                .collect::<Vec<_>>();
            let values = (0..count * 257)
                .map(|byte| ((byte * 17 + byte / 257) % 251) as u8)
                .collect::<Vec<_>>();
            let mut upload = TensorUpload::new(&device);
            upload.immutable_properties = true;
            let mut offsets = upload.required(&offsets)?;
            let mut bytes = upload.optional(&values)?;
            let pinned_offsets = offsets.clone();
            let pinned_bytes = bytes.clone();
            let original = row(&offsets, &bytes, 7)?;
            let mut allocator = Allocator::default();
            let replacement = (0..65_539)
                .map(|byte| (byte % 251) as u8)
                .collect::<Vec<_>>();
            (allocator, offsets, bytes) = allocator.patch(
                &offsets,
                &bytes,
                count,
                &[(7, replacement.clone())],
                &device,
            )?;
            assert_eq!(row(&offsets, &bytes, 7)?, replacement);
            assert_eq!(row(&pinned_offsets, &pinned_bytes, 7)?, original);
            assert_eq!(row(&offsets, &bytes, 8)?, values[8 * 257..9 * 257]);
            let maximum = bytes.as_ref().map_or(0, Tensor::elem_count);
            for pass in 0..64 {
                let width = [1_027, 0, 65_539, 7_111][pass % 4];
                let expected = vec![(pass % 251) as u8; width];
                (allocator, offsets, bytes) =
                    allocator.patch(&offsets, &bytes, count, &[(7, expected.clone())], &device)?;
                assert_eq!(row(&offsets, &bytes, 7)?, expected);
                assert!(
                    bytes.as_ref().map_or(0, Tensor::elem_count) <= maximum + 131_072,
                    "replacement history increased arena after pass {pass}"
                );
                assert_eq!(row(&pinned_offsets, &pinned_bytes, 7)?, original);
            }
            let before = row(&offsets, &bytes, 7)?;
            assert!(
                allocator
                    .patch(&offsets, &bytes, count, &[(count, vec![1])], &device)
                    .is_err()
            );
            assert!(
                allocator
                    .patch(&offsets, &bytes, usize::MAX, &[], &device)
                    .is_err()
            );
            assert_eq!(row(&offsets, &bytes, 7)?, before);
            eprintln!(
                "surgical native variable arena: rows={count}, initial_bytes={}, retained_arena_bytes={}",
                values.len(),
                bytes.as_ref().map_or(0, Tensor::elem_count)
            );
        }
        Ok(())
    }
}
