//! Memory mapping and parsing for locally installed model weights and derived caches.
use candle_core::{DType, Device, Error, Result, Shape, Tensor, safetensors::Load};
use candle_nn::{VarBuilder, var_builder::SimpleBackend};
use memmap2::Mmap;
use safetensors::{Dtype as SafeDType, SafeTensors, tensor::TensorView};
use std::{collections::BTreeMap, fs::File, ops::Range, path::Path};

/// Maps a regular file read-only. Callers own the file and must never truncate a live mapping.
///
/// # Errors
/// Returns an error if the file is absent, is not regular, or cannot be mapped.
pub fn map_owned_file(path: &Path) -> Result<Mmap> {
    let file = File::open(path).map_err(|error| Error::from(error).with_path(path))?;
    if !file
        .metadata()
        .map_err(|error| Error::from(error).with_path(path))?
        .is_file()
    {
        return Err(artifact_error(path, "mapped file is not regular"));
    }
    ffi::map_readonly(&file).map_err(|error| Error::from(error).with_path(path))
}

/// Loads tensors from a completed local safetensors file without checksumming its content.
/// The returned backend retains the mapping while tensors are loaded on the requested device.
///
/// # Errors
/// Returns an error for unreadable files, invalid safetensors metadata or invalid tensor ranges.
pub fn mmap_safetensors(path: &Path, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
    safetensors_builder(map_owned_file(path)?, path, dtype, device)
}

fn safetensors_builder(
    mapping: Mmap,
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let parsed =
        SafeTensors::deserialize(&mapping).map_err(|error| Error::from(error).with_path(path))?;
    let mut index = BTreeMap::new();
    let base = mapping.as_ptr() as usize;
    for (name, view) in parsed.tensors() {
        let start = (view.data().as_ptr() as usize)
            .checked_sub(base)
            .ok_or_else(|| artifact_error(path, "tensor data precedes its mapping"))?;
        let end = start
            .checked_add(view.data().len())
            .filter(|end| *end <= mapping.len())
            .ok_or_else(|| artifact_error(path, "tensor data exceeds its mapping"))?;
        if index
            .insert(
                name,
                TensorIndex {
                    dtype: view.dtype(),
                    shape: view.shape().to_vec(),
                    bytes: start..end,
                },
            )
            .is_some()
        {
            return Err(artifact_error(path, "safetensors field name is duplicated"));
        }
    }
    drop(parsed);
    let backend = SafetensorsBackend { mapping, index };
    Ok(VarBuilder::from_backend(
        Box::new(backend),
        dtype,
        device.clone(),
    ))
}

struct SafetensorsBackend {
    mapping: Mmap,
    index: BTreeMap<String, TensorIndex>,
}

struct TensorIndex {
    dtype: SafeDType,
    shape: Vec<usize>,
    bytes: Range<usize>,
}

impl SafetensorsBackend {
    fn load(&self, name: &str, device: &Device) -> Result<Tensor> {
        let index = self
            .index
            .get(name)
            .ok_or_else(|| Error::Msg(format!("safetensors field is absent: {name}")))?;
        let bytes = self
            .mapping
            .get(index.bytes.clone())
            .ok_or_else(|| Error::Msg(format!("safetensors field range is invalid: {name}")))?;
        TensorView::new(index.dtype, index.shape.clone(), bytes)?.load(device)
    }
}

impl SimpleBackend for SafetensorsBackend {
    fn get(
        &self,
        expected_shape: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let tensor = self.load(name, device)?.to_dtype(dtype)?;
        if tensor.shape() != &expected_shape {
            return Err(Error::Msg(format!(
                "shape mismatch for safetensors field {name}: expected {expected_shape:?}, got {:?}",
                tensor.shape()
            )));
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        self.load(name, device)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
}

fn artifact_error(path: &Path, message: &str) -> Error {
    Error::Msg(format!("{message}: {}", path.display()))
}

mod ffi {
    #![allow(unsafe_code)]
    use memmap2::{Mmap, MmapOptions};
    use std::{fs::File, io};
    pub fn map_readonly(file: &File) -> io::Result<Mmap> {
        // SAFETY: callers own installed artifacts or cache files. Published artifacts are never
        // modified in place; cache compaction replaces files rather than truncating live mappings.
        unsafe { MmapOptions::new().map(file) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::serialize;
    #[test]
    fn mapped_tensors_preserve_shape_and_convert_to_requested_dtype() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model.safetensors");
        let values = [1.0f32, -2.0, 3.5, 4.0];
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let tensor = TensorView::new(SafeDType::F32, vec![2, 2], &bytes)?;
        let encoded = serialize([("weights", tensor)], None)?;
        std::fs::write(&path, &encoded)?;
        let builder = mmap_safetensors(&path, DType::F64, &Device::Cpu)?;
        assert_eq!(
            builder
                .get((2, 2), "weights")?
                .flatten_all()?
                .to_vec1::<f64>()?,
            vec![1., -2., 3.5, 4.]
        );
        assert!(builder.get((4, 1), "weights").is_err());
        assert!(builder.get((2, 2), "missing").is_err());
        assert!(builder.contains_tensor("weights"));
        assert!(!builder.contains_tensor("absent"));
        assert_eq!(&*map_owned_file(&path)?, encoded);
        Ok(())
    }

    #[test]
    fn malformed_or_absent_artifacts_return_errors() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model.safetensors");
        assert!(mmap_safetensors(&path, DType::F32, &Device::Cpu).is_err());
        std::fs::write(&path, b"invalid safetensors header")?;
        assert!(mmap_safetensors(&path, DType::F32, &Device::Cpu).is_err());
        std::fs::write(&path, [])?;
        assert!(mmap_safetensors(&path, DType::F32, &Device::Cpu).is_err());
        assert!(map_owned_file(directory.path()).is_err());
        Ok(())
    }
}
