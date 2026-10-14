//! Bekko a8m's four-layer ModernBERT encoder. Padding is masked in every attention layer.
use crate::{Error, ErrorCode, Result};
use candle_core::{D, DType, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm_no_bias, linear_no_bias,
};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub(super) struct EncoderConfig {
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    max_position_embeddings: usize,
    layer_norm_eps: f64,
    global_attn_every_n_layers: usize,
    global_rope_theta: f64,
    local_attention: usize,
    local_rope_theta: f64,
    hidden_activation: String,
    pad_token_id: u32,
    attention_bias: bool,
    mlp_bias: bool,
    norm_bias: bool,
}
impl EncoderConfig {
    pub(super) fn validate(&self, maximum_input_tokens: usize) -> Result<()> {
        if self.hidden_size != 384
            || self.intermediate_size != 1152
            || self.vocab_size != 256_000
            || self.num_hidden_layers != 4
            || self.num_attention_heads != 6
            || self.max_position_embeddings != 8192
            || maximum_input_tokens != self.max_position_embeddings
            || self.layer_norm_eps != 1e-5
            || self.global_attn_every_n_layers != 3
            || self.global_rope_theta != 160_000.
            || self.local_rope_theta != 160_000.
            || self.local_attention != 128
            || self.hidden_activation != "gelu"
            || self.pad_token_id != 0
            || self.attention_bias
            || self.mlp_bias
            || self.norm_bias
        {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "embedding configuration is not the Bekko a8m architecture",
            ));
        }
        Ok(())
    }
    pub(super) fn vocabulary_size(&self) -> usize {
        self.vocab_size
    }
}

struct Layer {
    qkv: Linear,
    attention_out: Linear,
    wi: Linear,
    wo: Linear,
    attention_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
}
pub(super) struct Encoder {
    embeddings: Embedding,
    embedding_norm: LayerNorm,
    final_norm: LayerNorm,
    layers: Vec<Layer>,
    rope_sin: Tensor,
    rope_cos: Tensor,
    heads: usize,
    hidden: usize,
    global_every: usize,
    local_window: usize,
    maximum_tokens: usize,
}
impl Encoder {
    pub(super) fn load(vb: VarBuilder<'static>, config: &EncoderConfig) -> CandleResult<Self> {
        let hidden = config.hidden_size;
        let prefix = &vb;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let layer = prefix.pp(format!("layers.{index}"));
            layers.push(Layer {
                qkv: linear_no_bias(hidden, hidden * 3, layer.pp("attn.Wqkv"))?,
                attention_out: linear_no_bias(hidden, hidden, layer.pp("attn.Wo"))?,
                wi: linear_no_bias(hidden, config.intermediate_size * 2, layer.pp("mlp.Wi"))?,
                wo: linear_no_bias(config.intermediate_size, hidden, layer.pp("mlp.Wo"))?,
                attention_norm: if index == 0 {
                    None
                } else {
                    Some(layer_norm_no_bias(
                        hidden,
                        config.layer_norm_eps,
                        layer.pp("attn_norm"),
                    )?)
                },
                mlp_norm: layer_norm_no_bias(hidden, config.layer_norm_eps, layer.pp("mlp_norm"))?,
            });
        }
        // Bekko uses the same rotary frequency for local and global attention.
        let head_size = hidden / config.num_attention_heads;
        let mut sin = Vec::with_capacity(config.max_position_embeddings * head_size / 2);
        let mut cos = Vec::with_capacity(sin.capacity());
        for position in 0..config.max_position_embeddings {
            for channel in (0..head_size).step_by(2) {
                let angle = position as f32
                    / config
                        .global_rope_theta
                        .powf(channel as f64 / head_size as f64) as f32;
                sin.push(angle.sin());
                cos.push(angle.cos());
            }
        }
        let shape = (config.max_position_embeddings, head_size / 2);
        Ok(Self {
            embeddings: embedding(
                config.vocab_size,
                hidden,
                prefix.pp("embeddings.tok_embeddings"),
            )?,
            embedding_norm: layer_norm_no_bias(
                hidden,
                config.layer_norm_eps,
                prefix.pp("embeddings.norm"),
            )?,
            final_norm: layer_norm_no_bias(hidden, config.layer_norm_eps, prefix.pp("final_norm"))?,
            rope_sin: Tensor::from_vec(sin, shape, vb.device())?.to_dtype(vb.dtype())?,
            rope_cos: Tensor::from_vec(cos, shape, vb.device())?.to_dtype(vb.dtype())?,
            layers,
            heads: config.num_attention_heads,
            hidden,
            global_every: config.global_attn_every_n_layers,
            local_window: config.local_attention,
            maximum_tokens: config.max_position_embeddings,
        })
    }
    pub(super) fn forward(&self, ids: &Tensor, attention: &Tensor) -> CandleResult<Tensor> {
        let (batch, sequence) = ids.dims2()?;
        if batch == 0
            || sequence == 0
            || sequence > self.maximum_tokens
            || attention.dims() != ids.dims()
        {
            candle_core::bail!("embedding input and attention mask have invalid dimensions");
        }
        let global = ((1. - attention.to_dtype(DType::F32)?)? * f32::MIN as f64)?
            .unsqueeze(1)?
            .unsqueeze(1)?;
        let local: Vec<f32> = (0..sequence)
            .flat_map(|i| {
                (0..sequence).map(move |j| {
                    if i.abs_diff(j) > self.local_window / 2 {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let local =
            Tensor::from_vec(local, (sequence, sequence), ids.device())?.broadcast_add(&global)?;
        let cos = self.rope_cos.narrow(0, 0, sequence)?.contiguous()?;
        let sin = self.rope_sin.narrow(0, 0, sequence)?.contiguous()?;
        let mut hidden = ids.apply(&self.embeddings)?.apply(&self.embedding_norm)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let residual = hidden.clone();
            let normalized = if let Some(norm) = &layer.attention_norm {
                hidden.apply(norm)?
            } else {
                hidden.clone()
            };
            let qkv = normalized
                .apply(&layer.qkv)?
                .reshape((batch, sequence, 3, self.heads, self.hidden / self.heads))?
                .permute((2, 0, 3, 1, 4))?;
            let q = candle_nn::rotary_emb::rope(&qkv.i(0)?.contiguous()?, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope(&qkv.i(1)?.contiguous()?, &cos, &sin)?;
            let v = qkv.i(2)?.contiguous()?;
            let scores = (q.matmul(&k.t()?)? / (self.hidden as f64 / self.heads as f64).sqrt())?
                .to_dtype(DType::F32)?
                .broadcast_add(if index % self.global_every == 0 {
                    &global
                } else {
                    &local
                })?;
            // Masking and softmax stay FP32, including on FP16 devices.
            let attended = candle_nn::ops::softmax_last_dim(&scores)?
                .to_dtype(v.dtype())?
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((batch, sequence, self.hidden))?
                .apply(&layer.attention_out)?;
            hidden = (residual + attended)?;
            let halves = hidden
                .apply(&layer.mlp_norm)?
                .apply(&layer.wi)?
                .chunk(2, D::Minus1)?;
            let feed_forward = (halves[0].gelu_erf()? * &halves[1])?.apply(&layer.wo)?;
            hidden = (&hidden + feed_forward)?;
        }
        hidden.apply(&self.final_norm)
    }

    pub(super) fn resident_bytes(&self) -> usize {
        let tensor_bytes = |t: &Tensor| t.elem_count().saturating_mul(t.dtype().size_in_bytes());
        let linear_bytes = |l: &Linear| tensor_bytes(l.weight());
        let norm_bytes = |n: &LayerNorm| tensor_bytes(n.weight());
        let mut bytes = tensor_bytes(self.embeddings.embeddings())
            .saturating_add(norm_bytes(&self.embedding_norm))
            .saturating_add(norm_bytes(&self.final_norm))
            .saturating_add(tensor_bytes(&self.rope_sin))
            .saturating_add(tensor_bytes(&self.rope_cos));
        for layer in &self.layers {
            for linear in [&layer.qkv, &layer.attention_out, &layer.wi, &layer.wo] {
                bytes = bytes.saturating_add(linear_bytes(linear));
            }
            bytes = bytes.saturating_add(norm_bytes(&layer.mlp_norm));
            if let Some(norm) = &layer.attention_norm {
                bytes = bytes.saturating_add(norm_bytes(norm));
            }
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn small_encoder() -> CandleResult<Encoder> {
        let config = EncoderConfig {
            hidden_size: 8,
            intermediate_size: 12,
            vocab_size: 32,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            max_position_embeddings: 256,
            layer_norm_eps: 1e-5,
            global_attn_every_n_layers: 3,
            global_rope_theta: 160_000.,
            local_attention: 128,
            local_rope_theta: 160_000.,
            hidden_activation: "gelu".into(),
            pad_token_id: 0,
            attention_bias: false,
            mlp_bias: false,
            norm_bias: false,
        };
        let mut weights = std::collections::HashMap::new();
        let mut put = |name: String, shape: Vec<usize>, ones: bool| -> CandleResult<()> {
            let size = shape.iter().product();
            let values = (0..size)
                .map(|i| {
                    if ones {
                        1.
                    } else {
                        ((i * 17 + name.len()) % 101) as f32 / 100. - 0.5
                    }
                })
                .collect::<Vec<_>>();
            weights.insert(name, Tensor::from_vec(values, shape, &Device::Cpu)?);
            Ok(())
        };
        put(
            "embeddings.tok_embeddings.weight".into(),
            vec![32, 8],
            false,
        )?;
        for name in ["embeddings.norm", "final_norm"] {
            put(format!("{name}.weight"), vec![8], true)?;
        }
        for i in 0..4 {
            for (name, shape) in [
                ("attn.Wqkv", vec![24, 8]),
                ("attn.Wo", vec![8, 8]),
                ("mlp.Wi", vec![24, 8]),
                ("mlp.Wo", vec![8, 12]),
            ] {
                put(format!("layers.{i}.{name}.weight"), shape, false)?;
            }
            put(format!("layers.{i}.mlp_norm.weight"), vec![8], true)?;
            if i > 0 {
                put(format!("layers.{i}.attn_norm.weight"), vec![8], true)?;
            }
        }
        Encoder::load(
            VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu),
            &config,
        )
    }

    #[test]
    fn padding_cannot_change_visible_outputs_in_global_or_local_layers() -> CandleResult<()> {
        let encoder = small_encoder()?;
        let tokens = (0..137).map(|i| (i % 31 + 1) as u32).collect::<Vec<_>>();
        let ids = Tensor::from_slice(&tokens, (1, 137), &Device::Cpu)?;
        let visible = encoder.forward(&ids, &ids.ones_like()?)?;
        // A second full row makes the first row heavily padded past the local window.
        let mut batch = tokens.clone();
        batch.resize(192, 0);
        batch.extend((0..192).map(|i| (i % 31 + 1) as u32));
        let mut mask = vec![1u32; 137];
        mask.resize(192, 0);
        mask.extend(vec![1; 192]);
        let result = encoder.forward(
            &Tensor::from_vec(batch, (2, 192), &Device::Cpu)?,
            &Tensor::from_vec(mask, (2, 192), &Device::Cpu)?,
        )?;
        let actual = result
            .narrow(0, 0, 1)?
            .narrow(1, 0, 137)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = visible.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| a.is_finite() && (a - b).abs() < 1e-5)
        );
        assert!(encoder.resident_bytes() > 0);
        Ok(())
    }
    #[test]
    fn invalid_shapes_return_errors_instead_of_panics() -> CandleResult<()> {
        let encoder = small_encoder()?;
        let ids = Tensor::zeros((1, 257), DType::U32, &Device::Cpu)?;
        assert!(encoder.forward(&ids, &ids.ones_like()?).is_err());
        let ids = Tensor::zeros((1, 2), DType::U32, &Device::Cpu)?;
        assert!(
            encoder
                .forward(&ids, &Tensor::zeros((1, 1), DType::U32, &Device::Cpu)?)
                .is_err()
        );
        Ok(())
    }
}
