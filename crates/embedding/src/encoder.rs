//! Owned bidirectional encoder used by the pinned retrieval model.

use std::f32::consts::PI;

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};
use serde::Deserialize;

use crate::{Error, ErrorCode, Result};

use super::embedding::{PINNED_HIDDEN_SIZE, PINNED_MAXIMUM_INPUT_TOKENS};

#[derive(Clone, Debug, Deserialize)]
pub(super) struct BidirectionalEncoderConfig {
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f32,
    max_position_embeddings: usize,
    hidden_act: String,
    attention_bias: bool,
    mlp_bias: bool,
    pooling: String,
    use_bidirectional_attention: bool,
    tie_word_embeddings: bool,
    rope_scaling: RopeScaling,
}

#[derive(Clone, Debug, Deserialize)]
struct RopeScaling {
    factor: f32,
    high_freq_factor: f32,
    low_freq_factor: f32,
    original_max_position_embeddings: usize,
    rope_type: String,
}

impl BidirectionalEncoderConfig {
    pub(super) fn validate_pinned(&self, maximum_input_tokens: usize) -> Result<()> {
        let exact_shape = self.hidden_size == PINNED_HIDDEN_SIZE
            && self.intermediate_size == 8_192
            && self.vocab_size == 128_256
            && self.num_hidden_layers == 16
            && self.num_attention_heads == 32
            && self.num_key_value_heads == 8
            && self.head_dim == 64
            && self.hidden_act == "silu"
            && !self.attention_bias
            && !self.mlp_bias
            && self.pooling == "avg"
            && self.use_bidirectional_attention
            && self.tie_word_embeddings
            && self.rms_norm_eps.to_bits() == 1e-5_f64.to_bits()
            && self.rope_theta.to_bits() == 500_000_f32.to_bits()
            && self.rope_scaling.factor.to_bits() == 32_f32.to_bits()
            && self.rope_scaling.high_freq_factor.to_bits() == 4_f32.to_bits()
            && self.rope_scaling.low_freq_factor.to_bits() == 1_f32.to_bits()
            && self.rope_scaling.original_max_position_embeddings == 8_192
            && self.rope_scaling.rope_type == "llama3"
            && maximum_input_tokens == PINNED_MAXIMUM_INPUT_TOKENS
            && maximum_input_tokens <= self.max_position_embeddings;
        if !exact_shape {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "embedding configuration differs from the pinned encoder contract",
            ));
        }
        Ok(())
    }
}

pub(super) struct BidirectionalEncoder {
    token_embedding: Embedding,
    layers: Vec<EncoderLayer>,
    final_norm: RmsNorm,
    rotary: RotaryCache,
}

impl BidirectionalEncoder {
    pub(super) fn load(
        builder: VarBuilder<'static>,
        config: &BidirectionalEncoderConfig,
        maximum_input_tokens: usize,
    ) -> CandleResult<Self> {
        let token_embedding = embedding(
            config.vocab_size,
            config.hidden_size,
            builder.pp("embed_tokens"),
        )?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| EncoderLayer::load(builder.pp(format!("layers.{index}")), config))
            .collect::<CandleResult<Vec<_>>>()?;
        let final_norm = rms_norm(config.hidden_size, config.rms_norm_eps, builder.pp("norm"))?;
        let rotary = RotaryCache::new(
            config,
            maximum_input_tokens,
            builder.dtype(),
            builder.device(),
        )?;
        Ok(Self {
            token_embedding,
            layers,
            final_norm,
            rotary,
        })
    }

    pub(super) fn forward(&self, token_ids: &Tensor) -> CandleResult<Tensor> {
        let (_, sequence_length) = token_ids.dims2()?;
        if sequence_length == 0 || sequence_length > self.rotary.maximum_sequence_length {
            candle_core::bail!("embedding token sequence exceeds the encoder capacity")
        }
        let mut hidden = self.token_embedding.forward(token_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &self.rotary)?;
        }
        self.final_norm.forward(&hidden)
    }

    /// Resident tensor bytes measured from the allocations actually constructed on the selected
    /// device. This includes the immutable rotary tables because they share the same lifetime and
    /// physical memory budget as the embedding weights.
    pub(super) fn resident_bytes(&self) -> usize {
        let layers = self.layers.iter().fold(0_usize, |total, layer| {
            total.saturating_add(layer.resident_bytes())
        });
        tensor_bytes(self.token_embedding.embeddings())
            .saturating_add(layers)
            .saturating_add(tensor_bytes(self.final_norm.weight()))
            .saturating_add(self.rotary.resident_bytes())
    }
}

struct EncoderLayer {
    input_norm: RmsNorm,
    attention: BidirectionalAttention,
    post_attention_norm: RmsNorm,
    mlp: GatedMlp,
}

impl EncoderLayer {
    fn load(
        builder: VarBuilder<'static>,
        config: &BidirectionalEncoderConfig,
    ) -> CandleResult<Self> {
        Ok(Self {
            input_norm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                builder.pp("input_layernorm"),
            )?,
            attention: BidirectionalAttention::load(builder.pp("self_attn"), config)?,
            post_attention_norm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                builder.pp("post_attention_layernorm"),
            )?,
            mlp: GatedMlp::load(builder.pp("mlp"), config)?,
        })
    }

    fn forward(&self, hidden: &Tensor, rotary: &RotaryCache) -> CandleResult<Tensor> {
        let attention = self
            .attention
            .forward(&self.input_norm.forward(hidden)?, rotary)?;
        let hidden = (hidden + attention)?;
        let feed_forward = self
            .mlp
            .forward(&self.post_attention_norm.forward(&hidden)?)?;
        hidden + feed_forward
    }

    fn resident_bytes(&self) -> usize {
        tensor_bytes(self.input_norm.weight())
            .saturating_add(self.attention.resident_bytes())
            .saturating_add(tensor_bytes(self.post_attention_norm.weight()))
            .saturating_add(self.mlp.resident_bytes())
    }
}

struct BidirectionalAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    query_heads: usize,
    key_value_heads: usize,
    head_dimension: usize,
}

impl BidirectionalAttention {
    fn load(
        builder: VarBuilder<'static>,
        config: &BidirectionalEncoderConfig,
    ) -> CandleResult<Self> {
        let query_width = config.num_attention_heads * config.head_dim;
        let key_value_width = config.num_key_value_heads * config.head_dim;
        Ok(Self {
            query: linear_no_bias(config.hidden_size, query_width, builder.pp("q_proj"))?,
            key: linear_no_bias(config.hidden_size, key_value_width, builder.pp("k_proj"))?,
            value: linear_no_bias(config.hidden_size, key_value_width, builder.pp("v_proj"))?,
            output: linear_no_bias(query_width, config.hidden_size, builder.pp("o_proj"))?,
            query_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dimension: config.head_dim,
        })
    }

    fn forward(&self, hidden: &Tensor, rotary: &RotaryCache) -> CandleResult<Tensor> {
        let (batch, sequence_length, _) = hidden.dims3()?;
        let query = self
            .query
            .forward(hidden)?
            .reshape((
                batch,
                sequence_length,
                self.query_heads,
                self.head_dimension,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let key = self
            .key
            .forward(hidden)?
            .reshape((
                batch,
                sequence_length,
                self.key_value_heads,
                self.head_dimension,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let value = self
            .value
            .forward(hidden)?
            .reshape((
                batch,
                sequence_length,
                self.key_value_heads,
                self.head_dimension,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let query = rotary.apply(&query, sequence_length)?;
        let key = rotary.apply(&key, sequence_length)?;
        let scale = 1.0_f32 / (self.head_dimension as f32).sqrt();
        let attended = bidirectional_attention(&query, &key, &value, scale)?;
        let attended = attended.transpose(1, 2)?.reshape((
            batch,
            sequence_length,
            self.query_heads * self.head_dimension,
        ))?;
        self.output.forward(&attended)
    }

    fn resident_bytes(&self) -> usize {
        linear_bytes(&self.query)
            .saturating_add(linear_bytes(&self.key))
            .saturating_add(linear_bytes(&self.value))
            .saturating_add(linear_bytes(&self.output))
    }
}

fn bidirectional_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    scale: f32,
) -> CandleResult<Tensor> {
    let device = query.device();
    if device.is_cpu() {
        let query = query.transpose(1, 2)?.contiguous()?;
        let key = key.transpose(1, 2)?.contiguous()?;
        let value = value.transpose(1, 2)?.contiguous()?;
        return candle_nn::attention::flash_attn::<f32>(
            &query,
            &key,
            &value,
            scale,
            candle_nn::attention::AttnMask::None,
            None,
            None,
        );
    }
    if device.is_metal() {
        return candle_nn::ops::sdpa(query, key, value, None, false, scale, 1.0);
    }

    // CUDA currently has no fused SDPA in the selected tensor backend. This exact fallback keeps
    // semantics correct; the CUDA performance gate remains open until its native kernel lands.
    let key = candle_transformers::utils::repeat_kv(key.clone(), query.dims()[1] / key.dims()[1])?;
    let value =
        candle_transformers::utils::repeat_kv(value.clone(), query.dims()[1] / value.dims()[1])?;
    let input_dtype = query.dtype();
    let scores = query
        .to_dtype(DType::F32)?
        .matmul(&key.to_dtype(DType::F32)?.t()?)?
        .affine(f64::from(scale), 0.0)?;
    candle_nn::ops::softmax_last_dim(&scores)?
        .matmul(&value.to_dtype(DType::F32)?.contiguous()?)?
        .to_dtype(input_dtype)
}

struct GatedMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl GatedMlp {
    fn load(
        builder: VarBuilder<'static>,
        config: &BidirectionalEncoderConfig,
    ) -> CandleResult<Self> {
        Ok(Self {
            gate: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                builder.pp("gate_proj"),
            )?,
            up: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                builder.pp("up_proj"),
            )?,
            down: linear_no_bias(
                config.intermediate_size,
                config.hidden_size,
                builder.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> CandleResult<Tensor> {
        let gated =
            (candle_nn::ops::silu(&self.gate.forward(hidden)?)? * self.up.forward(hidden)?)?;
        self.down.forward(&gated)
    }

    fn resident_bytes(&self) -> usize {
        linear_bytes(&self.gate)
            .saturating_add(linear_bytes(&self.up))
            .saturating_add(linear_bytes(&self.down))
    }
}

struct RotaryCache {
    cosine: Tensor,
    sine: Tensor,
    maximum_sequence_length: usize,
}

impl RotaryCache {
    fn new(
        config: &BidirectionalEncoderConfig,
        maximum_sequence_length: usize,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Self> {
        let low_frequency_wavelength = config.rope_scaling.original_max_position_embeddings as f32
            / config.rope_scaling.low_freq_factor;
        let high_frequency_wavelength = config.rope_scaling.original_max_position_embeddings as f32
            / config.rope_scaling.high_freq_factor;
        let inverse_frequencies = (0..config.head_dim)
            .step_by(2)
            .map(|index| {
                1_f32
                    / config
                        .rope_theta
                        .powf(index as f32 / config.head_dim as f32)
            })
            .map(|frequency| {
                let wavelength = 2.0 * PI / frequency;
                if wavelength < high_frequency_wavelength {
                    frequency
                } else if wavelength > low_frequency_wavelength {
                    frequency / config.rope_scaling.factor
                } else {
                    let smooth = (config.rope_scaling.original_max_position_embeddings as f32
                        / wavelength
                        - config.rope_scaling.low_freq_factor)
                        / (config.rope_scaling.high_freq_factor
                            - config.rope_scaling.low_freq_factor);
                    (1.0 - smooth) * frequency / config.rope_scaling.factor + smooth * frequency
                }
            })
            .collect::<Vec<_>>();
        let inverse_frequencies = Tensor::new(inverse_frequencies, device)?;
        let positions = Tensor::arange(0, maximum_sequence_length as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((maximum_sequence_length, 1))?;
        let phase = positions
            .matmul(&inverse_frequencies.reshape((1, inverse_frequencies.elem_count()))?)?;
        Ok(Self {
            cosine: phase.cos()?.to_dtype(dtype)?,
            sine: phase.sin()?.to_dtype(dtype)?,
            maximum_sequence_length,
        })
    }

    fn resident_bytes(&self) -> usize {
        tensor_bytes(&self.cosine).saturating_add(tensor_bytes(&self.sine))
    }

    fn apply(&self, tensor: &Tensor, sequence_length: usize) -> CandleResult<Tensor> {
        let cosine = self.cosine.narrow(0, 0, sequence_length)?;
        let sine = self.sine.narrow(0, 0, sequence_length)?;
        candle_nn::rotary_emb::rope(tensor, &cosine, &sine)
    }
}

fn tensor_bytes(tensor: &Tensor) -> usize {
    tensor
        .elem_count()
        .saturating_mul(tensor.dtype().size_in_bytes())
}

fn linear_bytes(linear: &Linear) -> usize {
    tensor_bytes(linear.weight()).saturating_add(linear.bias().map_or(0, tensor_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_configuration_accepts_only_the_owned_bidirectional_shape() -> crate::Result<()> {
        let config = BidirectionalEncoderConfig {
            hidden_size: PINNED_HIDDEN_SIZE,
            intermediate_size: 8_192,
            vocab_size: 128_256,
            num_hidden_layers: 16,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            max_position_embeddings: 131_072,
            hidden_act: "silu".to_owned(),
            attention_bias: false,
            mlp_bias: false,
            pooling: "avg".to_owned(),
            use_bidirectional_attention: true,
            tie_word_embeddings: true,
            rope_scaling: RopeScaling {
                factor: 32.0,
                high_freq_factor: 4.0,
                low_freq_factor: 1.0,
                original_max_position_embeddings: 8_192,
                rope_type: "llama3".to_owned(),
            },
        };
        config.validate_pinned(PINNED_MAXIMUM_INPUT_TOKENS)?;
        assert!(
            config
                .validate_pinned(PINNED_MAXIMUM_INPUT_TOKENS - 1)
                .is_err()
        );
        Ok(())
    }
}
