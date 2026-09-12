#!/usr/bin/env python3
"""Convert the verified pinned retrieval encoder into a fused Core ML program.

The result is a derived execution artifact, not a different model: it loads the exact locally
verified safetensors, makes Llama attention bidirectional, and fuses the product's masked-average,
384-coordinate Matryoshka, and second-normalization profile into one prediction.
"""

import argparse
import json
import pathlib
import shutil

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel


class BidirectionalLlamaModel(LlamaModel):
    """The pinned NVIDIA architecture on the Transformers version used for conversion."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_seen_tokens,
        output_attentions,
    ):
        del cache_position, past_seen_tokens, output_attentions
        if attention_mask is None:
            return None
        return _prepare_4d_attention_mask(attention_mask, input_tensor.dtype)


def rotate_half(hidden):
    return torch.cat((-hidden[..., 32:], hidden[..., :32]), dim=-1)


class FixedAttention(torch.nn.Module):
    """Llama GQA written with fixed bucket dimensions Core ML can lower."""

    def __init__(self, attention, cos, sin, batch: int, sequence: int) -> None:
        super().__init__()
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.batch = batch
        self.sequence = sequence

    def forward(self, hidden, attention_mask):
        query = (
            self.q_proj(hidden)
            .view(self.batch, self.sequence, 32, 64)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden)
            .view(self.batch, self.sequence, 8, 64)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden)
            .view(self.batch, self.sequence, 8, 64)
            .transpose(1, 2)
        )
        cos = self.cos.unsqueeze(1)
        sin = self.sin.unsqueeze(1)
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        key = (
            key[:, :, None, :, :]
            .expand(self.batch, 8, 4, self.sequence, 64)
            .reshape(self.batch, 32, self.sequence, 64)
        )
        value = (
            value[:, :, None, :, :]
            .expand(self.batch, 8, 4, self.sequence, 64)
            .reshape(self.batch, 32, self.sequence, 64)
        )
        scores = torch.matmul(query, key.transpose(2, 3)) / 8.0
        key_mask = (1.0 - attention_mask[:, None, None, :].to(scores.dtype)) * -10000.0
        weights = functional.softmax(scores + key_mask, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        output = torch.matmul(weights, value)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(self.batch, self.sequence, 2048)
        )
        return self.o_proj(output)


class FixedLayer(torch.nn.Module):
    def __init__(self, layer, cos, sin, batch: int, sequence: int) -> None:
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.attention = FixedAttention(layer.self_attn, cos, sin, batch, sequence)
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp

    def forward(self, hidden, attention_mask):
        residual = hidden
        hidden = residual + self.attention(
            self.input_layernorm(hidden), attention_mask
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class EmbeddingProfile(torch.nn.Module):
    def __init__(self, model: BidirectionalLlamaModel, batch: int, sequence: int) -> None:
        super().__init__()
        self.embed_tokens = model.embed_tokens
        position_ids = torch.arange(sequence).unsqueeze(0)
        with torch.no_grad():
            sample = torch.zeros((1, sequence, 2048), dtype=torch.float32)
            cos, sin = model.rotary_emb(sample, position_ids)
        self.layers = torch.nn.ModuleList(
            FixedLayer(layer, cos, sin, batch, sequence) for layer in model.layers
        )
        self.norm = model.norm

    def forward(self, input_ids, attention_mask):
        hidden = self.embed_tokens(input_ids.to(torch.long))
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        hidden = self.norm(hidden)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1)
        pooled = functional.normalize(pooled, p=2, dim=-1)
        matryoshka = pooled[:, :384]
        return functional.normalize(matryoshka, p=2, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--trace-length", type=int, default=128)
    args = parser.parse_args()

    config_data = json.loads((args.source / "config.json").read_text())
    config_data["model_type"] = "llama"
    config_data["_attn_implementation"] = "eager"
    config_data["use_cache"] = False
    config = LlamaConfig(**config_data)
    model = BidirectionalLlamaModel(config)
    missing, unexpected = model.load_state_dict(
        load_file(args.source / "model.safetensors"), strict=False
    )
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing}, unexpected={unexpected}")
    wrapper = EmbeddingProfile(model.eval(), args.max_batch, args.trace_length).eval()

    input_ids = torch.ones((args.max_batch, args.trace_length), dtype=torch.int32)
    attention_mask = torch.ones((args.max_batch, args.trace_length), dtype=torch.int32)
    torch.set_grad_enabled(False)
    traced = torch.jit.trace(wrapper, (input_ids, attention_mask), strict=False)

    # Llama's rotary implementation converts symbolic dimensions through Python integers. Core ML
    # cannot lower that operation for RangeDim inputs, so emit fixed execution buckets. The runtime
    # pads ordinary rows into this 512-token program and retains Candle for longer 513–8,192-token
    # inputs. More fixed buckets can be generated without changing model semantics.
    if args.max_length == args.trace_length:
        input_shape = (args.max_batch, args.trace_length)
    else:
        batch = ct.RangeDim(lower_bound=1, upper_bound=args.max_batch, default=1)
        sequence = ct.RangeDim(
            lower_bound=1, upper_bound=args.max_length, default=args.trace_length
        )
        input_shape = (batch, sequence)
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType("input_ids", shape=input_shape, dtype=np.int32),
            ct.TensorType("attention_mask", shape=input_shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType("embeddings")],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.is_dir():
            shutil.rmtree(args.output)
        else:
            args.output.unlink()
    converted.save(str(args.output))


if __name__ == "__main__":
    main()
