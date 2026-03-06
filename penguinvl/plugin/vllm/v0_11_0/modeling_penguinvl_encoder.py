"""PenguinVL Vision Encoder for vLLM inference.

Adapted from the HuggingFace PenguinVL vision encoder (Qwen3-based architecture)
for efficient inference with vLLM's tensor parallelism and flash attention.
"""

from typing import Optional, Iterable, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.utils import is_flash_attn_2_available
from vllm.config import QuantizationConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, RowParallelLinear, QKVParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.platforms import _Backend

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos().unsqueeze(1).float()
    sin = freqs.sin().unsqueeze(1).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.to(orig_dtype)


class VisionRotaryEmbedding(nn.Module):
    """2D rotary embedding for PenguinVL vision encoder.

    Uses full head_dim for inv_freq computation (matching PenguinVL's
    VisualRotaryEmbedding), then splits freqs between height and width dims.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


class PenguinVLVisionEmbeddings(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1, self.config.num_channels, self.patch_size, self.patch_size
        )
        patch_embeds = self.patch_embedding(hidden_states)
        embeddings = patch_embeds.view(-1, self.embed_dim)
        return embeddings


class PenguinVLVisionAttention(nn.Module):
    """Qwen3-style attention with GQA, per-head QK norms, and 2D RoPE."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.head_dim = getattr(config, 'head_dim', self.embed_dim // self.num_heads)

        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.num_heads_per_partition = dist_utils.divide(self.num_heads, self.tp_size)
        self.num_kv_heads_per_partition = dist_utils.divide(self.num_kv_heads, self.tp_size)

        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rms_norm_eps = getattr(config, 'rms_norm_eps', getattr(config, 'layer_norm_eps', 1e-6))
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.attn_backend: _Backend = get_vit_attn_backend(
            head_size=self.head_dim, dtype=torch.get_default_dtype())
        if self.attn_backend not in {
            _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS
        }:
            raise RuntimeError(
                f"PenguinVL does not support {self.attn_backend} backend now."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        q_len = hidden_states.size(0)
        qkv, _ = self.qkv_proj(hidden_states)

        q_dim = self.num_heads_per_partition * self.head_dim
        kv_dim = self.num_kv_heads_per_partition * self.head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

        q = q.view(q_len, self.num_heads_per_partition, self.head_dim)
        k = k.view(q_len, self.num_kv_heads_per_partition, self.head_dim)
        v = v.view(q_len, self.num_kv_heads_per_partition, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        if self.num_kv_heads_per_partition != self.num_heads_per_partition:
            repeat_factor = self.num_heads_per_partition // self.num_kv_heads_per_partition
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        if self.attn_backend == _Backend.FLASH_ATTN:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0,
                causal=False,
            )
        elif self.attn_backend == _Backend.TORCH_SDPA:
            outputs = []
            for i in range(1, len(cu_seqlens)):
                start_idx = cu_seqlens[i - 1]
                end_idx = cu_seqlens[i]
                q_i = q[start_idx:end_idx].permute(1, 0, 2).unsqueeze(0)
                k_i = k[start_idx:end_idx].permute(1, 0, 2).unsqueeze(0)
                v_i = v[start_idx:end_idx].permute(1, 0, 2).unsqueeze(0)
                output_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0)
                output_i = output_i.squeeze(0).permute(1, 0, 2)
                outputs.append(output_i)
            output = torch.cat(outputs, dim=0)
        elif self.attn_backend == _Backend.XFORMERS:
            from xformers import ops as xops
            from xformers.ops.fmha.attn_bias import BlockDiagonalMask

            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            attn_bias = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens, kv_seqlen=None, device=q.device
            )
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
            output = xops.memory_efficient_attention_forward(
                q, k, v, attn_bias=attn_bias, p=0, scale=None
            ).squeeze(0)

        output, _ = self.o_proj(output.view(q_len, -1))
        return output


class PenguinVLVisionMLP(nn.Module):
    """SwiGLU MLP matching Qwen3's MLP architecture."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size, self.intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = self.act_fn(gate) * up
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class PenguinVLVisionEncoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()
        rms_norm_eps = getattr(config, 'rms_norm_eps', getattr(config, 'layer_norm_eps', 1e-6))

        self.self_attn = PenguinVLVisionAttention(
            config, quant_config=quant_config, prefix=f"{prefix}.self_attn"
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)
        self.mlp = PenguinVLVisionMLP(
            config, quant_config=quant_config, prefix=f"{prefix}.mlp"
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states


class PenguinVLVisionTransformerEncoder(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        rope_theta = getattr(config, 'rope_theta', 10000.0)
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim, theta=rope_theta)
        self.layers = nn.ModuleList([
            PenguinVLVisionEncoderLayer(
                config, quant_config=quant_config, prefix=f"{prefix}.layers.{i}"
            )
            for i in range(config.num_hidden_layers)
        ])
        rms_norm_eps = getattr(config, 'rms_norm_eps', getattr(config, 'layer_norm_eps', 1e-6))
        self.norm = RMSNorm(config.hidden_size, eps=rms_norm_eps)

    def rot_pos_emb(self, grid_sizes, merge_sizes):
        h_pos_ids = []
        w_pos_ids = []
        for (t, h, w), merge_size in zip(grid_sizes, merge_sizes):
            hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos = hpos.reshape(
                h // merge_size, merge_size,
                w // merge_size, merge_size,
            )
            hpos = hpos.permute(0, 2, 1, 3).flatten()

            wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos = wpos.reshape(
                h // merge_size, merge_size,
                w // merge_size, merge_size,
            )
            wpos = wpos.permute(0, 2, 1, 3).flatten()

            h_pos_ids.append(hpos.repeat(t))
            w_pos_ids.append(wpos.repeat(t))

        h_pos_ids = torch.cat(h_pos_ids, dim=0)
        w_pos_ids = torch.cat(w_pos_ids, dim=0)

        max_grid_size = grid_sizes[:, 1:].max()
        freqs_table = self.rotary_pos_emb(max_grid_size)

        h_freqs = freqs_table[h_pos_ids]
        w_freqs = freqs_table[w_pos_ids]
        rotary_pos_emb = torch.cat([h_freqs, w_freqs], dim=-1)
        return rotary_pos_emb

    def forward(self, hidden_states, grid_sizes, merge_sizes) -> torch.Tensor:
        rotary_pos_emb = self.rot_pos_emb(grid_sizes, merge_sizes)

        cu_seqlens = torch.repeat_interleave(
            grid_sizes[:, 1] * grid_sizes[:, 2], grid_sizes[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.layers:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class PenguinVLVisionEncoderModel(nn.Module):
    """PenguinVL vision encoder for vLLM.

    Structure matches the HF model:
      embeddings -> encoder (transformer layers) -> norm -> downsample
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.embeddings = PenguinVLVisionEmbeddings(config)
        self.encoder = PenguinVLVisionTransformerEncoder(
            config, quant_config=quant_config, prefix=f"{prefix}.encoder"
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    def forward(self, pixel_values, grid_sizes, merge_sizes=None) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states, grid_sizes, merge_sizes)

        hidden_states_chunks = hidden_states.split(
            grid_sizes.prod(dim=1).tolist(), dim=0
        )
        outputs = []

        for hidden_states, grid_size, merge_size in zip(
            hidden_states_chunks, grid_sizes, merge_sizes
        ):
            c = hidden_states.shape[-1]
            hidden_states = hidden_states.view(
                grid_size[0],
                grid_size[1] // merge_size,
                grid_size[2] // merge_size,
                merge_size,
                merge_size,
                c,
            ).permute(0, 1, 3, 2, 4, 5)
            hidden_states = hidden_states.reshape(
                grid_size[0], grid_size[1], grid_size[2], c
            ).permute(0, 3, 1, 2)
            hidden_states = F.interpolate(
                hidden_states,
                size=(grid_size[1] // merge_size, grid_size[2] // merge_size),
                mode='bilinear',
            )
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(-1, c)
            outputs.append(hidden_states)

        return outputs

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0
                                 else loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
