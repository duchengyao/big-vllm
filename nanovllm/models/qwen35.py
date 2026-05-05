import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.attention import Attention
from nanovllm.layers.rotary_embedding import get_rope


class Qwen35RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            orig_dtype = x.dtype
            x = x.float()
            var = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (x * (1.0 + self.weight.float())).to(orig_dtype)
        else:
            orig_dtype = x.dtype
            x = x.float().add_(residual.float())
            residual = x.to(orig_dtype)
            var = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (x * (1.0 + self.weight.float())).to(orig_dtype), residual


class Qwen35Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int = 262144,
        rms_norm_eps: float = 1e-06,
        attn_bias: bool = False,
        rope_theta: float = 10000000,
        partial_factor: float = 1.0,
    ) -> None:
        super().__init__()
        assert dist.get_world_size() == 1
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=attn_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=attn_bias)
        self.q_norm = Qwen35RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(head_dim, eps=rms_norm_eps)

        self.rotary_emb = get_rope(head_dim, int(head_dim * partial_factor), max_position, rope_theta)
        self.attn = Attention(num_heads, head_dim, head_dim ** -0.5, num_kv_heads)

    def forward(self, positions, hidden_states):
        bs = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(bs, self.num_heads, -1)
        k = self.k_proj(hidden_states).view(bs, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bs, self.num_kv_heads, self.head_dim)
        q, gate = q.chunk(2, dim=-1)
        q, gate = q.contiguous(), gate.contiguous()
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.reshape(bs, -1) * torch.sigmoid(gate.reshape(bs, -1)))


class Qwen35MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        assert dist.get_world_size() == 1
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class NativeGatedDeltaNet(nn.Module):
    """Native Qwen3.5 GatedDeltaNet with external state buffers for CUDA graph."""

    def __init__(self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                 conv_kernel_size, layer_idx, rms_norm_eps):
        super().__init__()
        self.hidden_size = hidden_size
        self.key_dim = head_k_dim * num_k_heads
        self.value_dim = head_v_dim * num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_v_dim = head_v_dim
        self.head_k_dim = head_k_dim
        self.layer_idx = layer_idx
        self.norm_eps = rms_norm_eps

        self.in_proj_qkv = nn.Linear(hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, conv_kernel_size, groups=self.conv_dim,
                                padding=conv_kernel_size - 1, bias=False)
        self.norm = RMSNormGated(head_v_dim, rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(num_v_heads))

        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update
        self.chunk_delta = chunk_gated_delta_rule
        self.recurrent_delta = fused_recurrent_gated_delta_rule

    def forward(self, hidden_states, cache=None):
        batch, seq_len, _ = hidden_states.shape
        has_state = cache is not None and cache.has_prev(self.layer_idx)

        if has_state:
            conv_state = cache.get_conv(self.layer_idx)
            rec_state = cache.get_rec(self.layer_idx)

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).view(batch, seq_len, -1, self.head_v_dim)
        b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)

        if has_state and seq_len == 1:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, "silu",
            )
        else:
            if has_state:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if cache is not None:
                # Store conv_state: pad with kernel_size-1 zeros on left, then take last kernel_size
                pad = F.pad(mixed_qkv, (self.conv_kernel_size - 1, 0))
                cache.update_conv(pad[:, :, -self.conv_kernel_size:], self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias,
                    activation="silu",
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :mixed_qkv.shape[-1]])
            if has_state:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.view(batch, seq_len, -1, self.head_k_dim)
        key = key.view(batch, seq_len, -1, self.head_k_dim)
        value = value.view(batch, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if has_state and seq_len == 1:
            out, last = self.recurrent_delta(query, key, value, g=g, beta=beta,
                                             initial_state=rec_state, output_final_state=cache is not None,
                                             use_qk_l2norm_in_kernel=True)
        else:
            init = rec_state if has_state else None
            out, last = self.chunk_delta(query, key, value, g=g, beta=beta,
                                         initial_state=init, output_final_state=cache is not None,
                                         use_qk_l2norm_in_kernel=True)

        if cache is not None and last is not None:
            cache.update_rec(last, self.layer_idx)

        out = self.norm(out.view(batch, seq_len, -1, self.head_v_dim))
        out = (out * F.silu(z)).reshape(batch, seq_len, -1)
        return self.out_proj(out)


class RMSNormGated(nn.Module):
    """RMSNorm for GatedDeltaNet output. Uses x * weight (weight init=1), NOT x * (1+weight)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight.float()).to(orig_dtype)


class GDNStateCache:
    """External state cache for GatedDeltaNet, compatible with CUDA graph."""

    def __init__(self, num_layers, conv_dim, num_v_heads, head_k_dim, head_v_dim):
        self._conv = torch.zeros(num_layers, conv_dim, 4, device='cuda', dtype=torch.bfloat16)
        self._rec = torch.zeros(num_layers, num_v_heads, head_k_dim, head_v_dim,
                                device='cuda', dtype=torch.bfloat16)
        self._has = [False] * num_layers

    def has_prev(self, layer_idx):
        return self._has[layer_idx]

    def get_conv(self, layer_idx):
        return self._conv[layer_idx:layer_idx+1]

    def get_rec(self, layer_idx):
        return self._rec[layer_idx:layer_idx+1]

    def update_conv(self, state, layer_idx):
        self._conv[layer_idx] = state.squeeze(0)
        self._has[layer_idx] = True

    def update_rec(self, state, layer_idx):
        self._rec[layer_idx] = state.squeeze(0)
        self._has[layer_idx] = True

    def reset(self):
        self._has = [False] * len(self._has)


class Qwen35DecoderLayer(nn.Module):

    def __init__(self, text_config, layer_idx: int, gdn_cache: GDNStateCache | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        interval = text_config.full_attention_interval
        is_full = layer_idx % interval == interval - 1
        rope = getattr(text_config, "rope_parameters", {}) or {}

        if is_full:
            self.self_attn = Qwen35Attention(
                hidden_size=text_config.hidden_size,
                num_heads=text_config.num_attention_heads,
                num_kv_heads=text_config.num_key_value_heads,
                head_dim=text_config.head_dim,
                max_position=text_config.max_position_embeddings,
                attn_bias=getattr(text_config, "attention_bias", False),
                rope_theta=rope.get("rope_theta", 10000000),
                partial_factor=rope.get("partial_rotary_factor", 1.0),
            )
        else:
            self.linear_attn = NativeGatedDeltaNet(
                hidden_size=text_config.hidden_size,
                num_k_heads=text_config.linear_num_key_heads,
                num_v_heads=text_config.linear_num_value_heads,
                head_k_dim=text_config.linear_key_head_dim,
                head_v_dim=text_config.linear_value_head_dim,
                conv_kernel_size=text_config.linear_conv_kernel_dim,
                layer_idx=layer_idx,
                rms_norm_eps=text_config.rms_norm_eps,
            )

        self.mlp = Qwen35MLP(text_config.hidden_size, text_config.intermediate_size)
        self.input_layernorm = Qwen35RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual, cache=None):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if hasattr(self, "self_attn"):
            hidden_states = self.self_attn(positions, hidden_states)
        else:
            hidden_states = self.linear_attn(hidden_states.unsqueeze(0), cache=cache).squeeze(0)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen35Model(nn.Module):

    def __init__(self, text_config):
        super().__init__()
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        num_k = getattr(text_config, "linear_num_key_heads", 16)
        num_v = getattr(text_config, "linear_num_value_heads", 32)
        hk = getattr(text_config, "linear_key_head_dim", 128)
        hv = getattr(text_config, "linear_value_head_dim", 128)
        cdim = hk * num_k * 2 + hv * num_v
        self._state_cache = GDNStateCache(text_config.num_hidden_layers, cdim, num_v, hk, hv)

        self.layers = nn.ModuleList()
        for i in range(text_config.num_hidden_layers):
            self.layers.append(Qwen35DecoderLayer(text_config, i, self._state_cache))

        self.norm = Qwen35RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def reset_cache(self):
        self._state_cache.reset()

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, self._state_cache)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen35ForCausalLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, hf_config):
        super().__init__()
        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config
        self.model = nn.Module()
        self.model.language_model = Qwen35Model(hf_config)
        self.lm_head = nn.Linear(hf_config.hidden_size, hf_config.vocab_size, bias=False)
        if hf_config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(self, input_ids, positions):
        return self.model.language_model(input_ids, positions)

    def compute_logits(self, hidden_states):
        from nanovllm.utils.context import get_context
        ctx = get_context()
        if ctx.is_prefill and ctx.cu_seqlens_q is not None:
            hidden_states = hidden_states[ctx.cu_seqlens_q[1:] - 1].contiguous()
        return self.lm_head(hidden_states)
