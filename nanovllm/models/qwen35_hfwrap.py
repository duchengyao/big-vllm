import torch
from torch import nn
import torch.nn.functional as F


class Qwen35ForCausalLM(nn.Module):
    """Full HuggingFace Qwen3.5 model with DynamicCache, transparent to engine."""
    packed_modules_mapping = {}

    def __init__(self, hf_config):
        super().__init__()
        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config

        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5TextModel, Qwen3_5TextRotaryEmbedding,
        )
        hf_cfg = Qwen3_5TextConfig(**hf_config.to_dict())
        self._hf_config = hf_cfg
        self.model = nn.Module()
        self.model.language_model = Qwen3_5TextModel(hf_cfg)
        self.model.language_model.reset_cache = self.reset_cache
        self._past_key_values = None
        self._allocated = False

    def reset_cache(self):
        self._past_key_values = None

    def _ensure_rotary(self, device, dtype):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
        rotary = Qwen3_5TextRotaryEmbedding(self._hf_config)
        return rotary

    def forward(self, input_ids, positions):
        from nanovllm.utils.context import get_context
        ctx = get_context()

        if ctx.is_prefill and ctx.cu_seqlens_q is not None:
            num_seqs = len(ctx.cu_seqlens_q) - 1
            lengths = ctx.cu_seqlens_q[1:] - ctx.cu_seqlens_q[:-1]
            max_len = lengths.max().item()
            batch_ids = torch.zeros(num_seqs, max_len, dtype=input_ids.dtype, device=input_ids.device)
            batch_pos = torch.zeros(num_seqs, max_len, dtype=positions.dtype, device=positions.device)
            attn_mask = torch.zeros(num_seqs, max_len, dtype=torch.bool, device=input_ids.device)
            for i in range(num_seqs):
                start = ctx.cu_seqlens_q[i].item()
                end = ctx.cu_seqlens_q[i+1].item()
                seq_len = end - start
                batch_ids[i, :seq_len] = input_ids[start:end]
                batch_pos[i, :seq_len] = positions[start:end]
                attn_mask[i, :seq_len] = True
            pkv = None
        else:
            batch_ids = input_ids.unsqueeze(-1)
            batch_pos = positions.unsqueeze(-1)
            attn_mask = None
            pkv = self._past_key_values

        output = self.model.language_model(
            batch_ids, attention_mask=attn_mask, position_ids=batch_pos,
            past_key_values=pkv, use_cache=True,
        )
        self._past_key_values = output.past_key_values
        h = output.last_hidden_state
        
        if ctx.is_prefill and ctx.cu_seqlens_q is not None:
            flat_parts = []
            for i in range(num_seqs):
                seq_len = lengths[i].item()
                flat_parts.append(h[i, :seq_len])
            h = torch.cat(flat_parts, dim=0)
        else:
            h = h.reshape(-1, h.shape[-1])
        return h

    def compute_logits(self, hidden_states):
        from nanovllm.utils.context import get_context
        ctx = get_context()
        if ctx.is_prefill and ctx.cu_seqlens_q is not None:
            last_indices = ctx.cu_seqlens_q[1:] - 1
            hidden_states = hidden_states[last_indices].contiguous()
        weight = self.model.language_model.embed_tokens.weight
        return F.linear(hidden_states, weight)

    def _allocate_cache(self):
        pass
