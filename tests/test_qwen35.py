import os
import sys

import pytest
import torch
import torch.distributed as dist

MODEL = os.path.expanduser("~/huggingface/Qwen3.5-4B")


@pytest.fixture(scope="module")
def model():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    from nanovllm.config import Config
    from nanovllm.models import get_model_cls
    from nanovllm.utils.loader import load_model

    cfg = Config(model=MODEL)
    m = get_model_cls(cfg.hf_config)(cfg.hf_config)
    load_model(m, MODEL)

    # Allocate dummy KV cache for tests
    for module in m.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = torch.zeros(100, 256, 4, 256, device='cuda', dtype=torch.bfloat16)
            module.v_cache = torch.zeros(100, 256, 4, 256, device='cuda', dtype=torch.bfloat16)
    yield m
    torch.cuda.synchronize()


def _prefill(model, tokens):
    """Run prefill with proper context. Returns logits."""
    from nanovllm.utils.context import set_context, reset_context
    inp = torch.tensor(tokens, dtype=torch.int64)
    pos = torch.arange(len(inp), dtype=torch.int64)
    n = len(inp)
    cu = torch.tensor([0, n], dtype=torch.int32, device='cuda')
    slot = torch.arange(n, dtype=torch.int32, device='cuda')
    set_context(True, cu, cu, n, n, slot_mapping=slot)
    with torch.inference_mode():
        out = model(inp, pos)
        logits = model.compute_logits(out).squeeze(0)
    reset_context()
    return logits


def _decode(model, n, token):
    """Run single decode step. Returns token."""
    from nanovllm.utils.context import set_context, reset_context
    inp = torch.tensor([token], dtype=torch.int64)
    pos = torch.tensor([n], dtype=torch.int64)
    slot = torch.tensor([n % 256], dtype=torch.int32, device='cuda')
    clen = torch.tensor([n], dtype=torch.int32, device='cuda')
    bt = torch.zeros(1, 1, dtype=torch.int32, device='cuda')
    set_context(False, slot_mapping=slot, context_lens=clen, block_tables=bt)
    with torch.inference_mode():
        out = model(inp, pos)
        logits = model.compute_logits(out).squeeze(0)
    reset_context()
    return logits.argmax().item()


class TestModelLoading:
    def test_embed_tokens_shape(self, model):
        w = model.model.language_model.embed_tokens.weight
        assert w.shape == (248320, 2560)

    def test_num_layers(self, model):
        assert len(model.model.language_model.layers) == 32

    def test_linear_attn_layers(self, model):
        count = sum(1 for l in model.model.language_model.layers if hasattr(l, "linear_attn"))
        assert count == 24

    def test_self_attn_layers(self, model):
        count = sum(1 for l in model.model.language_model.layers if hasattr(l, "self_attn"))
        assert count == 8


class TestPrefill:
    def test_single_prefill_runs(self, model):
        model.model.language_model.reset_cache()
        logits = _prefill(model, [9419, 11, 821, 803, 369])
        assert logits.shape == (248320,)

    def test_prefill_logits_reasonable(self, model):
        model.model.language_model.reset_cache()
        logits = _prefill(model, [9419, 11, 821, 803, 369])
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(MODEL)
        first = t.decode([logits.argmax().item()])
        assert first.strip(), f"First token should not be whitespace, got {first!r}"

    def test_prefill_hidden_shape(self, model):
        model.model.language_model.reset_cache()
        from nanovllm.utils.context import set_context, reset_context
        inp = torch.tensor([9419, 11, 821, 803, 369], dtype=torch.int64)
        pos = torch.arange(len(inp), dtype=torch.int64)
        n = len(inp)
        cu = torch.tensor([0, n], dtype=torch.int32, device='cuda')
        slot = torch.arange(n, dtype=torch.int32, device='cuda')
        set_context(True, cu, cu, n, n, slot_mapping=slot)
        with torch.inference_mode():
            out = model(inp, pos)
        reset_context()
        assert out.shape == (5, 2560)


class TestDecode:
    def test_decode_follows_prefill(self, model):
        model.model.language_model.reset_cache()
        logits = _prefill(model, [9419, 11, 821, 803, 369])
        n = 5
        next_tok = logits.argmax().item()
        tokens = [next_tok]
        for i in range(4):
            next_tok = _decode(model, n + i, next_tok)
            tokens.append(next_tok)
        assert len(tokens) == 5

    def test_decode_output_not_repeating(self, model):
        model.model.language_model.reset_cache()
        logits = _prefill(model, [9419, 11, 821, 803, 369])
        n = 5
        seen = set()
        next_tok = logits.argmax().item()
        for i in range(8):
            seen.add(next_tok)
            next_tok = _decode(model, n + i, next_tok)
        assert len(seen) >= 2


class TestBlockManager:
    def test_bypass_can_allocate(self):
        from nanovllm.engine.block_manager import BypassBlockManager
        from nanovllm.engine.sequence import Sequence
        bm = BypassBlockManager(256)
        seq = Sequence([1, 2, 3])
        assert bm.can_allocate(seq) == 0

    def test_bypass_allocate_and_append(self):
        from nanovllm.engine.block_manager import BypassBlockManager
        from nanovllm.engine.sequence import Sequence
        bm = BypassBlockManager(256)
        seq = Sequence([1] * 256)
        bm.allocate(seq, 0)
        assert len(seq.block_table) == 1
        bm.may_append(seq)
        assert len(seq.block_table) == 1
        seq.append_token(1)
        bm.may_append(seq)
        assert len(seq.block_table) == 2

    def test_bypass_deallocate(self):
        from nanovllm.engine.block_manager import BypassBlockManager
        from nanovllm.engine.sequence import Sequence
        bm = BypassBlockManager(256)
        seq = Sequence([1] * 100)
        bm.allocate(seq, 0)
        assert len(seq.block_table) > 0
        bm.deallocate(seq)
        assert len(seq.block_table) == 0
