"""Run quant tests in subprocesses to avoid OOM from 8B models."""

import os, subprocess, sys

TESTS = [
    ("qwen3-8B-original", "~/huggingface/Qwen3-8B"),
    ("qwen3-8B-rtn", "~/huggingface/Qwen3-8B-rtn"),
    ("qwen3-8B-W4A16", "~/huggingface/Qwen3-8B-W4A16-G128"),
    ("qwen35-0.8B-W4A16", "~/huggingface/Qwen3.5-0.8B-W4A16-G128"),
]

PYTHON = sys.executable
code = r"""
import os, torch, sys, glob
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ["MASTER_PORT"] = "29602"
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
import torch.distributed as dist
dist.init_process_group("nccl", world_size=1, rank=0)
torch.cuda.set_device(0)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")

from nanovllm.config import Config
from nanovllm.models import get_model_cls
from nanovllm.utils.loader import load_model
from nanovllm.utils.context import set_context, reset_context

path = os.path.expanduser(sys.argv[1])
cfg = Config(model=path)
m = get_model_cls(cfg.hf_config)(cfg.hf_config)
load_model(m, path)

tokens = [9419, 11, 821, 803, 369]
inp = torch.tensor(tokens, dtype=torch.int64)
pos = torch.arange(len(inp), dtype=torch.int64)
n = len(inp)
set_context(True, torch.tensor([0,n], dtype=torch.int32), torch.tensor([0,n], dtype=torch.int32), n, n)
with torch.inference_mode():
    out = m(inp, pos)
    logits = m.compute_logits(out).squeeze(0)
reset_context()
tok = logits.argmax().item()

from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(path)
text = t.decode([tok])
assert 0 <= tok < 248320, f"bad token: {tok}"
assert text.strip(), f"empty top token: {text!r}"

# Cosine similarity vs FP16 reference (skip for large models to avoid OOM)
is_8b = "8B" in path or "8b" in path
is_quantized = "-rtn" in path or "-W4A16" in path

if is_quantized and not is_8b:
    fp_path = path.replace("-W4A16-G128", "").replace("-rtn", "")
    if os.path.isdir(fp_path):
        from transformers import AutoConfig
        hf_cfg = AutoConfig.from_pretrained(fp_path, trust_remote_code=True)
        try:
            hf_cfg._attn_implementation = "flash_attention_2"
        except Exception:
            pass
        import transformers
        Model = transformers.AutoModelForCausalLM.from_config(hf_cfg, torch_dtype=torch.bfloat16)
        Model = Model.cuda()
        from safetensors.torch import load_file as ld
        state = {}
        for f in sorted(glob.glob(fp_path + "/*.safetensors")):
            state.update(ld(f))
        hs = {}
        for k, v in state.items():
            if k.startswith("model.language_model."):
                hs["model." + k[len("model.language_model."):]] = v.cuda().to(torch.bfloat16)
            else:
                hs[k] = v.cuda().to(torch.bfloat16)
        Model.load_state_dict(hs, strict=False)
        Model.eval()
        inp2 = torch.tensor([tokens], device='cuda')
        with torch.inference_mode():
            l_ref = Model(inp2).logits[0, -1]
        cos = torch.nn.functional.cosine_similarity(logits.float(), l_ref.float(), dim=0).item()
        assert cos > 0.95, f"cosine similarity too low: {cos:.4f}"
        print(f"OK tok={tok} cos={cos:.4f} text={text[:30]!r}")
        del Model; torch.cuda.empty_cache()
    else:
        print(f"OK tok={tok} text={text[:30]!r} (no fp16 ref)")
else:
    print(f"OK tok={tok} text={text[:30]!r}")
"""

for name, model in TESTS:
    path = os.path.expanduser(model)
    print(f"  {name}: ", end="", flush=True)
    r = subprocess.run([PYTHON, "-c", code, path], capture_output=True, text=True, timeout=180)
    if r.returncode == 0:
        print(r.stdout.strip())
    else:
        print(f"FAILED\n{r.stderr[-300:]}")
