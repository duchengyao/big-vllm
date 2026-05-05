"""Run quant tests in subprocesses to avoid OOM from 8B models."""

import os, subprocess, sys

TESTS = [
    ("test_original", "~/huggingface/Qwen3-8B"),
    ("test_rtn", "~/huggingface/Qwen3-8B-rtn"),
    ("test_g128", "~/huggingface/Qwen3-8B-W4A16-G128"),
]

PYTHON = sys.executable
code = """
import os, torch, sys
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ["MASTER_PORT"] = "29601"
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
assert 0 <= tok < 151936, f"bad token: {tok}"
assert text.strip(), f"whitespace: {text!r}"
assert t.decode([logits.topk(3).indices[0].item()]).strip(), "top-1 whitespace"
print(f"OK tok={tok} text={text[:40]!r}")
"""

for name, model in TESTS:
    path = os.path.expanduser(model)
    print(f"  {name}: ", end="", flush=True)
    r = subprocess.run([PYTHON, "-c", code, path], capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        print(r.stdout.strip())
    else:
        print(f"FAILED\n{r.stderr[-200:]}")
