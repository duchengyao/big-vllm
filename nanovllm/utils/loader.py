import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _get_quant_config(path: str) -> dict | None:
    import json
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config", {})
    if qc.get("quant_method") == "compressed-tensors":
        return qc
    return None


def _dequant_w4a16(packed: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    out_f, in_f_div8 = packed.shape
    in_f = in_f_div8 * 8

    packed_i32 = packed.to(torch.int32)
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    unpacked = (packed_i32.unsqueeze(-1) >> shifts) & 0xF  # (out_f, in_f_div8, 8)
    unpacked = unpacked.reshape(out_f, in_f).float()
    # Symmetric 4-bit: u4 (0-15) → s4 (-8 to 7)
    unpacked = unpacked - 8

    # scale: (out_f, in_f // group_size)
    n_groups = in_f // group_size
    scale_fp = scale.float().unsqueeze(-1)  # (out_f, n_groups, 1)
    expanded = unpacked.reshape(out_f, n_groups, group_size)
    expanded = expanded * scale_fp
    return expanded.reshape(out_f, in_f)


def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor, block_size: list) -> torch.Tensor:
    w_fp = weight.float()
    s_fp = scale.float()
    out_f, in_f = w_fp.shape
    b0, b1 = block_size
    n0, n1 = out_f // b0, in_f // b1
    expanded = w_fp.reshape(n0, b0, n1, b1)
    scale_view = s_fp.unsqueeze(1).unsqueeze(-1)
    expanded = expanded * scale_view
    return expanded.reshape(out_f, in_f)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    qconfig = _get_quant_config(path)

    # Build dequant info for quantized models
    scale_map = {}  # packed_weight_name → (scale_name, group_size or block_size)
    if qconfig is not None:
        groups = qconfig.get("config_groups", {})
        for gcfg in groups.values():
            w_cfg = gcfg.get("weights", {})
            num_bits = w_cfg.get("num_bits")
            group_size = w_cfg.get("group_size")
            block_structure = w_cfg.get("block_structure")
            fmt = gcfg.get("format", "")
            is_packed = "pack" in fmt
            is_float_quant = "float-quantized" in fmt

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            # First pass: collect scale tensors for dequant
            scale_tensors = {}
            for weight_name in f.keys():
                if weight_name.endswith("_scale"):
                    base = weight_name[:-6]  # remove "_scale"
                    scale_tensors[base] = f.get_tensor(weight_name)

            for weight_name in f.keys():
                if weight_name.endswith("_scale"):
                    continue  # handled by dequant below

                # Check if this is a packed quantized weight
                loaded = f.get_tensor(weight_name)
                is_packed_q = weight_name.endswith("_packed") or loaded.dtype == torch.int32
                is_fp8 = str(loaded.dtype) in ("torch.float8_e4m3fn", "torch.float8_e5m2")

                if is_packed_q or is_fp8:
                    # Find the corresponding scale
                    base = weight_name.replace("_packed", "")
                    scale = scale_tensors.get(base)
                    if scale is None and base in scale_tensors:
                        scale = scale_tensors[base]

                    if is_packed_q and scale is not None:
                        # Determine group_size from scale shape
                        # scale: (out_f, in_f // group_size)
                        group_size_calc = loaded.shape[1] * 8 // scale.shape[1]
                        loaded = _dequant_w4a16(loaded, scale, group_size_calc)
                        weight_name = base  # use base name for parameter lookup
                    elif is_fp8 and scale is not None:
                        # block_size from scale shape: (out_f // bs0, in_f // bs1)
                        bs0 = loaded.shape[0] // scale.shape[0]
                        bs1 = loaded.shape[1] // scale.shape[1]
                        loaded = _dequant_fp8(loaded, scale, [bs0, bs1])
                        weight_name = base if weight_name.endswith("_packed") else weight_name

                # Now load the (possibly dequantized) weight
                param_loaded = False
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        try:
                            param = model.get_parameter(param_name)
                        except AttributeError:
                            continue
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, loaded, shard_id)
                        param_loaded = True
                        break

                if param_loaded:
                    continue

                try:
                    param = model.get_parameter(weight_name)
                except AttributeError:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded)
