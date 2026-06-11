#!/usr/bin/env python3
"""Analyze the final parameter delta between a base HF checkpoint and an OPD checkpoint."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import re
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from tqdm import tqdm


ABS_THRESHOLDS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
REL_THRESHOLDS = [1e-4, 1e-3, 1e-2]
COORD_TOP_PCTS = [0.001, 0.01, 0.05]
SVD_TOPKS = [1, 2, 4, 8, 16, 32, 64]
PROJ_RATIOS = [0.01, 0.05, 0.10, 0.20]
MASK_OVERLAP_RATIOS = [0.01, 0.05, 0.10, 0.20]
EPS = 1e-12


def discover_checkpoint_files(model_dir: Path) -> Dict[str, Any]:
    """Discover supported checkpoint formats under a model/checkpoint directory."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    if safetensors_files:
        return {"kind": "safetensors", "files": safetensors_files}

    pt_files = sorted(model_dir.glob("pytorch_model*.bin"))
    if pt_files:
        return {"kind": "torch", "files": pt_files}

    model_world_files = sorted(model_dir.glob("model_world_size_*_rank_*.pt"))
    if not model_world_files:
        model_world_files = sorted(model_dir.glob("**/model_world_size_*_rank_*.pt"))
    if model_world_files:
        latest = _choose_latest_checkpoint_dir(model_world_files)
        files = sorted(latest.glob("model_world_size_*_rank_*.pt"), key=_rank_from_path)
        return {"kind": "dtensor_sharded_torch", "files": files, "checkpoint_dir": latest}

    single_pt = sorted(model_dir.glob("*.pt"))
    if single_pt:
        return {"kind": "torch", "files": single_pt}

    raise FileNotFoundError(f"No supported checkpoint files found under {model_dir}")


def _choose_latest_checkpoint_dir(files: Sequence[Path]) -> Path:
    dirs = sorted({p.parent for p in files})
    parent = dirs[0].parent.parent if len(dirs[0].parts) >= 2 else dirs[0].parent
    latest_file = parent / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        step = latest_file.read_text().strip()
        candidate = parent / f"global_step_{step}" / "actor"
        if candidate.exists():
            return candidate

    def step_num(path: Path) -> int:
        for part in path.parts:
            m = re.fullmatch(r"global_step_(\d+)", part)
            if m:
                return int(m.group(1))
        return -1

    return max(dirs, key=step_num)


def _rank_from_path(path: Path) -> int:
    m = re.search(r"_rank_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else 0


class TensorStore:
    def keys(self) -> List[str]:
        raise NotImplementedError

    def get(self, name: str) -> torch.Tensor:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SafetensorsStore(TensorStore):
    def __init__(self, files: Sequence[Path]):
        self.files = list(files)
        self.name_to_file: Dict[str, Path] = {}
        self.handles: Dict[Path, Any] = {}
        for path in self.files:
            handle = safe_open(str(path), framework="pt", device="cpu")
            self.handles[path] = handle
            for key in handle.keys():
                self.name_to_file[key] = path

    def keys(self) -> List[str]:
        return sorted(self.name_to_file)

    def get(self, name: str) -> torch.Tensor:
        return self.handles[self.name_to_file[name]].get_tensor(name)

    def close(self) -> None:
        self.handles.clear()


class TorchStore(TensorStore):
    def __init__(self, files: Sequence[Path]):
        self.objects: List[Mapping[str, Any]] = []
        self.name_to_obj: Dict[str, Mapping[str, Any]] = {}
        for path in files:
            obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            state = _extract_state_dict(obj)
            self.objects.append(state)
            for key in state.keys():
                self.name_to_obj[key] = state

    def keys(self) -> List[str]:
        return sorted(self.name_to_obj)

    def get(self, name: str) -> torch.Tensor:
        return _as_tensor(self.name_to_obj[name][name])


class DTensorShardedTorchStore(TensorStore):
    def __init__(self, files: Sequence[Path]):
        self.files = list(files)
        self.states: List[Mapping[str, Any]] = []
        for path in tqdm(self.files, desc="loading dtensor shards"):
            obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            self.states.append(_extract_state_dict(obj))
        common = set(self.states[0].keys())
        for state in self.states[1:]:
            common &= set(state.keys())
        self._keys = sorted(common)

    def keys(self) -> List[str]:
        return self._keys

    def get(self, name: str) -> torch.Tensor:
        vals = [state[name] for state in self.states]
        first = vals[0]
        global_shape = tuple(getattr(first, "shape", ()))
        placements = getattr(first, "placements", ())
        if placements and (getattr(placements[0], "is_shard", lambda: False)() or "Shard" in repr(placements[0])):
            shard_dim = getattr(placements[0], "dim", None)
            if shard_dim is None:
                m = re.search(r"Shard\(dim=(\d+)\)", repr(placements[0]))
                shard_dim = int(m.group(1)) if m else 0
            pieces = [_as_tensor(v) for v in vals]
            out = torch.cat(pieces, dim=shard_dim)
            if global_shape:
                slices = tuple(slice(0, s) for s in global_shape)
                out = out[slices]
            return out
        return _as_tensor(first)


def _extract_state_dict(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model", "module", "model_state_dict"):
            val = obj.get(key)
            if isinstance(val, Mapping):
                return val
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise TypeError(f"Could not extract state_dict from object of type {type(obj)}")


def _as_tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "to_local"):
        return value.to_local().detach().cpu()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    raise TypeError(f"Object is not tensor-like: {type(value)}")


def load_tensor_map(model_dir: Path) -> TensorStore:
    info = discover_checkpoint_files(model_dir)
    if info["kind"] == "safetensors":
        return SafetensorsStore(info["files"])
    if info["kind"] == "dtensor_sharded_torch":
        return DTensorShardedTorchStore(info["files"])
    return TorchStore(info["files"])


def iter_named_tensors(store: TensorStore) -> Iterator[Tuple[str, torch.Tensor]]:
    for key in store.keys():
        yield key, store.get(key)


def canonical_tensor_key(name: str) -> str:
    """Normalize common wrapper prefixes used by multimodal/FSDP checkpoints."""
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model.") :]
    if name.startswith("language_model."):
        return "model." + name[len("language_model.") :]
    if name.startswith("model.visual."):
        return "visual." + name[len("model.visual.") :]
    return name


def match_tensor_keys(src_keys: Sequence[str], opd_keys: Sequence[str]) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """Match checkpoint keys exactly first, then by conservative canonical aliases."""
    src_set = set(src_keys)
    opd_set = set(opd_keys)
    exact = sorted(src_set & opd_set)
    pairs: List[Tuple[str, str]] = [(key, key) for key in exact]
    matched_src = set(exact)
    matched_opd = set(exact)

    src_by_canon: Dict[str, List[str]] = {}
    opd_by_canon: Dict[str, List[str]] = {}
    for key in src_set - matched_src:
        src_by_canon.setdefault(canonical_tensor_key(key), []).append(key)
    for key in opd_set - matched_opd:
        opd_by_canon.setdefault(canonical_tensor_key(key), []).append(key)

    alias_pairs: List[Tuple[str, str]] = []
    ambiguous: List[Dict[str, Any]] = []
    for canon in sorted(set(src_by_canon) & set(opd_by_canon)):
        src_vals = sorted(src_by_canon[canon])
        opd_vals = sorted(opd_by_canon[canon])
        if len(src_vals) == 1 and len(opd_vals) == 1:
            src_key, opd_key = src_vals[0], opd_vals[0]
            pairs.append((src_key, opd_key))
            alias_pairs.append((src_key, opd_key))
            matched_src.add(src_key)
            matched_opd.add(opd_key)
        else:
            ambiguous.append({"canonical_key": canon, "src_keys": src_vals, "opd_keys": opd_vals})

    report = {
        "exact_matches": len(exact),
        "alias_matches": len(alias_pairs),
        "alias_examples": [{"src": s, "opd": o} for s, o in alias_pairs[:20]],
        "ambiguous_alias_matches": ambiguous,
        "missing_in_src": sorted(opd_set - matched_opd),
        "missing_in_opd": sorted(src_set - matched_src),
    }
    return sorted(pairs, key=lambda p: p[0]), report


def parse_tensor_name(name: str) -> Dict[str, Any]:
    layer_id = -1
    for pattern in (
        r"(?:^|\.)model\.language_model\.layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)language_model\.layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)model\.layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)transformer\.h\.(\d+)(?:\.|$)",
        r"(?:^|\.)blocks\.(\d+)(?:\.|$)",
    ):
        m = re.search(pattern, name)
        if m:
            layer_id = int(m.group(1))
            break

    lower = name.lower()
    module_type = "other"
    module_patterns = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
        "norm",
    ]
    for token in module_patterns:
        if token in lower:
            module_type = token
            break
    if module_type == "other":
        if any(token in lower for token in ("layernorm", "ln_")):
            module_type = "norm"
        elif re.search(r"(?:^|\.)(fc1|fc2|w1|w2|w3)(?:\.|$)", lower):
            module_type = re.search(r"(?:^|\.)(fc1|fc2|w1|w2|w3)(?:\.|$)", lower).group(1)
        elif "wte" in lower:
            module_type = "embed_tokens"

    if module_type in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        block_type = "attention"
    elif module_type in {"gate_proj", "up_proj", "down_proj", "fc1", "fc2", "w1", "w2", "w3"}:
        block_type = "mlp"
    elif module_type == "embed_tokens":
        block_type = "embedding"
    elif module_type == "lm_head":
        block_type = "lm_head"
    elif module_type == "norm":
        block_type = "norm"
    else:
        block_type = "other"
    return {"layer_id": layer_id, "module_type": module_type, "block_type": block_type}


def compute_coordinate_sparsity(delta: torch.Tensor, src_rms: float) -> Dict[str, float]:
    abs_delta = delta.abs()
    numel = max(delta.numel(), 1)
    out = {}
    out["frac_delta_eq_0"] = float((delta == 0).sum().item() / numel)
    zero = torch.zeros_like(delta)
    for eps in ABS_THRESHOLDS:
        out[f"frac_abs_delta_lt_{_fmt_thresh(eps)}"] = float((abs_delta < eps).sum().item() / numel)
        out[f"frac_isclose_zero_atol_{_fmt_thresh(eps)}"] = float(torch.isclose(delta, zero, atol=eps).sum().item() / numel)
    denom = max(float(src_rms), EPS)
    for tau in REL_THRESHOLDS:
        out[f"frac_abs_delta_lt_{_fmt_thresh(tau)}_src_rms"] = float((abs_delta < tau * denom).sum().item() / numel)
    return out


def compute_coordinate_energy_concentration(delta: torch.Tensor) -> Dict[str, float]:
    flat = delta.flatten()
    total = float(torch.sum(flat * flat).item())
    out = {}
    for pct in COORD_TOP_PCTS:
        col = f"coord_top_{_fmt_pct(pct)}_energy_ratio"
        if total <= 0 or flat.numel() == 0:
            out[col] = float("nan")
            continue
        k = max(1, int(math.ceil(pct * flat.numel())))
        vals = torch.topk(flat.abs(), k, largest=True).values
        out[col] = float(torch.sum(vals * vals).item() / total)
    return out


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def cast_for_delta(src: torch.Tensor, opd: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src_cast = src.detach().to(dtype).cpu()
    opd_cast = opd.detach().to(dtype).cpu()
    delta_cast = opd_cast - src_cast
    return src_cast, opd_cast, delta_cast


def compute_tensor_metrics(name: str, src: torch.Tensor, opd: torch.Tensor, analysis_dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
    src_cast, opd_cast, delta_cast = cast_for_delta(src, opd, analysis_dtype)
    src = src_cast.to(torch.float32)
    opd = opd_cast.to(torch.float32)
    delta = delta_cast.to(torch.float32)
    info = parse_tensor_name(name)
    has_bad = bool((~torch.isfinite(src)).any().item() or (~torch.isfinite(opd)).any().item() or (~torch.isfinite(delta)).any().item())
    src_energy = float(torch.sum(src * src).item())
    opd_energy = float(torch.sum(opd * opd).item())
    delta_energy = float(torch.sum(delta * delta).item())
    numel = delta.numel()
    src_fro = math.sqrt(max(src_energy, 0.0))
    opd_fro = math.sqrt(max(opd_energy, 0.0))
    delta_fro = math.sqrt(max(delta_energy, 0.0))
    src_rms = math.sqrt(src_energy / max(numel, 1))
    opd_rms = math.sqrt(opd_energy / max(numel, 1))
    delta_rms = math.sqrt(delta_energy / max(numel, 1))
    row = {
        "name": name,
        "shape": json.dumps(list(delta.shape)),
        "ndim": delta.ndim,
        "numel": numel,
        "analysis_dtype": str(analysis_dtype).replace("torch.", ""),
        **info,
        "src_fro_norm": src_fro,
        "opd_fro_norm": opd_fro,
        "delta_fro_norm": delta_fro,
        "relative_delta_fro": delta_fro / (src_fro + EPS),
        "delta_l2_norm": delta_fro,
        "delta_linf_norm": float(delta.abs().max().item()) if numel else 0.0,
        "delta_mean_abs": float(delta.abs().mean().item()) if numel else 0.0,
        "delta_rms": delta_rms,
        "src_rms": src_rms,
        "opd_rms": opd_rms,
        "has_nan_or_inf": has_bad,
        "src_energy": src_energy,
        "opd_energy": opd_energy,
        "delta_energy": delta_energy,
    }
    row.update(compute_coordinate_sparsity(delta_cast, src_rms))
    row.update(compute_coordinate_energy_concentration(delta))
    return row


def compute_svd_metrics(
    name: str,
    delta: torch.Tensor,
    device: torch.device,
    max_exact_svd_dim: int,
    topk_svd: int,
    approx_svd_max_numel: Optional[int],
) -> Optional[Dict[str, Any]]:
    if delta.ndim != 2:
        return None
    m, n = delta.shape
    min_dim = min(m, n)
    if min_dim == 0:
        return None
    info = parse_tensor_name(name)
    row = {
        "name": name,
        "shape": json.dumps([m, n]),
        **info,
        "matrix_m": m,
        "matrix_n": n,
        "min_dim": min_dim,
    }
    delta = delta.to(torch.float32)
    delta_fro_sq = float(torch.sum(delta * delta).item())
    if delta_fro_sq <= 0:
        return None

    if min_dim <= max_exact_svd_dim:
        sigma = torch.linalg.svdvals(delta.to(device)).detach().cpu().float()
        row.update(_svd_exact_row(sigma, delta_fro_sq))
        row["svd_mode"] = "exact"
        return row

    if topk_svd <= 0:
        row["svd_mode"] = "skipped"
        row["skipped_reason"] = "topk_svd<=0"
        return row
    if approx_svd_max_numel is not None and delta.numel() > approx_svd_max_numel:
        row["svd_mode"] = "skipped"
        row["skipped_reason"] = f"numel>{approx_svd_max_numel}"
        return row

    q = min(topk_svd, min_dim)
    _, sigma, _ = torch.svd_lowrank(delta.to(device), q=q, niter=2)
    sigma = sigma.detach().cpu().float()
    row.update(_svd_approx_row(sigma, delta_fro_sq))
    row["svd_mode"] = "approximate_topk"
    return row


def _svd_exact_row(sigma: torch.Tensor, fro_sq: float) -> Dict[str, float]:
    total = float(torch.sum(sigma * sigma).item())
    spectral = float(sigma[0].item()) if len(sigma) else 0.0
    probs = sigma / (float(torch.sum(sigma).item()) + EPS)
    entropy = float(torch.exp(-(probs * torch.log(probs + EPS)).sum()).item())
    min_dim = max(int(len(sigma)), 1)
    rank_tol_1e_6 = int((sigma > 1e-6 * spectral).sum().item())
    rank_tol_1e_5 = int((sigma > 1e-5 * spectral).sum().item())
    rank_tol_1e_4 = int((sigma > 1e-4 * spectral).sum().item())
    row = {
        "spectral_norm": spectral,
        "spectral_fro_ratio": spectral / (math.sqrt(fro_sq) + EPS),
        "stable_rank": fro_sq / (spectral * spectral + EPS),
        "effective_rank_entropy": entropy,
        "rank_tol_1e_6": rank_tol_1e_6,
        "rank_pct_tol_1e_6": rank_tol_1e_6 / min_dim,
        "rank_tol_1e_5": rank_tol_1e_5,
        "rank_pct_tol_1e_5": rank_tol_1e_5 / min_dim,
        "rank_tol_1e_4": rank_tol_1e_4,
        "rank_pct_tol_1e_4": rank_tol_1e_4 / min_dim,
    }
    for k in SVD_TOPKS:
        row[f"top{k}_energy_ratio"] = _sigma_energy_ratio(sigma, k, total)
    for pct in (0.01, 0.05, 0.10):
        k = max(1, int(pct * len(sigma)))
        row[f"top{int(pct * 100)}pct_energy_ratio"] = _sigma_energy_ratio(sigma, k, total)
    return row


def _svd_approx_row(sigma: torch.Tensor, fro_sq: float) -> Dict[str, float]:
    spectral = float(sigma[0].item()) if len(sigma) else float("nan")
    row = {
        "spectral_norm_approx": spectral,
        "spectral_fro_ratio_approx": spectral / (math.sqrt(fro_sq) + EPS),
        "effective_rank_entropy": float("nan"),
        "rank_tol_1e_6": float("nan"),
        "rank_pct_tol_1e_6": float("nan"),
        "rank_tol_1e_5": float("nan"),
        "rank_pct_tol_1e_5": float("nan"),
        "rank_tol_1e_4": float("nan"),
        "rank_pct_tol_1e_4": float("nan"),
    }
    for k in SVD_TOPKS:
        row[f"approx_top{k}_energy_ratio"] = _sigma_energy_ratio(sigma, k, fro_sq)
    return row


def _sigma_energy_ratio(sigma: torch.Tensor, k: int, denom: float) -> float:
    if denom <= 0 or len(sigma) == 0:
        return float("nan")
    k = min(k, len(sigma))
    return float(torch.sum(sigma[:k] * sigma[:k]).item() / denom)


def compute_base_geometry_metrics(
    name: str,
    src: torch.Tensor,
    opd: torch.Tensor,
    device: torch.device,
    max_exact_svd_dim: int,
) -> Optional[Dict[str, Any]]:
    if src.ndim != 2:
        return None
    m, n = src.shape
    min_dim = min(m, n)
    info = parse_tensor_name(name)
    row = {
        "name": name,
        "shape": json.dumps([m, n]),
        **info,
        "base_geometry_mode": "exact" if min_dim <= max_exact_svd_dim else "skipped",
    }
    if min_dim > max_exact_svd_dim:
        row["skipped_reason"] = f"min_dim>{max_exact_svd_dim}"
        return row
    src = src.to(torch.float32).to(device)
    opd = opd.to(torch.float32).to(device)
    delta = opd - src
    delta_fro_sq = torch.sum(delta * delta)
    if float(delta_fro_sq.item()) <= 0:
        row["skipped_reason"] = "zero_delta"
        return row
    u_src, s_src, vh_src = torch.linalg.svd(src, full_matrices=False)
    u_opd, s_opd, _ = torch.linalg.svd(opd, full_matrices=False)
    for ratio in PROJ_RATIOS:
        label = _fmt_ratio_label(ratio)
        k = max(1, int(ratio * min_dim))
        uk = u_src[:, :k]
        vk = vh_src[:k, :].T
        both = uk.T @ delta @ vk
        left = uk.T @ delta
        right = delta @ vk
        row[f"principal_projection_energy_ratio_{label}"] = float(torch.sum(both * both).item() / (float(delta_fro_sq.item()) + EPS))
        row[f"left_principal_projection_energy_ratio_{label}"] = float(torch.sum(left * left).item() / (float(delta_fro_sq.item()) + EPS))
        row[f"right_principal_projection_energy_ratio_{label}"] = float(torch.sum(right * right).item() / (float(delta_fro_sq.item()) + EPS))
        uok = u_opd[:, :k]
        _, cos, _ = torch.linalg.svd(uk.T @ uok, full_matrices=False)
        sin_dist = torch.sqrt(torch.clamp(1.0 - cos * cos, min=0.0))
        row[f"top_subspace_rotation_{label}"] = float(sin_dist.mean().item())
    row["spectral_drift_l2_ratio"] = float(torch.linalg.vector_norm(s_opd - s_src).item() / (torch.linalg.vector_norm(s_src).item() + EPS))
    return {k: (v.detach().cpu().item() if isinstance(v, torch.Tensor) else v) for k, v in row.items()}


def compute_coordinate_mask_overlap_metrics(
    name: str,
    src: torch.Tensor,
    delta: torch.Tensor,
    device: torch.device,
    max_exact_svd_dim: int,
    update_atol: float = 0.0,
    ratios: Sequence[float] = MASK_OVERLAP_RATIOS,
) -> Optional[Dict[str, Any]]:
    """Compare visible update coordinates with source principal/low-magnitude masks."""
    if src.ndim != 2:
        return None
    m, n = src.shape
    min_dim = min(m, n)
    info = parse_tensor_name(name)
    row: Dict[str, Any] = {
        "name": name,
        "shape": json.dumps([m, n]),
        **info,
        "mask_overlap_mode": "exact" if min_dim <= max_exact_svd_dim else "skipped",
        "update_mask_atol": update_atol,
    }
    if min_dim > max_exact_svd_dim:
        row["skipped_reason"] = f"min_dim>{max_exact_svd_dim}"
        return row

    src_d = src.to(torch.float32).to(device)
    delta_d = delta.to(torch.float32).to(device)
    numel = max(delta_d.numel(), 1)
    update_mask = delta_d.abs() > update_atol
    update_count = int(update_mask.sum().item())
    row["numel"] = int(delta_d.numel())
    row["update_count"] = update_count
    row["update_density"] = update_count / numel

    u_src, s_src, vh_src = torch.linalg.svd(src_d, full_matrices=False)
    src_abs = src_d.abs()
    for ratio in ratios:
        label = _fmt_ratio_label(ratio)
        rank = max(1, int(ratio * min_dim))
        score = torch.abs((u_src[:, :rank] * s_src[:rank]) @ vh_src[:rank, :])
        principal_mask = _fraction_mask(score, ratio, largest=True)
        low_mask = _fraction_mask(src_abs, ratio, largest=False)
        principal_not_low_mask = principal_mask & ~low_mask
        nonprincipal_or_low_mask = ~principal_mask | low_mask

        row[f"principal_rank_{label}"] = rank
        _add_mask_overlap_fields(row, f"principal_{label}", principal_mask, update_mask, numel, update_count)
        _add_mask_overlap_fields(row, f"low_magnitude_{label}", low_mask, update_mask, numel, update_count)
        _add_mask_overlap_fields(row, f"principal_not_low_{label}", principal_not_low_mask, update_mask, numel, update_count)
        _add_mask_overlap_fields(row, f"nonprincipal_or_low_{label}", nonprincipal_or_low_mask, update_mask, numel, update_count)
        row[f"principal_low_intersection_density_{label}"] = _mask_density(principal_mask & low_mask, numel)

        del score, principal_mask, low_mask, principal_not_low_mask, nonprincipal_or_low_mask

    return {k: (v.detach().cpu().item() if isinstance(v, torch.Tensor) else v) for k, v in row.items()}


def _fraction_mask(values: torch.Tensor, ratio: float, largest: bool) -> torch.Tensor:
    flat = values.flatten()
    numel = flat.numel()
    if numel == 0:
        return torch.zeros_like(flat, dtype=torch.bool).reshape_as(values)
    k = max(1, int(math.ceil(ratio * numel)))
    if k >= numel:
        return torch.ones_like(flat, dtype=torch.bool).reshape_as(values)
    indices = torch.topk(flat, k, largest=largest).indices
    mask = torch.zeros(numel, dtype=torch.bool, device=values.device)
    mask[indices] = True
    return mask.reshape_as(values)


def _mask_density(mask: torch.Tensor, numel: int) -> float:
    return int(mask.sum().item()) / max(numel, 1)


def _add_mask_overlap_fields(
    row: Dict[str, Any],
    prefix: str,
    mask: torch.Tensor,
    update_mask: torch.Tensor,
    numel: int,
    update_count: int,
) -> None:
    mask_count = int(mask.sum().item())
    intersection = int((mask & update_mask).sum().item())
    density = mask_count / max(numel, 1)
    update_overlap = intersection / update_count if update_count > 0 else float("nan")
    mask_covered_by_update = intersection / mask_count if mask_count > 0 else float("nan")
    row[f"{prefix}_density"] = density
    row[f"{prefix}_update_overlap_ratio"] = update_overlap
    row[f"{prefix}_covered_by_update_ratio"] = mask_covered_by_update
    row[f"{prefix}_enrichment_vs_random"] = update_overlap / (density + EPS) if update_count > 0 else float("nan")


def aggregate_metrics(tensor_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    total_delta_energy = float(tensor_df["delta_energy"].sum())
    out = {}
    for name, group_cols in {
        "layer_summary": ["layer_id"],
        "module_summary": ["module_type"],
        "block_summary": ["block_type"],
        "layer_module_summary": ["layer_id", "module_type"],
    }.items():
        rows = []
        for keys, group in tensor_df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            delta_energy = float(group["delta_energy"].sum())
            src_energy = float(group["src_energy"].sum())
            row.update(
                {
                    "num_tensors": int(len(group)),
                    "total_numel": int(group["numel"].sum()),
                    "total_delta_energy": delta_energy,
                    "delta_energy_share": delta_energy / (total_delta_energy + EPS),
                    "total_src_energy": src_energy,
                    "group_relative_delta": math.sqrt(delta_energy) / (math.sqrt(src_energy) + EPS),
                }
            )
            for col in _metric_cols(group, prefix="frac_abs_delta_lt_"):
                row[col] = _weighted_mean(group[col], group["numel"])
            if "frac_delta_eq_0" in group:
                row["frac_delta_eq_0"] = _weighted_mean(group["frac_delta_eq_0"], group["numel"])
            for col in _metric_cols(group, prefix="frac_isclose_zero_atol_"):
                row[col] = _weighted_mean(group[col], group["numel"])
            for col in _metric_cols(group, prefix="coord_top_"):
                row[col] = _weighted_mean(group[col], group["delta_energy"])
            rows.append(row)
        out[name] = pd.DataFrame(rows).sort_values("total_delta_energy", ascending=False)
    top = tensor_df.copy()
    top["delta_energy_share"] = top["delta_energy"] / (total_delta_energy + EPS)
    out["top_delta_tensors"] = top.sort_values("delta_energy_share", ascending=False).head(100)
    return out


def _metric_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def make_plots(out_dir: Path, tensor_df: pd.DataFrame, summaries: Dict[str, pd.DataFrame], svd_df: pd.DataFrame, base_df: pd.DataFrame) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"plots skipped: matplotlib import failed: {exc}"]

    warnings = []
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def save_bar(df: pd.DataFrame, x: str, y: str, title: str, path: str, limit: int = 30) -> None:
        data = df.head(limit).copy()
        plt.figure(figsize=(max(8, min(18, len(data) * 0.45)), 5))
        plt.bar(data[x].astype(str), data[y])
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        plt.savefig(plot_dir / path, dpi=180)
        plt.close()

    try:
        save_bar(summaries["layer_summary"].sort_values("layer_id"), "layer_id", "delta_energy_share", "Delta energy share by layer", "layer_delta_energy_share.png")
        save_bar(summaries["module_summary"], "module_type", "delta_energy_share", "Delta energy share by module", "module_delta_energy_share.png")
        save_bar(summaries["block_summary"], "block_type", "delta_energy_share", "Delta energy share by block", "block_delta_energy_share.png")
        save_bar(summaries["layer_summary"].sort_values("layer_id"), "layer_id", "group_relative_delta", "Relative delta by layer", "layer_relative_delta.png")
        save_bar(summaries["top_delta_tensors"], "name", "delta_energy_share", "Top delta tensors", "top_delta_tensors_bar.png", limit=20)

        sparsity_col = "frac_abs_delta_lt_1e-5"
        if sparsity_col in summaries["layer_summary"].columns:
            save_bar(summaries["layer_summary"].sort_values("layer_id"), "layer_id", sparsity_col, "Absolute sparsity by layer", "sparsity_by_layer.png")

        heat = summaries["layer_module_summary"].pivot(index="layer_id", columns="module_type", values="delta_energy_share").fillna(0)
        plt.figure(figsize=(10, 7))
        plt.imshow(heat.values, aspect="auto")
        plt.title("Layer-module delta energy share")
        plt.xlabel("module_type")
        plt.ylabel("layer_id")
        plt.xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
        plt.yticks(range(len(heat.index)), heat.index)
        plt.colorbar(label="delta_energy_share")
        plt.tight_layout()
        plt.savefig(plot_dir / "layer_module_delta_energy_heatmap.png", dpi=180)
        plt.close()

        if not svd_df.empty:
            col = "top8_energy_ratio" if "top8_energy_ratio" in svd_df.columns else "approx_top8_energy_ratio"
            if col in svd_df.columns:
                svd_df.groupby("module_type")[col].mean().plot(kind="bar", figsize=(8, 5), title="Top-8 spectral energy by module")
                plt.ylabel(col)
                plt.tight_layout()
                plt.savefig(plot_dir / "spectral_topk_energy_by_module.png", dpi=180)
                plt.close()
            if "effective_rank_entropy" in svd_df.columns:
                svd_df.groupby("module_type")["effective_rank_entropy"].mean().plot(kind="bar", figsize=(8, 5), title="Effective rank by module")
                plt.tight_layout()
                plt.savefig(plot_dir / "effective_rank_by_module.png", dpi=180)
                plt.close()
        if not base_df.empty and "principal_projection_energy_ratio_10pct" in base_df.columns:
            base_df.groupby("module_type")["principal_projection_energy_ratio_10pct"].mean().plot(kind="bar", figsize=(8, 5), title="10% principal projection by module")
            plt.tight_layout()
            plt.savefig(plot_dir / "principal_projection_by_module.png", dpi=180)
            plt.close()
    except Exception as exc:
        warnings.append(f"plot generation partially failed: {exc}")
    return warnings


def write_summary(
    out_dir: Path,
    tensor_df: pd.DataFrame,
    summaries: Dict[str, pd.DataFrame],
    svd_df: pd.DataFrame,
    base_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    mismatch_report: Dict[str, Any],
    failed_rows: List[Dict[str, Any]],
    extra_warnings: List[str],
) -> Dict[str, Any]:
    total_src_energy = float(tensor_df["src_energy"].sum())
    total_opd_energy = float(tensor_df["opd_energy"].sum())
    total_delta_energy = float(tensor_df["delta_energy"].sum())
    summary: Dict[str, Any] = {
        "total_tensors_analyzed": int(len(tensor_df)),
        "total_parameters_analyzed": int(tensor_df["numel"].sum()),
        "analysis_dtype": str(tensor_df["analysis_dtype"].iloc[0]) if "analysis_dtype" in tensor_df and len(tensor_df) else "unknown",
        "total_source_norm": math.sqrt(total_src_energy),
        "total_opd_norm": math.sqrt(total_opd_energy),
        "total_delta_norm": math.sqrt(total_delta_energy),
        "global_relative_delta_norm": math.sqrt(total_delta_energy) / (math.sqrt(total_src_energy) + EPS),
        "top_10_layers_by_delta_energy_share": summaries["layer_summary"].head(10).to_dict(orient="records"),
        "top_10_module_types_by_delta_energy_share": summaries["module_summary"].head(10).to_dict(orient="records"),
        "top_20_tensors_by_delta_energy_share": summaries["top_delta_tensors"].head(20).to_dict(orient="records"),
        "warnings": {
            "mismatched_tensors": len(mismatch_report.get("shape_mismatched", [])),
            "missing_in_src": len(mismatch_report.get("missing_in_src", [])),
            "missing_in_opd": len(mismatch_report.get("missing_in_opd", [])),
            "skipped_tensors": int(mismatch_report.get("skipped_non_floating", 0)),
            "failed_tensors": len(failed_rows),
            "approximate_svd_tensors": int((svd_df.get("svd_mode", pd.Series(dtype=str)) == "approximate_topk").sum()) if not svd_df.empty else 0,
            "skipped_base_geometry_tensors": int((base_df.get("base_geometry_mode", pd.Series(dtype=str)) == "skipped").sum()) if not base_df.empty else 0,
            "skipped_mask_overlap_tensors": int((mask_df.get("mask_overlap_mode", pd.Series(dtype=str)) == "skipped").sum()) if not mask_df.empty else 0,
            "extra": extra_warnings,
        },
    }
    for col in _metric_cols(tensor_df, "frac_abs_delta_lt_"):
        summary[f"global_{col}"] = _weighted_mean(tensor_df[col], tensor_df["numel"])
    if "frac_delta_eq_0" in tensor_df:
        summary["global_frac_delta_eq_0"] = _weighted_mean(tensor_df["frac_delta_eq_0"], tensor_df["numel"])
    for col in _metric_cols(tensor_df, "frac_isclose_zero_atol_"):
        summary[f"global_{col}"] = _weighted_mean(tensor_df[col], tensor_df["numel"])
    for col in _metric_cols(tensor_df, "coord_top_"):
        summary[f"global_{col}"] = _weighted_mean(tensor_df[col], tensor_df["delta_energy"])

    for col in (
        "spectral_fro_ratio",
        "effective_rank_entropy",
        "stable_rank",
        "rank_pct_tol_1e_6",
        "rank_pct_tol_1e_5",
        "rank_pct_tol_1e_4",
        "top1_energy_ratio",
        "top8_energy_ratio",
        "top16_energy_ratio",
    ):
        if col in svd_df.columns:
            vals = pd.to_numeric(svd_df[col], errors="coerce").dropna()
            summary[f"mean_{col}"] = float(vals.mean()) if len(vals) else float("nan")
            summary[f"median_{col}"] = float(vals.median()) if len(vals) else float("nan")
    for ratio in PROJ_RATIOS:
        col = f"principal_projection_energy_ratio_{_fmt_ratio_label(ratio)}"
        if col in base_df.columns:
            vals = pd.to_numeric(base_df[col], errors="coerce").dropna()
            summary[f"mean_{col}"] = float(vals.mean()) if len(vals) else float("nan")
            summary[f"median_{col}"] = float(vals.median()) if len(vals) else float("nan")
    for ratio in MASK_OVERLAP_RATIOS:
        label = _fmt_ratio_label(ratio)
        for prefix in ("principal", "low_magnitude", "principal_not_low", "nonprincipal_or_low"):
            col = f"{prefix}_{label}_update_overlap_ratio"
            if col in mask_df.columns:
                vals = pd.to_numeric(mask_df[col], errors="coerce").dropna()
                weights = pd.to_numeric(mask_df.get("update_count", pd.Series(dtype=float)), errors="coerce")
                if len(vals):
                    summary[f"mean_{col}"] = float(vals.mean())
                    summary[f"median_{col}"] = float(vals.median())
                    summary[f"update_weighted_mean_{col}"] = _weighted_mean(mask_df[col], weights)
            enrich_col = f"{prefix}_{label}_enrichment_vs_random"
            if enrich_col in mask_df.columns:
                vals = pd.to_numeric(mask_df[enrich_col], errors="coerce").dropna()
                if len(vals):
                    summary[f"mean_{enrich_col}"] = float(vals.mean())
                    summary[f"median_{enrich_col}"] = float(vals.median())

    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    write_summary_md(out_dir, summary, summaries, svd_df, base_df, mask_df)
    return summary


def write_summary_md(out_dir: Path, summary: Dict[str, Any], summaries: Dict[str, pd.DataFrame], svd_df: pd.DataFrame, base_df: pd.DataFrame, mask_df: pd.DataFrame) -> None:
    exact_zero = summary.get("global_frac_delta_eq_0", float("nan"))
    isclose_1e5 = summary.get("global_frac_isclose_zero_atol_1e-5", float("nan"))
    sparsity = summary.get("global_frac_abs_delta_lt_1e-5", float("nan"))
    rel_sparsity = summary.get("global_frac_abs_delta_lt_1e-3_src_rms", float("nan"))
    top_coord = summary.get("global_coord_top_1pct_energy_ratio", float("nan"))
    lines = [
        "# OPD Delta Analysis Summary",
        "",
        f"- Tensors analyzed: {summary['total_tensors_analyzed']}",
        f"- Parameters analyzed: {summary['total_parameters_analyzed']}",
        f"- Analysis dtype: {summary.get('analysis_dtype', 'unknown')}",
        f"- Global relative delta norm: {summary['global_relative_delta_norm']:.6g}",
        f"- Global exact-zero delta fraction: {_fmt_float(exact_zero)}",
        f"- Global `isclose(delta, 0, atol=1e-5)` fraction: {_fmt_float(isclose_1e5)}",
        f"- Global `abs(delta) < 1e-5` fraction: {_fmt_float(sparsity)}",
        f"- Global `abs(delta) < 1e-3 * src_rms` fraction: {_fmt_float(rel_sparsity)}",
        f"- Global top 1% coordinate energy ratio: {_fmt_float(top_coord)}",
        "",
        "## Interpretation",
        "",
        f"The final OPD delta has global relative Frobenius norm `{summary['global_relative_delta_norm']:.6g}`. This measures the update scale relative to the source checkpoint, not its downstream usefulness.",
        "",
        f"Coordinate sparsity should be interpreted cautiously. The exact-zero fraction is `{_fmt_float(exact_zero)}` and `isclose(delta, 0, atol=1e-5)` fraction is `{_fmt_float(isclose_1e5)}`. At strict threshold `abs(delta) < 1e-5`, the fraction below threshold is `{_fmt_float(sparsity)}`; at `1e-3 * src_rms`, it is `{_fmt_float(rel_sparsity)}`. The top 1% coordinate energy ratio is `{_fmt_float(top_coord)}`, indicating how concentrated the squared update energy is in the largest coordinates.",
        "",
        "The modules with the largest delta energy share are:",
    ]
    for row in summaries["module_summary"].head(8).to_dict(orient="records"):
        lines.append(f"- `{row['module_type']}`: {row['delta_energy_share']:.4%}")

    if not svd_df.empty:
        exact = svd_df[svd_df.get("svd_mode") == "exact"] if "svd_mode" in svd_df else pd.DataFrame()
        if not exact.empty and "top8_energy_ratio" in exact:
            lines += [
                "",
                f"Exact-SVD matrices have median top-8 spectral energy ratio `{exact['top8_energy_ratio'].median():.6g}` and median stable rank `{exact['stable_rank'].median():.6g}`. This suggests, but does not prove, the degree to which 2D updates are rank-k dominated.",
            ]
        if not exact.empty and "rank_pct_tol_1e_5" in exact:
            lines += [
                "",
                f"Exact-SVD matrices have median numerical rank percentage `{exact['rank_pct_tol_1e_5'].median():.6g}` at tolerance `1e-5 * spectral_norm` and `{exact['rank_pct_tol_1e_4'].median():.6g}` at tolerance `1e-4 * spectral_norm`.",
            ]
        approx = svd_df[svd_df.get("svd_mode") == "approximate_topk"] if "svd_mode" in svd_df else pd.DataFrame()
        if not approx.empty:
            lines.append(f"{len(approx)} matrices used approximate top-k SVD, so their spectral metrics should be read as approximate lower-bound style diagnostics.")

    if not base_df.empty:
        col = "principal_projection_energy_ratio_10pct"
        vals = pd.to_numeric(base_df.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        if len(vals):
            lines += [
                "",
                f"The median 10% source principal-subspace projection energy ratio is `{vals.median():.6g}` among tensors where exact base geometry was computed. This suggests, but does not prove, how much update energy lies in the leading singular directions of the base weights.",
            ]

    if not mask_df.empty:
        exact_mask = mask_df[mask_df.get("mask_overlap_mode") == "exact"] if "mask_overlap_mode" in mask_df else mask_df
        if not exact_mask.empty:
            principal_col = "principal_10pct_update_overlap_ratio"
            low_col = "low_magnitude_10pct_update_overlap_ratio"
            safe_col = "nonprincipal_or_low_10pct_update_overlap_ratio"
            if principal_col in exact_mask.columns and low_col in exact_mask.columns:
                principal = pd.to_numeric(exact_mask[principal_col], errors="coerce").dropna()
                low = pd.to_numeric(exact_mask[low_col], errors="coerce").dropna()
                safe = pd.to_numeric(exact_mask.get(safe_col, pd.Series(dtype=float)), errors="coerce").dropna()
                if len(principal) and len(low):
                    sentence = (
                        f"Coordinate-mask overlap at the 10% principal/low-magnitude setting has median update overlap "
                        f"`{principal.median():.6g}` for source principal coordinates and `{low.median():.6g}` for source low-magnitude coordinates."
                    )
                    if len(safe):
                        sentence += f" The median overlap with the non-principal-or-low mask is `{safe.median():.6g}`."
                    lines += ["", sentence]

    lines += [
        "",
        "Functional evaluation is needed to determine whether these directions are actually useful. Static delta analysis cannot reveal training-time dynamics.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_intervention_checkpoints(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("Intervention checkpoint writing is intentionally not enabled in this implementation.")


def copy_model_metadata_files(src_model: Path, out_dir: Path) -> None:
    for path in src_model.iterdir():
        if path.is_file() and not fnmatch.fnmatch(path.name, "*.safetensors") and not fnmatch.fnmatch(path.name, "*.bin"):
            shutil.copy2(path, out_dir / path.name)


def _fmt_thresh(value: float) -> str:
    base, exp = f"{value:.0e}".split("e")
    return f"{base}e{int(exp)}"


def _fmt_pct(pct: float) -> str:
    return {0.001: "0p1pct", 0.01: "1pct", 0.05: "5pct"}[pct]


def _fmt_ratio_label(ratio: float) -> str:
    return f"{int(round(ratio * 100))}pct"


def _fmt_float(value: Any) -> str:
    try:
        if math.isnan(float(value)):
            return "NaN"
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _write_failed(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = ["name", "stage", "error_message", "traceback_short"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def _failure(name: str, stage: str, exc: BaseException) -> Dict[str, str]:
    return {
        "name": name,
        "stage": stage,
        "error_message": str(exc),
        "traceback_short": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_model", type=Path, required=True)
    parser.add_argument("--opd_model", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["float32", "bfloat16", "bf16", "float16", "fp16"], help="Arithmetic dtype used before forming parameter deltas.")
    parser.add_argument("--max_exact_svd_dim", type=int, default=2048)
    parser.add_argument("--topk_svd", type=int, default=64)
    parser.add_argument("--approx_svd_max_numel", type=int, default=None, help="Skip approximate SVD for matrices larger than this many elements.")
    parser.add_argument("--mask_overlap_update_atol", type=float, default=0.0, help="Visible-update mask threshold: abs(delta) > atol.")
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--write_interventions", action="store_true")
    parser.add_argument("--intervention_out_dir", type=Path)
    parser.add_argument("--intervention_modes", default="")
    parser.add_argument("--limit_tensors", type=int, default=None, help="Debug/test option.")
    args = parser.parse_args()

    if args.write_interventions:
        raise SystemExit("Intervention checkpoint generation is not enabled; run analysis without --write_interventions.")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    analysis_dtype = _dtype_from_name(args.dtype)

    src_store = load_tensor_map(args.src_model)
    opd_store = load_tensor_map(args.opd_model)
    src_keys = src_store.keys()
    opd_keys = opd_store.keys()
    common_pairs, key_match_report = match_tensor_keys(src_keys, opd_keys)
    mismatch_report: Dict[str, Any] = {
        **key_match_report,
        "shape_mismatched": [],
        "skipped_non_floating": 0,
    }
    failed_rows: List[Dict[str, Any]] = []
    tensor_rows: List[Dict[str, Any]] = []
    svd_rows: List[Dict[str, Any]] = []
    base_rows: List[Dict[str, Any]] = []
    mask_rows: List[Dict[str, Any]] = []

    if args.limit_tensors is not None:
        common_pairs = common_pairs[: args.limit_tensors]

    for src_name, opd_name in tqdm(common_pairs, desc="analyzing tensors"):
        name = src_name
        try:
            src = src_store.get(src_name)
            opd = opd_store.get(opd_name)
        except Exception as exc:
            failed_rows.append(_failure(name, "load", exc))
            continue

        if tuple(src.shape) != tuple(opd.shape):
            mismatch_report["shape_mismatched"].append({"src_name": src_name, "opd_name": opd_name, "src_shape": list(src.shape), "opd_shape": list(opd.shape)})
            continue
        if not (src.is_floating_point() and opd.is_floating_point()):
            mismatch_report["skipped_non_floating"] += 1
            continue

        try:
            row = compute_tensor_metrics(name, src, opd, analysis_dtype)
            tensor_rows.append(row)
        except Exception as exc:
            failed_rows.append(_failure(name, "delta_metrics", exc))
            continue

        src_cast, _, delta_cast = cast_for_delta(src, opd, analysis_dtype)
        src_f = src_cast.to(torch.float32)
        delta = delta_cast.to(torch.float32)
        opd_f = src_f + delta
        try:
            svd_row = compute_svd_metrics(name, delta, device, args.max_exact_svd_dim, args.topk_svd, args.approx_svd_max_numel)
            if svd_row:
                svd_rows.append(svd_row)
        except Exception as exc:
            failed_rows.append(_failure(name, "svd_metrics", exc))
        try:
            base_row = compute_base_geometry_metrics(name, src_f, opd_f, device, args.max_exact_svd_dim)
            if base_row:
                base_rows.append(base_row)
        except Exception as exc:
            failed_rows.append(_failure(name, "base_geometry", exc))
        try:
            mask_row = compute_coordinate_mask_overlap_metrics(name, src_f, delta, device, args.max_exact_svd_dim, update_atol=args.mask_overlap_update_atol)
            if mask_row:
                mask_rows.append(mask_row)
        except Exception as exc:
            failed_rows.append(_failure(name, "mask_overlap", exc))

        del src, opd, src_f, opd_f, delta

    (out_dir / "mismatch_report.json").write_text(json.dumps(_jsonable(mismatch_report), indent=2), encoding="utf-8")
    _write_failed(out_dir / "failed_tensors.csv", failed_rows)
    if not tensor_rows:
        raise SystemExit("No tensors were analyzed; inspect mismatch_report.json and failed_tensors.csv")

    tensor_df = pd.DataFrame(tensor_rows)
    tensor_df.to_csv(out_dir / "tensor_metrics.csv", index=False)
    svd_df = pd.DataFrame(svd_rows)
    svd_df.to_csv(out_dir / "svd_metrics.csv", index=False)
    base_df = pd.DataFrame(base_rows)
    base_df.to_csv(out_dir / "base_geometry_metrics.csv", index=False)
    mask_df = pd.DataFrame(mask_rows)
    mask_df.to_csv(out_dir / "mask_overlap_metrics.csv", index=False)

    summaries = aggregate_metrics(tensor_df)
    for name, df in summaries.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    plot_warnings = make_plots(out_dir, tensor_df, summaries, svd_df, base_df) if args.make_plots else []
    write_summary(out_dir, tensor_df, summaries, svd_df, base_df, mask_df, mismatch_report, failed_rows, plot_warnings)
    src_store.close()
    opd_store.close()


if __name__ == "__main__":
    main()
