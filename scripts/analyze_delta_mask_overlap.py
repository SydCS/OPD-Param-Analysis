#!/usr/bin/env python3
"""Compare zero/nonzero delta masks for two tuned checkpoints against one base."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from analyze_opd_delta import canonical_tensor_key, discover_checkpoint_files, load_tensor_map, parse_tensor_name

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover - plotting is optional.
    plt = None
    sns = None


def resolve_dtype(name: str) -> torch.dtype:
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16"}:
        return torch.float16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug[:80] or "checkpoint"


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def write_safetensor_shard(tensors: Dict[str, torch.Tensor], out_dir: Path, shard_idx: int) -> Dict[str, Any]:
    filename = f"model-{shard_idx:05d}.safetensors"
    save_file(tensors, str(out_dir / filename))
    return {
        "file": filename,
        "num_tensors": len(tensors),
        "num_bytes": sum(tensor_nbytes(t) for t in tensors.values()),
    }


def materialize_to_safetensors(model_path: Path, out_dir: Path, *, max_shard_bytes: int, force: bool) -> Dict[str, Any]:
    manifest_path = out_dir / "merge_manifest.json"
    if not force and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete") and list(out_dir.glob("*.safetensors")):
            return {**manifest, "reused": True}

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.safetensors"):
        stale.unlink()

    store = load_tensor_map(model_path)
    shards: List[Dict[str, Any]] = []
    current: Dict[str, torch.Tensor] = {}
    current_bytes = 0
    num_tensors = 0
    numel = 0
    try:
        for name in tqdm(store.keys(), desc=f"materializing {model_path.name}"):
            tensor = store.get(name)
            if not isinstance(tensor, torch.Tensor):
                continue
            tensor = tensor.detach().cpu().contiguous()
            size = tensor_nbytes(tensor)
            if current and current_bytes + size > max_shard_bytes:
                shards.append(write_safetensor_shard(current, out_dir, len(shards) + 1))
                current = {}
                current_bytes = 0
            current[name] = tensor
            current_bytes += size
            num_tensors += 1
            numel += int(tensor.numel())
        if current:
            shards.append(write_safetensor_shard(current, out_dir, len(shards) + 1))
    finally:
        store.close()

    manifest = {
        "source_model": str(model_path),
        "merged_model": str(out_dir),
        "format": "safetensors",
        "complete": True,
        "reused": False,
        "num_tensors": num_tensors,
        "numel": numel,
        "max_shard_bytes": max_shard_bytes,
        "shards": shards,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def maybe_materialize_model(
    model_path: Path,
    *,
    slot: str,
    label: str,
    merge_enabled: bool,
    merge_root: Path,
    max_shard_bytes: int,
    force: bool,
) -> Tuple[Path, Dict[str, Any]]:
    info = discover_checkpoint_files(model_path)
    record: Dict[str, Any] = {
        "slot": slot,
        "label": label,
        "original_model": str(model_path),
        "checkpoint_kind": info["kind"],
        "used_model": str(model_path),
        "merged": False,
    }
    if not merge_enabled or info["kind"] != "dtensor_sharded_torch":
        return model_path, record

    digest = hashlib.sha1(str(model_path).encode("utf-8")).hexdigest()[:10]
    target = merge_root / f"{slot}_{safe_slug(label)}_{digest}"
    manifest = materialize_to_safetensors(model_path, target, max_shard_bytes=max_shard_bytes, force=force)
    record.update(
        {
            "used_model": str(target),
            "merged": True,
            "merged_model": str(target),
            "merge_manifest": str(target / "merge_manifest.json"),
            "merge_reused": bool(manifest.get("reused")),
            "merge_num_tensors": manifest.get("num_tensors"),
            "merge_numel": manifest.get("numel"),
            "merge_shards": manifest.get("shards", []),
        }
    )
    return target, record


def unique_canonical_key_map(keys: Iterable[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    grouped: Dict[str, List[str]] = {}
    for key in keys:
        grouped.setdefault(canonical_tensor_key(key), []).append(key)
    unique = {canon: vals[0] for canon, vals in grouped.items() if len(vals) == 1}
    ambiguous = {canon: sorted(vals) for canon, vals in grouped.items() if len(vals) > 1}
    return unique, ambiguous


def match_three_tensor_keys(base_keys: Iterable[str], a_keys: Iterable[str], b_keys: Iterable[str]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    base_map, base_ambiguous = unique_canonical_key_map(base_keys)
    a_map, a_ambiguous = unique_canonical_key_map(a_keys)
    b_map, b_ambiguous = unique_canonical_key_map(b_keys)

    all_ambiguous = set(base_ambiguous) | set(a_ambiguous) | set(b_ambiguous)
    matched_canons = sorted((set(base_map) & set(a_map) & set(b_map)) - all_ambiguous)
    triples = [
        {
            "canonical": canon,
            "base_name": base_map[canon],
            "a_name": a_map[canon],
            "b_name": b_map[canon],
        }
        for canon in matched_canons
    ]
    exact_matches = sum(1 for row in triples if row["base_name"] == row["a_name"] == row["b_name"])
    alias_rows = [row for row in triples if not (row["base_name"] == row["a_name"] == row["b_name"])]
    report = {
        "matched_tensors": len(triples),
        "exact_matches": exact_matches,
        "alias_matches": len(alias_rows),
        "alias_examples": alias_rows[:20],
        "missing_in_base_from_a_or_b": sorted((set(a_map) | set(b_map)) - set(base_map)),
        "missing_in_a": sorted(set(base_map) - set(a_map)),
        "missing_in_b": sorted(set(base_map) - set(b_map)),
        "ambiguous_canonical_keys": {
            "base": base_ambiguous,
            "model_a": a_ambiguous,
            "model_b": b_ambiguous,
        },
    }
    return triples, report


def new_counts(group: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **group,
        "numel": 0,
        "a_nonzero": 0,
        "b_nonzero": 0,
        "both_nonzero": 0,
        "both_zero": 0,
        "a_only_nonzero": 0,
        "b_only_nonzero": 0,
    }


def add_counts(dst: Dict[str, Any], counts: Dict[str, int]) -> None:
    for key, value in counts.items():
        dst[key] += int(value)


def finalize(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        numel = row["numel"]
        union_nonzero = row["both_nonzero"] + row["a_only_nonzero"] + row["b_only_nonzero"]
        union_zero = row["both_zero"] + row["a_only_nonzero"] + row["b_only_nonzero"]
        a_nonzero = row["a_nonzero"]
        b_nonzero = row["b_nonzero"]
        a_zero = numel - a_nonzero
        b_zero = numel - b_nonzero
        finalized = dict(row)
        finalized.update(
            {
                "a_nonzero_frac": a_nonzero / numel if numel else 0.0,
                "b_nonzero_frac": b_nonzero / numel if numel else 0.0,
                "a_zero_frac": a_zero / numel if numel else 0.0,
                "b_zero_frac": b_zero / numel if numel else 0.0,
                "both_nonzero_frac": row["both_nonzero"] / numel if numel else 0.0,
                "both_zero_frac": row["both_zero"] / numel if numel else 0.0,
                "a_only_nonzero_frac": row["a_only_nonzero"] / numel if numel else 0.0,
                "b_only_nonzero_frac": row["b_only_nonzero"] / numel if numel else 0.0,
                "nonzero_jaccard": row["both_nonzero"] / union_nonzero if union_nonzero else 1.0,
                "zero_jaccard": row["both_zero"] / union_zero if union_zero else 1.0,
                "a_nonzero_covered_by_b": row["both_nonzero"] / a_nonzero if a_nonzero else 1.0,
                "b_nonzero_covered_by_a": row["both_nonzero"] / b_nonzero if b_nonzero else 1.0,
                "a_zero_covered_by_b": row["both_zero"] / a_zero if a_zero else 1.0,
                "b_zero_covered_by_a": row["both_zero"] / b_zero if b_zero else 1.0,
            }
        )
        out.append(finalized)
    return out


def write_csv(rows: List[Dict[str, Any]], path: Path, sort_by: str = "numel") -> None:
    df = pd.DataFrame(rows)
    if not df.empty and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    df.to_csv(path, index=False)


def pct(x: float) -> str:
    return f"{100.0 * x:.4f}%"


def random_baseline(row: Dict[str, Any]) -> Dict[str, float]:
    p_a = row["a_nonzero_frac"]
    p_b = row["b_nonzero_frac"]
    q_a = 1.0 - p_a
    q_b = 1.0 - p_b
    random_both_nonzero = p_a * p_b
    random_nonzero_union = p_a + p_b - random_both_nonzero
    random_both_zero = q_a * q_b
    random_zero_union = q_a + q_b - random_both_zero
    return {
        "random_both_nonzero_frac": random_both_nonzero,
        "random_a_nonzero_covered_by_b": p_b,
        "random_b_nonzero_covered_by_a": p_a,
        "random_nonzero_jaccard": random_both_nonzero / random_nonzero_union if random_nonzero_union else 1.0,
        "random_both_zero_frac": random_both_zero,
        "random_a_zero_covered_by_b": q_b,
        "random_b_zero_covered_by_a": q_a,
        "random_zero_jaccard": random_both_zero / random_zero_union if random_zero_union else 1.0,
    }


def add_baseline(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    base = random_baseline(row)
    out.update(base)
    for key, value in base.items():
        metric = key.removeprefix("random_")
        observed = row.get(metric)
        if observed is not None:
            out[f"{metric}_over_random"] = observed / value if value else float("inf")
    return out


def make_one_sided_rows(rows: Iterable[Dict[str, Any]], group_cols: List[str]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        row = add_baseline(row)
        base = {col: row[col] for col in group_cols if col in row}
        out.append(
            {
                **base,
                "numel": row["numel"],
                "a_nonzero_frac": row["a_nonzero_frac"],
                "b_nonzero_frac": row["b_nonzero_frac"],
                "a_nonzero_covered_by_b": row["a_nonzero_covered_by_b"],
                "random_a_nonzero_covered_by_b": row["random_a_nonzero_covered_by_b"],
                "a_nonzero_coverage_over_random": row["a_nonzero_covered_by_b_over_random"],
                "b_nonzero_covered_by_a": row["b_nonzero_covered_by_a"],
                "random_b_nonzero_covered_by_a": row["random_b_nonzero_covered_by_a"],
                "b_nonzero_coverage_over_random": row["b_nonzero_covered_by_a_over_random"],
                "both_nonzero_frac": row["both_nonzero_frac"],
                "random_both_nonzero_frac": row["random_both_nonzero_frac"],
                "both_nonzero_over_random": row["both_nonzero_frac_over_random"],
                "nonzero_jaccard": row["nonzero_jaccard"],
                "random_nonzero_jaccard": row["random_nonzero_jaccard"],
                "nonzero_jaccard_over_random": row["nonzero_jaccard_over_random"],
            }
        )
    return out


def plot_layer_module_heatmaps(layer_rows: List[Dict[str, Any]], out_dir: Path, label_a: str, label_b: str) -> List[str]:
    if plt is None or sns is None:
        return []
    df = pd.DataFrame(layer_rows)
    df = df[df["layer_id"] >= 0].copy()
    if df.empty:
        return []
    metrics = [
        ("a_nonzero_frac", f"{label_a} nonzero fraction", "layer_module_a_nonzero_frac.png"),
        ("b_nonzero_frac", f"{label_b} nonzero fraction", "layer_module_b_nonzero_frac.png"),
        ("both_nonzero_frac", "Both nonzero fraction", "layer_module_both_nonzero_frac.png"),
        ("a_nonzero_covered_by_b", f"{label_a} nonzero covered by {label_b}", "layer_module_a_covered_by_b.png"),
        ("b_nonzero_covered_by_a", f"{label_b} nonzero covered by {label_a}", "layer_module_b_covered_by_a.png"),
        ("nonzero_jaccard", "Nonzero Jaccard", "layer_module_nonzero_jaccard.png"),
    ]
    written = []
    for metric, title, filename in metrics:
        pivot = df.pivot(index="layer_id", columns="module_type", values=metric).sort_index()
        plt.figure(figsize=(max(8, 0.7 * len(pivot.columns)), max(6, 0.22 * len(pivot.index))))
        sns.heatmap(pivot, vmin=0.0, vmax=1.0, cmap="viridis", cbar_kws={"format": "%.1f"})
        plt.title(title)
        plt.xlabel("module_type")
        plt.ylabel("layer_id")
        plt.tight_layout()
        path = out_dir / filename
        plt.savefig(path, dpi=180)
        plt.close()
        written.append(filename)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", type=Path, required=True)
    parser.add_argument("--model_a", type=Path, required=True, help="First tuned checkpoint, e.g. DeepScaleR.")
    parser.add_argument("--model_b", type=Path, required=True, help="Second tuned checkpoint, e.g. OPD.")
    parser.add_argument("--label_a", default="model_a")
    parser.add_argument("--label_b", default="model_b")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5, help="Delta is zero when abs(delta) <= atol.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit_tensors", type=int, default=None)
    parser.add_argument(
        "--merge_dtensor_checkpoints",
        action="store_true",
        help="Materialize DTensor/FSDP sharded tuned checkpoints to safetensors before analysis.",
    )
    parser.add_argument(
        "--merged_model_dir",
        type=Path,
        default=None,
        help="Directory used for materialized safetensors checkpoints. Defaults to <out_dir>/merged_checkpoints.",
    )
    parser.add_argument(
        "--merge_shard_size_gb",
        type=float,
        default=4.0,
        help="Maximum safetensors shard size when materializing DTensor checkpoints.",
    )
    parser.add_argument("--force_remerge", action="store_true", help="Rebuild materialized safetensors even if a manifest already exists.")
    parser.add_argument("--materialize_only", action="store_true", help="Only materialize requested DTensor checkpoints and write merged_checkpoints.json.")
    args = parser.parse_args()

    dtype = resolve_dtype(args.dtype)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merge_root = args.merged_model_dir or (args.out_dir / "merged_checkpoints")
    max_shard_bytes = max(1, int(args.merge_shard_size_gb * 1024**3))

    base_path, base_materialization = maybe_materialize_model(
        args.base_model,
        slot="base",
        label="base",
        merge_enabled=False,
        merge_root=merge_root,
        max_shard_bytes=max_shard_bytes,
        force=args.force_remerge,
    )
    model_a_path, model_a_materialization = maybe_materialize_model(
        args.model_a,
        slot="model_a",
        label=args.label_a,
        merge_enabled=args.merge_dtensor_checkpoints,
        merge_root=merge_root,
        max_shard_bytes=max_shard_bytes,
        force=args.force_remerge,
    )
    model_b_path, model_b_materialization = maybe_materialize_model(
        args.model_b,
        slot="model_b",
        label=args.label_b,
        merge_enabled=args.merge_dtensor_checkpoints,
        merge_root=merge_root,
        max_shard_bytes=max_shard_bytes,
        force=args.force_remerge,
    )
    materialization = {
        "enabled": args.merge_dtensor_checkpoints,
        "merge_root": str(merge_root),
        "base": base_materialization,
        "model_a": model_a_materialization,
        "model_b": model_b_materialization,
    }
    (args.out_dir / "merged_checkpoints.json").write_text(json.dumps(materialization, indent=2) + "\n")
    if args.materialize_only:
        return

    base = load_tensor_map(base_path)
    model_a = load_tensor_map(model_a_path)
    model_b = load_tensor_map(model_b_path)

    try:
        base_keys = set(base.keys())
        a_keys = set(model_a.keys())
        b_keys = set(model_b.keys())
        matched_keys, key_match_report = match_three_tensor_keys(base_keys, a_keys, b_keys)

        global_row = new_counts({"group": "global"})
        by_module: Dict[str, Dict[str, Any]] = {}
        by_block: Dict[str, Dict[str, Any]] = {}
        by_layer_module: Dict[tuple, Dict[str, Any]] = {}
        tensor_rows: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []

        if args.limit_tensors is not None:
            matched_keys = matched_keys[: args.limit_tensors]

        for matched in tqdm(matched_keys, desc="comparing delta masks"):
            name = matched["base_name"]
            try:
                base_t = base.get(matched["base_name"])
                a_t = model_a.get(matched["a_name"])
                b_t = model_b.get(matched["b_name"])
                if not (torch.is_floating_point(base_t) and torch.is_floating_point(a_t) and torch.is_floating_point(b_t)):
                    skipped.append({"name": name, "a_name": matched["a_name"], "b_name": matched["b_name"], "reason": "non_floating"})
                    continue
                if base_t.shape != a_t.shape or base_t.shape != b_t.shape:
                    skipped.append(
                        {
                            "name": name,
                            "a_name": matched["a_name"],
                            "b_name": matched["b_name"],
                            "reason": "shape_mismatch",
                            "base_shape": list(base_t.shape),
                            "a_shape": list(a_t.shape),
                            "b_shape": list(b_t.shape),
                        }
                    )
                    continue

                with torch.inference_mode():
                    base_d = base_t.to(device=device, dtype=dtype, non_blocking=True)
                    a_d = a_t.to(device=device, dtype=dtype, non_blocking=True)
                    b_d = b_t.to(device=device, dtype=dtype, non_blocking=True)
                    mask_a = (a_d - base_d).abs() > args.atol
                    mask_b = (b_d - base_d).abs() > args.atol

                    both_mask = mask_a & mask_b
                    union_mask = mask_a | mask_b
                    both_nonzero = torch.count_nonzero(both_mask).item()
                    a_nonzero = torch.count_nonzero(mask_a).item()
                    b_nonzero = torch.count_nonzero(mask_b).item()
                    union_nonzero = torch.count_nonzero(union_mask).item()
                    numel = mask_a.numel()
                    both_zero = numel - union_nonzero
                counts = {
                    "numel": numel,
                    "a_nonzero": a_nonzero,
                    "b_nonzero": b_nonzero,
                    "both_nonzero": both_nonzero,
                    "both_zero": both_zero,
                    "a_only_nonzero": a_nonzero - both_nonzero,
                    "b_only_nonzero": b_nonzero - both_nonzero,
                }

                info = parse_tensor_name(name)
                tensor_row = {"name": name, "a_name": matched["a_name"], "b_name": matched["b_name"], "shape": list(base_t.shape), **info, **counts}
                tensor_rows.extend(finalize([tensor_row]))
                add_counts(global_row, counts)

                module = info["module_type"]
                block = info["block_type"]
                layer_module = (info["layer_id"], module)
                by_module.setdefault(module, new_counts({"module_type": module}))
                by_block.setdefault(block, new_counts({"block_type": block}))
                by_layer_module.setdefault(layer_module, new_counts({"layer_id": info["layer_id"], "module_type": module}))
                add_counts(by_module[module], counts)
                add_counts(by_block[block], counts)
                add_counts(by_layer_module[layer_module], counts)
            except Exception as exc:
                skipped.append({"name": name, "a_name": matched.get("a_name"), "b_name": matched.get("b_name"), "reason": f"error: {type(exc).__name__}: {exc}"})

        finalized_global = add_baseline(finalize([global_row])[0])
        finalized_modules = finalize(by_module.values())
        finalized_blocks = finalize(by_block.values())
        finalized_layer_modules = finalize(by_layer_module.values())

        summary = {
            "base_model": str(args.base_model),
            "model_a": str(args.model_a),
            "model_b": str(args.model_b),
            "used_base_model": str(base_path),
            "used_model_a": str(model_a_path),
            "used_model_b": str(model_b_path),
            "label_a": args.label_a,
            "label_b": args.label_b,
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
            "requested_device": args.device,
            "atol": args.atol,
            "materialization": materialization,
            "common_tensors": len(matched_keys),
            "skipped_tensors": len(skipped),
            "key_match_report": key_match_report,
            "missing": {
                "missing_in_base_from_a_or_b": key_match_report["missing_in_base_from_a_or_b"],
                "missing_in_a": key_match_report["missing_in_a"],
                "missing_in_b": key_match_report["missing_in_b"],
            },
            **finalized_global,
        }

        write_csv(tensor_rows, args.out_dir / "tensor_overlap.csv")
        write_csv(finalized_modules, args.out_dir / "overlap_by_module.csv")
        write_csv(finalized_blocks, args.out_dir / "overlap_by_block.csv")
        write_csv(finalized_layer_modules, args.out_dir / "overlap_by_layer_module.csv")
        write_csv(make_one_sided_rows([finalized_global], ["group"]), args.out_dir / "one_sided_overlap_global.csv")
        write_csv(make_one_sided_rows(finalized_modules, ["module_type"]), args.out_dir / "one_sided_overlap_by_module.csv")
        write_csv(make_one_sided_rows(finalized_blocks, ["block_type"]), args.out_dir / "one_sided_overlap_by_block.csv")
        write_csv(make_one_sided_rows(finalized_layer_modules, ["layer_id", "module_type"]), args.out_dir / "one_sided_overlap_by_layer_module.csv")
        plot_files = plot_layer_module_heatmaps(finalized_layer_modules, args.out_dir, args.label_a, args.label_b)
        pd.DataFrame(skipped).to_csv(args.out_dir / "skipped_tensors.csv", index=False)
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        lines = [
            f"# Delta Mask Overlap: {args.label_a} vs {args.label_b}",
            "",
            f"- Base: `{args.base_model}`",
            f"- {args.label_a}: `{args.model_a}`",
            f"- {args.label_b}: `{args.model_b}`",
            f"- dtype: `{summary['dtype']}`; device: `{summary['device']}`; zero threshold: `abs(delta) <= {args.atol:g}`",
        ]
        merged_lines = []
        for key in ("model_a", "model_b"):
            rec = materialization[key]
            if rec.get("merged"):
                merged_lines.append(f"- {rec['label']} merged checkpoint: `{rec['merged_model']}`")
        if merged_lines:
            lines.extend(["", "## Materialized Checkpoints", *merged_lines, "- `merged_checkpoints.json`"])
        lines.extend(
            [
                "",
                "## Global",
            ]
        )
        lines += [
            f"- {args.label_a} nonzero: `{pct(summary['a_nonzero_frac'])}`",
            f"- {args.label_b} nonzero: `{pct(summary['b_nonzero_frac'])}`",
            f"- both nonzero: `{pct(summary['both_nonzero_frac'])}`",
            f"- both zero: `{pct(summary['both_zero_frac'])}`",
            f"- nonzero Jaccard: `{pct(summary['nonzero_jaccard'])}`",
            f"- zero Jaccard: `{pct(summary['zero_jaccard'])}`",
            f"- {args.label_a} nonzero covered by {args.label_b}: `{pct(summary['a_nonzero_covered_by_b'])}`",
            f"- random baseline for {args.label_a} nonzero covered by {args.label_b}: `{pct(summary['random_a_nonzero_covered_by_b'])}`; observed/random `{summary['a_nonzero_covered_by_b_over_random']:.4f}x`",
            f"- {args.label_b} nonzero covered by {args.label_a}: `{pct(summary['b_nonzero_covered_by_a'])}`",
            f"- random baseline for {args.label_b} nonzero covered by {args.label_a}: `{pct(summary['random_b_nonzero_covered_by_a'])}`; observed/random `{summary['b_nonzero_covered_by_a_over_random']:.4f}x`",
            f"- random baseline for nonzero Jaccard: `{pct(summary['random_nonzero_jaccard'])}`; observed/random `{summary['nonzero_jaccard_over_random']:.4f}x`",
            "",
            "## Files",
            "- `merged_checkpoints.json`",
            "- `summary.json`",
            "- `tensor_overlap.csv`",
            "- `overlap_by_module.csv`",
            "- `overlap_by_block.csv`",
            "- `overlap_by_layer_module.csv`",
            "- `one_sided_overlap_global.csv`",
            "- `one_sided_overlap_by_module.csv`",
            "- `one_sided_overlap_by_block.csv`",
            "- `one_sided_overlap_by_layer_module.csv`",
            "- `skipped_tensors.csv`",
        ]
        if plot_files:
            lines.extend(["", "## Heatmaps", *[f"- `{name}`" for name in plot_files]])
        (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    finally:
        base.close()
        model_a.close()
        model_b.close()


if __name__ == "__main__":
    main()
