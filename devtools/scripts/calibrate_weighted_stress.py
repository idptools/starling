#!/usr/bin/env python
"""Calibrate paired MD/VAE errors and validate weighted coordinate reconstruction."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from devtools.scripts.fit_weighted_stress import (
    current_error_profile, fit_error_model, leave_one_family_out, load_paired_record,
    predict_error_profile, sha256,
)

LENGTH_BANDS = [(30, 80), (80, 120), (120, 180), (180, 240), (240, 300), (300, 380)]
CHEMISTRIES = ("acidic", "basic", "hydrophobic", "balanced")
CONTACT_CUTOFF_A = 8.0


def distance_maps(xyz):
    return np.linalg.norm(xyz[:, :, None] - xyz[:, None, :], axis=-1)


def map_features(maps):
    n = maps.shape[-1]
    bonds = np.diagonal(maps, offset=1, axis1=1, axis2=2)
    sep2 = np.diagonal(maps, offset=2, axis1=1, axis2=2)
    cosine = (bonds[:, :-1] ** 2 + bonds[:, 1:] ** 2 - sep2**2) / np.maximum(
        2 * bonds[:, :-1] * bonds[:, 1:], 1e-12
    )
    local_rg = np.stack(
        [
            np.sqrt(np.square(maps[:, k : k + 10, k : k + 10]).sum((1, 2)) / 200)
            for k in range(n - 9)
        ],
        axis=1,
    )
    return {
        "rg": np.sqrt(np.square(maps).sum((1, 2)) / (2 * n**2)),
        "ree": maps[:, 0, -1],
        "bonds": bonds,
        "angles": np.degrees(np.arccos(np.clip(cosine, -1, 1))),
        "local_rg": local_rg,
    }


def paired_metrics(predicted, truth):
    i, j = np.triu_indices(truth.shape[-1], 1)
    residual = predicted[:, i, j] - truth[:, i, j]
    p, t = map_features(predicted), map_features(truth)
    frames = {
        "pair_rmse_A": np.sqrt(np.square(residual).mean(axis=1)),
        "long_pair_rmse_A": np.sqrt(np.square(residual[:, j - i >= 7]).mean(axis=1)),
        "bond_rmse_A": np.sqrt(np.square(p["bonds"] - t["bonds"]).mean(axis=1)),
        "rg_mae_A": np.abs(p["rg"] - t["rg"]),
        "ree_mae_A": np.abs(p["ree"] - t["ree"]),
        "local_rg_mae_A": np.abs(p["local_rg"] - t["local_rg"]).mean(axis=1),
        "angle_mae_deg": np.abs(p["angles"] - t["angles"]).mean(axis=1),
    }
    metrics = {key: float(value.mean()) for key, value in frames.items()}
    metrics["bond_std_abs_error_A"] = float(abs(p["bonds"].std() - t["bonds"].std()))
    nonlocal_pairs = j - i >= 3
    metrics["contact_probability_mae"] = float(
        np.abs(
            (predicted[:, i, j][:, nonlocal_pairs] < CONTACT_CUTOFF_A).mean(0)
            - (truth[:, i, j][:, nonlocal_pairs] < CONTACT_CUTOFF_A).mean(0)
        ).mean()
    )
    return metrics, frames


def family_name(name):
    accession = re.search(r"(?:tr|sp)___([^_]+)___", name)
    return accession.group(1) if accession else name.split("_")[0]


def select_panel(rows, old_families, old_sequences, calibration, held_out,
                 family_cap, seed, cells=None):
    cells = cells or [(i, c) for i in range(6) for c in CHEMISTRIES]
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    selected, seen, families = [], set(old_sequences), Counter()
    test_families = set()
    for split, target in [("held_out", held_out), ("calibration", calibration)]:
        counts = Counter()
        for row in rows:
            cell = (row["length_bin"], row["chemistry"])
            family, seq = row["family"], row["sequence"]
            if (cell not in cells or counts[cell] >= target or seq in seen
                    or families[family] >= family_cap):
                continue
            if split == "held_out" and family in old_families:
                continue
            if split == "calibration" and family in test_families:
                continue
            if any(not Path(row[k]).is_file() for k in ("md_xtc", "md_topology") if k in row):
                continue
            selected.append(dict(row, split=split))
            seen.add(seq)
            families[family] += 1
            counts[cell] += 1
            if split == "held_out":
                test_families.add(family)
            if all(counts[cell] == target for cell in cells):
                break
        missing = {str(c): target - counts[c] for c in cells if counts[c] != target}
        if missing:
            raise ValueError(f"{split}: unfilled cells {missing}; do not silently relax selection")
    return selected


def read_csv(path):
    with open(path) as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_panel(rows):
    required = {"sequence_id", "sequence", "family", "split", "md_xtc", "md_topology"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("Panel requires sequence_id, sequence, family, split, md_xtc, md_topology")
    if any(row["split"] not in ("calibration", "held_out") or not row["family"] for row in rows):
        raise ValueError("Panel requires identified families and calibration/held_out splits")
    calibration = {r["family"] for r in rows if r["split"] == "calibration"}
    held_out = {r["family"] for r in rows if r["split"] == "held_out"}
    if calibration & held_out:
        raise ValueError("Panel has cross-split family overlap")
    if len(calibration) < 3 or len(held_out) < 2:
        raise ValueError("Panel needs >=3 calibration and >=2 held-out families")
    if len({r["sequence_id"] for r in rows}) != len(rows) or len({r["sequence"] for r in rows}) != len(rows):
        raise ValueError("Panel contains duplicate identifiers or sequences")
    if any(not 30 <= len(r["sequence"]) < 380 for r in rows):
        raise ValueError("Panel lengths must be in [30,380)")


def paired_summary(report, seed=20261003, repetitions=2000):
    """Equal-sequence deltas with exploratory family-cluster bootstrap intervals.

    Resample name/accession families, retaining each sampled family's sequences.
    This respects known grouping, not unknown homology or correlated design sets.
    Intervals are descriptive, not a predeclared noninferiority/release test.
    """
    rows = list(report["held_out"].values())
    families = sorted({r["family"] for r in rows})
    if len(families) < 2:
        raise ValueError("Summary requires at least two test families")
    labels = [p for p in ("mode", "sample") if p in rows[0]]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(families), np.full(len(families), 1 / len(families)),
                             size=repetitions)
    sizes = np.array([sum(r["family"] == f for r in rows) for f in families])
    result = {}
    for posterior in labels:
        result[posterior] = {}
        for metric in rows[0][posterior]["delta"]:
            values = np.array([r[posterior]["delta"][metric] for r in rows])
            sums = np.array([sum(r[posterior]["delta"][metric] for r in rows if r["family"] == f)
                             for f in families])
            bootstrap = (counts @ sums) / (counts @ sizes)
            result[posterior][metric] = {
                "mean_delta": float(values.mean()), "minimum_delta": float(values.min()),
                "maximum_delta": float(values.max()),
                "candidate_better_sequences": int((values < 0).sum()),
                "exploratory_family_bootstrap_95pct": np.quantile(bootstrap, [.025, .975]).tolist(),
            }
    return result


def archive_sources(root):
    source = root / "source"
    if not source.is_dir():
        source.mkdir()
        repo = Path(__file__).resolve().parents[2]
        paths = [p for p in (repo / "starling").rglob("*.py") if "tests" not in p.parts]
        paths += [repo / "devtools/scripts" / name for name in
                  ("calibrate_weighted_stress.py", "fit_weighted_stress.py")]
        for path in paths:
            target = source / path.relative_to(repo)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    return source


def save_paired_cache(path, truth, decoded, blocks, provenance):
    """Write paired maps and return their input provenance and cache hash."""
    np.savez_compressed(path, truth=truth, decoded=decoded, blocks=blocks,
                        provenance=json.dumps(provenance))
    return dict(provenance, cache_sha256=sha256(path))


def build_panel(args):
    sequences, name = {}, None
    with open(args.fasta) as stream:
        for line in stream:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
                sequences[name] = ""
            elif name:
                sequences[name] += line
    old = [r for path in args.exclude_panel for r in read_csv(path)]
    old_families = {family_name(r["sequence_id"]) for r in old}
    by_name = {key.split("|")[0]: value for key, value in sequences.items()}
    old_sequences = set()
    for row in old:
        sequence = row.get("sequence") or by_name.get(row["sequence_id"])
        if not sequence:
            raise ValueError(f"Cannot resolve previously inspected sequence {row['sequence_id']}")
        old_sequences.add(sequence)
    candidates = []
    with open(args.mapping) as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            seq = sequences.get(row["fasta_id"], "")
            n = len(seq)
            if not 30 <= n < 380:
                continue
            ncpr = (seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E")) / n
            hydrophobic = sum(seq.count(a) for a in "AILMFWVY") / n
            chemistry = ("acidic" if ncpr <= -.15 else "basic" if ncpr >= .15
                         else "hydrophobic" if hydrophobic >= .45 else "balanced")
            directory = args.md_root / row["sim_dir"]
            candidates.append({
                "sequence_id": row["seq_in_name"], "sequence": seq,
                "family": family_name(row["seq_in_name"]),
                "length_bin": next(i for i, (lo, hi) in enumerate(LENGTH_BANDS) if lo <= n < hi),
                "chemistry": chemistry,
                "md_xtc": str(directory / "__traj_pbcfix.xtc"),
                "md_topology": str(directory / "__START_pbcfix.pdb"),
            })
    return select_panel(candidates, old_families, old_sequences,
                        args.calibration_per_cell, args.test_per_cell,
                        args.family_cap, args.seed)


def run(args, rows):
    import mdtraj as md
    import torch
    from starling.models.vae import VAE
    from starling.inference.constraints import symmetrize_distance_maps
    from starling.structure.coordinates import distance_matrix_to_3d_structure_torch_mds
    from starling.structure.weighted_stress import map_error_weights

    started = time.perf_counter()
    source = archive_sources(args.output)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint_hash = sha256(args.vae)
    vae = VAE.load_from_checkpoint(str(args.vae), map_location=args.device).eval().to(args.device)
    report = {
        "purpose": "candidate calibration, not release certification",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                      if k != "exclude_panel"},
        "input_hashes": {str(p): sha256(p) for p in [args.vae, args.output / "panel.csv",
            *([args.panel] if args.panel else [args.fasta, args.mapping, *args.exclude_panel])]},
        "script_sha256": sha256(__file__), "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "source_hashes": {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*.py"))},
        "torch": torch.__version__, "numpy": np.__version__, "mdtraj": md.__version__,
        "cuda": torch.version.cuda, "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "frames_per_sequence": args.frames, "iterations": args.iterations,
        "family_counts": {split: dict(Counter(r["family"] for r in rows if r["split"] == split))
                          for split in ("calibration", "held_out")},
        "cache_timings": [], "cache_provenance": {}, "fits": {}, "held_out": {},
        "summary_caveat": "Exploratory family bootstrap; no equivalence or release test",
    }
    records = {p: [] for p in ("mode", "sample")}
    for index, row in enumerate(rows):
        tick = time.perf_counter()
        trajectory = md.load(row["md_xtc"], top=row["md_topology"])
        n = len(row["sequence"])
        ca = trajectory.topology.select("name CA")
        if len(ca) != n:
            raise ValueError(f"{row['sequence_id']}: CA count differs from sequence")
        indices = np.linspace(int(.2 * len(trajectory)), len(trajectory) - 1, args.frames, dtype=int)
        if len(np.unique(indices)) != args.frames:
            raise ValueError("Insufficient distinct post-equilibration frames")
        truth = distance_maps(trajectory.xyz[indices][:, ca].astype(np.float64) * 10).astype(np.float32)
        blocks = np.repeat(np.arange(8), args.frames // 8)
        provenance = dict(row, vae_sha256=checkpoint_hash,
                          xtc_sha256=sha256(row["md_xtc"]), topology_sha256=sha256(row["md_topology"]),
                          indices=indices.tolist(), latent_seed=args.seed + index)
        data = torch.zeros((args.frames, 1, vae.dimension, vae.dimension), device=args.device)
        data[:, 0, :n, :n] = torch.as_tensor(truth, device=args.device)
        report["cache_provenance"][row["sequence_id"]] = {}
        with torch.inference_mode():
            encoding = vae.encode(data)
            for posterior in records:
                directory = args.output / posterior
                directory.mkdir(exist_ok=True)
                torch.manual_seed(args.seed + index)
                latent = encoding.mode() if posterior == "mode" else encoding.sample()
                decoded = symmetrize_distance_maps(vae.decode(latent)[:, :, :n, :n])[:, 0].cpu().numpy()
                if not np.isfinite(decoded).all():
                    raise ValueError("Nonfinite decoder output")
                path = directory / f"{index:04d}.paired.npz"
                report["cache_provenance"][row["sequence_id"]][posterior] = save_paired_cache(
                    path, truth, decoded, blocks, provenance | {"posterior": posterior})
                if row["split"] == "calibration":
                    records[posterior].append(load_paired_record(row["sequence_id"], path))
        report["cache_timings"].append({"name": row["sequence_id"], "seconds": time.perf_counter() - tick})
        print(f"Cached {index + 1}/{len(rows)} ({n} residues)", flush=True)
        if time.perf_counter() - started > args.budget_seconds:
            raise RuntimeError("Time budget exceeded; caches preserved, no production change")
    for posterior, training in records.items():
        folds = leave_one_family_out(training)
        report["fits"][posterior] = {
            "model": fit_error_model(training), "family_folds": folds,
            "family_loo_rmse_A": float(np.mean([f["rmse"] for f in folds])),
            "baseline_profile_rmse_A": float(np.mean([f["current_rmse"] for f in folds])),
        }
    # Freeze both coefficients before reading any held-out reconstruction results.
    (args.output / "fit.json").write_text(json.dumps(report["fits"], indent=2) + "\n")
    for index, row in enumerate(rows):
        if row["split"] != "held_out":
            continue
        n = len(row["sequence"])
        result = dict(length=n, family=row["family"], chemistry=row["chemistry"])
        for posterior in records:
            with np.load(args.output / posterior / f"{index:04d}.paired.npz") as cache:
                evaluation = cache["blocks"] % 2 == 1
                truth, decoded = cache["truth"][evaluation], cache["decoded"][evaluation]
            errors = predict_error_profile(report["fits"][posterior]["model"], n)
            if not np.isfinite(errors).all() or np.any(errors <= 0):
                raise ValueError("Invalid fitted errors")
            separation = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
            weights = np.zeros((n, n), dtype=np.float32)
            mask = separation > 0
            weights[mask] = 1 / errors[separation[mask] - 1] ** 2
            residual = decoded.astype(np.float64) - truth
            observed = np.array([np.sqrt(np.mean(np.diagonal(
                residual, offset=s, axis1=1, axis2=2) ** 2)) for s in range(1, n)])
            metrics = {}
            for label, w in [("baseline", map_error_weights(n).numpy()), ("candidate", weights)]:
                w = w / w[np.triu_indices(n, 1)].mean()
                xyz, _ = distance_matrix_to_3d_structure_torch_mds(
                    decoded, device=args.device, weights=w, n_iter=args.iterations,
                    tol=1e-6, progress_bar=False)
                recovered = distance_maps(xyz)
                if not np.isfinite(recovered).all():
                    raise ValueError("Nonfinite reconstructed coordinates")
                metrics[label], frames = paired_metrics(recovered, truth)
                metrics[label]["pair_frame_rmse_p95_A"] = float(np.quantile(frames["pair_rmse_A"], .95))
                profile = current_error_profile(n) if label == "baseline" else errors
                metrics[label]["decoder_profile_rmse_A"] = float(np.sqrt(np.mean((profile - observed) ** 2)))
            metrics["delta"] = {k: metrics["candidate"][k] - metrics["baseline"][k]
                                for k in metrics["baseline"]}
            result[posterior] = metrics
        report["held_out"][row["sequence_id"]] = result
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Validated {len(report['held_out'])} new test sequences", flush=True)
        if time.perf_counter() - started > args.budget_seconds:
            raise RuntimeError("Time budget exceeded; partial report preserved")
    report["total_seconds"] = time.perf_counter() - started
    report["paired_summary"] = paired_summary(report)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Completed in {report['total_seconds']:.1f}s: {args.output / 'report.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, help="Frozen panel; bypass automatic selection")
    parser.add_argument("--fasta", type=Path, default=Path("data/completed_sequences_unique.fasta"))
    parser.add_argument("--mapping", type=Path, default=Path("data/completed_sequences_unique_mapping.CLEAN.tsv"))
    parser.add_argument("--md-root", type=Path, default=Path("/work/j.lotthammer/projects"))
    parser.add_argument("--exclude-panel", type=Path, action="append", default=[])
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-per-cell", type=int, default=8)
    parser.add_argument("--test-per-cell", type=int, default=2)
    parser.add_argument("--family-cap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--budget-seconds", type=float, default=900)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    if not args.panel and not args.exclude_panel:
        parser.error("Supply previously inspected panels with --exclude-panel")
    if (args.frames < 8 or args.frames % 8 or args.iterations < 1 or args.budget_seconds <= 0
            or min(args.calibration_per_cell, args.test_per_cell, args.family_cap) < 1):
        parser.error("Positive counts required; frames must be a multiple of eight")
    if not args.select_only and args.vae is None:
        parser.error("--vae is required for decoding")
    if args.output.exists():
        parser.error("Output directory already exists; use a new path to preserve results")
    rows = read_csv(args.panel) if args.panel else build_panel(args)
    try:
        validate_panel(rows)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True)
    write_csv(args.output / "panel.csv", rows)
    print(f"Selected {len(rows)} sequences; manifest: {args.output / 'panel.csv'}", flush=True)
    if not args.select_only:
        run(args, rows)


if __name__ == "__main__":
    main()
