"""Protect panel independence and balanced selection without GPU/data dependencies."""
import importlib.util
from pathlib import Path
from argparse import Namespace
import hashlib
import json
import numpy as np

import pytest

PATH = Path(__file__).parents[1] / "scripts" / "calibrate_weighted_stress.py"


def module():
    spec = importlib.util.spec_from_file_location("calibrate", PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_panel_selection_balances_cells_and_excludes_previously_seen_test_families():
    study = module()
    rows = [
        {"sequence_id": f"{family}_{i}", "sequence": sequence,
         "family": family, "length_bin": band, "chemistry": chemistry}
        for family, i, sequence, band, chemistry in [
            ("old", 1, "D" * 40, 0, "acidic"),
            ("new1", 1, "E" * 40, 0, "acidic"),
            ("new2", 1, "K" * 40, 0, "basic"),
            ("new3", 1, "R" * 40, 0, "basic"),
        ]
    ]
    selected = study.select_panel(rows, {"old"}, set(), 1, 1, 1, 7,
                                  cells=[(0, "acidic"), (0, "basic")])
    assert selected == study.select_panel(rows, {"old"}, set(), 1, 1, 1, 7,
                                         cells=[(0, "acidic"), (0, "basic")])
    assert len(selected) == 4
    assert {r["family"] for r in selected if r["split"] == "held_out"}.isdisjoint({"old"})
    assert {r["family"] for r in selected if r["split"] == "held_out"}.isdisjoint(
        r["family"] for r in selected if r["split"] == "calibration")
    assert sorted((r["chemistry"], r["split"]) for r in selected) == [
        ("acidic", "calibration"), ("acidic", "held_out"),
        ("basic", "calibration"), ("basic", "held_out")]


def test_selection_uses_families_not_string_similarity():
    study = module()
    assert study.family_name("IDR_sp___P12345___FOO_20_70") == "P12345"
    assert study.family_name("P12345_80_130") == "P12345"
    rows = [
        {"sequence_id": "a", "sequence": "DEKRA" * 10, "family": "a",
         "length_bin": 0, "chemistry": "balanced"},
        {"sequence_id": "b", "sequence": "DEKRA" * 9 + "DEKRG", "family": "b",
         "length_bin": 0, "chemistry": "balanced"},
    ]
    selected = study.select_panel(rows, set(), set(), 1, 1, 1, 7,
                                  cells=[(0, "balanced")])
    assert len(selected) == 2
    assert {r["family"] for r in selected if r["split"] == "held_out"}.isdisjoint(
        r["family"] for r in selected if r["split"] == "calibration")


def test_old_manifest_without_sequences_resolves_them_from_fasta(tmp_path):
    study = module()
    fasta = tmp_path / "sequences.fasta"
    fasta.write_text(">old|simulation/1\n" + "D" * 40 + "\n")
    old = tmp_path / "old.csv"
    old.write_text("sequence_id\nold\n")
    mapping = tmp_path / "mapping.tsv"
    mapping.write_text("fasta_id\tsim_dir\tseq_in_name\n")
    args = Namespace(fasta=fasta, mapping=mapping, exclude_panel=[old],
                     md_root=tmp_path, calibration_per_cell=1, test_per_cell=1,
                     family_cap=1, seed=7)
    # Empty source should fail for absent candidates, not missing old-manifest columns.
    with pytest.raises(ValueError, match="unfilled"):
        study.build_panel(args)


def test_paired_summary_uses_equal_sequence_errors_and_reports_wins():
    study = module()
    report = {"held_out": {
        "a": {"family": "one", "chemistry": "acidic", "mode": {"delta": {"bond_rmse_A": 1.0}}},
        "b": {"family": "one", "chemistry": "basic", "mode": {"delta": {"bond_rmse_A": 1.0}}},
        "c": {"family": "two", "chemistry": "basic", "mode": {"delta": {"bond_rmse_A": -2.0}}},
    }}
    summary = study.paired_summary(report, seed=7, repetitions=100)
    bond = summary["mode"]["bond_rmse_A"]
    assert bond["mean_delta"] == 0.0
    assert bond["candidate_better_sequences"] == 1
    assert bond["minimum_delta"] == -2.0
    assert bond["maximum_delta"] == 1.0
    assert bond["exploratory_family_bootstrap_95pct"] == [-2.0, 1.0]


def test_cache_record_preserves_inputs_and_hashes_the_written_arrays(tmp_path):
    study = module()
    path = tmp_path / "paired.npz"
    truth = np.zeros((2, 3, 3), dtype=np.float32)
    decoded = np.ones_like(truth)
    provenance = {"sequence": "DDD", "family": "a", "split": "calibration",
                  "indices": [2, 8], "xtc_sha256": "raw-md-hash"}
    record = study.save_paired_cache(path, truth, decoded, np.array([0, 1]), provenance)
    assert record["indices"] == [2, 8]
    assert record["xtc_sha256"] == "raw-md-hash"
    assert record["cache_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache["truth"], truth)
        np.testing.assert_array_equal(cache["decoded"], decoded)
        np.testing.assert_array_equal(cache["blocks"], [0, 1])
        assert json.loads(cache["provenance"].item()) == provenance


def test_frozen_panel_rejects_cross_split_family_overlap():
    study = module()
    rows = [dict(sequence_id=name, sequence=seq, family="same", split=split,
                 md_xtc="example.xtc", md_topology="example.pdb")
            for name, seq, split in [("a", "D" * 30, "calibration"),
                                     ("b", "K" * 30, "held_out")]]
    with pytest.raises(ValueError, match="family"):
        study.validate_panel(rows)


def test_metrics_detect_exact_reconstruction():
    study = module()
    xyz = np.random.default_rng(0).normal(size=(4, 12, 3))
    maps = study.distance_maps(xyz)
    metrics, frames = study.paired_metrics(maps, maps)
    assert all(abs(value) < 1e-12 for value in metrics.values())
    assert all(np.allclose(values, 0) for values in frames.values())
