"""
Tests for the --remove-errors / remove_errors error-filtering pathway.

These exercise the sample-screen-replace loop that backs the CLI flag. The
sampler itself is stubbed out so we can control exactly which conformers are
"good" and which are physically impossible, which means the tests are fast and
deterministic and never touch the diffusion model.
"""

import numpy as np
import pytest
import torch

from starling.inference import generation

SEQ = "MKTAYIAKQRQ"
L = len(SEQ)

# a fully extended chain sits at 3.81 A per bond, comfortably inside the
# 4.81 A per-bond physical bound used by check_distance_map_for_error
GOOD_SPACING = 3.8

# 10 A per bond is far beyond the bound, so every off-diagonal pair is flagged
BAD_SPACING = 10.0


def _distance_map(spacing, n_res=L):
    """Build a symmetric (n_res, n_res) distance map with a fixed per-bond spacing."""
    sep = np.abs(np.subtract.outer(np.arange(n_res), np.arange(n_res)))
    return (sep * spacing).astype(np.float32)


def _stack(spacings):
    """Stack one distance map per entry in spacings into a torch tensor."""
    return torch.from_numpy(np.stack([_distance_map(s) for s in spacings]))


def test_good_map_passes_and_bad_map_fails_the_underlying_check():
    """Sanity check the fixtures themselves before relying on them."""
    from starling.utilities import check_distance_map_for_error

    assert check_distance_map_for_error(_distance_map(GOOD_SPACING)) is False
    assert check_distance_map_for_error(_distance_map(BAD_SPACING)) is True


class StubSampler:
    """Records how many conformers were requested on each call."""

    def __init__(self):
        self.requests = []


def _patch_sampling(monkeypatch, spacing_plan):
    """
    Replace _sample_distance_maps with a stub driven by ``spacing_plan``.

    ``spacing_plan`` is a list of lists: one inner list per expected round,
    each giving the per-conformer spacing to hand back for that round.
    """
    rounds = iter(spacing_plan)
    calls = []

    def fake_sample(sampler, sequence, conformations, *args, **kwargs):
        calls.append(conformations)
        try:
            spacings = next(rounds)
        except StopIteration:  # pragma: no cover - defensive
            spacings = [GOOD_SPACING] * conformations
        return _stack(spacings)

    monkeypatch.setattr(generation, "_sample_distance_maps", fake_sample)
    return calls


def _run(conformations, max_rounds=10):
    """Invoke the filter loop with structures disabled (distance-map stage only)."""
    return generation._generate_error_filtered_conformers(
        sampler=StubSampler(),
        sequence=SEQ,
        conformations=conformations,
        batch_size=100,
        show_per_step_progress_bar=False,
        constraint=None,
        return_structures=False,
        device="cpu",
        num_cpus_mds=1,
        num_mds_init=1,
        show_progress_bar=False,
        verbose=False,
        max_rounds=max_rounds,
    )


def test_all_good_returns_requested_count_in_one_round(monkeypatch):
    calls = _patch_sampling(monkeypatch, [[GOOD_SPACING] * 5])
    maps, coords, discarded = _run(5)
    assert len(maps) == 5
    assert coords is None
    assert discarded == 0
    assert calls == [5], "should not have needed a second round"


def test_bad_conformers_are_discarded_and_replaced(monkeypatch):
    """Round 1 yields 2 good of 5; the loop must top up to the full request."""
    calls = _patch_sampling(
        monkeypatch,
        [
            [GOOD_SPACING, BAD_SPACING, BAD_SPACING, GOOD_SPACING, BAD_SPACING],
            [GOOD_SPACING, GOOD_SPACING, GOOD_SPACING],
        ],
    )
    maps, _, discarded = _run(5)

    assert len(maps) == 5, "must return exactly the requested number"
    assert discarded == 3
    assert calls == [5, 3], "second round should request only the shortfall"


def test_returned_conformers_are_all_clean(monkeypatch):
    """Nothing that fails the physical check may survive into the output."""
    from starling.utilities import check_distance_map_for_error

    _patch_sampling(
        monkeypatch,
        [
            [BAD_SPACING, GOOD_SPACING, BAD_SPACING, BAD_SPACING],
            [GOOD_SPACING, GOOD_SPACING, GOOD_SPACING],
        ],
    )
    maps, _, _ = _run(4)
    assert len(maps) == 4
    for dm in maps:
        assert check_distance_map_for_error(dm) is False


def test_multiple_top_up_rounds(monkeypatch):
    """A shortfall that persists across rounds keeps being topped up."""
    calls = _patch_sampling(
        monkeypatch,
        [
            [GOOD_SPACING, BAD_SPACING, BAD_SPACING],
            [BAD_SPACING, GOOD_SPACING],
            [GOOD_SPACING],
        ],
    )
    maps, _, discarded = _run(3)
    assert len(maps) == 3
    assert discarded == 3
    assert calls == [3, 2, 1]


def test_round_that_is_entirely_bad_is_survivable(monkeypatch):
    """A round where every conformer fails must not abort or lose the count."""
    calls = _patch_sampling(
        monkeypatch,
        [
            [BAD_SPACING, BAD_SPACING],
            [GOOD_SPACING, GOOD_SPACING],
        ],
    )
    maps, _, discarded = _run(2)
    assert len(maps) == 2
    assert discarded == 2
    assert calls == [2, 2]


def test_overshoot_is_trimmed_to_requested_count(monkeypatch):
    """If a round returns more than needed, the output is trimmed exactly."""

    def fake_sample(sampler, sequence, conformations, *args, **kwargs):
        # deliberately hand back more than asked for
        return _stack([GOOD_SPACING] * (conformations + 4))

    monkeypatch.setattr(generation, "_sample_distance_maps", fake_sample)
    maps, _, _ = _run(3)
    assert len(maps) == 3


def test_raises_when_it_cannot_converge(monkeypatch):
    """An unfixable sequence must fail loudly rather than loop forever."""

    def all_bad(sampler, sequence, conformations, *args, **kwargs):
        return _stack([BAD_SPACING] * conformations)

    monkeypatch.setattr(generation, "_sample_distance_maps", all_bad)
    with pytest.raises(RuntimeError) as excinfo:
        _run(4, max_rounds=3)
    msg = str(excinfo.value)
    assert "Unable to generate 4 error-free conformation" in msg
    assert "3 rounds" in msg


def test_max_rounds_is_respected(monkeypatch):
    """The loop must stop after max_rounds attempts, not keep going."""
    calls = []

    def all_bad(sampler, sequence, conformations, *args, **kwargs):
        calls.append(conformations)
        return _stack([BAD_SPACING] * conformations)

    monkeypatch.setattr(generation, "_sample_distance_maps", all_bad)
    with pytest.raises(RuntimeError):
        _run(2, max_rounds=4)
    assert len(calls) == 4


def test_generate_rejects_non_bool_remove_errors():
    """generate() validates the flag like its other booleans."""
    from starling.frontend.ensemble_generation import generate

    with pytest.raises(ValueError, match="remove_errors must be True or False"):
        generate(SEQ, remove_errors="yes", output_directory=".")


def test_cli_exposes_remove_errors_flag():
    """The CLI flag exists and defaults to off."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv=['starling','--help']; "
            "from starling.scripts.starling_main_cli import main; main()",
        ],
        capture_output=True,
        text=True,
    )
    assert "--remove-errors" in result.stdout


#
# Tests below exercise the 3D/trajectory stage, so they run real MDS
# reconstruction. They are still driven by the stubbed sampler, so they stay
# fast, but they cover the coordinate accumulation and trimming logic that the
# distance-map-only tests above cannot reach.
#


def _run_with_structures(conformations, max_rounds=10):
    return generation._generate_error_filtered_conformers(
        sampler=StubSampler(),
        sequence=SEQ,
        conformations=conformations,
        batch_size=100,
        show_per_step_progress_bar=False,
        constraint=None,
        return_structures=True,
        device="cpu",
        num_cpus_mds=1,
        num_mds_init=1,
        show_progress_bar=False,
        verbose=False,
        max_rounds=max_rounds,
    )


def test_structures_path_returns_matching_maps_and_coordinates(monkeypatch):
    """Coordinates must come back aligned 1:1 with the surviving distance maps."""
    _patch_sampling(monkeypatch, [[GOOD_SPACING] * 3])
    maps, coords, discarded = _run_with_structures(3)

    assert len(maps) == 3
    assert coords is not None
    assert coords.shape == (3, L, 3), "expect (n_conformers, n_residues, xyz)"
    assert discarded == 0


def test_structures_path_tops_up_and_keeps_coordinates_in_sync(monkeypatch):
    """
    The coordinate array must be topped up alongside the distance maps.

    This is the case that matters for ``--remove-errors -r``: coordinates are
    accumulated across rounds and concatenated, so an off-by-one here would
    write a trajectory whose frame count disagrees with the distance maps.
    """
    calls = _patch_sampling(
        monkeypatch,
        [
            [GOOD_SPACING, BAD_SPACING, BAD_SPACING, GOOD_SPACING],
            [GOOD_SPACING, GOOD_SPACING],
        ],
    )
    maps, coords, discarded = _run_with_structures(4)

    assert len(maps) == 4
    assert len(coords) == 4, "coordinates must match the distance map count"
    assert coords.shape == (4, L, 3)
    assert discarded == 2
    assert calls == [4, 2]


def test_structures_path_trims_overshoot_consistently(monkeypatch):
    """Trimming an overshoot must trim maps and coordinates identically."""

    def fake_sample(sampler, sequence, conformations, *args, **kwargs):
        return _stack([GOOD_SPACING] * (conformations + 3))

    monkeypatch.setattr(generation, "_sample_distance_maps", fake_sample)
    maps, coords, _ = _run_with_structures(2)
    assert len(maps) == 2
    assert len(coords) == 2
