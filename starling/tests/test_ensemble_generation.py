"""
Comprehensive test suite for STARLING ensemble generation components.

Tests cover:
- Input handling and validation (handle_input, check_positive_int)
- generate() parameter validation and error paths
- Ensemble class construction, properties, and analysis methods
- Constraint classes (Bond, StericClash, Distance, Rg, Re, Helicity, Multi)
- Distance map symmetrization
- Serialization / deserialization round-trips
- Integration tests requiring model weights (marked slow)
"""

import os
import tempfile
import shutil

import numpy as np
import pytest
import torch

from starling import configs
from starling.frontend.ensemble_generation import (
    check_positive_int,
    generate,
    handle_input,
)
from starling.inference.constraints import (
    BondConstraint,
    Constraint,
    DistanceConstraint,
    HelicityConstraint,
    MultiConstraint,
    ReConstraint,
    RgConstraint,
    StericClashConstraint,
)
from starling.inference.generation import symmetrize_distance_map
from starling.structure.ensemble import Ensemble
from starling import load_ensemble


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# Short test sequence (within MAX_SEQUENCE_LENGTH)
SHORT_SEQ = "ASAPASPAPSPAP"

# Longer test sequence used in existing tests
MEDIUM_SEQ = "ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP"

# Sequence that exceeds MAX_SEQUENCE_LENGTH
LONG_SEQ = "AP" * 200  # 400 residues


def _make_synthetic_distance_maps(n_conformations, seq_len, seed=42):
    """Create synthetic positive-definite-ish distance maps for testing."""
    rng = np.random.RandomState(seed)
    dms = np.zeros((n_conformations, seq_len, seq_len), dtype=np.float32)
    for k in range(n_conformations):
        # Build distances from random coordinates so they satisfy triangle inequality
        coords = rng.randn(seq_len, 3).astype(np.float32) * 10
        for i in range(seq_len):
            for j in range(seq_len):
                dms[k, i, j] = np.linalg.norm(coords[i] - coords[j])
    return dms


@pytest.fixture
def synthetic_ensemble():
    """Return an Ensemble built from synthetic distance maps."""
    seq = "ACDEFGHIKLMNPQRSTVWY"  # 20 residues, all valid AAs
    dms = _make_synthetic_distance_maps(10, len(seq))
    return Ensemble(dms, seq)


@pytest.fixture
def tmpdir():
    """Provide a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ===========================================================================
# 1. check_positive_int
# ===========================================================================

class TestCheckPositiveInt:
    def test_positive_integer(self):
        assert check_positive_int(1) is True
        assert check_positive_int(100) is True

    def test_zero(self):
        assert check_positive_int(0) is False

    def test_negative(self):
        assert check_positive_int(-1) is False

    def test_float(self):
        assert check_positive_int(1.5) is False

    def test_string(self):
        assert check_positive_int("1") is False

    def test_none(self):
        assert check_positive_int(None) is False

    def test_numpy_int(self):
        assert check_positive_int(np.int64(5)) is True

    def test_numpy_zero(self):
        assert check_positive_int(np.int32(0)) is False

    def test_bool_true(self):
        # bool is subclass of int; True == 1 so should be True
        assert check_positive_int(True) is True

    def test_bool_false(self):
        assert check_positive_int(False) is False


# ===========================================================================
# 2. handle_input
# ===========================================================================

class TestHandleInput:
    """Tests for the input normalisation logic."""

    # --- string input (raw sequence) ---
    def test_single_sequence_string(self):
        result = handle_input("ACDEFG")
        assert result == {"sequence_1": "ACDEFG"}

    def test_single_sequence_string_custom_name(self):
        result = handle_input("ACDEFG", output_name="my_protein")
        assert result == {"my_protein": "ACDEFG"}

    def test_single_sequence_string_custom_index(self):
        result = handle_input("ACDEFG", seq_index_start=5)
        assert result == {"sequence_5": "ACDEFG"}

    def test_single_sequence_lowercase_converted(self):
        result = handle_input("acdefg")
        assert result == {"sequence_1": "ACDEFG"}

    def test_invalid_residue_raises(self):
        with pytest.raises(ValueError, match="Invalid amino acid"):
            handle_input("ACDEFGX")

    # --- list input ---
    def test_list_of_sequences(self):
        result = handle_input(["ACDE", "FGHI"])
        assert "sequence_1" in result
        assert "sequence_2" in result
        assert result["sequence_1"] == "ACDE"
        assert result["sequence_2"] == "FGHI"

    def test_list_custom_start_index(self):
        result = handle_input(["ACDE"], seq_index_start=10)
        assert "sequence_10" in result

    def test_empty_list(self):
        result = handle_input([])
        assert result == {}

    # --- dict input ---
    def test_dict_of_sequences(self):
        inp = {"prot_a": "ACDE", "prot_b": "FGHI"}
        result = handle_input(inp)
        assert result == {"prot_a": "ACDE", "prot_b": "FGHI"}

    def test_dict_lowercase_keys_preserved(self):
        inp = {"Prot_A": "acde"}
        result = handle_input(inp)
        assert "Prot_A" in result
        assert result["Prot_A"] == "ACDE"

    def test_dict_invalid_residue(self):
        with pytest.raises(ValueError, match="Invalid amino acid"):
            handle_input({"prot": "ACDXFG"})

    # --- file input ---
    def test_fasta_file(self, tmpdir):
        fasta = os.path.join(tmpdir, "test.fasta")
        with open(fasta, "w") as f:
            f.write(">protein_1\nACDEFGHIK\n>protein_2\nLMNPQRSTV\n")
        result = handle_input(fasta)
        assert "protein_1" in result
        assert "protein_2" in result
        assert result["protein_1"] == "ACDEFGHIK"

    def test_tsv_file(self, tmpdir):
        tsv_path = os.path.join(tmpdir, "test.tsv")
        with open(tsv_path, "w") as f:
            f.write("seq_a\tACDEFG\nseq_b\tGHIKLM\n")
        result = handle_input(tsv_path)
        assert result == {"seq_a": "ACDEFG", "seq_b": "GHIKLM"}

    def test_tsv_duplicate_names_raises(self, tmpdir):
        tsv_path = os.path.join(tmpdir, "dup.tsv")
        with open(tsv_path, "w") as f:
            f.write("same\tACDE\nsame\tFGHI\n")
        with pytest.raises(ValueError, match="Duplicate sequence name"):
            handle_input(tsv_path)

    def test_seq_in_file(self, tmpdir):
        seq_in = os.path.join(tmpdir, "test.seq.in")
        with open(seq_in, "w") as f:
            f.write("prot1\tACDEFG\n")
        result = handle_input(seq_in)
        assert result == {"prot1": "ACDEFG"}

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            handle_input("/nonexistent/path.fasta")

    # --- invalid type ---
    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid input type"):
            handle_input(12345)

    def test_invalid_type_tuple_raises(self):
        with pytest.raises(ValueError, match="Invalid input type"):
            handle_input(("ACDE",))


# ===========================================================================
# 3. generate() validation / error paths
# ===========================================================================

class TestGenerateValidation:
    """Test parameter validation without actually running the model."""

    def test_return_data_false_no_output_dir_raises(self):
        with pytest.raises(ValueError, match="no return data"):
            generate(SHORT_SEQ, conformations=10, return_data=False)

    def test_multiple_seqs_with_single_ensemble_raises(self):
        seqs = {"a": "ACDE" * 5, "b": "FGHI" * 5}
        with pytest.raises(ValueError, match="single ensemble"):
            generate(seqs, conformations=10, return_single_ensemble=True)

    def test_single_ensemble_and_no_return_data_raises(self):
        with pytest.raises(ValueError, match="single ensemble"):
            generate(
                SHORT_SEQ,
                conformations=10,
                return_single_ensemble=True,
                return_data=False,
            )

    def test_invalid_conformations_zero(self):
        with pytest.raises(ValueError, match="Conformations"):
            generate(SHORT_SEQ, conformations=0)

    def test_invalid_conformations_negative(self):
        with pytest.raises(ValueError, match="Conformations"):
            generate(SHORT_SEQ, conformations=-5)

    def test_invalid_conformations_float(self):
        with pytest.raises(ValueError, match="Conformations"):
            generate(SHORT_SEQ, conformations=1.5)

    def test_invalid_steps_zero(self):
        with pytest.raises(ValueError, match="Steps"):
            generate(SHORT_SEQ, conformations=10, steps=0)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            generate(SHORT_SEQ, conformations=10, batch_size=-1)

    def test_invalid_num_cpus_mds(self):
        with pytest.raises(ValueError, match="num_cpus_mds"):
            generate(SHORT_SEQ, conformations=10, num_cpus_mds=0)

    def test_invalid_num_mds_init(self):
        with pytest.raises(ValueError, match="num_mds_init"):
            generate(SHORT_SEQ, conformations=10, num_mds_init=0)

    def test_invalid_sampler_type(self):
        with pytest.raises(ValueError, match="sampler"):
            generate(SHORT_SEQ, conformations=10, sampler=123)

    def test_invalid_return_structures_type(self):
        with pytest.raises(ValueError, match="return_structures"):
            generate(SHORT_SEQ, conformations=10, return_structures="yes")

    def test_invalid_verbose_type(self):
        with pytest.raises(ValueError, match="verbose"):
            generate(SHORT_SEQ, conformations=10, verbose="yes")

    def test_invalid_show_progress_bar_type(self):
        with pytest.raises(ValueError, match="show_progress_bar"):
            generate(SHORT_SEQ, conformations=10, show_progress_bar="yes")

    def test_invalid_show_per_step_progress_bar_type(self):
        with pytest.raises(ValueError, match="show_per_step_progress_bar"):
            generate(SHORT_SEQ, conformations=10, show_per_step_progress_bar="yes")

    def test_nonexistent_output_directory_raises(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            generate(SHORT_SEQ, conformations=10, output_directory="/nonexistent/dir")

    def test_invalid_residue_raises(self):
        with pytest.raises(ValueError, match="Invalid amino acid"):
            generate("ACXEFG", conformations=10)


# ===========================================================================
# 4. Ensemble class
# ===========================================================================

class TestEnsembleConstruction:
    """Tests for Ensemble object creation and validation."""

    def test_basic_construction(self):
        seq = "ACDEF"
        dms = _make_synthetic_distance_maps(5, len(seq))
        ens = Ensemble(dms, seq)
        assert len(ens) == 5
        assert ens.sequence == "ACDEF"
        assert ens.sequence_length == 5
        assert ens.number_of_conformations == 5

    def test_invalid_distance_maps_not_ndarray(self):
        with pytest.raises(ValueError, match="numpy ndarray"):
            Ensemble([[1, 2], [3, 4]], "AC")

    def test_invalid_distance_maps_not_2d(self):
        dms = np.ones((3, 5))  # 2D, not 3D array of 2D matrices
        with pytest.raises(ValueError):
            Ensemble(dms, "ACDEF")

    def test_invalid_distance_maps_not_square(self):
        dms = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="square matrices"):
            Ensemble(dms, "ABC")

    def test_invalid_distance_maps_size_mismatch(self):
        dms = np.ones((2, 3, 3))
        with pytest.raises(ValueError, match="same size as the sequence"):
            Ensemble(dms, "AB")  # seq len 2, DM 3x3

    def test_invalid_sequence_type(self):
        dms = np.ones((2, 3, 3))
        with pytest.raises((ValueError, TypeError)):
            Ensemble(dms, 123)

    def test_invalid_sequence_characters(self):
        dms = np.ones((2, 3, 3))
        with pytest.raises(ValueError, match="valid amino acid"):
            Ensemble(dms, "XYZ")

    def test_invalid_ssprot_type(self):
        dms = _make_synthetic_distance_maps(2, 5)
        with pytest.raises((ValueError, TypeError), match="SSProtein"):
            Ensemble(dms, "ACDEF", ssprot_ensemble="not_an_ssprot")

    def test_has_structures_false_by_default(self, synthetic_ensemble):
        assert synthetic_ensemble.has_structures is False


class TestEnsembleProperties:
    """Tests for Ensemble analysis methods using synthetic data."""

    def test_len(self, synthetic_ensemble):
        assert len(synthetic_ensemble) == 10

    def test_str_repr(self, synthetic_ensemble):
        s = str(synthetic_ensemble)
        assert "ENSEMBLE" in s
        assert "len=20" in s
        assert "ensemble_size=10" in s
        assert "[ ]" in s  # no structures

    def test_repr(self, synthetic_ensemble):
        assert repr(synthetic_ensemble) == str(synthetic_ensemble)

    def test_distance_maps_shape(self, synthetic_ensemble):
        dms = synthetic_ensemble.distance_maps()
        assert dms.shape == (10, 20, 20)

    def test_distance_maps_mean(self, synthetic_ensemble):
        dm_mean = synthetic_ensemble.distance_maps(return_mean=True)
        assert dm_mean.shape == (20, 20)

    def test_rij(self, synthetic_ensemble):
        vals = synthetic_ensemble.rij(0, 5)
        assert len(vals) == 10
        assert all(v >= 0 for v in vals)

    def test_rij_mean(self, synthetic_ensemble):
        val = synthetic_ensemble.rij(0, 5, return_mean=True)
        assert isinstance(val, (float, np.floating))

    def test_rij_invalid_index(self, synthetic_ensemble):
        with pytest.raises(ValueError, match="Invalid residue index"):
            synthetic_ensemble.rij(-1, 5)

    def test_end_to_end_distance(self, synthetic_ensemble):
        re = synthetic_ensemble.end_to_end_distance()
        assert len(re) == 10
        assert all(v >= 0 for v in re)

    def test_end_to_end_distance_mean(self, synthetic_ensemble):
        re_mean = synthetic_ensemble.end_to_end_distance(return_mean=True)
        assert isinstance(re_mean, (float, np.floating))

    def test_radius_of_gyration(self, synthetic_ensemble):
        rg = synthetic_ensemble.radius_of_gyration()
        assert len(rg) == 10
        assert all(v > 0 for v in rg)

    def test_radius_of_gyration_mean(self, synthetic_ensemble):
        rg_mean = synthetic_ensemble.radius_of_gyration(return_mean=True)
        assert isinstance(rg_mean, (float, np.floating))
        assert rg_mean > 0

    def test_radius_of_gyration_caching(self, synthetic_ensemble):
        rg1 = synthetic_ensemble.radius_of_gyration()
        rg2 = synthetic_ensemble.radius_of_gyration()
        assert np.array_equal(rg1, rg2)

    def test_radius_of_gyration_force_recompute(self):
        seq = "ACDEFGHIKLMNPQRSTVWY"
        dms = _make_synthetic_distance_maps(10, len(seq))
        ens = Ensemble(dms, seq)
        rg1 = ens.radius_of_gyration()
        # Build a fresh ensemble to test force_recompute without stale cache
        ens2 = Ensemble(dms, seq)
        rg2 = ens2.radius_of_gyration(force_recompute=True)
        assert np.allclose(rg1, rg2)

    def test_local_radius_of_gyration(self, synthetic_ensemble):
        local_rg = synthetic_ensemble.local_radius_of_gyration(2, 10)
        assert len(local_rg) == 10
        assert all(v > 0 for v in local_rg)

    def test_local_radius_of_gyration_mean(self, synthetic_ensemble):
        val = synthetic_ensemble.local_radius_of_gyration(2, 10, return_mean=True)
        assert isinstance(val, (float, np.floating))

    def test_hydrodynamic_radius_nygaard(self, synthetic_ensemble):
        rh = synthetic_ensemble.hydrodynamic_radius(mode="nygaard")
        assert len(rh) == 10

    def test_hydrodynamic_radius_kr(self, synthetic_ensemble):
        rh = synthetic_ensemble.hydrodynamic_radius(mode="kr")
        assert len(rh) == 10
        assert all(v > 0 for v in rh)

    def test_hydrodynamic_radius_mean(self, synthetic_ensemble):
        rh_mean = synthetic_ensemble.hydrodynamic_radius(return_mean=True)
        assert isinstance(rh_mean, (float, np.floating))

    def test_hydrodynamic_radius_invalid_mode(self, synthetic_ensemble):
        with pytest.raises(ValueError, match="must be either 'kr' or 'nygaard'"):
            synthetic_ensemble.hydrodynamic_radius(mode="invalid")

    def test_hydrodynamic_radius_caching_and_mode_switch(self, synthetic_ensemble):
        rh_ny = synthetic_ensemble.hydrodynamic_radius(mode="nygaard")
        rh_kr = synthetic_ensemble.hydrodynamic_radius(mode="kr")
        # switching mode should recompute; values generally differ
        assert rh_ny is not rh_kr

    def test_contact_map(self, synthetic_ensemble):
        cm = synthetic_ensemble.contact_map()
        assert cm.shape == (10, 20, 20)
        assert set(np.unique(cm)).issubset({0, 1})

    def test_contact_map_mean(self, synthetic_ensemble):
        cm_mean = synthetic_ensemble.contact_map(return_mean=True)
        assert cm_mean.shape == (20, 20)
        assert np.all(cm_mean >= 0) and np.all(cm_mean <= 1)

    def test_contact_map_summed(self, synthetic_ensemble):
        cm_sum = synthetic_ensemble.contact_map(return_summed=True)
        assert cm_sum.shape == (20, 20)
        assert np.all(cm_sum >= 0) and np.all(cm_sum <= 10)

    def test_contact_map_mean_and_summed_raises(self, synthetic_ensemble):
        with pytest.raises(ValueError, match="cannot both be set"):
            synthetic_ensemble.contact_map(return_mean=True, return_summed=True)

    def test_contact_map_custom_threshold(self, synthetic_ensemble):
        cm_tight = synthetic_ensemble.contact_map(contact_thresh=5)
        cm_loose = synthetic_ensemble.contact_map(contact_thresh=50)
        # Looser threshold should have >= contacts
        assert np.sum(cm_loose) >= np.sum(cm_tight)


class TestEnsembleSerialization:
    """Tests for save / load round-trips."""

    def test_save_load_uncompressed(self, synthetic_ensemble, tmpdir):
        path = os.path.join(tmpdir, "test_ens")
        synthetic_ensemble.save(path, compress=False, verbose=False)
        loaded = load_ensemble(path + ".starling")
        assert len(loaded) == len(synthetic_ensemble)
        assert loaded.sequence == synthetic_ensemble.sequence
        assert np.allclose(
            loaded.distance_maps(), synthetic_ensemble.distance_maps()
        )

    def test_save_load_compressed_lzma(self, synthetic_ensemble, tmpdir):
        path = os.path.join(tmpdir, "test_lzma")
        synthetic_ensemble.save(
            path, compress=True, compression_algorithm="lzma", verbose=False
        )
        loaded = load_ensemble(path + ".starling.xz")
        assert len(loaded) == len(synthetic_ensemble)
        assert np.allclose(
            loaded.end_to_end_distance(),
            synthetic_ensemble.end_to_end_distance(),
            atol=0.1,
        )

    def test_save_load_compressed_gzip(self, synthetic_ensemble, tmpdir):
        path = os.path.join(tmpdir, "test_gzip")
        synthetic_ensemble.save(
            path, compress=True, compression_algorithm="gzip", verbose=False
        )
        loaded = load_ensemble(path + ".starling.gzip")
        assert len(loaded) == len(synthetic_ensemble)
        assert np.allclose(
            loaded.end_to_end_distance(),
            synthetic_ensemble.end_to_end_distance(),
            atol=0.1,
        )

    def test_save_load_compressed_no_reduce_precision(self, synthetic_ensemble, tmpdir):
        path = os.path.join(tmpdir, "test_full_prec")
        synthetic_ensemble.save(
            path,
            compress=True,
            reduce_precision=False,
            verbose=False,
        )
        loaded = load_ensemble(path + ".starling.xz")
        assert np.allclose(
            loaded.distance_maps(), synthetic_ensemble.distance_maps()
        )

    def test_save_load_preserves_sequence(self, synthetic_ensemble, tmpdir):
        path = os.path.join(tmpdir, "test_seq")
        synthetic_ensemble.save(path, verbose=False)
        loaded = load_ensemble(path + ".starling")
        assert loaded.sequence == synthetic_ensemble.sequence
        assert loaded.sequence_length == synthetic_ensemble.sequence_length


class TestEnsembleErrorChecking:
    """Tests for the check_for_errors method."""

    def test_no_errors_in_good_ensemble(self, synthetic_ensemble):
        bad = synthetic_ensemble.check_for_errors(verbose=False)
        # synthetic data from coordinates should be OK
        assert isinstance(bad, list)

    def test_check_for_errors_remove(self):
        seq = "ACDEF"
        dms = _make_synthetic_distance_maps(5, len(seq))
        # Corrupt one frame: set adjacent residues to huge distance
        dms[2, 0, 1] = 0.1  # impossibly close
        dms[2, 1, 0] = 0.1
        ens = Ensemble(dms, seq)
        bad = ens.check_for_errors(remove_errors=True, verbose=False)
        if len(bad) > 0:
            assert ens.number_of_conformations == 5 - len(bad)


# ===========================================================================
# 5. Distance map symmetrization
# ===========================================================================

class TestSymmetrizeDistanceMap:
    def test_basic_symmetrization(self):
        dm = torch.zeros(5, 5)
        dm[0, 1] = 10.0
        dm[1, 0] = 5.0  # will be overwritten
        dm[2, 3] = 7.0
        result = symmetrize_distance_map(dm)
        assert result[0, 1] == result[1, 0] == 10.0
        assert result[2, 3] == result[3, 2] == 7.0

    def test_diagonal_is_zero(self):
        dm = torch.ones(5, 5) * 3.0
        result = symmetrize_distance_map(dm)
        for i in range(5):
            assert result[i, i] == 0.0

    def test_shape_preserved(self):
        dm = torch.randn(10, 10)
        result = symmetrize_distance_map(dm)
        assert result.shape == (10, 10)

    def test_3d_input_squeezed(self):
        dm = torch.randn(1, 8, 8)
        result = symmetrize_distance_map(dm)
        assert result.shape == (8, 8)

    def test_result_is_symmetric(self):
        dm = torch.randn(12, 12)
        result = symmetrize_distance_map(dm)
        assert torch.allclose(result, result.T)


# ===========================================================================
# 6. Constraint classes
# ===========================================================================

class TestConstraintBase:
    """Tests for the abstract Constraint base class functionality."""

    def test_default_parameters(self):
        c = BondConstraint()  # concrete subclass
        assert c.constraint_weight == 1.0
        assert c.schedule == "cosine"
        assert c.guidance_start == 0.0
        assert c.guidance_end == 1.0

    def test_custom_parameters(self):
        c = BondConstraint(
            constraint_weight=2.5,
            schedule="linear",
            guidance_start=0.1,
            guidance_end=0.9,
        )
        assert c.constraint_weight == 2.5
        assert c.schedule == "linear"
        assert c.guidance_start == 0.1
        assert c.guidance_end == 0.9

    def test_should_apply_guidance_full_range(self):
        c = BondConstraint(guidance_start=0.0, guidance_end=1.0)
        c.n_steps = 100
        # At timestep 0 (reversed: 1.0) should apply
        assert c.should_apply_guidance(0, 100) is True
        # At timestep 100 (reversed: 0.0) should apply
        assert c.should_apply_guidance(100, 100) is True
        # At timestep 50 (reversed: 0.5) should apply
        assert c.should_apply_guidance(50, 100) is True

    def test_should_apply_guidance_restricted_range(self):
        c = BondConstraint(guidance_start=0.3, guidance_end=0.7)
        c.n_steps = 100
        # reversed 0.0 outside [0.3, 0.7]
        assert c.should_apply_guidance(100, 100) is False
        # reversed 0.5 inside [0.3, 0.7]
        assert c.should_apply_guidance(50, 100) is True
        # reversed 1.0 outside [0.3, 0.7]
        assert c.should_apply_guidance(0, 100) is False

    def test_cosine_weight(self):
        c = BondConstraint()
        # At t=0, cosine weight should be ~1.0
        assert abs(c.cosine_weight(0, 100) - 1.0) < 0.01
        # At t=total_steps, cosine weight should be ~0.0
        assert abs(c.cosine_weight(100, 100)) < 0.01

    def test_get_time_scale_cosine(self):
        c = BondConstraint(schedule="cosine")
        c.n_steps = 100
        ts = c.get_time_scale(0)
        assert abs(ts - 1.0) < 0.01

    def test_get_time_scale_linear(self):
        c = BondConstraint(schedule="linear_fallback")
        c.n_steps = 100
        # Fallback to linear: 1.0 - (t / n_steps)
        ts = c.get_time_scale(50)
        assert abs(ts - 0.5) < 0.01

    def test_get_adaptive_clip_threshold(self):
        c = BondConstraint()
        c.n_steps = 100
        # At beginning (timestep=100), it should be near max_threshold
        t_begin = c.get_adaptive_clip_threshold(100)
        # At end (timestep=0), it should be near min_threshold + something
        t_end = c.get_adaptive_clip_threshold(0)
        assert t_begin >= 1.0
        assert t_end >= 1.0


class TestBondConstraint:
    def test_default_params(self):
        c = BondConstraint()
        assert c.bond_length == 3.81
        assert c.tolerance == 0.0
        assert c.force_constant == 2.0

    def test_custom_params(self):
        c = BondConstraint(bond_length=4.0, tolerance=0.5, force_constant=1.0)
        assert c.bond_length == 4.0
        assert c.tolerance == 0.5
        assert c.force_constant == 1.0

    def test_compute_loss_shape(self):
        c = BondConstraint()
        c.sequence_length = 10
        # distance maps: (B, C, H, W) = (4, 1, 10, 10)
        dm = torch.ones(4, 1, 10, 10) * 3.81
        per_batch, mean = c.compute_loss(dm)
        assert per_batch.shape == (4,)
        assert mean.shape == ()

    def test_zero_loss_at_ideal_bond(self):
        c = BondConstraint()
        c.sequence_length = 10
        dm = torch.ones(2, 1, 10, 10) * 5.0
        # Set diagonal+1 to ideal bond length
        for i in range(9):
            dm[:, 0, i, i + 1] = 3.81
        per_batch, mean = c.compute_loss(dm)
        assert torch.allclose(mean, torch.tensor(0.0), atol=1e-6)


class TestStericClashConstraint:
    def test_default_params(self):
        c = StericClashConstraint()
        assert c.steric_clash_definition == 5.0
        assert c.force_constant == 2.0

    def test_compute_loss_no_clashes(self):
        c = StericClashConstraint(steric_clash_definition=5.0)
        c.sequence_length = 8
        # All distances > 5.0 → no clashes
        dm = torch.ones(3, 1, 8, 8) * 10.0
        per_batch, mean = c.compute_loss(dm)
        assert torch.allclose(mean, torch.tensor(0.0), atol=1e-6)

    def test_compute_loss_with_clashes(self):
        c = StericClashConstraint(steric_clash_definition=5.0)
        c.sequence_length = 8
        dm = torch.ones(3, 1, 8, 8) * 2.0  # all < 5.0 → clashes
        per_batch, mean = c.compute_loss(dm)
        assert mean.item() > 0


class TestDistanceConstraint:
    def test_construction(self):
        c = DistanceConstraint(resid1=5, resid2=10, target=15.0)
        assert c.resid1 == 5
        assert c.resid2 == 10
        assert c.target == 15.0

    def test_compute_loss_at_target(self):
        c = DistanceConstraint(resid1=2, resid2=5, target=10.0)
        c.sequence_length = 8
        dm = torch.ones(3, 1, 8, 8) * 5.0
        dm[:, 0, 2, 5] = 10.0
        per_batch, mean = c.compute_loss(dm)
        assert torch.allclose(mean, torch.tensor(0.0), atol=1e-6)

    def test_compute_loss_away_from_target(self):
        c = DistanceConstraint(resid1=2, resid2=5, target=10.0)
        c.sequence_length = 8
        dm = torch.ones(3, 1, 8, 8) * 20.0
        per_batch, mean = c.compute_loss(dm)
        assert mean.item() > 0

    def test_tolerance(self):
        c = DistanceConstraint(resid1=2, resid2=5, target=10.0, tolerance=2.0)
        c.sequence_length = 8
        dm = torch.ones(3, 1, 8, 8) * 5.0
        dm[:, 0, 2, 5] = 11.0  # deviation=1.0, within tolerance=2.0
        per_batch, mean = c.compute_loss(dm)
        assert torch.allclose(mean, torch.tensor(0.0), atol=1e-6)


class TestRgConstraint:
    def test_construction(self):
        c = RgConstraint(target=25.0)
        assert c.target == 25.0
        assert c.tolerance == 0.0
        assert c.force_constant == 2.0

    def test_compute_loss_shape(self):
        c = RgConstraint(target=25.0)
        c.sequence_length = 10
        c.device = torch.device("cpu")
        dm = torch.ones(4, 1, 10, 10) * 5.0
        per_batch, mean = c.compute_loss(dm)
        assert per_batch.shape == (4,)
        assert mean.shape == ()


class TestReConstraint:
    def test_construction(self):
        c = ReConstraint(target=50.0)
        assert c.target == 50.0
        assert c.tolerance == 0.0
        assert c.force_constant == 2.0


class TestHelicityConstraint:
    def test_construction(self):
        c = HelicityConstraint(resid_start=10, resid_end=30)
        assert c.resid_start == 10
        assert c.resid_end == 30
        assert c.tolerance == 0.0

    def test_setup_not_called_without_model(self):
        c = HelicityConstraint(resid_start=5, resid_end=15)
        assert c.helix_ref is None
        assert c.mask is None


class TestMultiConstraint:
    def test_construction(self):
        c1 = BondConstraint(constraint_weight=1.0)
        c2 = StericClashConstraint(constraint_weight=0.5)
        mc = MultiConstraint([c1, c2])
        assert len(mc.constraints) == 2
        assert mc.constraint_weights == [1.0, 0.5]

    def test_compute_loss_combines(self):
        c1 = BondConstraint(constraint_weight=1.0)
        c1.sequence_length = 8
        c2 = StericClashConstraint(constraint_weight=1.0, steric_clash_definition=5.0)
        c2.sequence_length = 8
        mc = MultiConstraint([c1, c2])
        mc.sequence_length = 8
        mc.constraint_weights = [1.0, 1.0]

        dm = torch.ones(2, 1, 8, 8) * 3.0  # close enough for clashes
        per_batch, total = mc.compute_loss(dm)
        assert per_batch.shape == (2,)
        assert total.item() > 0

    def test_guidance_starts_ends_extracted(self):
        c1 = RgConstraint(target=20.0, guidance_start=0.1, guidance_end=0.8)
        c2 = DistanceConstraint(
            resid1=0, resid2=5, target=10.0, guidance_start=0.2, guidance_end=0.9
        )
        mc = MultiConstraint([c1, c2])
        assert mc.guidance_starts == [0.1, 0.2]
        assert mc.guidance_ends == [0.8, 0.9]


# ===========================================================================
# 7. Config sanity checks
# ===========================================================================

class TestConfigs:
    def test_defaults_exist(self):
        assert configs.DEFAULT_NUMBER_CONFS > 0
        assert configs.DEFAULT_BATCH_SIZE > 0
        assert configs.DEFAULT_STEPS > 0
        assert configs.MAX_SEQUENCE_LENGTH > 0
        assert configs.DEFAULT_IONIC_STRENGTH > 0
        assert configs.DEFAULT_MDS_NUM_INIT > 0

    def test_valid_aa_string(self):
        assert len(configs.VALID_AA) == 20
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            assert aa in configs.VALID_AA

    def test_default_sampler(self):
        assert configs.DEFAULT_SAMPLER in ("ddim", "ddpm", "plms")


# ===========================================================================
# 8. Integration tests (require model weights, marked slow)
# ===========================================================================

@pytest.mark.slow
class TestIntegrationGenerate:
    """
    Integration tests that run the full generation pipeline.
    These require model weights to be available and are slow.
    Run with: pytest -m slow
    """

    def test_generate_single_sequence_dict_return(self):
        C = generate(
            MEDIUM_SEQ,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
        )
        assert isinstance(C, dict)
        assert "sequence_1" in C
        E = C["sequence_1"]
        assert len(E) == 10
        assert E.sequence == MEDIUM_SEQ

    def test_generate_single_ensemble_return(self):
        E = generate(
            MEDIUM_SEQ,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
        )
        assert isinstance(E, Ensemble)
        assert len(E) == 10

    def test_generate_multiple_sequences(self):
        seqs = {"prot_a": "AP" * 20, "prot_b": "GS" * 25}
        C = generate(
            seqs,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
        )
        assert "prot_a" in C
        assert "prot_b" in C
        assert len(C["prot_a"]) == 10
        assert len(C["prot_b"]) == 10

    def test_generate_list_input(self):
        seqs = ["AP" * 20, "GS" * 25]
        C = generate(
            seqs,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
        )
        assert len(C) == 2

    def test_generate_skip_long_sequences(self):
        seqs = {"short": "AP" * 20, "toolong": LONG_SEQ}
        C = generate(
            seqs,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
        )
        assert len(C) == 1
        assert "short" in C

    def test_generate_with_output_directory(self, tmpdir):
        E = generate(
            MEDIUM_SEQ,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
            output_directory=tmpdir,
        )
        assert len(E) == 10
        # Verify files were written
        files = os.listdir(tmpdir)
        assert any(f.endswith(".starling") or f.endswith(".xz") for f in files)

    def test_generate_with_structures(self):
        E = generate(
            MEDIUM_SEQ,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=True,
            return_single_ensemble=True,
        )
        assert E.has_structures is True

    def test_generate_biophysical_properties(self):
        E = generate(
            MEDIUM_SEQ,
            conformations=50,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
        )
        rg = E.radius_of_gyration(return_mean=True)
        re = E.end_to_end_distance(return_mean=True)
        assert rg > 0
        assert re > 0
        # Sanity: for a disordered protein, Rg << Re in general
        assert re > rg

    def test_generate_different_samplers(self):
        for sampler_name in ("ddim", "ddpm", "plms"):
            E = generate(
                SHORT_SEQ,
                conformations=5,
                verbose=False,
                show_progress_bar=False,
                return_data=True,
                return_structures=False,
                return_single_ensemble=True,
                sampler=sampler_name,
            )
            assert len(E) == 5

    def test_generate_batch_size_larger_than_conformations(self):
        E = generate(
            SHORT_SEQ,
            conformations=5,
            batch_size=100,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
        )
        assert len(E) == 5

    def test_generate_batch_size_smaller_than_conformations(self):
        E = generate(
            SHORT_SEQ,
            conformations=15,
            batch_size=4,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
        )
        assert len(E) == 15

    def test_generate_custom_steps(self):
        E = generate(
            SHORT_SEQ,
            conformations=5,
            steps=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
        )
        assert len(E) == 5

    def test_generate_device_cpu(self):
        E = generate(
            SHORT_SEQ,
            conformations=5,
            device="cpu",
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
        )
        assert len(E) == 5

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_generate_device_mps(self):
        E = generate(
            SHORT_SEQ,
            conformations=5,
            device="mps",
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
        )
        assert len(E) == 5

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_generate_device_cuda(self):
        E = generate(
            SHORT_SEQ,
            conformations=5,
            device="cuda",
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
        )
        assert len(E) == 5

    def test_generate_from_fasta(self, tmpdir):
        fasta = os.path.join(tmpdir, "input.fasta")
        with open(fasta, "w") as f:
            f.write(f">test_protein\n{MEDIUM_SEQ}\n")
        C = generate(
            fasta,
            conformations=5,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
        )
        assert "test_protein" in C
        assert len(C["test_protein"]) == 5

    def test_generate_no_return_data_with_output_dir(self, tmpdir):
        result = generate(
            SHORT_SEQ,
            conformations=5,
            verbose=False,
            show_progress_bar=False,
            return_data=False,
            output_directory=tmpdir,
        )
        # Should return None when return_data=False
        # but files should exist
        files = os.listdir(tmpdir)
        assert len(files) > 0


@pytest.mark.slow
class TestIntegrationConstraints:
    """Integration tests for constrained generation."""

    def test_generate_with_rg_constraint(self):
        constraint = RgConstraint(target=20.0, force_constant=0.1)
        E = generate(
            MEDIUM_SEQ,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
            constraint=constraint,
        )
        assert len(E) == 10

    def test_generate_with_bond_constraint(self):
        constraint = BondConstraint()
        E = generate(
            MEDIUM_SEQ,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
            constraint=constraint,
        )
        assert len(E) == 10

    def test_generate_with_distance_constraint(self):
        constraint = DistanceConstraint(resid1=5, resid2=50, target=30.0)
        E = generate(
            MEDIUM_SEQ,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
            constraint=constraint,
        )
        assert len(E) == 10

    def test_generate_with_steric_clash_constraint(self):
        constraint = StericClashConstraint()
        E = generate(
            MEDIUM_SEQ,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
            constraint=constraint,
        )
        assert len(E) == 10

    def test_generate_with_multi_constraint(self):
        c1 = BondConstraint(constraint_weight=0.5)
        c2 = StericClashConstraint(constraint_weight=0.5)
        constraint = MultiConstraint([c1, c2])
        E = generate(
            MEDIUM_SEQ,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
            constraint=constraint,
        )
        assert len(E) == 10


@pytest.mark.slow
class TestIntegrationEnsembleTrajectory:
    """Integration tests for trajectory building and analysis."""

    def test_trajectory_build_and_rg_consistency(self):
        E = generate(
            MEDIUM_SEQ,
            conformations=20,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=True,
            return_single_ensemble=True,
        )
        traj = E.trajectory
        traj_rg = np.mean(traj.get_radius_of_gyration())
        dm_rg = E.radius_of_gyration(return_mean=True)
        assert np.isclose(traj_rg, dm_rg, rtol=0.05, atol=1.0)

    def test_save_trajectory_pdb(self, tmpdir):
        E = generate(
            MEDIUM_SEQ,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=True,
            return_single_ensemble=True,
        )
        prefix = os.path.join(tmpdir, "test_traj")
        E.save_trajectory(prefix, pdb_trajectory=True)
        assert os.path.exists(prefix + ".pdb")

    def test_save_trajectory_xtc(self, tmpdir):
        E = generate(
            MEDIUM_SEQ,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=True,
            return_single_ensemble=True,
        )
        prefix = os.path.join(tmpdir, "test_traj")
        E.save_trajectory(prefix, pdb_trajectory=False)
        assert os.path.exists(prefix + ".pdb")
        assert os.path.exists(prefix + ".xtc")

    def test_lazy_trajectory_build(self):
        E = generate(
            MEDIUM_SEQ,
            conformations=10,
            verbose=False,
            show_progress_bar=False,
            return_data=True,
            return_structures=False,
            return_single_ensemble=True,
        )
        assert E.has_structures is False
        # Accessing trajectory triggers lazy build
        traj = E.trajectory
        assert E.has_structures is True
        assert traj is not None
