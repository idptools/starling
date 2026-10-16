"""
Regression tests for bugs fixed in STARLING's inference front end.

Covers constraint guidance (batch-size independence, schedules, index checks),
generate()/handle_input() validation, output naming, weights metadata, the
command-line interface, user configuration loading, compilation options,
sequence/ensemble encoders and PyTorch Hub loading. Everything runs on the CPU
with stub models, so no model weights are needed.
"""

import copy
import importlib.util
import os
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from einops import rearrange

import starling
from starling import configs
from starling.data.schedulers import cosine_beta_schedule
from starling.frontend import ensemble_generation as frontend
from starling.frontend.ensemble_generation import generate, handle_input
from starling.inference import generation, model_loading
from starling.inference.constraints import (
    REFERENCE_BATCH_SIZE,
    BondConstraint,
    DistanceConstraint,
    HelicityConstraint,
    MultiConstraint,
    RgConstraint,
    StericClashConstraint,
)
from starling.samplers import SAMPLERS
from starling.structure.ensemble import load_ensemble
from starling.utilities import symmetrize_tensor_distance_maps

# A short sequence well inside the 380-residue limit
SEQUENCE = "ACDEFGHIK"

# Sequence one residue longer than STARLING can generate
TOO_LONG_SEQUENCE = "A" * (configs.MAX_SEQUENCE_LENGTH + 1)

# Number of diffusion timesteps the stub model was "trained" with, matching STARLING's models
STUB_TRAINING_TIMESTEPS = 1000


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class IdentityDecoder:
    """VAE stand-in whose decoded distance maps are the latents themselves."""

    device = torch.device("cpu")

    @staticmethod
    def decode(latents):
        return latents


def stub_diffusion():
    """Return a minimal diffusion model with a real cosine noise schedule (as in test_samplers.py)."""
    betas = cosine_beta_schedule(STUB_TRAINING_TIMESTEPS)
    alpha = torch.cumprod(1 - betas, dim=0)
    alpha_prev = torch.cat([torch.ones(1), alpha[:-1]])
    return SimpleNamespace(
        device=torch.device("cpu"),
        num_timesteps=STUB_TRAINING_TIMESTEPS,
        alphas_cumprod=alpha,
        betas=betas,
        sqrt_recip_alphas=torch.sqrt(1 / (1 - betas)),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1 - alpha),
        posterior_variance=betas * (1 - alpha_prev) / (1 - alpha),
        latent_space_scaling_factor=2.0,
        sequence2labels=lambda tokens, mask, salt: tokens,
        model=lambda x, t, c, mask: 0.1 * x + t[:, None, None, None] * 0.0001,
    )


def stub_decoder():
    """Decoder that turns latents into small, non-negative 384 x 384 'distance maps'."""
    return SimpleNamespace(
        device=torch.device("cpu"),
        decode=lambda x: torch.nn.functional.interpolate(x.abs(), size=(384, 384)),
    )


@pytest.fixture
def stub_models(monkeypatch):
    """Make generate() sample with stub models instead of loading weights."""
    models = (stub_decoder(), stub_diffusion())
    monkeypatch.setattr(generation.model_manager, "get_models", lambda **kwargs: models)
    return models


@pytest.fixture
def no_model_loading(monkeypatch):
    """Fail the test if generate() tries to load models."""

    def unexpected_load(**kwargs):
        pytest.fail("models were loaded before the input was rejected")

    monkeypatch.setattr(generation.model_manager, "get_models", unexpected_load)


@pytest.fixture
def captured_backend(monkeypatch):
    """Replace generate_backend with a stub that records its arguments."""
    seen = {}

    def backend(*args, **kwargs):
        seen["args"] = args
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(frontend.generation, "generate_backend", backend)
    return seen


def quiet_generate(user_input, **kwargs):
    """Call generate() on the CPU without progress bars."""
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("show_progress_bar", False)
    kwargs.setdefault("show_per_step_progress_bar", False)
    return generate(user_input, **kwargs)


# ---------------------------------------------------------------------------
# Constraint guidance does not depend on batch size
# ---------------------------------------------------------------------------


def _pre_fix_guided_latents(constraint, latents, timestep):
    """
    Reproduce Constraint.apply() as it was before the batch-size fix.

    The only difference from the current code is that the gradient is taken
    of the batch-MEAN loss, so each conformer's update scaled as 1 / batch_size.
    """
    latents_copy = latents.clone().requires_grad_(True)
    maps = symmetrize_tensor_distance_maps(
        constraint.encoder_model.decode(latents_copy / constraint.latent_space_scaling_factor)
    )
    per_batch_loss, mean_loss = constraint.compute_loss(maps)
    base_grad = torch.autograd.grad(mean_loss, latents_copy)[0]

    if per_batch_loss.mean() > 1e-6:
        loss_scale = (per_batch_loss / per_batch_loss.mean()).clamp(max=2.0)
    else:
        loss_scale = torch.ones_like(per_batch_loss)
    update = -constraint.constraint_weight * constraint.get_time_scale(timestep) * rearrange(
        loss_scale, "b -> b 1 1 1"
    ) * base_grad
    norms = rearrange(update, "b c h w -> b (c h w)").norm(dim=1, keepdim=True)
    clip_factors = rearrange((1.0 / (norms + 1e-6)).clamp(max=1.0), "b 1 -> b 1 1 1")
    return latents + (update * clip_factors).detach()


def _rg_constraint():
    """An Rg constraint on a 10-residue sequence, decoded through the identity decoder."""
    constraint = RgConstraint(target=2.0, verbose=False)
    return constraint.initialize(IdentityDecoder(), torch.tensor(1.0), STUB_TRAINING_TIMESTEPS, 10)


def test_guidance_at_the_reference_batch_size_is_unchanged():
    torch.manual_seed(0)
    constraint = _rg_constraint()
    latents = 3.0 + torch.randn(REFERENCE_BATCH_SIZE, 1, 24, 24)
    torch.testing.assert_close(
        constraint.apply(latents, timestep=200),
        _pre_fix_guided_latents(constraint, latents, timestep=200),
    )


def test_a_conformers_guidance_does_not_depend_on_batch_size():
    torch.manual_seed(0)
    constraint = _rg_constraint()
    conformer = 3.0 + torch.randn(1, 1, 24, 24)
    batch = conformer.repeat(4, 1, 1, 1)

    alone = constraint.apply(conformer, timestep=200) - conformer
    in_batch = (constraint.apply(batch, timestep=200) - batch)[:1]

    # the update must be a real (unclipped, nonzero) step for the comparison to mean anything
    assert 0 < alone.norm() < 1
    torch.testing.assert_close(in_batch, alone)

    # before the fix, a conformer in a batch of 4 moved a quarter as far
    pre_fix_alone = _pre_fix_guided_latents(constraint, conformer, 200) - conformer
    pre_fix_in_batch = (_pre_fix_guided_latents(constraint, batch, 200) - batch)[:1]
    torch.testing.assert_close(pre_fix_alone, 4 * pre_fix_in_batch, rtol=1e-3, atol=1e-8)


# ---------------------------------------------------------------------------
# Constraint schedules, empty steric-clash masks and residue index checks
# ---------------------------------------------------------------------------


def test_none_schedule_is_a_constant_weight_of_one():
    constraint = BondConstraint(schedule=None)
    constraint.n_steps = 100
    assert [constraint.get_time_scale(t) for t in (0, 50, 100)] == [1.0, 1.0, 1.0]


def test_linear_schedule_is_an_explicit_option():
    constraint = BondConstraint(schedule="linear")
    constraint.n_steps = 100
    assert constraint.get_time_scale(25) == pytest.approx(0.75)


@pytest.mark.parametrize("schedule", ["cosin", "Linear", "constant", 1.0])
def test_unknown_schedules_are_rejected(schedule):
    with pytest.raises(ValueError, match="schedule must be one of"):
        BondConstraint(schedule=schedule)


def test_multiconstraint_validates_its_schedule_and_warns_about_ignored_child_schedules():
    with pytest.raises(ValueError, match="schedule must be one of"):
        MultiConstraint([BondConstraint()], schedule="bell")
    with pytest.warns(UserWarning, match="RgConstraint"):
        MultiConstraint([BondConstraint(), RgConstraint(target=20.0, schedule="bell_shaped")])


@pytest.mark.parametrize("sequence_length", [1, 2])
def test_steric_clash_loss_is_zero_not_nan_without_residue_pairs(sequence_length):
    constraint = StericClashConstraint(verbose=False)
    constraint.initialize(IdentityDecoder(), torch.tensor(1.0), STUB_TRAINING_TIMESTEPS, sequence_length)
    latents = torch.rand(3, 1, 24, 24)

    per_batch, mean = constraint.compute_loss(latents)
    torch.testing.assert_close(per_batch, torch.zeros(3))
    torch.testing.assert_close(mean, torch.tensor(0.0))

    guided = constraint.apply(latents, timestep=200)
    assert torch.isfinite(guided).all()


@pytest.mark.parametrize("resid1,resid2", [(0, 10), (10, 0), (-1, 3), (3, 12)])
def test_distance_constraint_indices_must_lie_in_the_sequence(resid1, resid2):
    constraint = DistanceConstraint(resid1, resid2, target=10.0)
    with pytest.raises(ValueError, match="0-based"):
        constraint.initialize(IdentityDecoder(), torch.tensor(1.0), STUB_TRAINING_TIMESTEPS, 10)


def test_distance_constraint_accepts_the_first_and_last_residue():
    constraint = DistanceConstraint(0, 9, target=10.0)
    constraint.initialize(IdentityDecoder(), torch.tensor(1.0), STUB_TRAINING_TIMESTEPS, 10)


@pytest.mark.parametrize("resid_start,resid_end", [(-1, 5), (2, 11), (5, 5), (5, 6), (7, 3)])
def test_helicity_region_must_fit_the_sequence(resid_start, resid_end):
    constraint = HelicityConstraint(resid_start, resid_end)
    with pytest.raises(ValueError, match="0-based"):
        constraint.initialize(IdentityDecoder(), torch.tensor(1.0), STUB_TRAINING_TIMESTEPS, 10)


def test_helicity_region_may_end_at_the_last_residue():
    # resid_end is exclusive, so a 10-residue sequence allows resid_end=10
    constraint = HelicityConstraint(0, 10)
    constraint.initialize(IdentityDecoder(), torch.tensor(1.0), STUB_TRAINING_TIMESTEPS, 10)
    assert constraint.mask.sum() == 45  # every pair i < j among 10 residues


# ---------------------------------------------------------------------------
# generate(): constraint lists, input validation and early failures
# ---------------------------------------------------------------------------


def test_a_list_of_constraints_is_combined_into_a_multiconstraint(captured_backend):
    bond, rg = BondConstraint(constraint_weight=0.5), RgConstraint(target=20.0)
    quiet_generate(SEQUENCE, constraint=[bond, rg])
    combined = captured_backend["constraint"]
    assert isinstance(combined, MultiConstraint)
    assert combined.constraints == [bond, rg]
    assert combined.constraint_weights == [0.5, 1.0]


def test_a_tuple_of_constraints_is_combined_and_a_single_constraint_is_passed_through(captured_backend):
    bond = BondConstraint()
    quiet_generate(SEQUENCE, constraint=(bond,))
    assert isinstance(captured_backend["constraint"], MultiConstraint)
    quiet_generate(SEQUENCE, constraint=bond)
    assert captured_backend["constraint"] is bond


@pytest.mark.parametrize("constraint", [object(), "rg", {"target": 20.0}, [BondConstraint(), object()]])
def test_non_constraints_raise_type_error_before_models_load(no_model_loading, constraint):
    with pytest.raises(TypeError, match="Constraint"):
        quiet_generate(SEQUENCE, constraint=constraint)


def test_an_empty_constraint_list_is_rejected(no_model_loading):
    with pytest.raises(ValueError, match="empty list"):
        quiet_generate(SEQUENCE, constraint=[])


def test_dpmpp_constraint_error_names_every_sampler_that_supports_constraints(no_model_loading):
    with pytest.raises(ValueError, match="use DDIM, PLMS or DDPM"):
        quiet_generate(SEQUENCE, sampler="dpmpp", constraint=BondConstraint())


@pytest.mark.parametrize("ionic_strength", [-1, -0.5, None, float("nan"), float("inf"), "150", True])
def test_invalid_ionic_strength_is_rejected_before_models_load(no_model_loading, ionic_strength):
    with pytest.raises(ValueError, match="ionic_strength"):
        quiet_generate(SEQUENCE, ionic_strength=ionic_strength)


@pytest.mark.parametrize("ionic_strength", [0, 20, 150.5, np.float32(300)])
def test_valid_ionic_strengths_reach_the_backend(captured_backend, ionic_strength):
    quiet_generate(SEQUENCE, ionic_strength=ionic_strength)
    assert captured_backend["ionic_strength"] == ionic_strength


@pytest.mark.parametrize("single", [False, True])
def test_nothing_left_after_the_length_filter_fails_before_models_load(no_model_loading, single):
    with pytest.raises(ValueError, match="longer than"):
        quiet_generate(TOO_LONG_SEQUENCE, return_single_ensemble=single)


def test_empty_input_fails_before_models_load(no_model_loading):
    with pytest.raises(ValueError, match="no sequences"):
        quiet_generate([])
    with pytest.raises(ValueError, match="no sequences"):
        quiet_generate({})


def test_names_with_path_separators_are_rejected_when_saving(no_model_loading, tmp_path):
    with pytest.raises(ValueError, match="path separator"):
        quiet_generate({"sp/P12345": SEQUENCE}, output_directory=str(tmp_path))


def test_names_with_path_separators_are_fine_without_an_output_directory(captured_backend):
    quiet_generate({"sp/P12345": SEQUENCE})
    assert list(captured_backend["args"][0]) == ["sp/P12345"]


# ---------------------------------------------------------------------------
# handle_input()
# ---------------------------------------------------------------------------


def test_fasta_conversion_warns_with_the_sequence_name_and_counts(tmp_path):
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">needs_fixing\nACDXBUZ*K\n>clean\nACDEF\n")
    with pytest.warns(UserWarning) as record:
        result = handle_input(str(fasta))

    assert result == {"needs_fixing": "ACDGNCQK", "clean": "ACDEF"}
    messages = [str(warning.message) for warning in record]
    assert len(messages) == 1
    assert "needs_fixing" in messages[0]
    assert "4 converted" in messages[0]
    assert "1 removed" in messages[0]


def test_fasta_remove_action_warns_about_each_removed_sequence(tmp_path):
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">bad\nACDX\n>good\nACDE\n")
    with pytest.warns(UserWarning, match="bad"):
        result = handle_input(str(fasta), invalid_sequence_action="remove")
    assert result == {"good": "ACDE"}


def test_fasta_record_that_converts_to_nothing_is_rejected(tmp_path):
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">only_stops\n**\n")
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="empty"):
        handle_input(str(fasta))


def test_tsv_blank_lines_are_skipped(tmp_path):
    tsv = tmp_path / "input.tsv"
    tsv.write_text("a\tACDE\n\nb\tFGHI\n   \n")
    assert handle_input(str(tsv)) == {"a": "ACDE", "b": "FGHI"}


def test_malformed_tsv_lines_name_the_line(tmp_path):
    tsv = tmp_path / "input.tsv"
    tsv.write_text("a\tACDE\nb ACDE\n")
    with pytest.raises(ValueError, match="Line 2"):
        handle_input(str(tsv))


def test_path_objects_are_read_as_files(tmp_path):
    tsv = tmp_path / "input.in"
    tsv.write_text("a\tACDE\n")
    assert handle_input(tsv) == {"a": "ACDE"}


def test_path_objects_need_a_supported_extension(tmp_path):
    with pytest.raises(ValueError, match="extensions"):
        handle_input(tmp_path / "sequences.txt")


@pytest.mark.parametrize("user_input", ["", [""], ("ACDE", ""), {"empty": ""}])
def test_empty_sequences_are_rejected(user_input):
    with pytest.raises(ValueError, match="Empty sequence"):
        handle_input(user_input)


# ---------------------------------------------------------------------------
# Saved outputs and their metadata
# ---------------------------------------------------------------------------


def test_names_with_dots_are_saved_to_separate_files(stub_models, tmp_path):
    quiet_generate(
        {"NP_000537.3": "AAAA", "NP_000537.4": "CCCC"},
        conformations=2,
        output_directory=str(tmp_path),
        return_data=False,
    )
    assert sorted(os.listdir(tmp_path)) == ["NP_000537.3.starling", "NP_000537.4.starling"]
    assert load_ensemble(str(tmp_path / "NP_000537.3.starling")).sequence == "AAAA"
    assert load_ensemble(str(tmp_path / "NP_000537.4.starling")).sequence == "CCCC"


def test_metadata_records_the_weights_that_were_loaded(stub_models, monkeypatch):
    # changing configs after import does not change what ModelManager loads,
    # so it must not change what the metadata claims was loaded either
    monkeypatch.setattr(configs, "DEFAULT_ENCODER_WEIGHTS_PATH", "/never/loaded/encoder.ckpt")
    monkeypatch.setattr(configs, "DEFAULT_DDPM_WEIGHTS_PATH", "/never/loaded/ddpm.ckpt")
    ensemble = quiet_generate("AAAA", conformations=2, return_single_ensemble=True)
    assert ensemble.metadata["DEFAULT_ENCODER_WEIGHTS_PATH"] == model_loading.DEFAULT_ENCODER_WEIGHTS_PATH
    assert ensemble.metadata["DEFAULT_DDPM_WEIGHTS_PATH"] == model_loading.DEFAULT_DDPM_WEIGHTS_PATH

    custom = quiet_generate(
        "AAAA",
        conformations=2,
        return_single_ensemble=True,
        encoder_path=pathlib.Path("/custom/encoder.ckpt"),
    )
    assert custom.metadata["DEFAULT_ENCODER_WEIGHTS_PATH"] == "/custom/encoder.ckpt"


def test_model_manager_and_metadata_agree_on_weight_references(monkeypatch):
    manager = model_loading.ModelManager()
    loaded = []

    def load(encoder_path, ddpm_path, device):
        loaded.append((encoder_path, ddpm_path))
        return torch.nn.Identity(), torch.nn.Identity()

    monkeypatch.setattr(manager, "load_models", load)
    monkeypatch.setitem(configs.TORCH_COMPILATION, "enabled", False)
    manager.get_models(encoder_path=None, ddpm_path=pathlib.Path("/custom/ddpm.ckpt"), device="cpu")
    assert loaded == [model_loading.weights_references(None, pathlib.Path("/custom/ddpm.ckpt"))]
    assert loaded[0][1] == "/custom/ddpm.ckpt"


def test_resolve_weights_path_accepts_path_objects(tmp_path, monkeypatch):
    monkeypatch.setenv("STARLING_OFFLINE", "1")
    weights = tmp_path / "weights.ckpt"
    assert configs.resolve_weights_path(weights, "https://example.invalid/w.ckpt") == str(weights)


# ---------------------------------------------------------------------------
# Error-filter message and verbose summary
# ---------------------------------------------------------------------------


def test_error_filter_failure_does_not_suggest_an_unexposed_option(monkeypatch):
    # every map has 10 A bonds, beyond the 4.81 A per-bond physical maximum
    separation = np.abs(np.subtract.outer(np.arange(len(SEQUENCE)), np.arange(len(SEQUENCE))))
    bad_maps = torch.from_numpy(np.stack([separation * 10.0] * 2).astype(np.float32))
    monkeypatch.setattr(generation, "_sample_distance_maps", lambda *args, **kwargs: bad_maps)

    with pytest.raises(RuntimeError) as error:
        generation._generate_error_filtered_conformers(
            None, SEQUENCE, 2, 2, False, None, False, "cpu", False, max_rounds=1
        )
    assert "remove_errors=False" in str(error.value)
    assert "increase the number of allowed rounds" not in str(error.value)


def test_verbose_summary_reports_the_steps_actually_run(stub_models, capsys):
    quiet_generate("AAAA", conformations=2, steps=12, sampler="ddim", verbose=True)
    steps_run = SAMPLERS["ddim"](stub_models[1], stub_models[0], 12).n_steps
    assert f"Number of denoising steps           : {steps_run} (DDIM; 12 requested)" in capsys.readouterr().out


def test_verbose_summary_does_not_split_timing_when_filtering_errors(stub_models, monkeypatch, capsys):
    clean_maps = np.zeros((2, 4, 4), dtype=np.float32)
    monkeypatch.setattr(generation, "_generate_error_filtered_conformers", lambda *a, **k: (clean_maps, None, 0))
    quiet_generate("AAAA", conformations=2, remove_errors=True, verbose=True)
    output = capsys.readouterr().out
    assert "Total time for sampling, structures and error filtering" in output
    assert "Total time for prediction" not in output


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def _run_cli(monkeypatch, tmp_path, *arguments):
    """Run the starling CLI with generate() stubbed; return generate's kwargs."""
    from starling.scripts import starling_main_cli as cli

    seen = {}
    monkeypatch.setattr(cli, "generate", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["starling", SEQUENCE, "-o", str(tmp_path), *arguments])
    cli.main()
    return seen


@pytest.mark.parametrize(
    "arguments,expected_relax",
    [
        (["-r"], "standard"),
        (["-r", "--relax"], "standard"),
        (["-r", "--relax", "standard"], "standard"),
        (["-r", "--relax", "fast"], "fast"),
        (["--relax", "fast", "-r"], "fast"),
        (["-r", "--no-relax"], False),
        ([], False),
        (["--no-relax"], False),
    ],
)
def test_cli_relaxes_structures_by_default(monkeypatch, tmp_path, arguments, expected_relax):
    relax = _run_cli(monkeypatch, tmp_path, *arguments)["relax"]
    assert type(relax) is type(expected_relax)
    assert relax == expected_relax


@pytest.mark.parametrize("arguments", [["--relax"], ["--relax", "fast"]])
def test_cli_explicit_relax_still_requires_structures(monkeypatch, tmp_path, arguments):
    with pytest.raises(SystemExit) as error:
        _run_cli(monkeypatch, tmp_path, *arguments)
    assert error.value.code == 1


def test_cli_rejects_unknown_relaxation_modes(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as error:
        _run_cli(monkeypatch, tmp_path, "-r", "--relax", "quick")
    assert error.value.code == 2


@pytest.mark.parametrize("arguments", [["--compile-forces", "-r", "--no-relax"], ["--compile-forces"]])
def test_cli_compile_forces_requires_relaxed_structures(monkeypatch, tmp_path, arguments):
    with pytest.raises(SystemExit) as error:
        _run_cli(monkeypatch, tmp_path, *arguments)
    assert error.value.code == 2


def test_cli_compile_forces_uses_the_default_relaxation(monkeypatch, tmp_path):
    seen = _run_cli(monkeypatch, tmp_path, "--compile-forces", "-r")
    assert seen["compile_forces"] is True
    assert seen["relax"] == "standard"


def test_cli_ionic_strength_is_a_float(monkeypatch, tmp_path):
    assert _run_cli(monkeypatch, tmp_path, "--ionic_strength", "137.5")["ionic_strength"] == 137.5


def test_cli_disable_progress_bar_disables_both_bars(monkeypatch, tmp_path):
    seen = _run_cli(monkeypatch, tmp_path, "--disable_progress_bar")
    assert seen["show_progress_bar"] is False
    assert seen["show_per_step_progress_bar"] is False


# ---------------------------------------------------------------------------
# User configuration file
# ---------------------------------------------------------------------------

# Environment variables that take precedence over ~/.starling_weights/configs.py
CONFIG_ENVIRONMENT_VARIABLES = (
    "STARLING_ENCODER_PATH",
    "STARLING_DDPM_PATH",
    "STARLING_FAISS_INDEX_PATH",
    "STARLING_SEQSTORE_PATH",
    "STARLING_FAISS_MANIFEST_PATH",
    "STARLING_FAISS_INDEX_MD5",
)


def _fresh_configs(monkeypatch, tmp_path, user_config):
    """Import a private copy of starling.configs with a temporary home and user config."""
    home = tmp_path / "home"
    (home / ".starling_weights").mkdir(parents=True)
    (home / ".starling_weights" / "configs.py").write_text(user_config)
    monkeypatch.setenv("HOME", str(home))
    for variable in CONFIG_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    spec = importlib.util.spec_from_file_location("fresh_starling_configs", configs.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_user_config_overrides_settings_defined_after_it_was_loaded(monkeypatch, tmp_path):
    fresh = _fresh_configs(
        monkeypatch,
        tmp_path,
        "DEFAULT_ENCODER_WEIGHTS_PATH = '/my/encoder.ckpt'\n"
        "DEFAULT_SEARCH_DIR = '/my/search'\n"
        "DEFAULT_FAISS_INDEX_NAME = 'index.faiss'\n"
        "FAISS_INDEX_MD5 = 'abc123'\n",
    )
    assert fresh.DEFAULT_ENCODER_WEIGHTS_PATH == "/my/encoder.ckpt"
    assert fresh.FAISS_INDEX_MD5 == "abc123"

    # settings derived from the overridden ones follow them
    assert fresh.DEFAULT_SEQSTORE_NAME == "index.faiss.seqs.sqlite"
    assert fresh.DEFAULT_FAISS_INDEX_PATH == os.path.join("/my/search", "index.faiss")
    assert fresh.DEFAULT_SEQSTORE_DB_PATH == os.path.join("/my/search", "index.faiss.seqs.sqlite")
    assert fresh.ensure_search_artifacts(download=False)[0] == os.path.join("/my/search", "index.faiss")


def test_overriding_weight_file_names_rebuilds_the_download_urls(monkeypatch, tmp_path):
    fresh = _fresh_configs(monkeypatch, tmp_path, "DEFAULT_DDPM_WEIGHTS = 'custom_ddpm.ckpt'\n")
    assert fresh.GITHUB_DDPM_URL.endswith("/custom_ddpm.ckpt")
    assert fresh.DEFAULT_DDPM_WEIGHTS_PATH == fresh.GITHUB_DDPM_URL


def test_environment_variables_win_over_the_user_config(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / ".starling_weights").mkdir(parents=True)
    (home / ".starling_weights" / "configs.py").write_text("DEFAULT_ENCODER_WEIGHTS_PATH = '/from/config.ckpt'\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("STARLING_ENCODER_PATH", "/from/environment.ckpt")

    spec = importlib.util.spec_from_file_location("fresh_starling_configs", configs.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh.DEFAULT_ENCODER_WEIGHTS_PATH == "/from/environment.ckpt"


def test_unknown_user_settings_warn_and_helpers_are_ignored(monkeypatch, tmp_path):
    with pytest.warns(UserWarning, match="DEFAULT_STEP,"):
        fresh = _fresh_configs(
            monkeypatch,
            tmp_path,
            "import os\nhelper = 'not a setting'\nDEFAULT_STEP = 50\n",
        )
    assert fresh.DEFAULT_STEPS == configs.DEFAULT_STEPS
    assert not hasattr(fresh, "helper")


# ---------------------------------------------------------------------------
# Compilation options
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_compilation_settings(monkeypatch):
    """Let a test change the compilation settings without affecting other tests."""
    monkeypatch.setattr(configs, "TORCH_COMPILATION", copy.deepcopy(configs.TORCH_COMPILATION))


def test_docstring_compilation_example_is_accepted_by_torch_compile(isolated_compilation_settings):
    settings = starling.set_compilation_options(
        enabled=False,
        backend="inductor",
        fullgraph=False,
        dynamic=True,
        options={"triton.cudagraphs": True},
    )
    assert "mode" not in settings["options"]

    manager = model_loading.ModelManager()
    manager.encoder_model = SimpleNamespace(decoder=torch.nn.Identity())
    manager.diffusion_model = SimpleNamespace(model=torch.nn.Identity())
    manager.compile()  # torch.compile raises if it is given both mode and options


def test_setting_mode_after_options_replaces_them(isolated_compilation_settings):
    starling.set_compilation_options(options={"triton.cudagraphs": True})
    settings = starling.set_compilation_options(mode="max-autotune")
    assert settings["options"]["mode"] == "max-autotune"
    assert "options" not in settings["options"]


def test_mode_and_options_together_are_rejected(isolated_compilation_settings):
    with pytest.raises(ValueError, match="mode or options"):
        starling.set_compilation_options(mode="default", options={"triton.cudagraphs": True})


# ---------------------------------------------------------------------------
# Sequence and ensemble encoders
# ---------------------------------------------------------------------------


class _ShapeRecordingDiffusion:
    """Diffusion stand-in that records the shape of every batch it embeds."""

    def __init__(self):
        self.batch_shapes = []

    def sequence2labels(self, sequences, sequence_mask, ionic_strength):
        self.batch_shapes.append(tuple(sequences.shape))
        return sequences.float()[..., None]


@pytest.mark.parametrize("bucket,expected_shapes", [(False, [(4, 5)]), (True, [(2, 5), (2, 3)])])
def test_bucketing_splits_batches_at_bucket_boundaries(bucket, expected_shapes):
    diffusion = _ShapeRecordingDiffusion()
    manager = SimpleNamespace(get_models=lambda **kwargs: (None, diffusion))
    # token lengths 5 and 4 fall in bucket 5 // 4 = 4 // 4 = 1; lengths 3 and 2 in bucket 0
    tokens = {"a": [1] * 5, "b": [1] * 4, "c": [1] * 3, "d": [1] * 2}

    embeddings = generation.sequence_encoder_backend(
        tokens,
        device="cpu",
        batch_size=4,
        ionic_strength=150,
        aggregate=False,
        model_manager=manager,
        pretokenized=True,
        bucket=bucket,
        bucket_size=4,
    )
    assert diffusion.batch_shapes == expected_shapes
    assert {name: len(embedding) for name, embedding in embeddings.items()} == {"a": 5, "b": 4, "c": 3, "d": 2}


class _Float32Encoder:
    """VAE stand-in with float32 weights, which reject float64 input like the real VAE."""

    def __init__(self):
        self.convolution = torch.nn.Conv2d(1, 1, kernel_size=16, stride=16)

    def encode(self, distance_maps):
        latents = self.convolution(distance_maps)
        return SimpleNamespace(mode=lambda: latents)


def test_ensemble_encoder_accepts_float64_maps():
    manager = SimpleNamespace(get_models=lambda **kwargs: (_Float32Encoder(), None))
    maps = np.random.default_rng(0).random((3, 10, 10))  # float64, numpy's default
    latents = generation.ensemble_encoder_backend(maps, device="cpu", batch_size=2, model_manager=manager)
    assert latents.shape == (3, 24, 24)
    assert latents.dtype == np.float32


def test_ensemble_encoder_no_longer_takes_an_unused_output_directory(no_model_loading):
    with pytest.raises(TypeError, match="output_directory"):
        frontend.ensemble_encoder(np.zeros((1, 4, 4)), output_directory=".")


# ---------------------------------------------------------------------------
# PyTorch Hub entry point
# ---------------------------------------------------------------------------


def _load_hubconf():
    """Import hubconf.py from the repository root."""
    path = pathlib.Path(starling.__file__).resolve().parents[1] / "hubconf.py"
    spec = importlib.util.spec_from_file_location("starling_hubconf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hub_rejects_untrained_models(monkeypatch):
    monkeypatch.setattr(model_loading.ModelManager, "get_models", lambda self, **kwargs: pytest.fail())
    with pytest.raises(NotImplementedError, match="pretrained"):
        _load_hubconf().starling_model(pretrained=False)


def test_hub_returns_pretrained_models(monkeypatch):
    monkeypatch.setattr(model_loading.ModelManager, "get_models", lambda self, **kwargs: ("encoder", "diffusion"))
    assert _load_hubconf().starling_model() == ("encoder", "diffusion")
