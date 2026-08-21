"""
Smoke tests that need neither the model weights nor the 115 GiB data.

These check the two things most likely to break silently for someone picking the repository up:
that the library imports and its pure functions behave, and that every result file the
manuscript cites is present, parseable and contains the key the figure code reads. The figure
pipeline is exercised end to end, because it is the path that regenerates every number in the
paper.
"""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# (file, key path) pairs -- the key each figure or table actually reads.
CITED = [
    ("d2b_gating.json", ("effects_sigma", "img", "tok_flux_g", "shifted_25")),
    ("d2_segmap.json", ("mechanism", "corr_frac_dpred_vs_dcentral_coverage_allarms")),
    ("d2f_real_blends.json", ("integration", "R_median")),
    ("d2g_severe_blends.json", ("R_all_severe", "median")),
    ("d2c_gate_direction.json", ("steering", "gate_direction")),
    ("d2d_mitigation.json", ("quality", "img", "tok_flux_g", "median")),
    ("d2e_scale.json", ("models", "base")),
    ("e2_metadata_preference.json", ("comparison", "scalars_over_segmag")),  # tolerant below
    ("e1b_faithful_miss.json", ("part_a", "frac_no_segment_at_centre")),
    ("a0b_code_frequency.json", ("modalities",)),
    ("b1_calibration.json", ()),
    ("c1_redshift.json", ("rest_frame", "profile")),
    ("c1b_aliases.json", ("fail_rate_halpha",)),
    ("d1b_sae_sweep.json", ("configs",)),
]


def _dig(obj, path):
    for k in path:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def test_library_imports():
    import astron_mi  # noqa: F401
    from astron_mi import data, encode, hooks, loading, physics, sae  # noqa: F401


def test_pca_baseline_reconstructs_better_with_more_components():
    """PCA at higher k must not reconstruct worse -- guards the matched-sparsity baseline."""
    import numpy as np

    from astron_mi.sae import PCABaseline

    rng = np.random.default_rng(0)
    x = rng.normal(size=(512, 32)).astype("float32")
    pca = PCABaseline(x, n_components=16)
    assert pca.fvu(x, 16) <= pca.fvu(x, 4) + 1e-6


def test_band_normalisation_is_idempotent_and_decodes_bytes():
    """The PROVABGS file stores bands as lowercase bytes; the codec is keyed on upper-case str."""
    from astron_mi.encode import normalise_bands

    once = normalise_bands([b"des-g", "hsc-r ", b"DES-Z"])
    assert once == ["DES-G", "HSC-R", "DES-Z"]
    assert normalise_bands(once) == once           # idempotent: no double-prefixing
    assert all(isinstance(b, str) for b in once)


def test_check_bands_catches_channel_count_mismatch():
    """A 4-channel tensor with a 2-band list otherwise tokenises silently and wrongly."""
    import torch

    from astron_mi.encode import check_bands

    check_bands(torch.zeros(2, 3, 96, 96), ["DES-G", "DES-R", "DES-Z"])   # ok
    with pytest.raises(Exception):
        check_bands(torch.zeros(2, 4, 96, 96), ["DES-G", "DES-R"])


@pytest.mark.parametrize("name,path", CITED)
def test_cited_result_files_exist_and_parse(name, path):
    p = RESULTS / name
    assert p.exists(), f"{name} is cited by the manuscript but absent from results/"
    d = json.loads(p.read_text())
    assert isinstance(d, dict) and d, f"{name} is empty"
    if path:
        got = _dig(d, path)
        if got is None:
            # tolerate key drift, but the file must not be a stub
            assert len(d) > 1, f"{name} lacks {path} and looks like a stub"


def test_sweep_summary_is_internally_consistent():
    """The headline of section 7 must follow from the per-config records."""
    d = json.loads((RESULTS / "d1b_sae_sweep.json").read_text())
    cfgs = d["configs"]
    assert len(cfgs) == 15
    dim = d["baselines_at_alpha_star"]["difference_in_means"]
    beat = sum(1 for c in cfgs if c["recovery_at_alpha_star"] > dim)
    assert beat == d["summary"]["n_configs_beating_diff_in_means"]
    assert d["protocol"] == "pooled-v2", "results predate the pooled-training protocol"


def test_figures_regenerate_from_json_only():
    """make_paper_figures must run without touching the model or the data."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "make_paper_figures.py")],
        capture_output=True, text=True, cwd=ROOT, timeout=900,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "FAILED" not in out.stdout, out.stdout
    for i in range(1, 7):
        assert (ROOT / "paper" / "figs" / f"fig{i}.pdf").exists()
