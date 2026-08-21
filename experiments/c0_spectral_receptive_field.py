"""
C0 — Spectral codec receptive field, on REAL DESI spectra.   (AGENDA.md, Thrust C)

Every wavelength-resolved claim in this project depends on knowing how localised a
spectrum token is. The nominal mapping is 25.6 A per token (stride 32 on a 0.8 A grid),
but the codec is a 4-stage ConvNeXt-1D whose GRN reduces over the sequence axis, so the
formal receptive field is global. The question is the EFFECTIVE one.

Two directions, both measured:

  FORWARD  inject a Gaussian emission line of realistic equivalent width at a known
           wavelength, re-encode, and see which token ids change and by how much.
  REVERSE  flip one token id, decode, and see over what wavelength range the
           reconstructed flux moves.

Uses the AION authors' own 128 real DESI spectra (tests/test_data), not synthetic ones.

Kill criterion from the agenda: if the FWHM is >= 6 tokens, every wavelength-resolved
claim must be restated in terms of spectral REGIONS rather than lines.

Run:  python experiments/c0_spectral_receptive_field.py
Out:  results/c0_receptive_field.json, results/c0_receptive_field.png
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import astron_mi as A
from astron_mi import physics as P

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

N_SPEC = 48                      # real spectra to average over
EWS = [5.0, 15.0, 30.0]          # injected line equivalent widths, Angstrom
# probe wavelengths: avoid the grid edges, span the range, hit real line locations
PROBE_LAMBDA = [3900.0, 4400.0, 4861.0, 5007.0, 5800.0, 6563.0, 7500.0, 8500.0, 9500.0]
LINE_SIGMA = 3.0                 # A, ~unresolved for DESI at R~2500


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def encode_spectra(cm, flux, ivar, mask, lam):
    """Codec tokens are bit-identical on MPS vs CPU and ~56x faster; verified."""
    out = A.encode_spectrum(cm, flux, ivar, mask, lam, survey="desi", device=DEVICE)
    t = list(out.values())[0]
    return t.reshape(t.shape[0], -1).long().cpu()


def main():
    cm = A.load_codecs(DEVICE)
    print(f"device: {DEVICE}")
    sp = A.load_fixture_spectra()
    flux, ivar, mask, lam = (sp[k][:N_SPEC] for k in ("flux", "ivar", "mask", "lambda"))
    print(f"real DESI spectra: {tuple(flux.shape)}  lambda "
          f"[{float(lam[0].min()):.1f}, {float(lam[0].max()):.1f}] A")

    base = encode_spectra(cm, flux, ivar, mask, lam)
    print(f"baseline tokens: {tuple(base.shape)}")

    results = {"n_spectra": int(N_SPEC), "nominal_A_per_token": P.SPEC_A_PER_TOKEN,
               "forward": {}, "reverse": {}}

    # ------------------------------------------------------------------ FORWARD
    # median continuum level per spectrum, so the injected EW is physically meaningful
    cont = flux.median(dim=1, keepdim=True).values.clamp(min=1e-3)

    for ew in EWS:
        offsets_flip: dict[int, list] = {}
        n_changed, mapped_tok, expect_tok = [], [], []
        for lam0 in PROBE_LAMBDA:
            amp = ew * cont / (LINE_SIGMA * np.sqrt(2 * np.pi))
            line = amp * torch.exp(-0.5 * ((lam - lam0) / LINE_SIGMA) ** 2)
            tok = encode_spectra(cm, flux + line, ivar, mask, lam)
            diff = (tok != base)                                  # (N, 273)
            p_expect = P.lambda_token(lam0)
            for b in range(diff.shape[0]):
                idx = torch.nonzero(diff[b]).flatten().tolist()
                if not idx:
                    continue
                n_changed.append(len(idx))
                # which token moved: take the one nearest the expected position
                nearest = min(idx, key=lambda i: abs(i - p_expect))
                mapped_tok.append(nearest)
                expect_tok.append(p_expect)
                for i in idx:
                    offsets_flip.setdefault(i - p_expect, []).append(1)

        # flip probability as a function of offset from the injection token
        total = len(PROBE_LAMBDA) * diff.shape[0]
        pflip = {int(k): len(v) / total for k, v in sorted(offsets_flip.items())}
        fwhm_tok = _fwhm_from_profile(pflip)
        results["forward"][str(ew)] = {
            "mean_tokens_changed": float(np.mean(n_changed)) if n_changed else 0.0,
            "median_tokens_changed": float(np.median(n_changed)) if n_changed else 0.0,
            "flip_profile": pflip,
            "flip_FWHM_tokens": fwhm_tok,
            "flip_FWHM_A": None if fwhm_tok is None else fwhm_tok * P.SPEC_A_PER_TOKEN,
            "mapping_exact_frac": float(np.mean(np.array(mapped_tok) == np.array(expect_tok)))
            if mapped_tok else 0.0,
            "mapping_within1_frac": float(np.mean(np.abs(np.array(mapped_tok) - np.array(expect_tok)) <= 1))
            if mapped_tok else 0.0,
        }
        r = results["forward"][str(ew)]
        fwhm_a = "n/a" if r["flip_FWHM_A"] is None else f"{r['flip_FWHM_A']:.0f} A"
        print(f"\nEW={ew:5.1f} A: {r['mean_tokens_changed']:.1f} tokens changed on average; "
              f"flip FWHM = {r['flip_FWHM_tokens']} tok ({fwhm_a}); "
              f"token mapping exact {r['mapping_exact_frac']:.2f}, "
              f"within1 {r['mapping_within1_frac']:.2f}")

    # ------------------------------------------------------------------ REVERSE
    # flip a single token and measure where the decoded flux moves
    from aion.modalities import DESISpectrum

    codec = cm._load_codec(DESISpectrum)
    probe_tokens = [40, 90, 140, 190, 240]
    prof_all, fwhms, w01s = [], [], []
    with torch.no_grad():
        rec0 = codec.decode(base[:8].to(DEVICE))
        rec0 = rec0.flux if hasattr(rec0, "flux") else rec0
        for p in probe_tokens:
            tk = base[:8].clone()
            tk[:, p] = (tk[:, p] + 512) % 1024          # a large, arbitrary code change
            rec1 = codec.decode(tk.to(DEVICE))
            rec1 = rec1.flux if hasattr(rec1, "flux") else rec1
            d = (rec1 - rec0).abs().mean(0).cpu().numpy()
            if d.max() <= 0:
                continue
            d = d / d.max()
            prof_all.append((p, d))
            grid = np.arange(len(d)) * 0.8 + 3500.0
            fwhms.append(_width_at(d, grid, 0.5))
            w01s.append(_width_at(d, grid, 0.01))

    results["reverse"] = {
        "probe_tokens": probe_tokens,
        "median_FWHM_A": float(np.median(fwhms)) if fwhms else None,
        "median_width_at_1pct_A": float(np.median(w01s)) if w01s else None,
        "median_FWHM_tokens": float(np.median(fwhms) / P.SPEC_A_PER_TOKEN) if fwhms else None,
    }
    print(f"\nREVERSE single-token flip: decoded flux FWHM = {results['reverse']['median_FWHM_A']:.1f} A "
          f"({results['reverse']['median_FWHM_tokens']:.1f} tokens); "
          f"width at 1% of peak = {results['reverse']['median_width_at_1pct_A']:.0f} A")

    # ------------------------------------------------------------------ verdict
    fw = [v["flip_FWHM_tokens"] for v in results["forward"].values() if v["flip_FWHM_tokens"]]
    core = float(np.median(fw)) if fw else float("nan")
    results["verdict"] = {
        "effective_FWHM_tokens": core,
        "effective_FWHM_A": core * P.SPEC_A_PER_TOKEN,
        "passes_agenda_kill_criterion": bool(core < 6),
    }
    print(f"\nVERDICT: effective FWHM = {core:.0f} tokens = {core * P.SPEC_A_PER_TOKEN:.0f} A. "
          f"Agenda kill criterion (<6 tokens): {'PASS' if core < 6 else 'FAIL -> restate as regions'}")

    with open(RESULTS / "c0_receptive_field.json", "w") as f:
        json.dump(results, f, indent=2)

    # -------------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for ew in EWS:
        prof = results["forward"][str(ew)]["flip_profile"]
        ks = sorted(int(k) for k in prof)
        ax.plot(ks, [prof[str(k)] if str(k) in prof else prof[k] for k in ks],
                marker="o", ms=3, lw=1.4, label=f"EW = {ew:.0f} A")
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.set_xlim(-14, 14)
    ax.set_xlabel("token offset from the injected line")
    ax.set_ylabel("P(token id changes)")
    ax.set_title("FORWARD: line injection -> which tokens flip", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for p, d in prof_all:
        grid = np.arange(len(d)) * 0.8 + 3500.0
        lam_c = 3500.0 + (p - 1) * P.SPEC_A_PER_TOKEN
        ax.semilogy(grid - lam_c, np.maximum(d, 1e-4), lw=1.1, label=f"token {p}")
    ax.set_xlim(-600, 600); ax.set_ylim(1e-4, 1.5)
    ax.set_xlabel("wavelength offset from the flipped token (A)")
    ax.set_ylabel("|delta flux| / peak")
    ax.set_title("REVERSE: single token flip -> where flux moves\n(note the low-level global tail)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("C0: effective spectral receptive field of AION's spectrum codec (real DESI spectra)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "c0_receptive_field.png", dpi=150)
    print(f"wrote {RESULTS / 'c0_receptive_field.json'} and .png")


def _fwhm_from_profile(prof: dict) -> float | None:
    if not prof:
        return None
    ks = np.array(sorted(prof.keys()))
    vs = np.array([prof[k] for k in ks], dtype=float)
    if vs.max() <= 0:
        return None
    half = vs.max() / 2
    above = ks[vs >= half]
    return float(above.max() - above.min() + 1) if len(above) else None


def _width_at(d: np.ndarray, grid: np.ndarray, frac: float) -> float:
    above = grid[d >= frac * d.max()]
    return float(above.max() - above.min()) if len(above) else 0.0


if __name__ == "__main__":
    main()
