"""
D2d — Is withholding the detection mask a viable mitigation?   (Tier 1, after D2/D2b/D2c)

D2 established that AION defers to the segmentation mask over the pixels, on every readout.
D2 also noticed, in passing, that supplying NO mask cost only 2.1% on flux. If that holds up
across readouts and against ground truth, the finding stops being "here is a worrying
property" and becomes "here is a worrying property and a one-line fix".

Two quantities per configuration, measured against catalogue truth:

  QUALITY        how close the readout is to the true value (Legacy FLUX_*, PROVABGS Z_HP)
  VULNERABILITY  how much the readout moves when the mask is swapped for another galaxy's
                 real mask (zero by construction when no mask is supplied)

Configurations (the model is never given the scalar it is asked to predict):

  img                 imaging only, no segmap                 -> vulnerability 0
  img + seg_real      imaging + the pipeline's mask           -> the status quo
  img + seg_disc      imaging + a FIXED CENTRED DISC          -> the interesting middle option
  ... and the same three with DESI spectra added

`seg_disc` is the same mask for every object: it carries "there is a source at the centre"
and nothing else, so there is nothing for a detection pipeline to get wrong. If quality with
a generic disc matches quality with the real mask, then what the channel actually supplies is
a presence flag, not object-specific information -- and the mitigation is free.

All numbers carry bootstrap 16-84% intervals over objects (AGENDA.md 11: bootstrap anything
you intend to explain).

Run:  python experiments/d2d_mitigation.py [N]
Out:  results/d2d_mitigation.json, results/d2d_mitigation.png
"""

from __future__ import annotations

import json
import pathlib
import random
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 480
BATCH = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
CROP = 96
N_BOOT = 1000

# readout -> (truth key in the batch, kind)
TARGETS = {
    "tok_flux_g": ("FLUX_G", "flux"),
    "tok_flux_z": ("FLUX_Z", "flux"),
    "tok_z": ("z", "redshift"),
}
NICE = {"tok_flux_g": "flux g", "tok_flux_z": "flux z", "tok_z": "redshift"}
CONFIGS = ["img", "img+seg_real", "img+seg_disc",
           "spec", "spec+seg_real", "spec+seg_disc"]


def centre_crop(x, n=CROP):
    h, w = x.shape[-2:]
    r, c = (h - n) // 2, (w - n) // 2
    return x[..., r:r + n, c:c + n]


def disc_mask(like: torch.Tensor, radius_px: float = 12.0) -> torch.Tensor:
    """A fixed centred disc, identical for every object."""
    h, w = like.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    d = ((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2).sqrt()
    return (d <= radius_px).float().expand(like.shape[0], h, w).clone()


def boot_ci(vals, fn=np.median, n=N_BOOT, seed=SEED):
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = [fn(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(n)]
    return float(fn(vals)), float(np.percentile(stats, 16)), float(np.percentile(stats, 84))


def main():
    from aion.modalities import (LegacySurveyFluxG, LegacySurveyFluxZ,
                                 LegacySurveySegmentationMap, Z)

    targets = [LegacySurveyFluxG, LegacySurveyFluxZ, Z]
    rng = np.random.default_rng(SEED)
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device=DEVICE)
    T = P.tables()
    print(f"device={DEVICE}  N={N}")

    def seg(binary):
        t = cm.encode(LegacySurveySegmentationMap(field=binary.clone().to(DEVICE)))
        return {k: v.reshape(v.shape[0], -1) for k, v in t.items()}

    def read(tok):
        # one forward per target: batching targets triggers the decoder-mask bug and
        # destroys the posteriors (see astron_mi.hooks.read_targets)
        return A.read_targets(model, tok, targets, T, device=DEVICE, seed=SEED)

    pred: dict = {c: {} for c in CONFIGS}
    swapped: dict = {c: {} for c in CONFIGS if "seg_real" in c}
    truth: dict = {}

    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), N, replace=False))
        for s in range(0, N, BATCH):
            b = r.aion_batch(rows[s:s + BATCH], with_segmap=True)
            ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            ts = A.encode_spectrum(cm, **b["spectrum"], survey="desi", device=DEVICE)
            binary = (b["segmap"] > 0).float()
            g = torch.Generator().manual_seed(s)
            swap = binary[torch.randperm(len(binary), generator=g)]

            sg_real, sg_disc, sg_swap = seg(binary), seg(disc_mask(binary)), seg(swap)
            bases = {"img": dict(ti), "spec": {**ti, **ts}}
            for bn, base in bases.items():
                for suffix, extra in (("", {}), ("+seg_real", sg_real), ("+seg_disc", sg_disc)):
                    name = bn + suffix
                    for k, v in read({**base, **extra}).items():
                        pred[name].setdefault(k, []).append(v)
                # vulnerability arm
                for k, v in read({**base, **sg_swap}).items():
                    swapped[bn + "+seg_real"].setdefault(k, []).append(v)

            for key, (tk, _) in TARGETS.items():
                src = b["labels"] if tk == "z" else b["scalars"]
                truth.setdefault(key, []).append(np.asarray(src[tk], dtype="float64"))
            print(f"  {min(s + BATCH, N)}/{N}", flush=True)

    truth = {k: np.concatenate(v) for k, v in truth.items()}
    pred = {c: {k: np.concatenate(v) for k, v in d.items()} for c, d in pred.items()}
    swapped = {c: {k: np.concatenate(v) for k, v in d.items()} for c, d in swapped.items()}

    def err(p, key):
        """
        Flux error in MAGNITUDES, not as a fraction. Fractional error explodes for the many
        objects with small FLUX_Z and produced meaningless medians in the first pass
        (values in the thousands). |dmag| = 2.5|log10(pred/true)| is the standard, bounded,
        astronomer-legible choice.
        """
        t = truth[key]
        if TARGETS[key][1] == "flux":
            ok = np.isfinite(t) & (t > 1e-2) & (p > 1e-2)
            e = np.full(len(t), np.nan)
            e[ok] = 2.5 * np.abs(np.log10(p[ok] / t[ok]))
            return e
        return np.abs(p - truth[key])

    out = {"n": int(N), "quality": {}, "vulnerability": {}}
    print(f"\nQUALITY vs catalogue truth  (flux: median |dmag|; redshift: median |dz|)")
    print(f"{'config':16s} " + " ".join(f"{NICE[k]:>26s}" for k in TARGETS))
    for c in CONFIGS:
        cells = []
        for k in TARGETS:
            m, lo, hi = boot_ci(err(pred[c][k], k))
            out["quality"].setdefault(c, {})[k] = {"median": m, "lo16": lo, "hi84": hi}
            cells.append(f"{m:.4f} [{lo:.4f},{hi:.4f}]")
        print(f"{c:16s} " + " ".join(f"{x:>26s}" for x in cells))

    print(f"\nVULNERABILITY: median |change| when the mask is swapped (flux in mag, z absolute)")
    print(f"{'config':16s} " + " ".join(f"{NICE[k]:>26s}" for k in TARGETS))
    for c in CONFIGS:
        cells = []
        for k in TARGETS:
            if c in swapped:
                if TARGETS[k][1] == "flux":
                    a, bq = pred[c][k], swapped[c][k]
                    ok = (a > 1e-2) & (bq > 1e-2)
                    rel = np.full(len(a), np.nan)
                    rel[ok] = 2.5 * np.abs(np.log10(bq[ok] / a[ok]))
                else:
                    rel = np.abs(swapped[c][k] - pred[c][k])
                m, lo, hi = boot_ci(rel)
                out["vulnerability"].setdefault(c, {})[k] = {"median": m, "lo16": lo, "hi84": hi}
                cells.append(f"{m:.4f} [{lo:.4f},{hi:.4f}]")
            else:
                out["vulnerability"].setdefault(c, {})[k] = {"median": 0.0, "note": "no mask supplied"}
                cells.append("0 (no mask)")
        print(f"{c:16s} " + " ".join(f"{x:>26s}" for x in cells))

    # ------------------------------------------------------------------ verdict
    print(f"\nVERDICT")
    for bn in ("img", "spec"):
        q_no = out["quality"][bn]["tok_flux_g"]["median"]
        q_re = out["quality"][bn + "+seg_real"]["tok_flux_g"]["median"]
        q_di = out["quality"][bn + "+seg_disc"]["tok_flux_g"]["median"]
        v_re = out["vulnerability"][bn + "+seg_real"]["tok_flux_g"]["median"]
        cost_drop = 100 * (q_no - q_re) / max(q_re, 1e-9)
        cost_disc = 100 * (q_di - q_re) / max(q_re, 1e-9)
        out.setdefault("summary", {})[bn] = {
            "quality_cost_of_dropping_mask_pct": float(cost_drop),
            "quality_cost_of_generic_disc_pct": float(cost_disc),
            "vulnerability_removed_pct": float(100 * v_re),
        }
        print(f"  base={bn}:  dropping the mask changes flux-g error by {cost_drop:+.1f}%, "
              f"a generic disc by {cost_disc:+.1f}%,")
        print(f"            and removes a {100 * v_re:.1f}% swap vulnerability.")

    with open(RESULTS / "d2d_mitigation.json", "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, k in zip(axes, TARGETS):
        x = np.arange(len(CONFIGS))
        med = [out["quality"][c][k]["median"] for c in CONFIGS]
        lo = [med[i] - out["quality"][c][k]["lo16"] for i, c in enumerate(CONFIGS)]
        hi = [out["quality"][c][k]["hi84"] - med[i] for i, c in enumerate(CONFIGS)]
        cols = ["C0" if "seg" not in c else ("C1" if "real" in c else "C2") for c in CONFIGS]
        ax.bar(x, med, yerr=[lo, hi], capsize=3, color=cols)
        for i, c in enumerate(CONFIGS):
            v = out["vulnerability"][c][k]["median"]
            if v > 0:
                ax.text(i, med[i] * 1.04, f"vuln\n{100 * v:.0f}%", ha="center",
                        fontsize=6, color="C3")
        ax.set_xticks(x); ax.set_xticklabels(CONFIGS, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("median error vs truth" + (" (mag)" if TARGETS[k][1] == "flux" else " |dz|"))
        ax.set_title(NICE[k], fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("D2d: cost of withholding the detection mask (bars) vs the vulnerability it removes (red)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "d2d_mitigation.png", dpi=150)
    print(f"wrote {RESULTS / 'd2d_mitigation.json'} and .png")


if __name__ == "__main__":
    main()
