"""
D0 — Token dilution and the cross-modal shared direction.   (AGENDA.md, Thrust D)

Two questions, both cheap, and together they decide whether D1 (a sparse dictionary) is
worth its GPU-days.

1. DILUTION. The AION paper's Table 6 shows mean-pooled redshift R^2 COLLAPSING when
   spectra are added to the input (0.995 attentive -> 0.800 mean). If probing from
   spectrum rows alone recovers the performance, the collapse is pure token dilution —
   576 image rows swamping 273 spectrum rows in the mean — and needs no dictionary,
   no feature interference story, nothing. Just don't mean-pool across modalities.

2. SHARED DIRECTION. Is the physical information in image rows and in spectrum rows
   carried along the SAME direction of the residual stream? Fit a ridge probe for a
   quantity on image rows only and on spectrum rows only, then measure
     (a) cosine between the two weight vectors
     (b) cross-applied R^2: does the image-trained probe work on spectrum rows?
   High => a genuinely shared, modality-independent physical direction exists, and a
   cross-modal dictionary has something to find. Near zero => AION is a bag of
   per-modality codes and D1 should be cancelled.

Baseline that must be beaten throughout: probing from the raw AION scalar inputs
(fluxes, colours) with the same ridge. If the representation does not beat its own
inputs, nothing downstream means anything.

Run:  python experiments/d0_dilution.py [cache.npz]
Out:  results/d0_dilution.json, results/d0_dilution.png
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

TARGETS = ["z", "log_mstar", "log_Z_mw", "t_age_mw", "log_sSFR"]
ALPHAS = np.logspace(-2, 5, 24)
SEED = 0


def cv_probe(X, y, n_splits: int = 5, seed: int = SEED):
    """Ridge with nested alpha selection. Returns out-of-fold R^2 and mean weights."""
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    if len(y) < 50:
        return float("nan"), None, 0
    pred = np.zeros_like(y)
    ws = []
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        ym = y[tr].mean()
        m = RidgeCV(alphas=ALPHAS).fit((X[tr] - mu) / sd, y[tr] - ym)
        pred[te] = m.predict((X[te] - mu) / sd) + ym
        ws.append(m.coef_ / sd)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(r2), np.mean(ws, axis=0), int(len(y))


def cross_r2(Xa, Xb, y, wa, seed: int = SEED):
    """Apply a probe trained on representation A to representation B (refit intercept
    and a single scale only, so this measures DIRECTION agreement, not a free refit)."""
    ok = np.isfinite(y) & np.isfinite(Xa).all(1) & np.isfinite(Xb).all(1)
    Xb, y = Xb[ok], y[ok]
    if len(y) < 50 or wa is None:
        return float("nan")
    s = Xb @ wa
    A_ = np.c_[s, np.ones_like(s)]
    coef, *_ = np.linalg.lstsq(A_, y, rcond=None)
    pred = A_ @ coef
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        RESULTS.glob("cache_N*.npz")
    )[-1]
    d = np.load(path)
    print(f"cache: {path.name}")

    layer = "final"
    reps = {
        "ALL rows (mean)": d[f"pool.{layer}.ALL"],
        "image rows": d[f"pool.{layer}.tok_image"],
        "spectrum rows": d[f"pool.{layer}.tok_spectrum_desi"],
        "img+spec concat": np.concatenate(
            [d[f"pool.{layer}.tok_image"], d[f"pool.{layer}.tok_spectrum_desi"]], axis=1
        ),
    }
    # baseline: the model's own scalar inputs, as magnitudes and colours
    fl = np.stack([d[f"scalar.FLUX_{b}"] for b in ["G", "R", "I", "Z", "W1", "W2"]], 1)
    mag = -2.5 * np.log10(np.clip(fl, 1e-3, None))
    base = np.concatenate([mag, mag[:, :-1] - mag[:, 1:]], axis=1)
    reps["BASELINE: input fluxes+colours"] = base

    out: dict = {"cache": path.name, "layer": layer, "n": int(len(base)), "probes": {}}
    print(f"\n{'target':11s} " + " ".join(f"{k[:17]:>18s}" for k in reps))
    weights: dict = {}
    for t in TARGETS:
        y = d[f"label.{t}"].astype("float64")
        row, wrow = [], {}
        for name, X in reps.items():
            r2, w, n = cv_probe(X.astype("float64"), y)
            row.append(r2)
            wrow[name] = w
            out["probes"].setdefault(t, {})[name] = {"r2": r2, "n": n}
        weights[t] = wrow
        print(f"{t:11s} " + " ".join(f"{v:18.4f}" for v in row))

    # ---------------------------------------------------- shared-direction test
    print(f"\n{'target':11s} {'cos(w_img,w_spec)':>19s} {'img probe on spec':>19s} {'spec probe on img':>19s}")
    Xi, Xs = reps["image rows"].astype("float64"), reps["spectrum rows"].astype("float64")
    for t in TARGETS:
        y = d[f"label.{t}"].astype("float64")
        wi, ws = weights[t]["image rows"], weights[t]["spectrum rows"]
        if wi is None or ws is None:
            continue
        cos = float(wi @ ws / (np.linalg.norm(wi) * np.linalg.norm(ws)))
        a = cross_r2(Xi, Xs, y, wi)
        b = cross_r2(Xs, Xi, y, ws)
        out["probes"][t]["shared_direction"] = {
            "cosine": cos, "img_probe_on_spec_r2": a, "spec_probe_on_img_r2": b
        }
        print(f"{t:11s} {cos:19.4f} {a:19.4f} {b:19.4f}")

    # ------------------------------------------------------------------ verdict
    zr = out["probes"]["z"]
    dilution = zr["spectrum rows"]["r2"] - zr["ALL rows (mean)"]["r2"]
    cos_z = zr.get("shared_direction", {}).get("cosine", float("nan"))
    out["verdict"] = {
        "dilution_gain_z": dilution,
        "dilution_explains_pooling_collapse": bool(dilution > 0.05),
        "cosine_z": cos_z,
        "d1_worth_running": bool(abs(cos_z) > 0.2),
    }
    print(f"\nVERDICT")
    print(f"  dilution: spectrum-only minus all-rows R^2 (z) = {dilution:+.4f}"
          f"  -> {'token dilution explains the paper pooling collapse' if dilution > 0.05 else 'NOT explained by dilution'}")
    print(f"  shared direction: cos(w_img, w_spec) for z = {cos_z:+.4f}"
          f"  -> D1 {'worth running' if abs(cos_z) > 0.2 else 'should be CANCELLED (bag of per-modality codes)'}")

    with open(RESULTS / "d0_dilution.json", "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    names = list(reps)
    x = np.arange(len(TARGETS))
    w = 0.8 / len(names)
    for i, nm in enumerate(names):
        vals = [out["probes"][t][nm]["r2"] for t in TARGETS]
        ax.bar(x + i * w - 0.4, vals, w, label=nm)
    ax.set_xticks(x); ax.set_xticklabels(TARGETS, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("out-of-fold $R^2$"); ax.set_ylim(0, 1.02)
    ax.set_title("Probing from different row subsets of the SAME forward pass", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    cs = [out["probes"][t].get("shared_direction", {}).get("cosine", np.nan) for t in TARGETS]
    xr = [out["probes"][t].get("shared_direction", {}).get("img_probe_on_spec_r2", np.nan) for t in TARGETS]
    ax.bar(x - 0.2, cs, 0.4, label="cos(w_img, w_spec)", color="C0")
    ax.bar(x + 0.2, xr, 0.4, label="image probe applied to spectrum rows ($R^2$)", color="C2")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(TARGETS, rotation=20, ha="right", fontsize=8)
    ax.set_title("Is the physical direction shared across modalities?", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("D0: token dilution and cross-modal shared directions in AION-base", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "d0_dilution.png", dpi=150)
    print(f"\nwrote {RESULTS / 'd0_dilution.json'} and .png")


if __name__ == "__main__":
    main()
