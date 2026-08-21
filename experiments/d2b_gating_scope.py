"""
D2b — How general is the detection gate?   (follow-on to D2)

D2 showed that AION's g-band flux estimate follows the segmentation mask rather than the
pixels. Three questions decide whether that is a curiosity about one readout or a general
property of the model:

  1. WHICH READOUTS?  flux (g, z), size (shape_r), ellipticity (shape_e1), redshift (z).
     If only flux is gated, it is a photometry quirk. If size and redshift are gated too,
     the mask is acting as a global "there is an object here" switch.

  2. DOES EVIDENCE OVERRIDE IT?  Each readout is measured twice: with imaging alone, and
     with imaging + DESI spectra. A spectrum is overwhelming independent evidence that a
     source exists. If the gate survives it, the model is not weighing evidence, it is
     deferring.

  3. IS IT WORSE WHERE IT MATTERS?  Stratified by
       - blendedness: the number of distinct source IDs in the cutout's segmentation map
         (this file carries real multi-source masks, 1-3 sources per cutout)
       - brightness: faint objects are where detection is genuinely ambiguous
     Rubin's problem is faint blended sources, so this is the deployment-relevant cut.

Interventions (pixels IDENTICAL): shifted_25 (mask displaced) and swapped (another
galaxy's real mask, fully on-manifold). Placebo: RA moved 250 codebook steps.

Effects are reported in ROBUST SIGMA units -- median |change| divided by the population
spread of that readout under the real mask -- so flux, size, ellipticity and redshift are
comparable. Fractional change is unstable for readouts that pass through zero (shape_e1).

Run:  python experiments/d2b_gating_scope.py [N]
Out:  results/d2b_gating.json, results/d2b_gating.png
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

READOUTS = ["tok_flux_g", "tok_flux_z", "tok_shape_r", "tok_shape_e1", "tok_z"]
NICE = {"tok_flux_g": "flux g", "tok_flux_z": "flux z", "tok_shape_r": "size R",
        "tok_shape_e1": "ellipticity e1", "tok_z": "redshift"}
ARMS = ["real", "shifted_25", "swapped", "placebo_ra"]


def centre_crop(x, n=CROP):
    h, w = x.shape[-2:]
    r, c = (h - n) // 2, (w - n) // 2
    return x[..., r:r + n, c:c + n]


def n_sources(segmap: torch.Tensor) -> np.ndarray:
    """Distinct nonzero source IDs inside the centre crop = blendedness proxy."""
    m = centre_crop(segmap)
    return np.array([len(torch.unique(mi)) - 1 for mi in m])


def main():
    from aion.modalities import (LegacySurveyFluxG, LegacySurveyFluxZ,
                                 LegacySurveySegmentationMap, LegacySurveyShapeE1,
                                 LegacySurveyShapeR, Ra, Z)

    targets = [LegacySurveyFluxG, LegacySurveyFluxZ, LegacySurveyShapeR,
               LegacySurveyShapeE1, Z]
    rng = np.random.default_rng(SEED)
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device=DEVICE)
    T = P.tables()
    print(f"device={DEVICE}  N={N}")

    def seg(binary):
        t = cm.encode(LegacySurveySegmentationMap(field=binary.clone().to(DEVICE)))
        return {k: v.reshape(v.shape[0], -1) for k, v in t.items()}

    def read_all(tokens):
        # ONE FORWARD PER TARGET. Passing all five targets to AION.forward at once triggers
        # the decoder-mask bug (model.py:99) and destroys the posteriors -- measured on
        # redshift: |dz| 0.0038 with one target, 2.35 with three. The first version of this
        # experiment batched the targets and its numbers were invalid.
        return A.read_targets(model, tokens, targets, T, device=DEVICE, seed=SEED)

    # acc[config][arm][readout] -> list of arrays
    acc: dict = {c: {a: {} for a in ARMS} for c in ("img", "img_spec")}
    meta: dict = {"nsrc": [], "flux_g": []}

    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), N, replace=False))
        for s in range(0, N, BATCH):
            b = r.aion_batch(rows[s:s + BATCH], with_segmap=True)
            ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            ts = A.encode_spectrum(cm, **b["spectrum"], survey="desi", device=DEVICE)
            binary = (b["segmap"] > 0).float()
            meta["nsrc"].append(n_sources(b["segmap"]))
            meta["flux_g"].append(np.asarray(b["scalars"]["FLUX_G"]))

            g = torch.Generator().manual_seed(s)
            variants = {
                "real": binary,
                "shifted_25": torch.roll(binary, shifts=(25, 25), dims=(-2, -1)),
                "swapped": binary[torch.randperm(len(binary), generator=g)],
                "placebo_ra": binary,
            }
            ra_tok = cm.encode(Ra(value=torch.from_numpy(
                np.asarray(b["ra"], dtype="float32")).to(DEVICE)))
            ra_tok = {k: (v.reshape(v.shape[0], -1) if v.dim() > 1 else v.unsqueeze(1))
                      for k, v in ra_tok.items()}
            ra_shift = {k: torch.clamp(v + 250, 0, 1023) for k, v in ra_tok.items()}

            for cfg, base in (("img", dict(ti)), ("img_spec", {**ti, **ts})):
                for arm, m in variants.items():
                    tok = {**base, **seg(m)}
                    if arm == "placebo_ra":
                        tok = {**tok, **ra_shift}
                    vals = read_all(tok)
                    for k, v in vals.items():
                        acc[cfg][arm].setdefault(k, []).append(v)
            print(f"  {min(s + BATCH, N)}/{N}", flush=True)

    nsrc = np.concatenate(meta["nsrc"])
    fluxg = np.concatenate(meta["flux_g"])
    pred = {c: {a: {k: np.concatenate(v) for k, v in d.items()}
                for a, d in arms.items()} for c, arms in acc.items()}

    def sigma_shift(cfg, arm, k, mask=None):
        """
        median |change| vs the real mask, in units of the readout's own spread.

        When comparing STRATA the scale must be computed WITHIN the stratum. Using the
        global spread would make brighter subsamples look more gated purely because their
        absolute flux changes are larger -- a scaling artifact, not a stronger gate.
        """
        a, b_ = pred[cfg][arm][k], pred[cfg]["real"][k]
        ref = pred[cfg]["real"][k]
        if mask is not None:
            a, b_, ref = a[mask], b_[mask], ref[mask]
        scale = np.subtract(*np.percentile(ref, [75, 25])) / 1.349
        return float(np.median(np.abs(a - b_)) / scale) if scale > 0 else float("nan")

    out: dict = {"n": int(len(nsrc)),
                 "blendedness_counts": {str(i): int((nsrc == i).sum())
                                        for i in np.unique(nsrc)},
                 "effects_sigma": {}}

    print(f"\nEffect size in sigma of each readout's own spread (median |change| vs real mask)")
    for cfg in ("img", "img_spec"):
        print(f"\n  config = {cfg}")
        print(f"  {'readout':16s} " + " ".join(f"{a:>12s}" for a in ARMS[1:]))
        for k in READOUTS:
            vals = [sigma_shift(cfg, a, k) for a in ARMS[1:]]
            out["effects_sigma"].setdefault(cfg, {})[k] = dict(zip(ARMS[1:], vals))
            print(f"  {NICE[k]:16s} " + " ".join(f"{v:12.3f}" for v in vals))

    # ------------------------------------------------------- stratified analysis
    blend = nsrc >= 2
    faint = fluxg < np.median(fluxg)
    strata = {"single source": ~blend, "blended (>=2)": blend,
              "bright half": ~faint, "faint half": faint,
              "faint AND blended": faint & blend}
    print(f"\nSwap-arm effect (sigma) by stratum, config=img")
    print(f"  {'stratum':20s} {'n':>5s} " + " ".join(f"{NICE[k]:>15s}" for k in READOUTS))
    out["strata"] = {}
    for nm, m in strata.items():
        if m.sum() < 20:
            continue
        vals = [sigma_shift("img", "swapped", k, m) for k in READOUTS]
        out["strata"][nm] = {"n": int(m.sum()), **dict(zip(READOUTS, vals))}
        print(f"  {nm:20s} {m.sum():5d} " + " ".join(f"{v:15.3f}" for v in vals))

    # ------------------------------------------------------------------ verdict
    e = out["effects_sigma"]
    gated = [k for k in READOUTS if e["img"][k]["swapped"] > 5 * max(e["img"][k]["placebo_ra"], 1e-3)]
    z_img, z_spec = e["img"]["tok_z"]["swapped"], e["img_spec"]["tok_z"]["swapped"]
    out["verdict"] = {
        "gated_readouts": gated,
        "redshift_gating_img": z_img,
        "redshift_gating_img_spec": z_spec,
        "spectra_protect_redshift": bool(z_spec < 0.5 * z_img),
    }
    print(f"\nVERDICT")
    print(f"  readouts gated by the mask (swap >> placebo): {', '.join(NICE[k] for k in gated)}")
    print(f"  redshift gating: {z_img:.3f} sigma with imaging alone -> {z_spec:.3f} sigma once spectra are added"
          f"  ({'evidence overrides the gate' if out['verdict']['spectra_protect_redshift'] else 'gate survives the evidence'})")

    with open(RESULTS / "d2b_gating.json", "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(len(READOUTS)); w = 0.26
    ax = axes[0]
    for i, a in enumerate(ARMS[1:]):
        ax.bar(x + (i - 1) * w, [e["img"][k][a] for k in READOUTS], w, label=a)
    ax.set_xticks(x); ax.set_xticklabels([NICE[k] for k in READOUTS], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("median |change| (sigma of readout)")
    ax.set_title("Which readouts are gated?  (imaging only)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x - 0.2, [e["img"][k]["swapped"] for k in READOUTS], 0.4, label="imaging only")
    ax.bar(x + 0.2, [e["img_spec"][k]["swapped"] for k in READOUTS], 0.4, label="+ spectra")
    ax.set_xticks(x); ax.set_xticklabels([NICE[k] for k in READOUTS], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("swap-arm effect (sigma)")
    ax.set_title("Does independent evidence override the gate?", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    names = [n for n in out["strata"]]
    xs = np.arange(len(names))
    for i, k in enumerate(READOUTS):
        ax.plot(xs, [out["strata"][n][k] for n in names], "o-", lw=1.4, ms=4, label=NICE[k])
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("swap-arm effect (sigma)")
    ax.set_title("Is it worse where detection is ambiguous?", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("D2b: how general is AION's detection gate?", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "d2b_gating.png", dpi=150)
    print(f"wrote {RESULTS / 'd2b_gating.json'} and .png")


if __name__ == "__main__":
    main()
