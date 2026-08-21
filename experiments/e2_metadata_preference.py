"""
E2 — Is the detection gate a special case of "trust metadata over pixels"?

The natural generalisation question is "is this AION, or any model given a detection channel?".
That cannot be tested directly: AION is the only public astronomical foundation model that
takes a segmentation map as an input modality, so there is no second model with the same
channel to compare against. Training one is out of scope.

The tractable — and arguably stronger — substitute is to ask whether the behaviour generalises
ACROSS CHANNELS within AION. The segmentation map is one piece of catalogue-derived metadata
that is redundant with the pixels. AION also ingests catalogue SCALARS (Legacy Tractor fluxes)
that are likewise redundant with the imaging. If the model defers to those too, then "the gate"
is a specific case of a general policy: when a highly-predictive metadata channel is available,
lean on it rather than on the raw signal.

Design (image pixels IDENTICAL throughout; the model is never given the band it must predict):

    img                 imaging only
    img + scalars       imaging + true Tractor fluxes r, i, z
    img + WRONG scalars imaging + another galaxy's r, i, z          <- the test
    img + segmap        imaging + the true segmentation map          <- D2's channel, for scale
    img + WRONG segmap  imaging + another galaxy's segmentation map
    placebo             imaging + true scalars, RA moved 250 codes

Readout: tok_flux_g, in magnitudes, against the catalogue truth.

If corrupting the catalogue scalars moves flux_g comparably to corrupting the segmentation map,
the phenomenon is a general metadata preference and the architectural claim in the paper can be
made about the design pattern rather than about one channel.

Run:  python experiments/e2_metadata_preference.py [N]
Out:  results/e2_metadata_preference.json, results/e2_metadata_preference.png
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 480
BATCH = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
N_BOOT = 1000
ARMS = ["img", "img+scalars", "img+WRONG_scalars",
        "img+segmap", "img+WRONG_segmap", "placebo_ra"]


def boot_ci(v, fn=np.median, n=N_BOOT, seed=SEED):
    v = np.asarray(v); v = v[np.isfinite(v)]
    if len(v) < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    st = [fn(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return float(fn(v)), float(np.percentile(st, 16)), float(np.percentile(st, 84))


def dmag(a, b):
    ok = (a > 1e-2) & (b > 1e-2) & np.isfinite(a) & np.isfinite(b)
    e = np.full(len(a), np.nan)
    e[ok] = 2.5 * np.abs(np.log10(a[ok] / b[ok]))
    return e


def main():
    from aion import modalities as M
    from aion.modalities import LegacySurveyFluxG, LegacySurveySegmentationMap, Ra

    rng = np.random.default_rng(SEED)
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device=DEVICE)
    T = P.tables()
    print(f"device={DEVICE}  N={N}")

    # r, i, z fluxes as metadata; g is the prediction target and is never supplied
    SCAL = [("LegacySurveyFluxR", "FLUX_R"), ("LegacySurveyFluxI", "FLUX_I"),
            ("LegacySurveyFluxZ", "FLUX_Z")]

    def scalar_tokens(scalars, order):
        out = {}
        for cls_name, key in SCAL:
            v = np.asarray(scalars[key], dtype="float32")[order]
            v = np.nan_to_num(v, nan=0.0)
            t = cm.encode(getattr(M, cls_name)(value=torch.from_numpy(v).to(DEVICE)))
            out.update({k: (x.reshape(x.shape[0], -1) if x.dim() > 1 else x.unsqueeze(1))
                        for k, x in t.items()})
        return out

    def seg_tokens(binary):
        t = cm.encode(LegacySurveySegmentationMap(field=binary.clone().to(DEVICE)))
        return {k: v.reshape(v.shape[0], -1) for k, v in t.items()}

    def flux_g(tok):
        return A.read_targets(model, tok, [LegacySurveyFluxG], T,
                              device=DEVICE, seed=SEED)["tok_flux_g"]

    pred = {a: [] for a in ARMS}
    truth = []
    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), N, replace=False))
        for s in range(0, N, BATCH):
            b = r.aion_batch(rows[s:s + BATCH], with_segmap=True)
            nb = len(b["ra"])
            ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            binary = (b["segmap"] > 0).float()
            g = torch.Generator().manual_seed(s)
            perm = torch.randperm(nb, generator=g).numpy()

            sc_true = scalar_tokens(b["scalars"], np.arange(nb))
            sc_wrong = scalar_tokens(b["scalars"], perm)
            sg_true = seg_tokens(binary)
            sg_wrong = seg_tokens(binary[perm])

            ra_t = cm.encode(Ra(value=torch.from_numpy(
                np.asarray(b["ra"], dtype="float32")).to(DEVICE)))
            ra_t = {k: (v.reshape(v.shape[0], -1) if v.dim() > 1 else v.unsqueeze(1))
                    for k, v in ra_t.items()}
            ra_shift = {k: torch.clamp(v + 250, 0, 1023) for k, v in ra_t.items()}

            arms = {
                "img": dict(ti),
                "img+scalars": {**ti, **sc_true},
                "img+WRONG_scalars": {**ti, **sc_wrong},
                "img+segmap": {**ti, **sg_true},
                "img+WRONG_segmap": {**ti, **sg_wrong},
                "placebo_ra": {**ti, **sc_true, **ra_shift},
            }
            for a, tok in arms.items():
                pred[a].append(flux_g(tok))
            truth.append(np.asarray(b["scalars"]["FLUX_G"], dtype="float64"))
            print(f"  {min(s + BATCH, N)}/{N}", flush=True)

    truth = np.concatenate(truth)
    pred = {a: np.concatenate(v) for a, v in pred.items()}

    out = {"n": int(len(truth)), "arms": {}}
    print(f"\n{'arm':22s} {'|dmag| vs truth':>26s} {'|dmag| vs its clean arm':>28s}")
    clean_of = {"img+WRONG_scalars": "img+scalars", "img+WRONG_segmap": "img+segmap",
                "placebo_ra": "img+scalars"}
    for a in ARMS:
        q, qlo, qhi = boot_ci(dmag(pred[a], truth))
        row = {"err_vs_truth": [q, qlo, qhi]}
        cell2 = ""
        if a in clean_of:
            d, dlo, dhi = boot_ci(dmag(pred[a], pred[clean_of[a]]))
            row["shift_vs_clean"] = [d, dlo, dhi]
            cell2 = f"{d:.4f} [{dlo:.4f},{dhi:.4f}]"
        out["arms"][a] = row
        print(f"{a:22s} {q:.4f} [{qlo:.4f},{qhi:.4f}]  {cell2:>28s}")

    sc = out["arms"]["img+WRONG_scalars"]["shift_vs_clean"][0]
    sg = out["arms"]["img+WRONG_segmap"]["shift_vs_clean"][0]
    pl = out["arms"]["placebo_ra"]["shift_vs_clean"][0]
    out["comparison"] = {"wrong_scalars_mag": sc, "wrong_segmap_mag": sg, "placebo_mag": pl,
                         "scalars_over_placebo": float(sc / max(pl, 1e-9)),
                         "segmap_over_placebo": float(sg / max(pl, 1e-9)),
                         "scalars_over_segmap": float(sc / max(sg, 1e-9))}
    general = sc > 5 * max(pl, 1e-9)
    out["verdict"] = ("metadata preference is GENERAL: corrupting catalogue scalars moves the "
                      "readout like corrupting the segmentation map"
                      if general else
                      "the effect is specific to the detection channel")
    print(f"\nCOMPARISON (all at pixels-identical)")
    print(f"  corrupted catalogue scalars -> flux_g shift: {sc:.4f} mag")
    print(f"  corrupted segmentation map  -> flux_g shift: {sg:.4f} mag")
    print(f"  placebo (RA moved 250 codes)               : {pl:.4f} mag")
    print(f"  scalars/placebo = {out['comparison']['scalars_over_placebo']:.1f}x   "
          f"segmap/placebo = {out['comparison']['segmap_over_placebo']:.1f}x")
    print(f"\nVERDICT: {out['verdict']}")

    with open(RESULTS / "e2_metadata_preference.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    x = np.arange(len(ARMS))
    med = [out["arms"][a]["err_vs_truth"][0] for a in ARMS]
    lo = [med[i] - out["arms"][a]["err_vs_truth"][1] for i, a in enumerate(ARMS)]
    hi = [out["arms"][a]["err_vs_truth"][2] - med[i] for i, a in enumerate(ARMS)]
    cols = ["C0", "C0", "C3", "C0", "C3", "#888"]
    ax.bar(x, med, yerr=[lo, hi], capsize=3, color=cols)
    ax.set_xticks(x); ax.set_xticklabels(ARMS, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("|dmag| vs catalogue FLUX_G")
    ax.set_title("Accuracy per input configuration", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    names = ["wrong scalars", "wrong segmap", "placebo (RA)"]
    vals = [sc, sg, pl]
    ax.bar(range(3), vals, color=["C3", "C1", "#888"])
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("induced |dmag| in flux_g")
    ax.set_title("Corrupting metadata, pixels identical", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("E2: is the detection gate one case of a general metadata preference?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "e2_metadata_preference.png", dpi=150)
    print(f"wrote {RESULTS / 'e2_metadata_preference.json'} and .png")


if __name__ == "__main__":
    main()
