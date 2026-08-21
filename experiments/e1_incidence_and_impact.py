"""
E1 — Does the detection gate fire in practice, and does it matter for cosmology?

The two objections a referee will raise about D2-D2g, answered together.

PART A — INCIDENCE. Everything so far perturbed masks by hand. How often does the Legacy
Survey pipeline actually produce a mask that is absent or displaced at the target position?
Measured directly from the survey's own segmentation maps over thousands of real cutouts:
    - fraction with NO segment covering the centre pixel  (a detection miss at the target)
    - distribution of the central segment's centroid offset from the frame centre
These are the real rates of exactly the failure mode D2 showed is damaging.

PART B — DOWNSTREAM IMPACT. Photometric-redshift requirements are stated on the MEAN redshift
of a tomographic bin, not on per-object scatter: the LSST SRD requires |<dz>|/(1+z) < 0.003.
So we take real galaxies, compute AION redshifts with the true mask, then again with masks
perturbed at the rate and magnitude measured in Part A, bin into tomographic bins, and compare
the induced shift in <z> against that requirement.

This converts "the model defers to the detection pipeline" into a number a cosmologist can act
on. If the induced |<dz>|/(1+z) is well under 0.003, the finding is a curiosity for this use
case and we say so; if it is comparable or larger, it is a systematic that has to be budgeted.

Run:  python experiments/e1_incidence_and_impact.py [N_SCAN] [N_IMPACT]
Out:  results/e1_incidence_impact.json, results/e1_incidence_impact.png
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
N_SCAN = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
N_IMPACT = int(sys.argv[2]) if len(sys.argv) > 2 else 960
BATCH = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
CROP = 96
N_BOOT = 500
LSST_REQ = 0.003              # |<dz>|/(1+z) per tomographic bin
TOMO_EDGES = [0.0, 0.15, 0.25, 0.35, 1.0]


def centre_crop(x, n=CROP):
    h, w = x.shape[-2:]
    r, c = (h - n) // 2, (w - n) // 2
    return x[..., r:r + n, c:c + n]


def boot_ci(v, fn=np.mean, n=N_BOOT, seed=SEED):
    v = np.asarray(v); v = v[np.isfinite(v)]
    if len(v) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    st = [fn(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return float(fn(v)), float(np.percentile(st, 16)), float(np.percentile(st, 84))


def main():
    from aion.modalities import LegacySurveySegmentationMap, Z

    rng = np.random.default_rng(SEED)
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device=DEVICE)
    T = P.tables()

    # ------------------------------------------------------------- PART A: incidence
    print(f"PART A — incidence, scanning {N_SCAN} real cutouts")
    no_centre, offsets, n_seen = 0, [], 0
    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), min(N_SCAN, r.n_rows()), replace=False))
        for s in range(0, len(rows), BATCH):
            b = r.aion_batch(rows[s:s + BATCH], with_segmap=True,
                             with_image=False, with_spectrum=False)
            seg = centre_crop(b["segmap"])
            for i in range(len(seg)):
                n_seen += 1
                cid = float(seg[i, CROP // 2, CROP // 2])
                if cid <= 0:
                    no_centre += 1
                    continue
                m = (seg[i] == cid)
                yy, xx = torch.nonzero(m, as_tuple=True)
                cy, cx = yy.float().mean().item(), xx.float().mean().item()
                offsets.append(float(np.hypot(cy - (CROP - 1) / 2, cx - (CROP - 1) / 2)))
    offsets = np.array(offsets)
    frac_no_centre = no_centre / max(n_seen, 1)
    print(f"  cutouts scanned                       : {n_seen}")
    print(f"  NO segment at the centre pixel        : {100 * frac_no_centre:.2f}%")
    print(f"  centroid offset of central segment (px): median {np.median(offsets):.2f}, "
          f"p90 {np.percentile(offsets, 90):.2f}, p99 {np.percentile(offsets, 99):.2f}")
    print(f"  offset > 5 px: {100 * np.mean(offsets > 5):.2f}%   "
          f"> 10 px: {100 * np.mean(offsets > 10):.2f}%")

    partA = {
        "n_scanned": int(n_seen),
        "frac_no_segment_at_centre": float(frac_no_centre),
        "centroid_offset_px": {
            "median": float(np.median(offsets)), "p90": float(np.percentile(offsets, 90)),
            "p99": float(np.percentile(offsets, 99)),
            "frac_gt_5px": float(np.mean(offsets > 5)),
            "frac_gt_10px": float(np.mean(offsets > 10)),
        },
    }

    # realistic perturbation: apply the measured miss rate, and displace by the measured
    # offset distribution (sampled empirically, not assumed)
    p_miss = frac_no_centre
    print(f"\n  -> realistic scenario: {100 * p_miss:.2f}% of objects get NO usable mask, "
          f"the rest are displaced by a draw from the measured offset distribution")

    # ------------------------------------------------------- PART B: downstream impact
    print(f"\nPART B — downstream impact on tomographic <z>, {N_IMPACT} galaxies")

    def seg_tok(binary):
        t = cm.encode(LegacySurveySegmentationMap(field=binary.clone().to(DEVICE)))
        return {k: v.reshape(v.shape[0], -1) for k, v in t.items()}

    def zmedian(tok):
        o = model.forward({k: v.to(DEVICE) for k, v in tok.items()}, target_modality=[Z])
        lg = o["tok_z"][:, 0, :].detach().float().cpu().numpy()
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        pm = p[:, :1024] / p[:, :1024].sum(1, keepdims=True)
        vals = T.index_to_value("tok_z")
        return vals[np.argmax(np.cumsum(pm, 1) >= 0.5, 1)]   # median: B1 says never the mean

    # TWO configurations. The one that matters for LSST is IMAGING-ONLY: a photometric
    # survey has no spectrum, and D2b showed spectra suppress the redshift gate ~100x. Running
    # only the with-spectra case would understate the impact by exactly that factor.
    # TWO perturbation scenarios, because they rest on different evidence:
    #   "miss_only"  -- only the MEASURED 3.7% detection-miss rate is applied. Fully
    #                   defensible: that rate is an unambiguous pipeline failure (no segment
    #                   covers the target position). This is the conservative lower bound.
    #   "miss_shift" -- additionally displaces the remaining masks by a draw from the observed
    #                   segment-centroid offset distribution. That distribution mixes genuine
    #                   astrometric/detection error with real morphological asymmetry, so it is
    #                   a PLAUSIBLE MAGNITUDE, not a validated error rate. Reported as such.
    CONFIGS = ("img", "img_spec")
    SCENARIOS = ("miss_only", "miss_shift")
    zt_l = []
    z_true_mask = {c: [] for c in CONFIGS}
    z_pert_mask = {(c, sc): [] for c in CONFIGS for sc in SCENARIOS}
    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), N_IMPACT, replace=False))
        for s in range(0, N_IMPACT, BATCH):
            b = r.aion_batch(rows[s:s + BATCH], with_segmap=True)
            ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            ts = A.encode_spectrum(cm, **b["spectrum"], survey="desi", device=DEVICE)
            base = {**ti, **ts}
            binary = (b["segmap"] > 0).float()

            miss = rng.random(len(binary)) < p_miss
            p_missonly = binary.clone(); p_missonly[miss] = 0.0
            p_missshift = binary.clone()
            for i in range(len(p_missshift)):
                if miss[i]:
                    p_missshift[i] = 0.0
                else:
                    sh = int(round(float(rng.choice(offsets))))
                    if sh:
                        p_missshift[i] = torch.roll(p_missshift[i], shifts=(sh, 0), dims=(-2, -1))
            perts = {"miss_only": p_missonly, "miss_shift": p_missshift}

            zt_l.append(np.asarray(b["labels"]["z"], dtype="float64"))
            bases = {"img": dict(ti), "img_spec": base}
            st = seg_tok(binary)
            stok = {sc: seg_tok(perts[sc]) for sc in SCENARIOS}
            for c in CONFIGS:
                z_true_mask[c].append(zmedian({**bases[c], **st}))
                for sc in SCENARIOS:
                    z_pert_mask[(c, sc)].append(zmedian({**bases[c], **stok[sc]}))
            print(f"  {min(s + BATCH, N_IMPACT)}/{N_IMPACT}", flush=True)

    zt_all = np.concatenate(zt_l)
    ok = np.isfinite(zt_all)
    zt = zt_all[ok]
    results_by_cfg = {}
    NICE = {"img": "IMAGING ONLY (the LSST photo-z case)", "img_spec": "imaging + spectra"}
    SNICE = {"miss_only": "measured 3.7% detection-miss rate ONLY (conservative)",
             "miss_shift": "misses + plausible-magnitude displacements"}
    for c, sc in [(c, sc) for c in CONFIGS for sc in SCENARIOS]:
        z0 = np.concatenate(z_true_mask[c])[ok]
        z1 = np.concatenate(z_pert_mask[(c, sc)])[ok]
        print(f"\n  {NICE[c]}  |  {SNICE[sc]} — tomographic bin means "
              f"(requirement |d<z>|/(1+z) < {LSST_REQ})")
        print(f"  {'bin':>14s} {'n':>5s} {'<z> true mask':>14s} {'<z> perturbed':>14s} "
              f"{'|d<z>|/(1+z)':>14s} {'vs req':>9s}")
        bins = []
        for lo, hi in zip(TOMO_EDGES[:-1], TOMO_EDGES[1:]):
            m = (zt >= lo) & (zt < hi)
            if m.sum() < 20:
                continue
            a, b_ = float(np.mean(z0[m])), float(np.mean(z1[m]))
            zc = float(np.mean(zt[m]))
            shift = abs(b_ - a) / (1 + zc)
            d, dlo, dhi = boot_ci(z1[m] - z0[m])
            bins.append({"lo": lo, "hi": hi, "n": int(m.sum()), "mean_true": a,
                         "mean_perturbed": b_, "shift_over_1pz": shift,
                         "delta_ci": [d, dlo, dhi],
                         "exceeds_requirement": bool(shift > LSST_REQ)})
            flag = "EXCEEDS" if shift > LSST_REQ else "ok"
            print(f"  [{lo:.2f},{hi:.2f}) {m.sum():5d} {a:14.5f} {b_:14.5f} "
                  f"{shift:14.5f} {flag:>9s}")
        worst = max((x["shift_over_1pz"] for x in bins), default=float("nan"))
        results_by_cfg[f"{c}|{sc}"] = {"bins": bins, "worst_shift_over_1pz": worst,
                             "n_bins_exceeding": int(sum(x["exceeds_requirement"] for x in bins)),
                             "worst_over_requirement": float(worst / LSST_REQ)}
        print(f"    worst bin {worst:.5f} = {worst / LSST_REQ:.1f}x the requirement")

    img = results_by_cfg["img|miss_only"]
    out = {"part_a_incidence": partA,
           "part_b": {"n": int(len(zt)), "lsst_requirement": LSST_REQ,
                      "by_config": results_by_cfg},
           "verdict": ("IMAGING-ONLY photo-z exceeds the LSST requirement from the "
                       "measured miss rate ALONE" if img["n_bins_exceeding"] else
                       "measured miss rate alone stays within the LSST requirement")}
    print(f"\nVERDICT: {out['verdict']}")

    with open(RESULTS / "e1_incidence_impact.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    ax = axes[0]
    ax.hist(offsets, bins=60, range=(0, 30), color="C0")
    ax.axvline(np.median(offsets), color="k", ls="--", lw=1.2,
               label=f"median {np.median(offsets):.1f} px")
    ax.axvline(np.percentile(offsets, 99), color="C3", ls=":", lw=1.2,
               label=f"p99 {np.percentile(offsets, 99):.1f} px")
    ax.set_xlabel("centroid offset of central segment (px)"); ax.set_ylabel("cutouts")
    ax.set_title(f"PART A: real mask offsets\n{100 * frac_no_centre:.2f}% have no segment at centre",
                 fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(z0, z1 - z0, s=5, alpha=0.3)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("AION z (true mask)"); ax.set_ylabel("Δz from realistic mask errors")
    ax.set_title("Per-object shift", fontsize=10); ax.grid(alpha=0.3)

    ax = axes[2]
    if bins:
        x = np.arange(len(bins))
        vals = [b["shift_over_1pz"] for b in bins]
        ax.bar(x, vals, color=["C3" if b["exceeds_requirement"] else "C0" for b in bins])
        ax.axhline(LSST_REQ, color="k", ls="--", lw=1.4, label=f"LSST req {LSST_REQ}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"[{b['lo']:.2f},{b['hi']:.2f})" for b in bins],
                           rotation=25, ha="right", fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel(r"$|\Delta\langle z\rangle|/(1+z)$")
        ax.legend(fontsize=8)
    ax.set_title("PART B: tomographic bias vs requirement", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("E1: how often the detection gate fires, and what it costs a cosmology analysis",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "e1_incidence_impact.png", dpi=150)
    print(f"wrote {RESULTS / 'e1_incidence_impact.json'} and .png")


if __name__ == "__main__":
    main()
