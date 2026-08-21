"""
D2g — Does the D2f null survive in SEVERE blends?   (closes D2f's caveat)

D2f found R = -0.012: AION completely ignores which pipeline segments the mask includes.
But that sample was Legacy Survey segmentation of bright BGS galaxies, where the central
segment holds a median 86% of the cutout light. Nearly-equal blends -- the regime that
actually matters for Rubin -- were rare, so the null could be an artifact of mild blending.

This selects on blend severity instead of taking whatever comes. Scanning many rows and
keeping only cutouts where the central segment holds a SMALL fraction of the light gives the
closest thing to a crowded field this dataset contains.

Reports R = d(AION flux) / d(true light added) in bins of central light fraction. If R stays
at zero as blends get severe, the null is a property of the model. If R climbs toward 1 in the
severe bin, D2f's null was a selection effect and the deblending worry is back.

Same tokenisation control as D2f: the two masks must actually differ as tokens.

Run:  python experiments/d2g_severe_blends.py [MAX_SCAN]
Out:  results/d2g_severe_blends.json, results/d2g_severe_blends.png
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
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import astron_mi as A
from astron_mi import physics as P

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
MAX_SCAN = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
SCAN_BATCH = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
CROP = 96
N_BOOT = 1000
SEVERE_MAX_FRAC = 0.70      # keep only cutouts where the centre holds <= this of the light


def centre_crop(x, n=CROP):
    h, w = x.shape[-2:]
    r, c = (h - n) // 2, (w - n) // 2
    return x[..., r:r + n, c:c + n]


def boot_ci(v, fn=np.median, n=N_BOOT, seed=SEED):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    st = [fn(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return float(fn(v)), float(np.percentile(st, 16)), float(np.percentile(st, 84))


def main():
    from aion.modalities import LegacySurveyFluxG, LegacySurveySegmentationMap

    rng = np.random.default_rng(SEED)
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device=DEVICE)
    T = P.tables()
    print(f"device={DEVICE}  scanning up to {MAX_SCAN} rows for severe blends "
          f"(central segment <= {SEVERE_MAX_FRAC:.0%} of the light)")

    def seg(binary):
        t = cm.encode(LegacySurveySegmentationMap(field=binary.clone().to(DEVICE)))
        return {k: v.reshape(v.shape[0], -1) for k, v in t.items()}

    def flux(tok):
        return A.read_targets(model, tok, [LegacySurveyFluxG], T,
                              device=DEVICE, seed=SEED)["tok_flux_g"]

    acc = {k: [] for k in ("f_central", "f_all", "L_central", "L_all", "truth", "frac", "tokdiff")}
    n_scanned = 0

    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), min(MAX_SCAN, r.n_rows()), replace=False))
        for s in range(0, len(rows), SCAN_BATCH):
            b = r.aion_batch(rows[s:s + SCAN_BATCH], with_segmap=True)
            segc = centre_crop(b["segmap"])
            imgc = centre_crop(b["image"]["flux"][:, 0])
            n_scanned += len(segc)

            keep, mc_l, ma_l, fr_l = [], [], [], []
            for i in range(len(segc)):
                cid = float(segc[i, CROP // 2, CROP // 2])
                ids = [v for v in torch.unique(segc[i]).tolist() if v > 0]
                if cid <= 0 or len(ids) < 2:
                    continue
                mcen = (segc[i] == cid).float()
                mall = (segc[i] > 0).float()
                Lc = float((imgc[i] * mcen).sum())
                La = float((imgc[i] * mall).sum())
                if La <= 0.05:
                    continue
                frac = Lc / La
                if frac > SEVERE_MAX_FRAC:          # not severe enough
                    continue
                keep.append(i); mc_l.append(mcen); ma_l.append(mall); fr_l.append(frac)
            if not keep:
                continue

            keep = np.array(keep)
            mc, ma = torch.stack(mc_l), torch.stack(ma_l)
            off = (b["segmap"].shape[-1] - CROP) // 2
            full = torch.zeros(len(keep), *b["segmap"].shape[-2:])
            fc = full.clone(); fc[:, off:off + CROP, off:off + CROP] = mc
            fa = full.clone(); fa[:, off:off + CROP, off:off + CROP] = ma

            ti_all = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            ti = {k: v[keep] for k, v in ti_all.items()}
            sc, sa = seg(fc), seg(fa)
            tc, ta = list(sc.values())[0], list(sa.values())[0]

            acc["tokdiff"].append((tc != ta).float().mean(-1).cpu().numpy())
            acc["f_central"].append(flux({**ti, **sc}))
            acc["f_all"].append(flux({**ti, **sa}))
            g = imgc[keep]
            acc["L_central"].append((g * mc).sum((-2, -1)).numpy())
            acc["L_all"].append((g * ma).sum((-2, -1)).numpy())
            acc["truth"].append(np.asarray(b["scalars"]["FLUX_G"])[keep])
            acc["frac"].append(np.array(fr_l))
            print(f"  severe blends: {sum(len(x) for x in acc['frac'])}  "
                  f"(scanned {n_scanned})", flush=True)

    d = {k: np.concatenate(v) for k, v in acc.items()}
    n = len(d["frac"])
    if n < 20:
        print(f"\nonly {n} severe blends found in {n_scanned} rows — this dataset does not "
              f"contain enough of them to settle the question. Reporting what there is.")
    print(f"\n{n} severe blends from {n_scanned} scanned "
          f"({100 * n / max(n_scanned, 1):.1f}% of cutouts)")
    print(f"  central light fraction: median {np.median(d['frac']):.3f}, "
          f"min {d['frac'].min():.3f}")
    print(f"  CONTROL segmap token difference: median {100 * np.median(d['tokdiff']):.1f}%, "
          f"zero-diff objects {100 * (d['tokdiff'] == 0).mean():.1f}%")

    dF = d["f_all"] - d["f_central"]
    dL = d["L_all"] - d["L_central"]
    ok = np.isfinite(dF) & np.isfinite(dL) & (dL > 0.05)
    R = dF[ok] / dL[ok]
    r_med, r_lo, r_hi = boot_ci(R)
    print(f"\nSEVERE-BLEND INTEGRATION TEST on {ok.sum()} objects")
    print(f"  median light added {np.median(dL[ok]):+.3f} nmgy, "
          f"median flux change {np.median(dF[ok]):+.4f} nmgy")
    print(f"  R = {r_med:+.4f} [{r_lo:+.4f}, {r_hi:+.4f}]   "
          f"(D2f, mild blends: R = -0.0117)")

    # ---------------------------------------------------- R vs blend severity
    bins = [(0.0, 0.4), (0.4, 0.55), (0.55, 0.70)]
    strata = {}
    print(f"\n  R in bins of central light fraction:")
    for lo, hi in bins:
        m = ok & (d["frac"] >= lo) & (d["frac"] < hi)
        if m.sum() < 10:
            print(f"    [{lo:.2f},{hi:.2f})  n={m.sum():3d}  (too few)")
            continue
        rm, rl, rh = boot_ci(dF[m] / dL[m])
        strata[f"{lo:.2f}-{hi:.2f}"] = {"n": int(m.sum()), "R": rm, "lo16": rl, "hi84": rh}
        print(f"    [{lo:.2f},{hi:.2f})  n={m.sum():3d}  R = {rm:+.4f} [{rl:+.4f},{rh:+.4f}]")

    rho = float(spearmanr(d["frac"][ok], R).statistic) if ok.sum() > 10 else float("nan")
    out = {
        "n_severe": int(n), "n_scanned": int(n_scanned),
        "severe_max_frac": SEVERE_MAX_FRAC,
        "median_central_frac": float(np.median(d["frac"])),
        "control_tokdiff_median": float(np.median(d["tokdiff"])),
        "R_all_severe": {"median": r_med, "lo16": r_lo, "hi84": r_hi, "n": int(ok.sum())},
        "R_by_severity": strata,
        "spearman_frac_vs_R": rho,
        "reference_d2f_mild_R": -0.0117,
    }
    verdict = ("null HOLDS in severe blends" if abs(r_med) < 0.15
               else "null BREAKS in severe blends — deblending does propagate here")
    out["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    with open(RESULTS / "d2g_severe_blends.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.scatter(dL[ok], dF[ok], s=10, alpha=0.5)
    lim = float(np.nanpercentile(dL[ok], 99)) if ok.sum() else 1.0
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="R = 1 (integrates)")
    ax.axhline(0, color="C3", ls=":", lw=1.2, label="R = 0 (ignores)")
    ax.set_xlim(0, lim)
    ax.set_xlabel("true g light added by neighbour segments (nmgy)")
    ax.set_ylabel(r"$\Delta$ AION E[flux_g] (nmgy)")
    ax.set_title(f"Severe blends only (centre ≤ {SEVERE_MAX_FRAC:.0%} of light)\n"
                 f"R = {r_med:+.3f} [{r_lo:+.3f},{r_hi:+.3f}]", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    if strata:
        xs = np.arange(len(strata))
        med = [v["R"] for v in strata.values()]
        lo = [v["R"] - v["lo16"] for v in strata.values()]
        hi = [v["hi84"] - v["R"] for v in strata.values()]
        ax.bar(xs, med, yerr=[lo, hi], capsize=4, color="C0")
        ax.set_xticks(xs); ax.set_xticklabels(list(strata), fontsize=8)
    ax.axhline(0, color="C3", ls=":", lw=1.2, label="ignores mask extent")
    ax.axhline(1, color="k", ls="--", lw=1.2, label="integrates the light")
    ax.set_xlabel("central segment's share of the light (more blended → left)")
    ax.set_ylabel("R")
    ax.set_title("Does R rise as blends get severe?", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("D2g: testing D2f's null where blending is severe", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "d2g_severe_blends.png", dpi=150)
    print(f"wrote {RESULTS / 'd2g_severe_blends.json'} and .png")


if __name__ == "__main__":
    main()
