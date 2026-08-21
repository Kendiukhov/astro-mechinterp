"""
D2f — Real blends: does AION track the pipeline's own deblending decision?  (Tier 2)

Everything in D2/D2b/D2d/D2e perturbed the mask synthetically (displace, swap, erase), which
invites the objection "that never happens". This experiment uses NO synthetic geometry. It
only chooses which of the pipeline's OWN segments to include.

In a blended cutout the Legacy Survey pipeline has already partitioned the light into
segments with distinct IDs. The catalogue FLUX_G refers to the CENTRAL object alone. So:

    central_only   mask = (segmap == id at the centre pixel)      -> "the target only"
    all_segments   mask = (segmap > 0)                            -> "target + neighbours"

Both masks are genuine pipeline output. The difference between them is exactly the
deblending decision.

THE QUANTITATIVE PREDICTION. Let dL be the true g-band light in the neighbour segments and
d(flux) the change in AION's readout between the two masks. If the model integrates the light
the mask assigns to the object, then

    R = d(flux) / dL  ~  1

R ~ 0 means the mask's extent is ignored (the readout comes from the pixels alone).
R >> 1 or R << 0 means the model responds to the mask but not by the amount of light involved
-- deference without integration, which is what D2's "detection gating, not aperture
photometry" result predicts.

Also measured, with no intervention at all: does AION's flux error against the catalogue grow
with how much neighbour light is present? That is the purely observational version.

Run:  python experiments/d2f_real_blends.py [N_BLENDED]
Out:  results/d2f_real_blends.json, results/d2f_real_blends.png
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
N_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 240   # blended objects wanted
SCAN_BATCH = 16
MAX_SCAN = 3000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
CROP = 96
N_BOOT = 1000


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
    print(f"device={DEVICE}  looking for {N_TARGET} genuinely blended cutouts")

    def seg(binary):
        t = cm.encode(LegacySurveySegmentationMap(field=binary.clone().to(DEVICE)))
        return {k: v.reshape(v.shape[0], -1) for k, v in t.items()}

    def flux(tok):
        return A.read_targets(model, tok, [LegacySurveyFluxG], T,
                              device=DEVICE, seed=SEED)["tok_flux_g"]

    rec = {k: [] for k in ("f_central", "f_all", "L_central", "L_all",
                           "truth", "n_src", "central_frac")}
    tok_diff: list = []
    n_have = 0

    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), min(MAX_SCAN, r.n_rows()), replace=False))
        for s in range(0, len(rows), SCAN_BATCH):
            if n_have >= N_TARGET:
                break
            b = r.aion_batch(rows[s:s + SCAN_BATCH], with_segmap=True)
            segc = centre_crop(b["segmap"])              # (B, 96, 96) integer IDs
            imgc = centre_crop(b["image"]["flux"][:, 0])  # g band, same crop

            keep, m_central, m_all = [], [], []
            for i in range(len(segc)):
                cid = float(segc[i, CROP // 2, CROP // 2])
                ids = [v for v in torch.unique(segc[i]).tolist() if v > 0]
                # require: a real source at the centre, and at least one other segment
                if cid <= 0 or len(ids) < 2:
                    continue
                keep.append(i)
                m_central.append((segc[i] == cid).float())
                m_all.append((segc[i] > 0).float())
            if not keep:
                continue

            keep = np.array(keep)
            mc = torch.stack(m_central)
            ma = torch.stack(m_all)
            # pad the 96-crop masks back to the 160 frame the codec expects
            full = torch.zeros(len(keep), *b["segmap"].shape[-2:])
            off = (b["segmap"].shape[-1] - CROP) // 2
            fc = full.clone(); fc[:, off:off + CROP, off:off + CROP] = mc
            fa = full.clone(); fa[:, off:off + CROP, off:off + CROP] = ma

            ti_all = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            ti = {k: v[keep] for k, v in ti_all.items()}

            sc, sa = seg(fc), seg(fa)
            # ESSENTIAL CONTROL: if the segmap codec collapses the two masks to the same
            # tokens, a null result measures the tokeniser, not the model.
            tc, ta = list(sc.values())[0], list(sa.values())[0]
            tok_diff.append((tc != ta).float().mean(-1).cpu().numpy())
            f_c = flux({**ti, **sc})
            f_a = flux({**ti, **sa})

            g = imgc[keep]
            L_c = (g * mc).sum((-2, -1)).numpy()
            L_a = (g * ma).sum((-2, -1)).numpy()

            rec["f_central"].append(f_c); rec["f_all"].append(f_a)
            rec["L_central"].append(L_c); rec["L_all"].append(L_a)
            rec["truth"].append(np.asarray(b["scalars"]["FLUX_G"])[keep])
            rec["n_src"].append(np.array([len([v for v in torch.unique(segc[i]).tolist() if v > 0])
                                          for i in keep]))
            rec["central_frac"].append(L_c / np.maximum(L_a, 1e-6))
            n_have += len(keep)
            print(f"  blended so far: {n_have}/{N_TARGET}  (scanned {min(s + SCAN_BATCH, len(rows))})",
                  flush=True)

    d = {k: np.concatenate(v)[:N_TARGET] for k, v in rec.items()}
    td = np.concatenate(tok_diff)[:N_TARGET]
    n = len(d["truth"])
    print(f"\ncollected {n} blended cutouts "
          f"(median {np.median(d['n_src']):.0f} segments, central segment holds "
          f"{100 * np.median(d['central_frac']):.0f}% of the light)")

    print(f"\nCONTROL: segmap tokens differing between the two masks: "
          f"median {100 * np.median(td):.1f}%  (mean {100 * td.mean():.1f}%, "
          f"objects with zero difference: {100 * (td == 0).mean():.1f}%)")
    if np.median(td) < 0.01:
        print("  *** the codec nearly collapses the two masks -- a null result would be a "
              "tokenisation artifact, NOT a model property ***")

    # ------------------------------------------------------- the integration test
    dF = d["f_all"] - d["f_central"]
    dL = d["L_all"] - d["L_central"]
    ok = np.isfinite(dF) & np.isfinite(dL) & (dL > 0.05)     # meaningful neighbour light
    R = dF[ok] / dL[ok]
    r_med, r_lo, r_hi = boot_ci(R)
    rho = float(spearmanr(dL[ok], dF[ok]).statistic)

    print(f"\nINTEGRATION TEST on {ok.sum()} objects with real neighbour light")
    print(f"  median d(AION flux) = {np.median(dF[ok]):+.4f} nmgy")
    print(f"  median d(true light in the added segments) = {np.median(dL[ok]):+.4f} nmgy")
    print(f"  R = d(flux)/d(light)  =  {r_med:+.4f}  [{r_lo:+.4f}, {r_hi:+.4f}]")
    print(f"  spearman(d light, d flux) = {rho:+.3f}")
    print(f"  R ~ 1 would mean the model integrates the light the mask assigns it.")

    # ------------------------------------------- observational, no intervention
    err_c = 2.5 * np.abs(np.log10(np.maximum(d["f_central"], 1e-3)
                                  / np.maximum(d["truth"], 1e-3)))
    err_a = 2.5 * np.abs(np.log10(np.maximum(d["f_all"], 1e-3)
                                  / np.maximum(d["truth"], 1e-3)))
    good = np.isfinite(err_c) & np.isfinite(err_a) & (d["truth"] > 1e-2)
    rho_err = float(spearmanr(1 - d["central_frac"][good], err_a[good]).statistic)
    ec, ec_lo, ec_hi = boot_ci(err_c[good])
    ea, ea_lo, ea_hi = boot_ci(err_a[good])

    print(f"\nAGAINST CATALOGUE TRUTH (FLUX_G is the CENTRAL object only)")
    print(f"  central-segment mask : |dmag| {ec:.4f} [{ec_lo:.4f},{ec_hi:.4f}]")
    print(f"  all-segments mask    : |dmag| {ea:.4f} [{ea_lo:.4f},{ea_hi:.4f}]")
    print(f"  spearman(neighbour light fraction, error with all-segments mask) = {rho_err:+.3f}")

    out = {
        "n_blended": int(n),
        "median_segments": float(np.median(d["n_src"])),
        "median_central_light_fraction": float(np.median(d["central_frac"])),
        "control_segmap_token_diff_median": float(np.median(td)),
        "control_segmap_token_diff_zero_frac": float((td == 0).mean()),
        "integration": {
            "n_used": int(ok.sum()),
            "median_dflux_nmgy": float(np.median(dF[ok])),
            "median_dlight_nmgy": float(np.median(dL[ok])),
            "R_median": r_med, "R_lo16": r_lo, "R_hi84": r_hi,
            "spearman_dlight_dflux": rho,
        },
        "vs_truth": {
            "central_mask_dmag": [ec, ec_lo, ec_hi],
            "all_segments_mask_dmag": [ea, ea_lo, ea_hi],
            "spearman_neighbour_fraction_vs_error": rho_err,
        },
    }
    verdict = ("integrates the assigned light" if 0.5 < r_med < 2.0
               else ("ignores mask extent" if abs(r_med) < 0.1
                     else "responds to the mask but NOT by the light involved"))
    out["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    with open(RESULTS / "d2f_real_blends.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    ax = axes[0]
    ax.scatter(dL[ok], dF[ok], s=8, alpha=0.45)
    lim = np.nanpercentile(dL[ok], 99)
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="R = 1 (integrates the light)")
    ax.axhline(0, color="C3", ls=":", lw=1.2, label="R = 0 (ignores mask extent)")
    ax.set_xlim(0, lim); ax.set_xlabel("true g light added by the neighbour segments (nmgy)")
    ax.set_ylabel(r"$\Delta$ AION E[flux_g] (nmgy)")
    ax.set_title(f"Does it integrate what the mask assigns?\nR = {r_med:+.3f} [{r_lo:+.3f},{r_hi:+.3f}]",
                 fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar([0, 1], [ec, ea], yerr=[[ec - ec_lo, ea - ea_lo], [ec_hi - ec, ea_hi - ea]],
           capsize=4, color=["C0", "C1"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["central segment\n(correct)", "all segments\n(under-deblended)"],
                                              fontsize=8)
    ax.set_ylabel("|dmag| vs catalogue FLUX_G")
    ax.set_title("Error against truth, real masks only", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ax.scatter(1 - d["central_frac"][good], err_a[good], s=8, alpha=0.45)
    ax.set_xlabel("fraction of cutout light in NEIGHBOUR segments")
    ax.set_ylabel("|dmag| with all-segments mask")
    ax.set_yscale("log")
    ax.set_title(f"Observational: error vs blend severity\nspearman {rho_err:+.3f}", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("D2f: real blends — only the pipeline's own segments, no synthetic edits",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "d2f_real_blends.png", dpi=150)
    print(f"wrote {RESULTS / 'd2f_real_blends.json'} and .png")


if __name__ == "__main__":
    main()
