"""
D2 — Does AION's flux estimate follow the pipeline's mask, or the light?  (AGENDA.md, Thrust D)

AION is, as far as I know, the only astronomical foundation model with an explicitly
editable *detection* channel: `tok_segmap`, the Legacy Survey segmentation map, supplied
as a modality alongside the pixels. So we can hold the image byte-identical and change
only the pipeline's opinion about where the sources are.

The question matters because ~60% of LSST sources will be blended, and every downstream
science case inherits whatever deblending error the detection pipeline makes. If a
foundation model reads the *mask* rather than the *sky*, it inherits those errors too —
and, worse, launders them as a model prediction.

Arms (pixels IDENTICAL throughout):
    image_only     no segmap supplied at all
    real           the true segmentation map                          <- reference
    erased         all-zero mask                    OFF-MANIFOLD, reported but discounted
    dilated_k      mask grown by k pixels           graded, near-manifold
    eroded_k       mask shrunk by k pixels          graded, near-manifold
    shifted_k      mask translated by k pixels      on-manifold shape, wrong location
    swapped        ANOTHER GALAXY'S REAL MASK       fully on-manifold -- the decisive arm

The swap arm is the one that carries the argument: every mask is a genuine Legacy Survey
mask of a real object, so nothing is out of distribution. If swapping masks between
objects moves the flux estimate, the model is reading the mask.

MECHANISM TEST. If the model is doing masked aperture photometry, its predicted flux
should track the light actually enclosed by the supplied mask. We compute, for every arm,
the true integrated image flux inside that arm's mask and correlate it with the
prediction. A high correlation is a positive, mechanistic statement, not just a
sensitivity result.

PLACEBO. `tok_ra` is swept over a comparable number of codebook steps. Any effect must be
reported as a ratio against it (AGENDA.md 11).

Run:  python experiments/d2_segmap_intervention.py [N]
Out:  results/d2_segmap.json, results/d2_segmap.png
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
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import astron_mi as A
from astron_mi import physics as P

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 240
BATCH = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
CROP = 96          # the codecs centre-crop 160 -> 96


def morph(mask: torch.Tensor, k: int) -> torch.Tensor:
    """Dilate (k>0) or erode (k<0) a binary mask by |k| pixels."""
    if k == 0:
        return mask
    x = mask.unsqueeze(1)
    size = 2 * abs(k) + 1
    if k > 0:
        y = F.max_pool2d(x, size, stride=1, padding=abs(k))
    else:
        y = -F.max_pool2d(-x, size, stride=1, padding=abs(k))
    return y.squeeze(1)


def centre_crop(x: torch.Tensor, n: int = CROP) -> torch.Tensor:
    h, w = x.shape[-2:]
    r, c = (h - n) // 2, (w - n) // 2
    return x[..., r:r + n, c:c + n]


def central_coverage(mask: torch.Tensor, n: int = 12) -> np.ndarray:
    """Fraction of the central n x n pixels the mask covers.

    This is the discriminator between two mechanisms:
      aperture photometry -> prediction tracks LIGHT ENCLOSED by the mask
      detection gating    -> prediction tracks whether the mask says
                             'there is a source at the centre'
    """
    m = centre_crop(mask)
    h = m.shape[-1]
    r = (h - n) // 2
    return (m[..., r:r + n, r:r + n] > 0).float().mean((-2, -1)).cpu().numpy()


def enclosed_light(image: torch.Tensor, mask: torch.Tensor, band: int = 0) -> np.ndarray:
    """Total g-band flux inside the mask, on the same centre crop the codecs use."""
    img = centre_crop(image[:, band])
    m = centre_crop(mask)
    return (img * (m > 0).float()).sum(dim=(-2, -1)).cpu().numpy()


def main():
    from aion.modalities import LegacySurveyFluxG, LegacySurveySegmentationMap, Ra

    rng = np.random.default_rng(SEED)
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device=DEVICE)
    T = P.tables()
    print(f"device={DEVICE}  N={N}")

    def seg_tokens(binary):
        t = cm.encode(LegacySurveySegmentationMap(field=binary.clone().to(DEVICE)))
        return {k: v.reshape(v.shape[0], -1) for k, v in t.items()}

    def read_flux(tokens):
        tk = {k: v.to(DEVICE) for k, v in tokens.items()}
        lg = model.forward(tk, target_modality=[LegacySurveyFluxG])["tok_flux_g"]
        return T.expected_value(lg[:, 0, :].detach().float().cpu().numpy(), "tok_flux_g")

    arms = (["image_only", "real", "erased", "swapped"]
            + [f"dilated_{k}" for k in (2, 5, 10)]
            + [f"eroded_{k}" for k in (2, 5)]
            + [f"shifted_{k}" for k in (10, 25)]
            + ["placebo_ra"])
    acc: dict[str, list] = {a: [] for a in arms}
    light: dict[str, list] = {a: [] for a in arms}
    cover: dict[str, list] = {a: [] for a in arms}
    truth, area_real = [], []

    with A.ProvabgsReader() as r:
        n_all = r.n_rows()
        rows = np.sort(rng.choice(n_all, N, replace=False))
        for s in range(0, N, BATCH):
            b = r.aion_batch(rows[s:s + BATCH], with_segmap=True)
            img = b["image"]["flux"]
            ti = A.encode_image(cm, img, b["image"]["bands"], device=DEVICE)
            binary = (b["segmap"] > 0).float()
            truth.append(np.asarray(b["scalars"]["FLUX_G"]))
            area_real.append((centre_crop(binary) > 0).float().mean((-2, -1)).numpy())

            variants = {
                "real": binary,
                "erased": torch.zeros_like(binary),
                "swapped": binary[torch.randperm(len(binary), generator=torch.Generator().manual_seed(s))],
            }
            for k in (2, 5, 10):
                variants[f"dilated_{k}"] = morph(binary, k)
            for k in (2, 5):
                variants[f"eroded_{k}"] = morph(binary, -k)
            for k in (10, 25):
                variants[f"shifted_{k}"] = torch.roll(binary, shifts=(k, k), dims=(-2, -1))

            acc["image_only"].append(read_flux(dict(ti)))
            light["image_only"].append(enclosed_light(img, torch.ones_like(binary)))
            cover["image_only"].append(central_coverage(torch.ones_like(binary)))

            for name, m in variants.items():
                acc[name].append(read_flux({**ti, **seg_tokens(m)}))
                light[name].append(enclosed_light(img, m))
                cover[name].append(central_coverage(m))

            # placebo: real segmap, but RA perturbed by ~250 codebook steps
            ra_v = torch.from_numpy(
                np.asarray(b["ra"], dtype="float32")
            ).to(DEVICE)
            tra = cm.encode(Ra(value=ra_v))
            tra = {k: (v.reshape(v.shape[0], -1) if v.dim() > 1 else v.unsqueeze(1))
                   for k, v in tra.items()}
            tra_before = {k: v.clone() for k, v in tra.items()}
            tra = {k: torch.clamp(v + 250, 0, 1023) for k, v in tra.items()}
            if s == 0:
                ch = float((list(tra.values())[0] != list(tra_before.values())[0]).float().mean())
                print(f"  [placebo check] RA tokens actually changed in {ch:.0%} of objects", flush=True)
            acc["placebo_ra"].append(read_flux({**ti, **seg_tokens(binary), **tra}))
            light["placebo_ra"].append(enclosed_light(img, binary))
            cover["placebo_ra"].append(central_coverage(binary))
            print(f"  {min(s + BATCH, N)}/{N}", flush=True)

    truth = np.concatenate(truth)
    area_real = np.concatenate(area_real)
    pred = {a: np.concatenate(v) for a, v in acc.items()}
    encl = {a: np.concatenate(v) for a, v in light.items()}
    cov = {a: np.concatenate(v) for a, v in cover.items()}
    ref = pred["real"]

    out: dict = {"n": int(len(truth)), "arms": {}}
    print(f"\n{'arm':14s} {'median E[flux_g]':>17s} {'vs real (%)':>12s} {'vs truth (%)':>13s} "
          f"{'r(pred, light in mask)':>23s}")
    for a in arms:
        p = pred[a]
        rel = 100 * np.median(np.abs(p - ref) / np.abs(ref))
        vt = 100 * np.median(np.abs(p - truth) / np.abs(truth))
        rr = float(np.corrcoef(p, encl[a])[0, 1]) if np.std(encl[a]) > 0 else float("nan")
        out["arms"][a] = {
            "median_pred": float(np.median(p)),
            "median_abs_rel_change_vs_real_pct": float(rel),
            "median_abs_rel_err_vs_truth_pct": float(vt),
            "corr_pred_vs_enclosed_light": rr,
        }
        print(f"{a:14s} {np.median(p):17.3f} {rel:12.1f} {vt:13.1f} {rr:23.3f}")

    # ------------------------------------------------------------ mechanism test
    # The naive pooled correlation is dominated by object-to-object brightness: bright
    # galaxies are bright under every mask, so r ~ 0.97 appears even for the swap arm and
    # means nothing. The intervention effect is only visible WITHIN object: does the
    # CHANGE in prediction track the CHANGE in enclosed light, relative to the real mask?
    grade = ["eroded_5", "eroded_2", "real", "dilated_2", "dilated_5", "dilated_10"]
    gp = np.concatenate([pred[a] for a in grade])
    gl = np.concatenate([encl[a] for a in grade])
    r_pooled = float(np.corrcoef(gp, gl)[0, 1])

    dp = np.concatenate([pred[a] - pred["real"] for a in grade if a != "real"])
    dl = np.concatenate([encl[a] - encl["real"] for a in grade if a != "real"])
    r_delta = float(np.corrcoef(dp, dl)[0, 1])
    # fractional version, so objects of different brightness contribute comparably
    fp = np.concatenate([(pred[a] - pred["real"]) / np.abs(pred["real"]) for a in grade if a != "real"])
    fl = np.concatenate([(encl[a] - encl["real"]) / np.abs(encl["real"]) for a in grade if a != "real"])
    keep = np.isfinite(fp) & np.isfinite(fl)
    r_frac = float(np.corrcoef(fp[keep], fl[keep])[0, 1])
    slope = float(np.polyfit(fl[keep], fp[keep], 1)[0])

    # the same statistic for the swap arm, where enclosed light also changes
    ds_p = (pred["swapped"] - pred["real"]) / np.abs(pred["real"])
    ds_l = (encl["swapped"] - encl["real"]) / np.abs(encl["real"])
    k2 = np.isfinite(ds_p) & np.isfinite(ds_l)
    r_swap = float(np.corrcoef(ds_p[k2], ds_l[k2])[0, 1])

    # DISCRIMINATOR: light enclosed vs central coverage
    allarms = [a for a in arms if a not in ("image_only", "placebo_ra", "real")]
    fp_all = np.concatenate([(pred[a] - pred["real"]) / np.abs(pred["real"]) for a in allarms])
    fl_all = np.concatenate([(encl[a] - encl["real"]) / np.abs(encl["real"]) for a in allarms])
    fc_all = np.concatenate([cov[a] - cov["real"] for a in allarms])
    kk = np.isfinite(fp_all) & np.isfinite(fl_all) & np.isfinite(fc_all)
    r_light_all = float(np.corrcoef(fp_all[kk], fl_all[kk])[0, 1])
    r_cover_all = float(np.corrcoef(fp_all[kk], fc_all[kk])[0, 1])
    print(f"\nDISCRIMINATOR across all {len(allarms)} intervention arms (within object):")
    print(f"  corr(fractional d prediction, fractional d ENCLOSED LIGHT)   = {r_light_all:+.3f}")
    print(f"  corr(fractional d prediction, d CENTRAL MASK COVERAGE)       = {r_cover_all:+.3f}")
    print(f"  -> {'DETECTION GATING' if abs(r_cover_all) > abs(r_light_all) else 'APERTURE PHOTOMETRY'}")

    out["mechanism"] = {
        "corr_frac_dpred_vs_denclosed_light_allarms": r_light_all,
        "corr_frac_dpred_vs_dcentral_coverage_allarms": r_cover_all,
        "discriminator_verdict": ("detection_gating" if abs(r_cover_all) > abs(r_light_all)
                                  else "aperture_photometry"),
        "graded_arms": grade,
        "corr_pooled_confounded_by_brightness": r_pooled,
        "corr_within_object_delta": r_delta,
        "corr_within_object_fractional": r_frac,
        "fractional_slope": slope,
        "corr_within_object_fractional_swap_arm": r_swap,
    }
    print(f"\nMECHANISM")
    print(f"  pooled corr(pred, enclosed light)          = {r_pooled:.3f}   "
          f"(CONFOUNDED by object brightness -- do not quote)")
    print(f"  within-object corr(d pred, d light)        = {r_delta:.3f}")
    print(f"  within-object corr, fractional             = {r_frac:.3f}  slope {slope:.3f}")
    print(f"  same statistic on the swap arm             = {r_swap:.3f}")
    print(f"  -> a slope near 1 with high r means masked aperture photometry;")
    print(f"     high r with slope << 1 means the mask modulates but does not set the flux.")

    # ------------------------------------------------------------------- verdict
    swap = out["arms"]["swapped"]["median_abs_rel_change_vs_real_pct"]
    plac = out["arms"]["placebo_ra"]["median_abs_rel_change_vs_real_pct"]
    out["verdict"] = {
        "swap_change_pct": swap,
        "placebo_change_pct": plac,
        "ratio_to_placebo": float(swap / plac) if plac > 0 else float("inf"),
        "follows_mask": bool(swap > 5 * max(plac, 1e-6)),
    }
    print(f"\nVERDICT (on-manifold arm vs placebo)")
    print(f"  swapping in another galaxy's REAL mask changes E[flux_g] by {swap:.1f}%")
    print(f"  placebo (RA shifted 250 codes)                              {plac:.1f}%")
    print(f"  ratio = {out['verdict']['ratio_to_placebo']:.1f}x  -> "
          f"{'FLUX FOLLOWS THE MASK' if out['verdict']['follows_mask'] else 'no mask dependence'}")

    with open(RESULTS / "d2_segmap.json", "w") as f:
        json.dump(out, f, indent=2)

    # -------------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    ax = axes[0]
    order = ["image_only", "eroded_5", "eroded_2", "real", "dilated_2", "dilated_5",
             "dilated_10", "shifted_10", "shifted_25", "swapped", "erased", "placebo_ra"]
    cols = ["#888" if a in ("image_only", "placebo_ra") else
            ("C3" if a == "erased" else ("C2" if a == "swapped" else "C0")) for a in order]
    ax.bar(range(len(order)), [np.median(pred[a]) for a in order], color=cols)
    ax.axhline(np.median(ref), color="k", ls=":", lw=1, label="real mask")
    ax.axhline(np.median(truth), color="C1", ls="--", lw=1, label="true FLUX_G")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("median E[flux_g]  (nmgy)")
    ax.set_title("Pixels identical; only the mask changes", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for a, c in [("real", "C0"), ("dilated_10", "C4"), ("eroded_5", "C5"), ("swapped", "C2")]:
        ax.scatter(encl[a], pred[a], s=6, alpha=0.4, label=a, color=c)
    ax.set_xscale("symlog"); ax.set_yscale("symlog")
    ax.set_xlabel("true g-band light enclosed by the supplied mask")
    ax.set_ylabel("AION E[flux_g]")
    ax.set_title(f"Enclosed light vs prediction\nwithin-object fractional r = {r_frac:.3f} (slope {slope:.2f})", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    xs = [np.median(encl[a]) for a in grade]
    ys = [np.median(pred[a]) for a in grade]
    ax.plot(xs, ys, "o-", lw=1.6)
    for a, x, y in zip(grade, xs, ys):
        ax.annotate(a, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("median enclosed light"); ax.set_ylabel("median E[flux_g]")
    ax.set_title("Graded erode -> dilate series", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("D2: does AION's flux estimate follow the detection mask or the light?", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "d2_segmap.png", dpi=150)
    print(f"wrote {RESULTS / 'd2_segmap.json'} and .png")


if __name__ == "__main__":
    main()
