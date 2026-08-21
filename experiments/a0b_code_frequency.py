"""
A0b — Direct code-frequency measurement, replacing A0's row-norm proxy.

A0 inferred codebook utilisation from embedding row norms: a code that is never emitted gets
no gradient and is pulled below its initialisation norm by weight decay. That gave 23.2% for
`tok_image` and 9.6% for `tok_image_hsc`, with every empirical-CDF scalar at 100% as the
positive control. It is a proxy, and the caveat has been carried in every write-up since.

This measures it directly: encode thousands of real cutouts and count which codes the codec
actually emits. Two questions:

  1. Does the direct count agree with the row-norm proxy?
  2. Is the "dead" set actually dead, or merely rare? A code emitted once in 10^6 tokens would
     look dead by row norm but is not structurally unreachable. The frequency distribution
     answers what the norm cannot.

Also reports the effective codebook size (perplexity of the code distribution), which is the
number a tokeniser paper would quote and which neither the AION paper nor A0 gives.

Legacy imaging only — this dataset has no HSC cutouts, so `tok_image_hsc`'s 9.6% remains a
proxy. Stated, not glossed.

Run:  python experiments/a0b_code_frequency.py [N]
Out:  results/a0b_code_frequency.json, results/a0b_code_frequency.png
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

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
BATCH = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
INIT_STD = 0.02

VOCAB = {"tok_image": 4375, "tok_spectrum_desi": 1024}
CROP, PATCH, GRID = 96, 4, 24     # image tokens are a 24x24 grid over the centre 96x96


def main():
    rng = np.random.default_rng(SEED)
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device="cpu")
    print(f"device={DEVICE}  N={N} cutouts")

    counts = {k: np.zeros(v, dtype="int64") for k, v in VOCAB.items()}
    # A 96x96 cutout is mostly blank sky, so a raw code histogram is dominated by whatever
    # code means "empty". Count SOURCE patches separately: a token is a source patch if its
    # 4x4 pixel footprint overlaps the pipeline segmentation map.
    counts_src = np.zeros(VOCAB["tok_image"], dtype="int64")
    counts_sky = np.zeros(VOCAB["tok_image"], dtype="int64")

    def source_patch_mask(segmap):
        m = segmap[..., (segmap.shape[-2] - CROP) // 2:(segmap.shape[-2] - CROP) // 2 + CROP,
                   (segmap.shape[-1] - CROP) // 2:(segmap.shape[-1] - CROP) // 2 + CROP]
        m = (m > 0).float().reshape(len(m), GRID, PATCH, GRID, PATCH)
        return (m.sum((2, 4)) > 0).reshape(len(m), -1).numpy()

    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), N, replace=False))
        for s in range(0, N, BATCH):
            b = r.aion_batch(rows[s:s + BATCH], with_segmap=True)
            ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            ts = A.encode_spectrum(cm, **b["spectrum"], survey="desi", device=DEVICE)
            for src in (ti, ts):
                for k, v in src.items():
                    if k in counts:
                        ids = v.reshape(-1).cpu().numpy()
                        np.add.at(counts[k], ids, 1)
            img_ids = ti["tok_image"].cpu().numpy()
            sm = source_patch_mask(b["segmap"])
            np.add.at(counts_src, img_ids[sm], 1)
            np.add.at(counts_sky, img_ids[~sm], 1)
            if (s // BATCH) % 20 == 0:
                print(f"  {min(s + BATCH, N)}/{N}", flush=True)

    out = {"n_cutouts": int(N), "modalities": {}}
    print(f"\n{'modality':20s} {'tokens':>12s} {'used':>12s} {'proxy (A0)':>11s} "
          f"{'perplexity':>11s} {'top-1%share':>12s}")
    for k, V in VOCAB.items():
        c = counts[k]
        tot = int(c.sum())
        used = int((c > 0).sum())
        p = c / max(tot, 1)
        nz = p[p > 0]
        perp = float(np.exp(-(nz * np.log(nz)).sum()))
        # row-norm proxy, recomputed here so the comparison is like-for-like
        w = model.encoder_embeddings[k].token_emb.weight.detach().float().numpy()
        thr = 0.5 * INIT_STD * np.sqrt(w.shape[1])
        proxy = float((np.linalg.norm(w, axis=1) > thr).mean())
        top1pct = float(np.sort(p)[::-1][:max(1, V // 100)].sum())
        # how rare is the rarest 10% of *used* codes?
        srt = np.sort(p[p > 0])
        out["modalities"][k] = {
            "vocab": V, "tokens_seen": tot,
            "codes_used": used, "frac_used": used / V,
            "rownorm_proxy_frac": proxy,
            "perplexity": perp, "effective_frac": perp / V,
            "top_1pct_share": top1pct,
            "codes_for_50pct_mass": int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.5) + 1),
            "codes_for_99pct_mass": int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.99) + 1),
            "min_used_freq": float(srt[0]) if len(srt) else 0.0,
        }
        print(f"{k:20s} {tot:12d} {used:5d}/{V:<6d} {100 * proxy:10.1f}% "
              f"{perp:11.1f} {100 * top1pct:11.1f}%")

    print(f"\ndetail:")
    for k, m in out["modalities"].items():
        print(f"  {k}")
        print(f"    directly used            : {100 * m['frac_used']:.1f}%   "
              f"(row-norm proxy said {100 * m['rownorm_proxy_frac']:.1f}%)")
        print(f"    effective size (perplexity): {m['perplexity']:.0f} / {m['vocab']}"
              f"  = {100 * m['effective_frac']:.1f}%")
        print(f"    codes holding 50% of mass : {m['codes_for_50pct_mass']}")
        print(f"    codes holding 99% of mass : {m['codes_for_99pct_mass']}")
        print(f"    rarest used code frequency: {m['min_used_freq']:.2e}")

    def summarise(c, label):
        tot = int(c.sum()); p = c / max(tot, 1); nz = p[p > 0]
        perp = float(np.exp(-(nz * np.log(nz)).sum()))
        return {"label": label, "tokens": tot, "codes_used": int((c > 0).sum()),
                "frac_used": float((c > 0).mean()), "perplexity": perp,
                "codes_for_50pct": int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.5) + 1)}

    src_stats = summarise(counts_src, "source patches")
    sky_stats = summarise(counts_sky, "sky patches")
    out["image_split"] = {"source": src_stats, "sky": sky_stats}
    print(f"\n  SPLITTING image tokens by whether the patch overlaps the segmentation map:")
    for st in (src_stats, sky_stats):
        print(f"    {st['label']:16s} {st['tokens']:9d} tokens   "
              f"{st['codes_used']:5d}/4375 used ({100 * st['frac_used']:4.1f}%)   "
              f"perplexity {st['perplexity']:8.1f}   50% mass in {st['codes_for_50pct']} codes")
    print(f"    -> the raw histogram is dominated by sky; the SOURCE-patch perplexity "
          f"({src_stats['perplexity']:.0f}) is the meaningful number")

    im = out["modalities"]["tok_image"]
    agree = abs(im["frac_used"] - im["rownorm_proxy_frac"]) < 0.10
    out["verdict"] = {
        "proxy_confirmed": bool(agree),
        "note": ("direct count agrees with the row-norm proxy" if agree else
                 "direct count DISAGREES with the row-norm proxy — A0's figure needs restating"),
    }
    print(f"\nVERDICT: {out['verdict']['note']}")
    print(f"  NB tok_image_hsc could not be measured directly — this dataset has no HSC "
          f"cutouts, so its 9.6% remains a proxy.")

    with open(RESULTS / "a0b_code_frequency.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, k in zip(axes, VOCAB):
        c = counts[k]
        p = np.sort(c / max(c.sum(), 1))[::-1]
        ax.semilogy(np.arange(1, len(p) + 1), np.maximum(p, 1e-9), lw=1.5)
        m = out["modalities"][k]
        ax.axvline(m["codes_used"], color="C3", ls="--", lw=1.2,
                   label=f"last used code ({m['codes_used']})")
        ax.axvline(m["perplexity"], color="C2", ls=":", lw=1.4,
                   label=f"perplexity ({m['perplexity']:.0f})")
        ax.axvline(m["vocab"], color="k", ls="-", lw=1, label=f"vocab ({m['vocab']})")
        ax.set_xlabel("code rank"); ax.set_ylabel("emission frequency")
        ax.set_title(f"{k}\n{100 * m['frac_used']:.1f}% used, "
                     f"proxy said {100 * m['rownorm_proxy_frac']:.1f}%", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(f"A0b: measured code frequencies on {N} real cutouts", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "a0b_code_frequency.png", dpi=150)
    print(f"wrote {RESULTS / 'a0b_code_frequency.json'} and .png")


if __name__ == "__main__":
    main()
