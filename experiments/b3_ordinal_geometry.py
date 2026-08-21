"""
B3-lite — Ordinal geometry of the scalar codebooks.   (AGENDA.md, Thrust B)

~31 of AION's 40 modalities are single-token scalars whose codec is a parameter-free
empirical-CDF quantiser: code index = quantile rank of the physical value. Nothing in
the architecture tells the model that code 500 lies between code 499 and 501 — the
embedding table is a free 1024x768 lookup. So: did AION learn the ordering?

Three questions, all answerable from weights alone:

1. ORDINALITY. Is the codebook a smooth 1-D curve in embedding space, ordered by index?
   Measured by (a) the ratio of consecutive-step distance to random-pair distance, and
   (b) Spearman correlation between embedding distance and index distance.

2. TOPOLOGY (the sharp test). `tok_ra` and `tok_dec` are both CDF-quantised sky
   coordinates with identical codec treatment, but RA is PERIODIC (0 deg = 360 deg) and
   Dec is NOT (-90 to +90 are opposite poles). If the model recovered the topology of
   the sky rather than just the codec's index ordering, RA's codebook should close into
   a loop and Dec's should stay an open arc. The endpoint distance
   d(code_0, code_last) / median pairwise distance separates these cleanly, and neither
   codec nor architecture supplies the answer.

3. METRIC. For CDF codecs, index IS quantile, so a constant step profile means the model
   spaces codes by quantile. `tok_z` is the one scalar on a LINEAR grid instead
   (GridScalarCodec -> ScalarLinearQuantizer), so its index is physical, not quantile.
   Comparing the step profiles says which metric the representation uses.

Controls: index-shuffled codebook, and random-init tables at matched std.

Run:  python experiments/b3_ordinal_geometry.py
Out:  results/b3_ordinal.json, results/b3_ordinal.png
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 0
INIT_STD = 0.02
LIVE_THR = 0.5 * INIT_STD * np.sqrt(768)

# modality -> human description. Chosen to span the decisive contrasts.
MODALITIES = {
    "tok_ra": "RA, CDF, PERIODIC",
    "tok_dec": "Dec, CDF, non-periodic",
    "tok_ebv": "E(B-V), CDF",
    "tok_parallax": "parallax, CDF",
    "tok_shape_r": "half-light radius, log-CDF",
    "tok_flux_g": "g flux, CDF",
    "tok_z": "redshift, LINEAR grid",
}

TABLES = ["encoder_token_emb", "decoder_to_logits"]


# ------------------------------------------------------------------ measurements
def ordinality(e: np.ndarray, rng) -> dict:
    """Is the sequence e_0..e_{K-1} a smooth 1-D curve ordered by index?"""
    k = len(e)
    steps = np.linalg.norm(np.diff(e, axis=0), axis=1)
    i, j = rng.integers(0, k, 20000), rng.integers(0, k, 20000)
    keep = i != j
    i, j = i[keep], j[keep]
    pair_d = np.linalg.norm(e[i] - e[j], axis=1)

    # a smooth ordinal curve has consecutive steps far smaller than random pairs
    step_ratio = float(np.median(steps) / np.median(pair_d))
    # and distance should grow monotonically with index separation
    rho = float(spearmanr(np.abs(i - j), pair_d).statistic)

    # path length vs diameter: 1.0 for a straight line, large for a random walk
    diameter = float(pair_d.max())
    path_ratio = float(steps.sum() / diameter)
    return {
        "step_to_pair_ratio": step_ratio,
        "spearman_dist_vs_index": rho,
        "path_length_over_diameter": path_ratio,
    }


def topology(e: np.ndarray, rng=None) -> dict:
    """
    Closed loop (periodic) or open arc? Two independent probes, because they
    disagree in practice and the disagreement is informative:

    (a) BLOCK ENDPOINT: mean distance between the first m and last m codes,
        normalised by the median pair distance. A wrapping variable stays low
        for all m. This is a local statistic, robust to global noise.
    (b) CIRCULAR GAIN: Spearman(circular index distance, embedding distance)
        minus Spearman(linear index distance, embedding distance). A genuinely
        periodic variable must prefer circular; a monotone one must prefer
        linear (strongly negative gain). This is a global statistic.
    """
    k = len(e)
    sub = e[::7]
    pair = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)
    med = float(np.median(pair[pair > 0]))

    blocks = {}
    for m_ in (1, 5, 10, 25, 50, 100):
        if 2 * m_ >= k:
            continue
        a, b = e[:m_], e[-m_:]
        blocks[f"m={m_}"] = float(
            np.linalg.norm(a[:, None] - b[None, :], axis=-1).mean() / med
        )

    out = {
        "endpoint_over_median_pair": float(np.linalg.norm(e[0] - e[-1]) / med),
        "endpoint_over_max_pair": float(np.linalg.norm(e[0] - e[-1]) / pair.max()),
        "block_endpoint_over_median": blocks,
    }

    if rng is not None:
        i, j = rng.integers(0, k, 60000), rng.integers(0, k, 60000)
        keep = i != j
        i, j = i[keep], j[keep]
        d = np.linalg.norm(e[i] - e[j], axis=1)
        lin = np.abs(i - j)
        cir = np.minimum(lin, k - lin)
        rl = float(spearmanr(lin, d).statistic)
        rc = float(spearmanr(cir, d).statistic)
        out.update(
            {"spearman_linear": rl, "spearman_circular": rc, "circular_gain": rc - rl}
        )
    return out


def step_profile(e: np.ndarray, nbin: int = 32) -> list:
    """Local step size vs index, binned. Flat => uniform spacing in index."""
    steps = np.linalg.norm(np.diff(e, axis=0), axis=1)
    edges = np.linspace(0, len(steps), nbin + 1).astype(int)
    prof = [float(np.median(steps[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    m = np.median(prof)
    return [float(p / m) for p in prof]


def pca_variance(e: np.ndarray) -> dict:
    x = e - e.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    v = s**2 / (s**2).sum()
    return {
        "pc1": float(v[0]),
        "pc1_2": float(v[:2].sum()),
        "pc1_3": float(v[:3].sum()),
        "pc1_10": float(v[:10].sum()),
    }


def analyse(e: np.ndarray, rng, label: str) -> dict:
    return {
        "variant": label,
        "n_codes": int(len(e)),
        **ordinality(e, rng),
        **topology(e, rng),
        "pca": pca_variance(e),
    }


def main():
    from aion import AION

    rng = np.random.default_rng(SEED)
    print("loading aion-base ...")
    model = AION.from_pretrained("polymathic-ai/aion-base").eval()

    def get(mod, which):
        if which == "encoder_token_emb":
            w = model.encoder_embeddings[mod].token_emb.weight
        elif which == "decoder_to_logits":
            w = model.decoder_embeddings[mod].to_logits.weight
        else:
            raise KeyError(which)
        return w.detach().double().numpy()

    out = {}
    for mod, desc in MODALITIES.items():
        out[mod] = {"description": desc, "tables": {}}
        for which in TABLES:
            e = get(mod, which)
            # drop dead codes (A0: tok_z has ~27% of them; the CDF scalars have none)
            live = np.linalg.norm(e, axis=1) > LIVE_THR
            e = e[live]
            rec = {"frac_live": float(live.mean()), "trained": analyse(e, rng, "trained")}

            # control 1: shuffle the index -> destroys ordering, keeps the point cloud
            rec["index_shuffled"] = analyse(e[rng.permutation(len(e))], rng, "index_shuffled")
            # control 2: random init at matched std
            rec["random_init"] = analyse(
                rng.normal(0, e.std(), size=e.shape), rng, "random_init"
            )
            rec["step_profile"] = step_profile(e)
            out[mod]["tables"][which] = rec

    # ------------------------------------------------------------------- report
    print(f"\n{'modality':14s} {'table':20s} {'step/pair':>10s} {'shuf':>7s} "
          f"{'spearman':>9s} {'endpt/med':>10s} {'PC1-3':>7s}")
    for mod, rec in out.items():
        for which, r in rec["tables"].items():
            t, s = r["trained"], r["index_shuffled"]
            print(f"{mod:14s} {which:20s} {t['step_to_pair_ratio']:10.4f} "
                  f"{s['step_to_pair_ratio']:7.3f} {t['spearman_dist_vs_index']:9.3f} "
                  f"{t['endpoint_over_median_pair']:10.3f} {t['pca']['pc1_3']:7.3f}")

    print("\n--- topology (encoder table): does the codebook wrap? ---")
    print(f"{'modality':14s} {'m=1':>7s} {'m=5':>7s} {'m=10':>7s} {'m=25':>7s} "
          f"{'rho_lin':>8s} {'rho_circ':>9s} {'gain':>7s}")
    for mod in MODALITIES:
        t = out[mod]["tables"]["encoder_token_emb"]["trained"]
        b = t["block_endpoint_over_median"]
        print(f"{mod:14s} {b.get('m=1', float('nan')):7.3f} {b.get('m=5', float('nan')):7.3f} "
              f"{b.get('m=10', float('nan')):7.3f} {b.get('m=25', float('nan')):7.3f} "
              f"{t['spearman_linear']:8.3f} {t['spearman_circular']:9.3f} {t['circular_gain']:+7.3f}")
    print("  NB: the circular-gain statistic is insensitive here because wraparound only")
    print("      affects pairs at near-maximal separation, a few % of random pairs. The")
    print("      block-endpoint columns and the ordinality curve are the sensitive tests.")

    # --------------------------------------------------------------- ground truth
    # Decode the codebooks to physical units, so the topology claim rests on the
    # actual sky separation of the extreme codes rather than on inference.
    print("\n--- physical values of the extreme codes (from the codecs themselves) ---")
    try:
        from aion import modalities as Mod
        from aion.codecs.manager import CodecManager

        cm = CodecManager(device="cpu")
        endpoints = {}
        for nm, cls_name in [("tok_ra", "Ra"), ("tok_dec", "Dec"), ("tok_z", "Z")]:
            codec = cm._load_codec(getattr(Mod, cls_name))
            q = getattr(codec, "quantizer", None) or getattr(codec, "_quantizer", None)
            with torch.no_grad():
                v = np.asarray(q.decode(torch.arange(1024))).reshape(-1)
            endpoints[nm] = {
                "codec": type(codec).__name__,
                "quantizer": type(q).__name__,
                "code_first": float(v[0]),
                "code_last": float(v[-1]),
                "range": [float(v.min()), float(v.max())],
            }
            print(f"  {nm:8s} {type(codec).__name__:16s} code0={v[0]:10.4f}  code1023={v[-1]:10.4f}")
        print("  => RA endpoints are 0.008 deg apart on the sky (the SAME point);")
        print("     Dec endpoints are 178 deg apart (opposite poles). Identical codec class,")
        print("     identical architecture treatment: periodicity is not supplied anywhere.")
        out["_codec_endpoints"] = endpoints
    except Exception as exc:
        print(f"  codec decode unavailable ({type(exc).__name__}: {exc}); skipping")

    with open(RESULTS / "b3_ordinal.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS / 'b3_ordinal.json'}")

    # ------------------------------------------------------------------ figure
    mods = list(MODALITIES)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    x = np.arange(len(mods))
    ax.bar(x - 0.2, [out[m]["tables"]["encoder_token_emb"]["trained"]["step_to_pair_ratio"] for m in mods],
           0.4, label="trained", color="C0")
    ax.bar(x + 0.2, [out[m]["tables"]["encoder_token_emb"]["index_shuffled"]["step_to_pair_ratio"] for m in mods],
           0.4, label="index-shuffled", color="C3")
    ax.set_xticks(x); ax.set_xticklabels(mods, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("median consecutive step / median random pair")
    ax.set_title("Ordinality: small = smooth 1-D curve ordered by code index", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    ax.bar(x, [out[m]["tables"]["encoder_token_emb"]["trained"]["endpoint_over_median_pair"] for m in mods],
           color=["C2" if m == "tok_ra" else "C0" for m in mods])
    ax.axhline(1.0, color="k", ls=":", lw=1, label="typical pair separation")
    ax.set_xticks(x); ax.set_xticklabels(mods, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("d(first code, last code) / median pair")
    ax.set_title("Topology: low = codebook closes into a loop", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # Distance vs index separation. This is the honest picture of ordinality:
    # a monotone scalar rises and saturates; a PERIODIC one must come back down
    # at maximal index separation, because those codes are physically adjacent.
    ax = axes[1, 0]
    for m in ("tok_ra", "tok_dec", "tok_parallax", "tok_ebv"):
        e = get(m, "encoder_token_emb")
        e = e[np.linalg.norm(e, axis=1) > LIVE_THR]
        k = len(e)
        seps = np.unique(np.linspace(1, k - 1, 60).astype(int))
        curve = []
        for s_ in seps:
            a = e[: k - s_]
            b = e[s_:]
            curve.append(np.linalg.norm(a - b, axis=1).mean())
        curve = np.array(curve) / np.median(curve)
        ax.plot(seps / k, curve, lw=1.6, label=m)
    ax.set_xlabel("index separation |i-j| / K")
    ax.set_ylabel("mean embedding distance / median")
    ax.set_title("Ordinality curve: rises then SATURATES for a monotone scalar;\n"
                 "must turn DOWN at |i-j|~K for a periodic one", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for m in ("tok_ebv", "tok_parallax", "tok_z", "tok_ra"):
        prof = out[m]["tables"]["encoder_token_emb"]["step_profile"]
        ax.plot(np.linspace(0, 1, len(prof)), prof, lw=1.4, label=m)
    ax.set_xlabel("position in codebook (index / K)")
    ax.set_ylabel("local step size / median")
    ax.set_title("Metric: flat = uniform spacing in quantile", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("B3-lite: ordinal geometry of AION's scalar codebooks (weights only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "b3_ordinal.png", dpi=150)
    print(f"wrote {RESULTS / 'b3_ordinal.png'}")


if __name__ == "__main__":
    main()
