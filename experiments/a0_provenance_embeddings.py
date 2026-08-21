"""
A0 — Shared-codebook, two-table provenance test.   (AGENDA.md, Thrust A)

AION gives HSC and Legacy Survey imaging ONE shared codec and ONE shared 4375-code
vocabulary, but TWO independent 4375x768 encoder embedding tables (`tok_image` for
Legacy, `tok_image_hsc` for HSC). Same structure for DESI vs SDSS spectra over a
shared 1024-code vocabulary.

Because the codebook index means the same physical reconstruction in both surveys,
any difference between the two embedding tables is pure learned provenance.

Question: does AION treat a shared codebook index as meaning the same thing
regardless of which telescope produced it?

  - Delta = E_b - E_a (near) rank-1 with a constant left singular vector
        => provenance is an ADDITIVE TAG; erasure/projection is well posed.
  - Delta high-rank and dependent on the FSQ lattice coordinates
        => SURVEY-CONDITIONED SEMANTICS; no single direction removes it.

CONFOUND handled here: most of the image codebook is never used, so its embedding
rows are weight-decayed near zero and carry no signal. Comparing them measures
nothing. We therefore estimate per-code usage from the row norm and restrict all
alignment statistics to live codes.

Controls (AGENDA.md 11):
  - random-init at matched std              -> null spectrum
  - index-permuted table                    -> is alignment index-specific?
  - anisotropy-corrected alignment          -> matched cosine vs shuffled cosine
  - CDF-scalar modalities as a usage control-> codecs that MUST be fully used

Run:  python experiments/a0_provenance_embeddings.py
Out:  results/a0_provenance.json, results/a0_provenance.png, results/a0_usage.png
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 0
INIT_STD = 0.02  # fourm/encoder_embeddings.py: nn.init.normal_(..., std=0.02)

# (label, reference table, comparison table, FSQ levels or None)
PAIRS = [
    ("spectrum_DESI_vs_SDSS", "tok_spectrum_sdss", "tok_spectrum_desi", None),
    ("image_HSC_vs_Legacy", "tok_image", "tok_image_hsc", [7, 5, 5, 5, 5]),
]

# Modalities whose codec makes every code (near-)equiprobable by construction.
# These are the positive control for the "dead row" reading of small row norms.
USAGE_CONTROL = [
    ("tok_ebv", "CDF scalar"),
    ("tok_flux_g", "CDF scalar"),
    ("tok_shape_r", "log-CDF scalar"),
    ("tok_parallax", "CDF scalar"),
    ("tok_ra", "CDF scalar"),
    ("tok_xp_bp", "Gaia XP"),
    ("tok_z", "LINEAR grid (not CDF)"),
    ("tok_segmap", "FSQ 1000"),
    ("tok_spectrum_sdss", "LFQ 1024"),
    ("tok_spectrum_desi", "LFQ 1024"),
    ("tok_image", "FSQ 4375 (Legacy)"),
    ("tok_image_hsc", "FSQ 4375 (HSC)"),
]


# ------------------------------------------------------------------ usage proxy
def live_mask(norms: np.ndarray) -> tuple[np.ndarray, float]:
    """
    A never-used embedding row is pulled toward zero by weight decay, ending up
    BELOW its initialisation norm (init_std * sqrt(D)). Live rows sit far above.
    Threshold at half the init norm; the distribution is strongly bimodal so the
    exact cut does not matter (verified by threshold sweep in the JSON).
    """
    d = 768
    init_norm = INIT_STD * np.sqrt(d)
    thr = 0.5 * init_norm
    return norms > thr, float(thr)


# ---------------------------------------------------------------------- metrics
def spectrum_stats(delta: np.ndarray) -> dict:
    s = np.linalg.svd(delta, compute_uv=False)
    energy = s**2
    total = energy.sum()
    frac = energy / total
    cum = np.cumsum(frac)
    p = frac[frac > 0]
    return {
        "rank1_frac": float(frac[0]),
        "rank5_frac": float(frac[:5].sum()),
        "n_dims_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "participation_ratio": float((total**2) / (energy**2).sum()),
        "effective_rank": float(np.exp(-(p * np.log(p)).sum())),
        "singular_values": s[:64].tolist(),
        "frobenius_norm": float(np.sqrt(total)),
    }


def additive_tag_frac(delta: np.ndarray) -> float:
    """Fraction of ||Delta||^2 explained by ONE constant offset (Delta ~ 1 mu^T).
    Stricter than rank-1, which lets the per-code coefficient vary."""
    mu = delta.mean(axis=0, keepdims=True)
    return float((delta.shape[0] * (mu**2).sum()) / (delta**2).sum())


def alignment(a: np.ndarray, b: np.ndarray, rng) -> dict:
    """Per-code cosine vs an index-shuffled null. The null is NOT zero when the
    embeddings are anisotropic, so the shuffled value is the baseline to beat."""
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    matched = (an * bn).sum(1)
    nulls = [
        np.median((an * bn[rng.permutation(len(bn))]).sum(1)) for _ in range(20)
    ]
    return {
        "cos_matched_median": float(np.median(matched)),
        "cos_shuffled_median": float(np.mean(nulls)),
        "cos_shuffled_std": float(np.std(nulls)),
        "excess_over_null": float(np.median(matched) - np.mean(nulls)),
        "frac_cos_above_half": float(np.mean(matched > 0.5)),
    }


def map_residuals(a: np.ndarray, b: np.ndarray, rng) -> dict:
    """How much of b is a linear/orthogonal image of a? Held-out split, because a
    768x768 free map on ~400 rows would otherwise fit noise perfectly."""
    n = len(a)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n) :]
    a_mu, b_mu = a[tr].mean(0, keepdims=True), b[tr].mean(0, keepdims=True)
    ac, bc = a - a_mu, b - b_mu

    u, _, vt = np.linalg.svd(ac[tr].T @ bc[tr], full_matrices=False)
    r = u @ vt
    unex_orth = float(((ac[te] @ r - bc[te]) ** 2).sum() / (bc[te] ** 2).sum())

    # ridge-regularised general linear map (needed when n < D)
    lam = 1e-2 * np.trace(ac[tr].T @ ac[tr]) / ac.shape[1]
    w = np.linalg.solve(
        ac[tr].T @ ac[tr] + lam * np.eye(ac.shape[1]), ac[tr].T @ bc[tr]
    )
    unex_lin = float(((ac[te] @ w - bc[te]) ** 2).sum() / (bc[te] ** 2).sum())

    alpha = float((ac[tr] * bc[tr]).sum() / (ac[tr] ** 2).sum())
    unex_scaled = float(
        ((alpha * ac[te] - bc[te]) ** 2).sum() / (bc[te] ** 2).sum()
    )
    return {
        "heldout_unexplained_after_orthogonal": unex_orth,
        "heldout_unexplained_after_ridge_linear": unex_lin,
        "heldout_unexplained_after_scaled_identity": unex_scaled,
        "best_scalar_alpha": alpha,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }


def fsq_lattice(levels: list[int]) -> np.ndarray:
    """Index -> FSQ lattice coordinate. Pure geometry, no learned parameters."""
    from aion.codecs.quantizers import FiniteScalarQuantizer

    q = FiniteScalarQuantizer(levels=levels)
    with torch.no_grad():
        codes = q.decode(torch.arange(int(np.prod(levels))))
    return codes.reshape(int(np.prod(levels)), -1).double().numpy()


def lattice_regression(delta: np.ndarray, lat: np.ndarray, rng) -> dict:
    """Is the survey difference a smooth function of the codebook's own geometry?
    Held-out R^2 of Delta regressed on lattice coords (linear+quadratic+pairwise)."""
    if lat is None:
        return {"available": False}
    x = (lat - lat.mean(0)) / (lat.std(0) + 1e-12)
    d = x.shape[1]
    feats = [np.ones((len(x), 1)), x, x**2]
    for i in range(d):
        for j in range(i + 1, d):
            feats.append((x[:, i] * x[:, j])[:, None])
    design = np.concatenate(feats, axis=1)

    y = delta - delta.mean(0, keepdims=True)
    n = len(y)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n) :]
    coef, *_ = np.linalg.lstsq(design[tr], y[tr], rcond=None)
    resid = y[te] - design[te] @ coef
    return {
        "available": True,
        "n_features": int(design.shape[1]),
        "heldout_r2": float(1.0 - (resid**2).sum() / (y[te] ** 2).sum()),
    }


def analyse(name, a, b, lat, variant, rng) -> dict:
    delta = b - a
    mu = delta.mean(0, keepdims=True)
    return {
        "pair": name,
        "variant": variant,
        "n_codes": int(a.shape[0]),
        "dim": int(a.shape[1]),
        "delta_rms": float(np.sqrt((delta**2).mean())),
        "relative_delta_norm": float(
            np.linalg.norm(delta) / np.linalg.norm(0.5 * (a + b))
        ),
        "additive_tag_frac": additive_tag_frac(delta),
        "delta_spectrum": spectrum_stats(delta),
        "deflated_spectrum": spectrum_stats(delta - mu),
        "alignment": alignment(a, b, rng),
        "map_residuals": map_residuals(a, b, rng),
        "lattice_regression_on_deflated": lattice_regression(delta - mu, lat, rng),
    }


# -------------------------------------------------------------------------- main
def main():
    from aion import AION

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print("loading aion-base ...")
    model = AION.from_pretrained("polymathic-ai/aion-base").eval()

    def table(k):
        return model.encoder_embeddings[k].token_emb.weight.detach().double().numpy()

    # ---------------------------------------------------- codebook usage survey
    print(f"\n=== codebook utilisation (proxy: row norm > {0.5 * INIT_STD * np.sqrt(768):.3f}"
          f", half the init norm {INIT_STD * np.sqrt(768):.3f}) ===")
    usage = []
    for key, kind in USAGE_CONTROL:
        n = np.linalg.norm(table(key), axis=1)
        live, thr = live_mask(n)
        usage.append(
            {
                "modality": key,
                "codec": kind,
                "n_codes": int(len(n)),
                "n_live": int(live.sum()),
                "frac_live": float(live.mean()),
                "norm_p01": float(np.percentile(n, 1)),
                "norm_median": float(np.median(n)),
                "norm_p99": float(np.percentile(n, 99)),
                "norm_cv": float(n.std() / n.mean()),
            }
        )
        print(f"  {key:20s} {kind:22s} K={len(n):5d}  live={live.sum():5d} ({live.mean():6.1%})")

    # ------------------------------------------------------------ the A0 test
    results = []
    for name, key_a, key_b, levels in PAIRS:
        ea, eb = table(key_a), table(key_b)
        na, nb = np.linalg.norm(ea, axis=1), np.linalg.norm(eb, axis=1)
        live = live_mask(na)[0] & live_mask(nb)[0]
        lat_full = fsq_lattice(levels) if levels else None

        print(f"\n=== {name}: {key_a} vs {key_b}  {ea.shape} ===")
        print(f"  live in BOTH: {live.sum()} / {len(live)}  ({live.mean():.1%})")
        # containment: is one survey's live set a subset of the other's?
        la, lb = live_mask(na)[0], live_mask(nb)[0]
        print(f"  live_a={la.sum()}  live_b={lb.sum()}  intersection={np.sum(la & lb)}"
              f"  jaccard={np.sum(la & lb) / np.sum(la | lb):.3f}")

        a, b = ea[live], eb[live]
        lat = lat_full[live] if lat_full is not None else None

        trained = analyse(name, a, b, lat, "trained_live", rng)
        trained["live_codes"] = int(live.sum())
        trained["frac_live"] = float(live.mean())
        trained["live_a"] = int(la.sum())
        trained["live_b"] = int(lb.sum())
        trained["live_jaccard"] = float(np.sum(la & lb) / np.sum(la | lb))
        results.append(trained)

        # control: matched-std random tables on the same number of codes
        ra = rng.normal(0, a.std(), size=a.shape)
        rb = rng.normal(0, b.std(), size=b.shape)
        results.append(analyse(name, ra, rb, lat, "random_init", rng))

        # control: index-permuted comparison table
        results.append(
            analyse(name, a, b[rng.permutation(len(b))], lat, "index_permuted", rng)
        )

        # threshold sweep, to show the cut does not drive the answer
        sweep = []
        for mult in [0.4, 0.5, 0.75, 1.0, 2.0, 4.0]:
            t = mult * INIT_STD * np.sqrt(768)
            msk = (na > t) & (nb > t)
            if msk.sum() < 50:
                continue
            sweep.append(
                {
                    "threshold": float(t),
                    "n_live": int(msk.sum()),
                    **alignment(ea[msk], eb[msk], rng),
                }
            )
        trained["threshold_sweep"] = sweep

        t = trained
        print(f"  relative ||delta||          : {t['relative_delta_norm']:.4f}")
        print(f"  additive-tag fraction       : {t['additive_tag_frac']:.4f}")
        print(f"  delta rank-1 fraction       : {t['delta_spectrum']['rank1_frac']:.4f}")
        print(f"  dims for 90% of delta       : {t['delta_spectrum']['n_dims_for_90pct']}")
        print(f"  cos matched / shuffled null : {t['alignment']['cos_matched_median']:+.4f}"
              f" / {t['alignment']['cos_shuffled_median']:+.4f}"
              f"  (excess {t['alignment']['excess_over_null']:+.4f})")
        print(f"  held-out unexpl. after ridge: {t['map_residuals']['heldout_unexplained_after_ridge_linear']:.4f}")
        if t["lattice_regression_on_deflated"].get("available"):
            print(f"  deflated ~ FSQ lattice R^2  : {t['lattice_regression_on_deflated']['heldout_r2']:+.4f}")

    payload = {"usage": usage, "pairs": results}
    with open(RESULTS / "a0_provenance.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {RESULTS / 'a0_provenance.json'}")

    # ------------------------------------------------------------------ figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, *_r) in zip(axes, PAIRS):
        for variant, style in [
            ("trained_live", dict(lw=2.0, color="C0")),
            ("random_init", dict(lw=1.2, color="C3", ls="--")),
            ("index_permuted", dict(lw=1.2, color="C2", ls=":")),
        ]:
            rec = next(r for r in results if r["pair"] == name and r["variant"] == variant)
            s = np.array(rec["delta_spectrum"]["singular_values"])
            ax.semilogy(np.arange(1, len(s) + 1), s / s[0], label=variant, **style)
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.set_xlabel("singular value index")
        ax.set_ylabel(r"$\sigma_i/\sigma_1$ of $\Delta_E$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("A0: is the survey difference between shared-codebook tables low-rank?  (no)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "a0_provenance.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    u = sorted(usage, key=lambda r: r["frac_live"])
    ax.barh(
        [f"{r['modality']}  ({r['codec']})" for r in u],
        [100 * r["frac_live"] for r in u],
        color=["C3" if r["frac_live"] < 0.95 else "C0" for r in u],
    )
    ax.set_xlabel("% of codebook in use (row-norm proxy)")
    ax.set_xlim(0, 105)
    ax.grid(alpha=0.3, axis="x")
    ax.set_title("AION-base codebook utilisation: the image FSQ codebook is mostly dead", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "a0_usage.png", dpi=150)
    print(f"wrote {RESULTS / 'a0_provenance.png'} and {RESULTS / 'a0_usage.png'}")


if __name__ == "__main__":
    main()
