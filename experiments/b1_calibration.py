"""
B1 — Calibration audit of AION's native single-token posteriors.   (AGENDA.md, Thrust B)

The AION paper draws a line: multi-token samples are "not calibrated joint posteriors",
but single-token categorical outputs are treated as legitimate posteriors. The side they
endorse is never tested. Everything in Thrusts C and D reads these posteriors, so this
goes first.

`tok_z` is the ideal target: one token, a 1024-bucket LINEAR grid over z in [0, 6]
(GridScalarCodec + ScalarLinearQuantizer, dz = 0.005865), a 1025th sentinel code with no
value bucket, and PROVABGS gives a trustworthy reference redshift.

Measured across an information ladder (photometry / +imaging / +spectra):
  - PIT histogram + KS distance from uniform. Randomised PIT, because a discrete
    predictor cannot give a uniform PIT otherwise.
  - central-interval coverage vs nominal
  - sharpness (posterior sd) and the error/uncertainty relationship
  - point-estimator comparison: posterior MEAN vs MEDIAN vs MODE. The mean is dragged
    upward by the long z<=6 grid whenever the posterior is broad, so this matters.
  - sentinel mass: how much probability lands on the unreachable 1025th code.

Run:  python experiments/b1_calibration.py [cache.npz]
Out:  results/b1_calibration.json, results/b1_calibration.png
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from astron_mi import physics as P

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
LADDERS = ["phot", "phot_img", "phot_img_spec"]
NICE = {"phot": "photometry", "phot_img": "+ imaging", "phot_img_spec": "+ spectra"}
SEED = 0


def softmax(z):
    z = np.asarray(z, dtype="float64")
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def summarise(logits, vals):
    """Point estimates and spread from the token pmf (sentinel code dropped)."""
    p = softmax(logits)
    sent = p[:, len(vals):].sum(1) if logits.shape[1] > len(vals) else np.zeros(len(p))
    p = p[:, : len(vals)]
    p = p / p.sum(1, keepdims=True)
    mean = p @ vals
    var = p @ (vals**2) - mean**2
    cdf = np.cumsum(p, 1)
    median = vals[np.argmax(cdf >= 0.5, axis=1)]
    mode = vals[np.argmax(p, axis=1)]
    return dict(p=p, mean=mean, median=median, mode=mode,
                sd=np.sqrt(np.clip(var, 0, None)), sentinel=sent, cdf=cdf)


def coverage(cdf, edges, truth, levels):
    """
    Central-interval coverage: fraction of truths inside the [a/2, 1-a/2] interval.

    Must use bucket EDGES, not centres. AION's redshift posterior given spectra is often
    concentrated in a single 0.0059-wide bucket; with centres the interval has zero width
    and coverage collapses to ~0 for a predictor that is in fact excellent. Using edges,
    a one-bucket posterior correctly yields an interval one bucket wide.
    """
    out = []
    for lv in levels:
        lo_i = np.argmax(cdf >= (1 - lv) / 2, axis=1)
        hi_i = np.argmax(cdf >= (1 + lv) / 2, axis=1)
        lo, hi = edges[lo_i], edges[np.minimum(hi_i + 1, len(edges) - 1)]
        out.append(float(np.mean((truth >= lo) & (truth <= hi))))
    return out


def main():
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        RESULTS.glob("cache_N*.npz")
    )[-1]
    d = np.load(path)
    print(f"cache: {path.name}")

    T = P.tables()
    vals = T.index_to_value("tok_z")
    edges = T.bin_edges("tok_z")
    z = d["label.z"].astype("float64")
    ok = np.isfinite(z)
    z = z[ok]
    print(f"n = {len(z)}  z range [{z.min():.4f}, {z.max():.4f}]  "
          f"grid dz = {np.diff(vals).mean():.6f}")

    rng = np.random.default_rng(SEED)
    levels = [0.5, 0.68, 0.90, 0.95, 0.99]
    out: dict = {"cache": path.name, "n": int(len(z)), "levels": levels, "ladders": {}}
    store = {}

    print(f"\n{'ladder':16s} {'KS(PIT)':>9s} {'bias':>9s} {'sd(mean)':>9s} "
          f"{'|err| mean':>11s} {'|err| med':>10s} {'|err| mode':>11s} {'sentinel':>10s}")
    for lad in LADDERS:
        lg = d[f"logits_z.{lad}"].astype("float64")[ok]
        s = summarise(lg, vals)
        store[lad] = s

        pit = T.pit(lg, z, "tok_z", randomized=True, rng=rng)
        pit = np.clip(np.asarray(pit, dtype="float64"), 0, 1)
        ks = float(np.max(np.abs(np.sort(pit) - np.linspace(0, 1, len(pit)))))
        cov = coverage(s["cdf"], edges, z, levels)

        rec = {
            "ks_pit": ks,
            "coverage": dict(zip(map(str, levels), cov)),
            "bias_mean": float(np.mean(s["mean"] - z)),
            "median_abs_err_mean": float(np.median(np.abs(s["mean"] - z))),
            "median_abs_err_median": float(np.median(np.abs(s["median"] - z))),
            "median_abs_err_mode": float(np.median(np.abs(s["mode"] - z))),
            "mean_posterior_sd": float(np.mean(s["sd"])),
            "sentinel_mass_mean": float(np.mean(s["sentinel"])),
            "sentinel_mass_max": float(np.max(s["sentinel"])),
            "err_vs_sd_spearman": float(
                _spearman(np.abs(s["median"] - z), s["sd"])
            ),
        }
        out["ladders"][lad] = rec
        print(f"{NICE[lad]:16s} {ks:9.4f} {rec['bias_mean']:+9.4f} "
              f"{rec['mean_posterior_sd']:9.4f} {rec['median_abs_err_mean']:11.4f} "
              f"{rec['median_abs_err_median']:10.4f} {rec['median_abs_err_mode']:11.4f} "
              f"{rec['sentinel_mass_mean']:10.2e}")

    print(f"\ncentral-interval coverage (nominal -> empirical)")
    print(f"{'ladder':16s} " + " ".join(f"{int(l*100):>7d}%" for l in levels))
    for lad in LADDERS:
        c = out["ladders"][lad]["coverage"]
        print(f"{NICE[lad]:16s} " + " ".join(f"{c[str(l)]:8.3f}" for l in levels))

    # -------------------------------------------------- stratify by design range
    # PROVABGS is DESI BGS, designed for z <~ 0.6. A small high-z tail exists and is
    # where nearly all catastrophic failures live; reporting a single pooled number
    # lets that subpopulation drive the headline. Split it out explicitly.
    print(f"\nstratified by BGS design range (z <= 0.6 vs above)")
    print(f"{'ladder':16s} {'subset':>10s} {'n':>6s} {'outlier%':>9s} {'med|dz|':>9s} "
          + " ".join(f"{int(l*100):>6d}%" for l in levels))
    out["stratified"] = {}
    for lad in LADDERS:
        s = store[lad]
        for tag, m in [("z<=0.6", z <= 0.6), ("z>0.6", z > 0.6)]:
            if m.sum() < 20:
                continue
            err = np.abs(s["median"][m] - z[m])
            cov = coverage(s["cdf"][m], edges, z[m], levels)
            out["stratified"].setdefault(lad, {})[tag] = {
                "n": int(m.sum()),
                "outlier_frac_dz_gt_0p05": float(np.mean(err > 0.05)),
                "median_abs_err": float(np.median(err)),
                "coverage": dict(zip(map(str, levels), cov)),
            }
            print(f"{NICE[lad]:16s} {tag:>10s} {m.sum():6d} "
                  f"{100 * np.mean(err > 0.05):8.2f}% {np.median(err):9.5f} "
                  + " ".join(f"{c:7.3f}" for c in cov))

    # ------------------------------------------------------------------ verdict
    best = out["ladders"]["phot_img_spec"]
    over = np.mean([best["coverage"][str(l)] - l for l in levels])
    out["verdict"] = {
        "mean_coverage_excess_full_ladder": float(over),
        "direction": "over-dispersed (too wide)" if over > 0.02
        else ("under-dispersed (over-confident)" if over < -0.02 else "well calibrated"),
        "mean_is_biased_by_grid": bool(
            best["median_abs_err_mean"] > 1.5 * best["median_abs_err_median"]
        ),
    }
    print(f"\nVERDICT: full ladder mean coverage excess = {over:+.3f} -> {out['verdict']['direction']}")
    if out["verdict"]["mean_is_biased_by_grid"]:
        print("  posterior MEAN is materially worse than the MEDIAN: use the median as the point estimate.")

    with open(RESULTS / "b1_calibration.json", "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    ax = axes[0]
    for lad in LADDERS:
        lg = d[f"logits_z.{lad}"].astype("float64")[ok]
        pit = np.clip(np.asarray(T.pit(lg, z, "tok_z", randomized=True,
                                       rng=np.random.default_rng(SEED))), 0, 1)
        ax.hist(pit, bins=20, histtype="step", lw=1.8, density=True, label=NICE[lad])
    ax.axhline(1.0, color="k", ls=":", lw=1, label="calibrated")
    ax.set_xlabel("PIT"); ax.set_ylabel("density")
    ax.set_title("PIT histogram (flat = calibrated)\nU-shape = over-confident, hump = too wide", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    for lad in LADDERS:
        c = out["ladders"][lad]["coverage"]
        ax.plot(levels, [c[str(l)] for l in levels], marker="o", lw=1.6, label=NICE[lad])
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="ideal")
    ax.set_xlim(0.4, 1.02); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("nominal central interval"); ax.set_ylabel("empirical coverage")
    ax.set_title("Coverage", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    s = store["phot_img_spec"]
    ax.scatter(z, s["median"], s=4, alpha=0.35, label="posterior median")
    ax.scatter(z, s["mean"], s=4, alpha=0.35, label="posterior mean")
    lim = [0, max(1.0, float(np.nanpercentile(z, 99.5)) * 1.5)]
    ax.plot(lim, lim, "k:", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("PROVABGS z"); ax.set_ylabel("AION point estimate")
    ax.set_title("Point estimators, full ladder", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("B1: calibration of AION's native single-token redshift posterior", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "b1_calibration.png", dpi=150)
    print(f"wrote {RESULTS / 'b1_calibration.json'} and .png")


def _spearman(a, b):
    from scipy.stats import spearmanr

    m = np.isfinite(a) & np.isfinite(b)
    return spearmanr(a[m], b[m]).statistic if m.sum() > 10 else float("nan")


if __name__ == "__main__":
    main()
