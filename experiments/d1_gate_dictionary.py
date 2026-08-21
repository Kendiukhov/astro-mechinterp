"""
D1 — A dictionary targeted at the detection gate.   (AGENDA.md Thrust D, after D2c)

Not a feature atlas. Wu & Walmsley (arXiv:2510.23749) already own SAE-on-galaxy-morphology
and already ran the PCA comparison, so a second atlas is incremental. D2c gave us something
better: a specific causal phenomenon with a measured baseline to beat.

  D2c result: the "there is a source here" gate is PARTIALLY one direction. A single
  difference-in-means vector recovers ~46% of the flux collapse at alpha=4 against a 5%
  matched-norm control, but 22 dimensions carry 90% of the residual object-to-object
  variation. That residual is what a dictionary is for.

PRE-REGISTERED SUCCESS CRITERION (fixed before looking at the result):
  a small set of dictionary features, used as a steering vector at matched intervention
  norm, must recover MORE of the flux collapse than the difference-in-means direction.
  If it does not, the honest conclusion is that the dictionary adds nothing here, and that
  is the result we report.

Baselines, both mandatory:
  - difference-in-means (D2c)
  - PCA at matched sparsity, and PCA top-k directions used for steering the same way

Everything is compared at MATCHED INTERVENTION NORM, because D2c showed that a large enough
random vector recovers 66% on its own. Norm, not method, drives most of the naive effect.

Run:  python experiments/d1_gate_dictionary.py [tokens.npz]
Out:  results/d1_gate_dictionary.json, results/d1_gate_dictionary.png
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
from astron_mi.hooks import Steer
from astron_mi.sae import PCABaseline, train_sae

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
LAYER = 6
N_FEATURES = 8192
K = 32
STEPS = 4000
N_EVAL = 192               # objects for the causal test
ALPHAS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
N_GATE_FEATURES = 16       # how many top gate features to combine into a steering vector


def main():
    from aion.modalities import LegacySurveyFluxG, LegacySurveySegmentationMap

    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        RESULTS.glob("tokens_L*_N*.npz"))[-1]
    d = np.load(path, allow_pickle=True)
    Xr = d["acts_real"].astype("float32")
    Xw = d["acts_wrong"].astype("float32")
    mod = d["modality"].astype(str)
    print(f"cache: {path.name}   real {Xr.shape}  wrong {Xw.shape}")
    print(f"  rows: " + ", ".join(f"{m} {int((mod == m).sum())}" for m in np.unique(mod)))

    # -------------------------------------------------------------- train the SAE
    # Train on BOTH conditions pooled: the dictionary must be able to represent gated and
    # ungated states, otherwise "which features move" is not a meaningful question.
    X = np.concatenate([Xr, Xw])
    print(f"\ntraining BatchTopK SAE: {N_FEATURES} features, k={K}, on {X.shape[0]} rows")
    sae, info = train_sae(X, n_features=N_FEATURES, k=K, steps=STEPS,
                          device=DEVICE, seed=SEED)

    # ----------------------------------------------------------------- baselines
    print("\nfitting PCA baseline (matched sparsity)")
    sub = X[np.random.default_rng(SEED).choice(len(X), min(40000, len(X)), replace=False)]
    pca = PCABaseline(sub, n_components=256)
    scale = sae.scale

    with torch.no_grad():
        xs = torch.from_numpy(sub[:8192] / scale).to(DEVICE)
        xh, f = sae(xs)
        fvu_sae = float(((xh - xs) ** 2).sum() / ((xs - xs.mean(0)) ** 2).sum())
    fvu_pca = pca.fvu(sub[:8192], K)
    print(f"  reconstruction FVU at k={K}:  SAE {fvu_sae:.4f}   PCA {fvu_pca:.4f}   "
          f"({'SAE better' if fvu_sae < fvu_pca else 'PCA BETTER -- dictionary not earning its keep'})")

    # ------------------------------------------------- which features are the gate?
    # Encode in CHUNKS. The full matrix would be 288k x 8192 floats (~9.4 GB) and the
    # batch-topk would run over 2.4e9 elements. Chunk size matches the training batch so
    # the BatchTopK threshold is computed over a comparable population.
    CH = 4096
    n_stat = min(120000, len(Xr))
    pick = np.random.default_rng(SEED).choice(len(Xr), n_stat, replace=False)

    def feature_stats(Xarr):
        s_sum = np.zeros(N_FEATURES, dtype="float64")
        s_cnt = np.zeros(N_FEATURES, dtype="float64")
        for i in range(0, n_stat, CH):
            chunk = Xarr[pick[i:i + CH]] / scale
            with torch.no_grad():
                f = sae.encode(torch.from_numpy(chunk).to(DEVICE)).cpu().numpy()
            s_sum += f.sum(0)
            s_cnt += (f > 0).sum(0)
        return s_sum / n_stat, s_cnt / n_stat

    mean_r, rate_r = feature_stats(Xr)
    mean_w, rate_w = feature_stats(Xw)
    delta = mean_r - mean_w
    alive = (rate_r + rate_w) > 1e-4
    order = np.argsort(-np.abs(delta) * alive)
    top = order[:N_GATE_FEATURES]
    print(f"\n  live features: {int(alive.sum())}/{N_FEATURES}")
    print(f"  top gate features (by mean activation difference real - displaced):")
    for i in top[:8]:
        print(f"    #{i:5d}  delta {delta[i]:+.4f}   fire rate {rate_r[i]:.3f} -> {rate_w[i]:.3f}")

    # steering vector from the dictionary: decoder rows weighted by their activation gap
    W = sae.W_dec.detach().cpu().numpy()
    v_sae = (delta[top][:, None] * W[top]).sum(0)
    v_sae = torch.from_numpy(v_sae * scale).float()

    # difference-in-means baseline, computed on exactly the same rows
    v_dim = torch.from_numpy((Xr[pick] - Xw[pick]).mean(0)).float()
    # PCA baseline: same construction, top-K PCs weighted by their real-vs-wrong gap
    cr, cw = pca.encode(Xr[pick], K), pca.encode(Xw[pick], K)
    gap = cr.mean(0) - cw.mean(0)
    sel = np.argsort(-np.abs(gap))[:N_GATE_FEATURES]
    v_pca = torch.from_numpy((gap[sel][:, None] * pca.components[sel]).sum(0)).float()
    rnd = torch.randn(len(v_dim), generator=torch.Generator().manual_seed(SEED))

    vecs = {"sae_gate_features": v_sae, "difference_in_means": v_dim,
            "pca_topk": v_pca, "random": rnd}
    # MATCHED NORM: every vector rescaled to the difference-in-means norm, so alpha means
    # the same intervention size for all methods.
    ref_norm = v_dim.norm()
    vecs = {k: v / v.norm() * ref_norm for k, v in vecs.items()}
    print(f"\n  all steering vectors matched to ||d_diff-in-means|| = {float(ref_norm):.3f}")
    print(f"  cos(sae, diff-in-means) = "
          f"{float(torch.dot(vecs['sae_gate_features'], vecs['difference_in_means']) / ref_norm**2):+.3f}")

    # ------------------------------------------------------------- causal test
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device=DEVICE)
    T = P.tables()

    def seg(binary):
        t = cm.encode(LegacySurveySegmentationMap(field=binary.clone().to(DEVICE)))
        return {k: v.reshape(v.shape[0], -1) for k, v in t.items()}

    def flux(tok, vec=None, alpha=0.0):
        tk = {k: v.to(DEVICE) for k, v in tok.items()}
        if vec is None:
            o = model.forward(tk, target_modality=[LegacySurveyFluxG])
        else:
            with Steer(model, vec, alpha=alpha, layers=[LAYER]):
                o = model.forward(tk, target_modality=[LegacySurveyFluxG])
        return T.expected_value(o["tok_flux_g"][:, 0, :].detach().float().cpu().numpy(),
                                "tok_flux_g")

    rng = np.random.default_rng(SEED + 1)      # different objects from the training cache
    toks_w, f_real, f_wrong = [], [], []
    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), N_EVAL, replace=False))
        for s in range(0, N_EVAL, 16):
            b = r.aion_batch(rows[s:s + 16], with_segmap=True)
            ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            binary = (b["segmap"] > 0).float()
            wrong = torch.roll(binary, shifts=(25, 25), dims=(-2, -1))
            tw = {**ti, **seg(wrong)}
            toks_w.append({k: v.cpu() for k, v in tw.items()})
            f_real.append(flux({**ti, **seg(binary)}))
            f_wrong.append(flux(tw))
    base, target = float(np.median(np.concatenate(f_wrong))), float(np.median(np.concatenate(f_real)))
    span = target - base
    print(f"\n  held-out objects: displaced {base:.3f} -> real {target:.3f}")

    curves = {}
    print(f"\n{'alpha':>6s} " + " ".join(f"{k:>22s}" for k in vecs))
    for alpha in ALPHAS:
        row = []
        for name, v in vecs.items():
            vals = np.concatenate([flux(t, v, alpha) for t in toks_w])
            rec = (float(np.median(vals)) - base) / span
            curves.setdefault(name, []).append(rec)
            row.append(rec)
        print(f"{alpha:6.1f} " + " ".join(f"{100 * v:21.1f}%" for v in row))

    # ------------------------------------------------------------------ verdict
    best = {k: (ALPHAS[int(np.argmax(v))], float(np.max(v))) for k, v in curves.items()}
    sae_a, sae_r = best["sae_gate_features"]
    dim_a, dim_r = best["difference_in_means"]
    out = {
        "cache": path.name, "layer": LAYER, "n_features": N_FEATURES, "k": K,
        "n_gate_features": N_GATE_FEATURES,
        "reconstruction": {"fvu_sae": fvu_sae, "fvu_pca_matched_k": fvu_pca,
                           "live_features": int(alive.sum())},
        "top_gate_features": [int(i) for i in top],
        "steering_curves": {k: [float(x) for x in v] for k, v in curves.items()},
        "alphas": ALPHAS, "best": {k: {"alpha": a, "recovered": r} for k, (a, r) in best.items()},
        "verdict": {
            "sae_beats_diff_in_means": bool(sae_r > dim_r + 0.05),
            "margin_pp": float(100 * (sae_r - dim_r)),
        },
    }
    print(f"\nPRE-REGISTERED TEST: does the dictionary beat difference-in-means at matched norm?")
    for k, (a, r) in best.items():
        print(f"  {k:22s} best {100 * r:6.1f}%  at alpha={a}")
    print(f"  margin over difference-in-means: {out['verdict']['margin_pp']:+.1f} pp"
          f"  -> {'DICTIONARY WINS' if out['verdict']['sae_beats_diff_in_means'] else 'dictionary does NOT beat the simple baseline'}")

    with open(RESULTS / "d1_gate_dictionary.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    ax = axes[0]
    ax.plot(info["fvu"], lw=1.6, label="SAE")
    ax.axhline(fvu_pca, color="C3", ls="--", lw=1.4, label=f"PCA at k={K}")
    ax.set_xlabel(f"training checkpoint (every 500 steps)"); ax.set_ylabel("fraction of variance unexplained")
    ax.set_title("Reconstruction vs the matched-sparsity baseline", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    o = np.argsort(-np.abs(delta))[:400]
    ax.plot(np.abs(delta[o]), lw=1.4)
    ax.set_yscale("log"); ax.set_xlabel("feature rank"); ax.set_ylabel("|mean activation gap|")
    ax.set_title("How concentrated is the gate in feature space?", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for k, v in curves.items():
        ax.plot(ALPHAS, [100 * x for x in v], "o-", lw=1.6, ms=4, label=k)
    ax.axhline(100, color="k", ls=":", lw=1, label="full recovery")
    ax.set_xlabel(r"$\alpha$ (matched norm)"); ax.set_ylabel("% of flux collapse recovered")
    ax.set_title("Causal test, all methods at matched norm", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("D1: a dictionary targeted at AION's detection gate", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "d1_gate_dictionary.png", dpi=150)
    print(f"wrote {RESULTS / 'd1_gate_dictionary.json'} and .png")


if __name__ == "__main__":
    main()
