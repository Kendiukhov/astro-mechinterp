"""
D1b — Is the sparse-dictionary null a property of SAEs, or of one under-tuned SAE?

D1 trained a single BatchTopK SAE (8192 features, k=32, seed 0) and found that a steering
vector built from its gate-selective features fails a causal test that difference-in-means
passes. The obvious referee objection is that this shows an under-tuned dictionary rather than
a limitation of dictionary learning, and it is a fair objection: one point in hyperparameter
space is not evidence about a method.

So this sweeps the dictionary: width x sparsity x seed, every configuration through the SAME
causal test, against the SAME baselines, at matched intervention norm.

PROTOCOL FIDELITY
-----------------
This experiment varies ONLY (n_features, k, seed). Everything else is held identical to D1,
because a sweep that also changes the protocol cannot be compared with the result it is meant
to test. In particular:

  * Dictionaries are trained on the POOLED activations, concat([Xr, Xw]) -- both the gated and
    ungated states. D1's reasoning applies unchanged: a dictionary that has never seen the
    gated state cannot sensibly be asked which of its features represent it, and testing such a
    dictionary would answer a weaker question than the referee asked.
  * The normaliser is `sae.scale`, the SAE's own internal one, used both to encode and to map
    the steering vector back into activation units.
  * PCA is fit on 40000 rows of the same pooled matrix, and its steering vector is rebuilt at
    each sparsity so that "matched sparsity" stays true as k varies. A PCA baseline pinned at
    k=32 while the SAE runs at k=16 or k=64 is not a matched baseline.
  * Gate features are ranked by |delta| masked by the fire-rate liveness test, exactly as D1.

An earlier version of this script trained on Xr alone, which handicapped every dictionary in
exactly the way the referee objection alleges; its results were discarded rather than reported.

Design notes
------------
* Baselines independent of the SAE (difference-in-means, random) are computed once; the PCA
  baseline once per distinct k. A batched forward over the 192 held-out objects costs ~24 s, so
  each extra (vector, alpha) pair is 24 s of wall clock.
* Every steering vector is rescaled to ||v_diff-in-means||, so alpha means the same
  intervention size for every method and the comparison is norm-matched.
* Recovery is (median flux under steering - median flux under displaced mask) / (median flux
  under true mask - median flux under displaced mask). Identical definition to D1.
* Evaluation objects are drawn with seed SEED+1, disjoint from the activation cache the
  dictionaries are trained on, so nothing is evaluated in-sample.
* Checkpointed after every configuration: this is a multi-hour job that has been interrupted.

Run:  python experiments/d1b_sae_sweep.py [tokens_cache.npz]
Out:  results/d1b_sae_sweep.json
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import astron_mi as A
from astron_mi import physics as P
from astron_mi.hooks import Steer
from astron_mi.sae import PCABaseline, train_sae

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
CKPT = RESULTS / "d1b_sae_sweep_partial.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
LAYER = 6
STEPS = 4000
N_EVAL = 192
N_GATE_FEATURES = 16
CH = 4096                    # encode chunk, matches D1 (and the training batch)
N_STAT = 120_000             # rows used for feature statistics, matches D1

WIDTHS = [2048, 8192, 32768]        # 2.7x to 43x overcomplete; brackets D1's 8192
KS = [16, 32, 64]                   # brackets D1's 32
SEEDS = [0, 1, 2]
CONFIGS = ([(w, k, 0) for w in WIDTHS for k in KS]
           + [(w, 32, sd) for w in WIDTHS for sd in SEEDS[1:]])
ALPHAS_BASE = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
ALPHAS_SWEEP = [2.0, 4.0, 6.0, 8.0]
ALPHA_STAR = 4.0


def main():
    from aion.modalities import LegacySurveyFluxG, LegacySurveySegmentationMap

    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        RESULTS.glob("tokens_L*_N*.npz"))[-1]
    print(f"activations: {path.name}", flush=True)
    d = np.load(path, allow_pickle=True)
    Xr_l = d["acts_real"].astype("float32")
    Xw_l = d["acts_wrong"].astype("float32")
    n = len(Xr_l)
    X = np.concatenate([Xr_l, Xw_l])          # POOLED -- D1 protocol
    del Xr_l, Xw_l
    Xr, Xw = X[:n], X[n:]                     # views into the pooled matrix, no copy
    print(f"  {n} paired rows, d={X.shape[1]}; training on {len(X)} pooled rows", flush=True)

    rng0 = np.random.default_rng(SEED)
    pick = rng0.choice(n, min(N_STAT, n), replace=False)

    # ---------------------------------------------------------------- baselines
    sub = X[np.random.default_rng(SEED).choice(len(X), min(40000, len(X)), replace=False)]
    pca = PCABaseline(sub, n_components=256)
    v_dim = torch.from_numpy((Xr[pick] - Xw[pick]).mean(0)).float()
    ref_norm = v_dim.norm()
    v_rnd = torch.randn(len(v_dim), generator=torch.Generator().manual_seed(SEED))

    def pca_vec(k):
        """PCA steering vector at sparsity k -- rebuilt per k so the match to the SAE holds."""
        cr, cw = pca.encode(Xr[pick], k), pca.encode(Xw[pick], k)
        gap = cr.mean(0) - cw.mean(0)
        sel = np.argsort(-np.abs(gap))[:N_GATE_FEATURES]
        return torch.from_numpy((gap[sel][:, None] * pca.components[sel]).sum(0)).float()

    def matched(v):
        return v / v.norm() * ref_norm

    print(f"  matched norm = {float(ref_norm):.3f}", flush=True)

    # ---------------------------------------------------------------- causal harness
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device=DEVICE)
    T = P.tables()

    def seg(binary):
        t = cm.encode(LegacySurveySegmentationMap(field=binary.clone().to(DEVICE)))
        return {kk: v.reshape(v.shape[0], -1) for kk, v in t.items()}

    def flux(tok, vec=None, alpha=0.0):
        tk = {kk: v.to(DEVICE) for kk, v in tok.items()}
        if vec is None:
            o = model.forward(tk, target_modality=[LegacySurveyFluxG])
        else:
            with Steer(model, vec, alpha=alpha, layers=[LAYER]):
                o = model.forward(tk, target_modality=[LegacySurveyFluxG])
        return T.expected_value(o["tok_flux_g"][:, 0, :].detach().float().cpu().numpy(),
                                "tok_flux_g")

    rng = np.random.default_rng(SEED + 1)
    toks_w, f_real, f_wrong = [], [], []
    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), N_EVAL, replace=False))
        for s in range(0, N_EVAL, 16):
            b = r.aion_batch(rows[s:s + 16], with_segmap=True)
            ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            binary = (b["segmap"] > 0).float()
            tw = {**ti, **seg(torch.roll(binary, shifts=(25, 25), dims=(-2, -1)))}
            toks_w.append({kk: v.cpu() for kk, v in tw.items()})
            f_real.append(flux({**ti, **seg(binary)}))
            f_wrong.append(flux(tw))
    base = float(np.median(np.concatenate(f_wrong)))
    target = float(np.median(np.concatenate(f_real)))
    span = target - base
    print(f"  held-out {N_EVAL} objects: displaced {base:.3f} -> real {target:.3f}\n", flush=True)

    def recovery(vec, alpha):
        vals = np.concatenate([flux(t, vec, alpha) for t in toks_w])
        return (float(np.median(vals)) - base) / span

    # ---------------------------------------------------------------- resume
    ckpt = {}
    if CKPT.exists():
        try:
            ckpt = json.loads(CKPT.read_text())
            if ckpt.get("protocol") != "pooled-v2":
                print("  checkpoint is from the superseded protocol; discarding", flush=True)
                ckpt = {}
            else:
                print(f"  resuming: {len(ckpt.get('configs', []))} configs done", flush=True)
        except json.JSONDecodeError:
            ckpt = {}

    def save_ckpt(bc, pc, done):
        CKPT.write_text(json.dumps({"protocol": "pooled-v2", "baseline_curves": bc,
                                    "pca_curves": pc, "configs": done}, indent=1))

    base_curves = ckpt.get("baseline_curves")
    pca_curves = ckpt.get("pca_curves")
    if not base_curves:
        print("baseline curves", flush=True)
        base_curves = {}
        for name, v in (("difference_in_means", v_dim), ("random", v_rnd)):
            base_curves[name] = [recovery(matched(v), a) for a in ALPHAS_BASE]
            print(f"  {name:20s} " + " ".join(f"{100*x:6.1f}" for x in base_curves[name]),
                  flush=True)
        pca_curves = {}
        for k in KS:
            pca_curves[str(k)] = [recovery(matched(pca_vec(k)), a) for a in ALPHAS_BASE]
            print(f"  {'pca k=%d' % k:20s} "
                  + " ".join(f"{100*x:6.1f}" for x in pca_curves[str(k)]), flush=True)
        save_ckpt(base_curves, pca_curves, [])
    else:
        print("baseline curves (from checkpoint)", flush=True)

    # ---------------------------------------------------------------- the sweep
    print(f"\nsweeping {len(CONFIGS)} dictionaries, trained on the pooled matrix\n", flush=True)
    out = list(ckpt.get("configs", []))
    done = {(c["width"], c["k"], c["seed"]) for c in out}
    t0 = time.time()
    for idx, (w, k, sd) in enumerate(CONFIGS, 1):
        if (w, k, sd) in done:
            print(f"  [{idx:2d}/{len(CONFIGS)}] w={w:5d} k={k:2d} s={sd}  cached", flush=True)
            continue
        ts = time.time()
        sae, info = train_sae(X, n_features=w, k=k, steps=STEPS,
                              device=DEVICE, seed=sd, verbose=False)
        sc = sae.scale

        def feature_stats(Xarr):
            s_sum = np.zeros(w, dtype="float64")
            s_cnt = np.zeros(w, dtype="float64")
            for i0 in range(0, len(pick), CH):
                chunk = Xarr[pick[i0:i0 + CH]] / sc
                with torch.no_grad():
                    f = sae.encode(torch.from_numpy(chunk).to(DEVICE)).cpu().numpy()
                s_sum += f.sum(0)
                s_cnt += (f > 0).sum(0)
            return s_sum / len(pick), s_cnt / len(pick)

        mean_r, rate_r = feature_stats(Xr)
        mean_w, rate_w = feature_stats(Xw)
        delta = mean_r - mean_w
        alive = (rate_r + rate_w) > 1e-4
        top = np.argsort(-np.abs(delta) * alive)[:N_GATE_FEATURES]
        W = sae.W_dec.detach().cpu().numpy()
        v = matched(torch.from_numpy((delta[top][:, None] * W[top]).sum(0) * sc).float())

        with torch.no_grad():
            xs = torch.from_numpy(sub[:8192] / sc).to(DEVICE)
            o = sae(xs)
            xh = o[0] if isinstance(o, tuple) else o
            fvu = float(((xh - xs) ** 2).sum() / ((xs - xs.mean(0)) ** 2).sum())

        curve = {f"{a}": recovery(v, a) for a in ALPHAS_SWEEP}
        rec = {"width": w, "k": k, "seed": sd, "fvu": fvu,
               "fvu_pca_matched_k": float(pca.fvu(sub[:8192], k)),
               "frac_alive": float(alive.mean()),
               "cos_with_dim": float(torch.dot(v, matched(v_dim)) / ref_norm ** 2),
               "recovery": curve, "recovery_at_alpha_star": curve[f"{ALPHA_STAR}"]}
        out.append(rec)
        del sae, W
        if DEVICE == "mps":
            torch.mps.empty_cache()
        save_ckpt(base_curves, pca_curves, out)
        print(f"  [{idx:2d}/{len(CONFIGS)}] w={w:5d} k={k:2d} s={sd}  fvu {fvu:.3f} "
              f"(pca {rec['fvu_pca_matched_k']:.3f})  alive {100*rec['frac_alive']:5.1f}%  "
              f"rec@a{ALPHA_STAR:.0f} {100*rec['recovery_at_alpha_star']:6.1f}%  "
              f"({time.time()-ts:.0f}s, {(time.time()-t0)/60:.0f}m)", flush=True)

    # ---------------------------------------------------------------- verdict
    i_star = ALPHAS_BASE.index(ALPHA_STAR)
    dim_star = base_curves["difference_in_means"][i_star]
    rnd_star = base_curves["random"][i_star]
    sae_star = np.array([c["recovery_at_alpha_star"] for c in out])
    pca_star = {k: pca_curves[str(k)][i_star] for k in KS}
    beat_pca = sum(1 for c in out if c["recovery_at_alpha_star"] > pca_star[c["k"]])
    res = {
        "protocol": "pooled-v2", "n_eval_objects": N_EVAL, "layer": LAYER,
        "alpha_star": ALPHA_STAR, "matched_norm": float(ref_norm),
        "flux_displaced": base, "flux_true": target,
        "alphas_baseline": ALPHAS_BASE, "alphas_sweep": ALPHAS_SWEEP,
        "baseline_curves": base_curves, "pca_curves": pca_curves,
        "baselines_at_alpha_star": {"difference_in_means": dim_star, "random": rnd_star,
                                    "pca_by_k": pca_star},
        "grid": {"widths": WIDTHS, "ks": KS, "seeds": SEEDS, "n_configs": len(CONFIGS)},
        "configs": out,
        "summary": {
            "sae_recovery_at_alpha_star": {
                "min": float(sae_star.min()), "median": float(np.median(sae_star)),
                "max": float(sae_star.max()), "std": float(sae_star.std())},
            "n_configs_beating_random": int((sae_star > rnd_star).sum()),
            "n_configs_beating_pca_matched_k": int(beat_pca),
            "n_configs_beating_diff_in_means": int((sae_star > dim_star).sum()),
        },
        "verdict": (f"across {len(out)} dictionaries the SAE steering vector recovers "
                    f"{100*sae_star.min():.1f}-{100*sae_star.max():.1f}% at alpha={ALPHA_STAR} "
                    f"(random {100*rnd_star:.1f}%, diff-in-means {100*dim_star:.1f}%); "
                    f"{int((sae_star > dim_star).sum())}/{len(out)} beat difference-in-means"),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "d1b_sae_sweep.json").write_text(json.dumps(res, indent=1))
    print("\n" + res["verdict"], flush=True)


if __name__ == "__main__":
    main()
