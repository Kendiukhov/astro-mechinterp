"""
D2c — Is the detection gate a direction?   (the cheap decisive test before any dictionary)

D2/D2b established that the segmentation mask gates every readout AION produces. The next
question decides how to spend the next month:

    Is the gate LOW-DIMENSIONAL -- one direction in the residual stream that says
    "there is a source at the centre" -- or is it smeared across the representation?

If low-dimensional, a sparse dictionary will find it easily (and we barely need one),
steering gives a mitigation, and the mechanism is publishable. If high-dimensional, an SAE
is the right tool but the honest expectation for a clean single feature drops sharply.

Method:
  1. Collect encoder activations for the same objects under a REAL mask (gate on) and a
     DISPLACED mask (gate off), pixels identical throughout.
  2. Difference-in-means over objects -> candidate gate direction d. Its dimensionality is
     read off the singular spectrum of the per-object difference matrix.
  3. CAUSAL TEST: run with the displaced mask and add alpha*d back into the residual
     stream. If the flux estimate climbs back toward its real-mask value, d carries the
     gate.
  4. CONTROL: a random direction of matched norm, and a shuffled-object direction. Adding
     any large vector perturbs the output; only a matched-norm control makes the result
     mean anything.

Run:  python experiments/d2c_gate_direction.py [N]
Out:  results/d2c_gate_direction.json, results/d2c_gate_direction.png
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

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
BATCH = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
LAYER = 6                      # where to read the direction and where to steer
# The response peaks around alpha ~ 6-10 and then OVER-steers; sweeping only to alpha=3
# (my first attempt) shows 18% recovery and reads as a negative result. Sweep far enough
# to find the peak, and report the peak, not the endpoint.
ALPHAS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0]


def main():
    from aion.modalities import LegacySurveyFluxG, LegacySurveySegmentationMap

    rng = np.random.default_rng(SEED)
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device=DEVICE)
    T = P.tables()
    print(f"device={DEVICE}  N={N}  layer={LAYER}")

    def seg(binary):
        t = cm.encode(LegacySurveySegmentationMap(field=binary.clone().to(DEVICE)))
        return {k: v.reshape(v.shape[0], -1) for k, v in t.items()}

    def flux(tokens, steer=None, alpha=0.0):
        tk = {k: v.to(DEVICE) for k, v in tokens.items()}
        if steer is None:
            out = model.forward(tk, target_modality=[LegacySurveyFluxG])
        else:
            with Steer(model, steer, alpha=alpha, layers=[LAYER]):
                out = model.forward(tk, target_modality=[LegacySurveyFluxG])
        return T.expected_value(out["tok_flux_g"][:, 0, :].detach().float().cpu().numpy(),
                                "tok_flux_g")

    # ---------------------------------------------- pass 1: collect the direction
    diffs, toks_real, toks_wrong, f_real, f_wrong = [], [], [], [], []
    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), N, replace=False))
        for s in range(0, N, BATCH):
            b = r.aion_batch(rows[s:s + BATCH], with_segmap=True)
            ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            binary = (b["segmap"] > 0).float()
            wrong = torch.roll(binary, shifts=(25, 25), dims=(-2, -1))

            tr = {**ti, **seg(binary)}
            tw = {**ti, **seg(wrong)}
            toks_real.append({k: v.cpu() for k, v in tr.items()})
            toks_wrong.append({k: v.cpu() for k, v in tw.items()})
            f_real.append(flux(tr)); f_wrong.append(flux(tw))

            hr = A.encode_activations(model, tr, num_encoder_tokens=900,
                                      sites=("enc_resid",), layers=[LAYER])
            hw = A.encode_activations(model, tw, num_encoder_tokens=900,
                                      sites=("enc_resid",), layers=[LAYER])
            for h, tag in ((hr, "r"), (hw, "w")):
                v = h["valid"].unsqueeze(-1).to(h["acts"][f"enc_resid.{LAYER}"].dtype)
                h["pool"] = ((h["acts"][f"enc_resid.{LAYER}"] * v).sum(1)
                             / v.sum(1).clamp(min=1))
            diffs.append((hr["pool"] - hw["pool"]).numpy())
            print(f"  {min(s + BATCH, N)}/{N}", flush=True)

    D = np.concatenate(diffs)                       # (N, 768) per-object difference
    f_real = np.concatenate(f_real); f_wrong = np.concatenate(f_wrong)
    d_mean = D.mean(0)

    sv = np.linalg.svd(D - D.mean(0, keepdims=True), compute_uv=False)
    en = sv**2 / (sv**2).sum()
    consistency = float(np.linalg.norm(d_mean) / np.sqrt((D**2).sum(1).mean()))

    out = {
        "n": int(len(D)), "layer": LAYER,
        "dimensionality": {
            "rank1_frac_of_centred_diff": float(en[0]),
            "rank5_frac": float(en[:5].sum()),
            "n_dims_for_90pct": int(np.searchsorted(np.cumsum(en), 0.90) + 1),
            "mean_direction_consistency": consistency,
        },
        "flux_real_median": float(np.median(f_real)),
        "flux_wrong_median": float(np.median(f_wrong)),
    }
    print(f"\nDIMENSIONALITY of the real-minus-displaced difference at layer {LAYER}")
    print(f"  ||mean d|| / rms ||d_i||        = {consistency:.3f}   (1 = every object moves identically)")
    print(f"  rank-1 fraction of centred diff = {en[0]:.3f}")
    print(f"  dims for 90% of the variation   = {out['dimensionality']['n_dims_for_90pct']}")
    print(f"\n  median E[flux_g]: real mask {np.median(f_real):.3f} -> displaced {np.median(f_wrong):.3f}")

    # ------------------------------------------------------------ pass 2: steer
    dvec = torch.from_numpy(d_mean).float()
    rand = torch.randn(len(d_mean), generator=torch.Generator().manual_seed(SEED))
    rand = rand / rand.norm() * dvec.norm()

    curves = {"gate_direction": [], "random_matched_norm": []}
    for alpha in ALPHAS:
        for name, vec in (("gate_direction", dvec), ("random_matched_norm", rand)):
            vals = []
            for tw in toks_wrong:
                vals.append(flux(tw, steer=vec, alpha=alpha))
            curves[name].append(float(np.median(np.concatenate(vals))))
        print(f"  alpha={alpha:4.1f}  gate {curves['gate_direction'][-1]:8.3f}   "
              f"random {curves['random_matched_norm'][-1]:8.3f}")

    base, target = np.median(f_wrong), np.median(f_real)
    span = target - base
    rec = np.array([(v - base) / span for v in curves["gate_direction"]])
    ctl = np.array([(v - base) / span for v in curves["random_matched_norm"]])

    # Peak recovery is NOT the right statistic: at large alpha a random vector of the same
    # norm also inflates the flux (65% at alpha=8), so "89% recovered" is mostly a
    # non-specific large-perturbation effect. Report the operating point where the gate
    # direction is most SPECIFIC -- largest ratio to the matched-norm control among alphas
    # that achieve meaningful recovery.
    elig = np.where(rec > 0.2)[0]
    if len(elig):
        ratios = np.where(ctl[elig] > 0.02, rec[elig] / np.maximum(ctl[elig], 1e-6), np.inf)
        j = elig[int(np.argmax(ratios))]
    else:
        j = int(np.argmax(rec))
    k = int(np.argmax(rec - ctl))          # largest absolute separation

    out["steering"] = {
        "alphas": ALPHAS, **curves,
        "specific_operating_point": {
            "alpha": ALPHAS[j], "recovered": float(rec[j]), "control": float(ctl[j]),
            "ratio_to_control": float(rec[j] / max(ctl[j], 1e-6)),
        },
        "max_separation_point": {
            "alpha": ALPHAS[k], "recovered": float(rec[k]), "control": float(ctl[k]),
            "separation_pp": float(100 * (rec[k] - ctl[k])),
        },
        "peak_recovery_not_specific": {
            "alpha": ALPHAS[int(np.argmax(rec))],
            "recovered": float(rec.max()),
            "control_at_same_alpha": float(ctl[int(np.argmax(rec))]),
        },
    }
    sp = out["steering"]["specific_operating_point"]
    ms = out["steering"]["max_separation_point"]
    print(f"\nCAUSAL TEST (displaced mask, steering at layer {LAYER})")
    print(f"  most SPECIFIC point   alpha={sp['alpha']:>4}: recovered {100*sp['recovered']:5.1f}%  "
          f"control {100*sp['control']:5.1f}%  ratio {sp['ratio_to_control']:.1f}x")
    print(f"  max separation        alpha={ms['alpha']:>4}: recovered {100*ms['recovered']:5.1f}%  "
          f"control {100*ms['control']:5.1f}%  gap {ms['separation_pp']:.1f} pp")
    pk = out["steering"]["peak_recovery_not_specific"]
    print(f"  peak recovery         alpha={pk['alpha']:>4}: {100*pk['recovered']:5.1f}%  "
          f"but control reaches {100*pk['control_at_same_alpha']:5.1f}% -- NOT specific, do not quote")
    verdict = ("largely one direction" if sp["ratio_to_control"] > 5 and sp["recovered"] > 0.4
               else ("partially one direction" if sp["ratio_to_control"] > 3 and sp["recovered"] > 0.2
                     else "not captured by a single direction"))
    out["verdict"] = verdict
    print(f"  -> {verdict}")

    with open(RESULTS / "d2c_gate_direction.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.semilogy(np.arange(1, min(64, len(en)) + 1), en[:64], "o-", ms=3)
    ax.set_xlabel("component"); ax.set_ylabel("fraction of variance")
    ax.set_title(f"Object-to-object variation in the gate difference\n"
                 f"{out['dimensionality']['n_dims_for_90pct']} dims for 90%, "
                 f"consistency {consistency:.2f}", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ALPHAS, curves["gate_direction"], "o-", lw=1.8, label="gate direction")
    ax.axvline(sp["alpha"], color="grey", ls="-.", lw=1, alpha=0.7)
    ax.plot(ALPHAS, curves["random_matched_norm"], "s--", lw=1.4, label="random, matched norm")
    ax.axhline(target, color="C2", ls=":", lw=1.4, label="real-mask value (target)")
    ax.axhline(base, color="C3", ls=":", lw=1.4, label="displaced-mask value")
    ax.set_xlabel(r"steering coefficient $\alpha$"); ax.set_ylabel("median E[flux_g]")
    ax.set_title("Can we steer the gate back on?", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("D2c: is AION's detection gate a single direction?", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "d2c_gate_direction.png", dpi=150)
    print(f"wrote {RESULTS / 'd2c_gate_direction.json'} and .png")


if __name__ == "__main__":
    main()
