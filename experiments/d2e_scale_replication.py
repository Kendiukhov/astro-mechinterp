"""
D2e — Does the detection gate replicate at 2.7x scale?   (Tier 1, last item)

Everything so far is n=1 model. "AION-base has a quirk" is a live alternative to "models
given a detection channel defer to it", and only a second model can separate them.

aion-large: 860M params, 24/24 blocks, d=1024 (base: 314M, 12/12, d=768). Same architecture
family, same codecs, same 40 modality tables — so the comparison is clean.

Measured on IDENTICAL objects for both models:
  quality        |readout - catalogue truth|, with and without the segmentation mask
  vulnerability  change when the mask is swapped for another galaxy's real mask
  utilisation    (weights only) does the larger model also leave its image codebook mostly
                 dead? A0 found base uses 23% of 4375 codes.

Three outcomes, all informative:
  gate persists or grows  -> a property of the design, not of one checkpoint
  gate shrinks with scale -> capacity-limited, may vanish in future models
  gate absent in large    -> base-specific, and the finding is much weaker

Run:  python experiments/d2e_scale_replication.py [N]
Out:  results/d2e_scale.json, results/d2e_scale.png
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

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 240
BATCH = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
N_BOOT = 1000
MODELS = ["base", "large"]
INIT_STD = 0.02


def boot_ci(v, fn=np.median, n=N_BOOT, seed=SEED):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    st = [fn(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return float(fn(v)), float(np.percentile(st, 16)), float(np.percentile(st, 84))


def dmag(p, t):
    ok = np.isfinite(t) & (t > 1e-2) & (p > 1e-2)
    e = np.full(len(t), np.nan)
    e[ok] = 2.5 * np.abs(np.log10(p[ok] / t[ok]))
    return e


def codebook_use(model, key, dim):
    w = model.encoder_embeddings[key].token_emb.weight.detach().float().numpy()
    thr = 0.5 * INIT_STD * np.sqrt(dim)
    return float((np.linalg.norm(w, axis=1) > thr).mean())


def main():
    from aion.modalities import LegacySurveyFluxG, LegacySurveySegmentationMap, Z

    targets = [LegacySurveyFluxG, Z]
    rng = np.random.default_rng(SEED)
    cm = A.load_codecs(DEVICE)
    T = P.tables()
    out: dict = {"n": N, "models": {}}

    # ------------------------------------------------- weights-only: utilisation
    print("codebook utilisation (A0's row-norm proxy), weights only:")
    for size in MODELS:
        m = A.load_model(size, device="cpu")
        dim = m.encoder_embeddings["tok_image"].token_emb.weight.shape[1]
        u = {k: codebook_use(m, k, dim) for k in
             ("tok_image", "tok_image_hsc", "tok_spectrum_desi", "tok_ebv", "tok_z")}
        out["models"].setdefault(size, {})["codebook_use"] = u
        out["models"][size]["d_model"] = int(dim)
        print(f"  {size:6s} d={dim}  " + "  ".join(f"{k.replace('tok_',''):12s}{100 * v:5.1f}%"
                                                   for k, v in u.items()))

    # --------------------------------------------------------- the intervention
    for size in MODELS:
        model = A.load_model(size, device=DEVICE)
        print(f"\n=== {size} ===", flush=True)
        pred = {"img": {}, "seg": {}, "swap": {}}
        truth = {"tok_flux_g": [], "tok_z": []}

        with A.ProvabgsReader() as r:
            rows = np.sort(np.random.default_rng(SEED).choice(r.n_rows(), N, replace=False))
            for s in range(0, N, BATCH):
                b = r.aion_batch(rows[s:s + BATCH], with_segmap=True)
                ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
                binary = (b["segmap"] > 0).float()
                g = torch.Generator().manual_seed(s)
                sw = binary[torch.randperm(len(binary), generator=g)]

                def seg(x):
                    t = cm.encode(LegacySurveySegmentationMap(field=x.clone().to(DEVICE)))
                    return {k: v.reshape(v.shape[0], -1) for k, v in t.items()}

                for tag, extra in (("img", {}), ("seg", seg(binary)), ("swap", seg(sw))):
                    for k, v in A.read_targets(model, {**ti, **extra}, targets, T,
                                               device=DEVICE, seed=SEED).items():
                        pred[tag].setdefault(k, []).append(v)
                truth["tok_flux_g"].append(np.asarray(b["scalars"]["FLUX_G"], dtype="float64"))
                truth["tok_z"].append(np.asarray(b["labels"]["z"], dtype="float64"))
                print(f"  {min(s + BATCH, N)}/{N}", flush=True)

        pred = {t: {k: np.concatenate(v) for k, v in d.items()} for t, d in pred.items()}
        truth = {k: np.concatenate(v) for k, v in truth.items()}

        rec = {}
        for k in ("tok_flux_g", "tok_z"):
            e = (lambda p: dmag(p, truth[k])) if k == "tok_flux_g" \
                else (lambda p: np.abs(p - truth[k]))
            q_img = boot_ci(e(pred["img"][k]))
            q_seg = boot_ci(e(pred["seg"][k]))
            if k == "tok_flux_g":
                vul = boot_ci(dmag(pred["swap"][k], pred["seg"][k]))
            else:
                vul = boot_ci(np.abs(pred["swap"][k] - pred["seg"][k]))
            rec[k] = {"quality_no_mask": q_img, "quality_with_mask": q_seg,
                      "vulnerability_swap": vul}
            print(f"  {k:12s} no-mask {q_img[0]:.4f} [{q_img[1]:.4f},{q_img[2]:.4f}]   "
                  f"with-mask {q_seg[0]:.4f} [{q_seg[1]:.4f},{q_seg[2]:.4f}]   "
                  f"swap vulnerability {vul[0]:.4f} [{vul[1]:.4f},{vul[2]:.4f}]")
        out["models"][size]["intervention"] = rec

    # ------------------------------------------------------------------ verdict
    print(f"\n{'=' * 78}\nVERDICT — does the gate replicate at 2.7x scale?")
    for k, nice in (("tok_flux_g", "flux g (mag)"), ("tok_z", "redshift (|dz|)")):
        vb = out["models"]["base"]["intervention"][k]["vulnerability_swap"][0]
        vl = out["models"]["large"]["intervention"][k]["vulnerability_swap"][0]
        print(f"  {nice:18s} vulnerability  base {vb:.4f}  ->  large {vl:.4f}   "
              f"({'PERSISTS' if vl > 0.5 * vb else 'shrinks'}, ratio {vl / max(vb, 1e-9):.2f}x)")
    for k, nice in (("tok_flux_g", "flux g (mag)"), ("tok_z", "redshift (|dz|)")):
        for size in MODELS:
            r = out["models"][size]["intervention"][k]
            d = r["quality_no_mask"][0] - r["quality_with_mask"][0]
            print(f"  {nice:18s} {size:6s} cost of dropping the mask: {d:+.4f} "
                  f"({'free or better' if d <= 0 else 'worse without'})")
    out["verdict"] = {
        "gate_persists_at_scale": bool(
            out["models"]["large"]["intervention"]["tok_flux_g"]["vulnerability_swap"][0]
            > 0.5 * out["models"]["base"]["intervention"]["tok_flux_g"]["vulnerability_swap"][0]
        )
    }

    with open(RESULTS / "d2e_scale.json", "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    x = np.arange(len(MODELS))
    for ax, (k, nice) in zip(axes[:2], (("tok_flux_g", "flux g (mag)"),
                                        ("tok_z", "redshift |dz|"))):
        for i, (lab, key) in enumerate((("no mask", "quality_no_mask"),
                                        ("with mask", "quality_with_mask"))):
            m = [out["models"][s]["intervention"][k][key][0] for s in MODELS]
            lo = [m[j] - out["models"][s]["intervention"][k][key][1] for j, s in enumerate(MODELS)]
            hi = [out["models"][s]["intervention"][k][key][2] - m[j] for j, s in enumerate(MODELS)]
            ax.bar(x + (i - 0.5) * 0.35, m, 0.35, yerr=[lo, hi], capsize=3, label=lab)
        v = [out["models"][s]["intervention"][k]["vulnerability_swap"][0] for s in MODELS]
        ax.plot(x, v, "r*--", ms=12, label="swap vulnerability")
        ax.set_xticks(x); ax.set_xticklabels([f"aion-{s}" for s in MODELS])
        ax.set_ylabel(nice); ax.set_title(nice, fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    keys = ["tok_image", "tok_image_hsc", "tok_spectrum_desi", "tok_ebv", "tok_z"]
    for i, s in enumerate(MODELS):
        ax.bar(np.arange(len(keys)) + (i - 0.5) * 0.35,
               [100 * out["models"][s]["codebook_use"][k] for k in keys], 0.35,
               label=f"aion-{s}")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([k.replace("tok_", "") for k in keys], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("% of codebook in use"); ax.set_title("Codebook utilisation vs scale", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("D2e: does the detection gate replicate from aion-base (314M) to aion-large (860M)?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "d2e_scale.png", dpi=150)
    print(f"\nwrote {RESULTS / 'd2e_scale.json'} and .png")


if __name__ == "__main__":
    main()
