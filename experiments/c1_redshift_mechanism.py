"""
C1 — How does AION compute redshift: lines or continuum?   (AGENDA.md, Thrust C)

The question every photo-z systematic hangs on. A model that locks onto emission/absorption
LINES fails catastrophically and unpredictably when it misidentifies one. A model that fits
the CONTINUUM SHAPE (the 4000 A break, broadband curvature) degrades gracefully. The failure
modes are completely different, and which one AION has determines where its outliers live.

Method — causal ablation, not attention:
  1. Ablate a WINDOW of spectrum tokens via `input_mask` (validated: masking all 273 tokens
     reproduces the image-only redshift error exactly, 0.1207 vs 0.1207).
  2. Measure how much the tok_z posterior moves (Jensen-Shannon divergence, plus |d E[z]|).
  3. STACK the resulting 273-vector in the REST frame using the known redshift.

Window size is 8 tokens (~205 A) with stride 4. C0 measured the codec's effective resolution
at ~11 tokens FWHM, so ablating single tokens would be finer than the tokeniser resolves and
would only blur the answer. This is the resolution the instrument actually supports.

THE DISCRIMINATOR is the rest-frame vs observed-frame contrast:
  sharper in REST frame     -> the model tracks features that move with redshift, i.e. LINES
  sharper in OBSERVED frame -> a fixed-wavelength effect (instrument, sky, or the tokeniser)

Restricted to objects the model already gets right (|dz| < 0.01 with the full spectrum), so
this measures how it works when it works, not how it fails.

Run:  python experiments/c1_redshift_mechanism.py [N]
Out:  results/c1_redshift.json, results/c1_redshift.png
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 192
BATCH = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
KEY = "tok_spectrum_desi"
WIN, STRIDE = 8, 4              # tokens; C0 says the codec resolves ~11
GOOD_DZ = 0.01                  # only objects the model already gets right

# rest-frame features worth annotating (Angstrom)
LINES = [
    (3727, "[O II]"), (3950, "Ca H&K"), (4000, "4000 A break"), (4102, r"H$\delta$"),
    (4861, r"H$\beta$"), (5007, "[O III]"), (5175, "Mg b"), (5892, "Na D"),
    (6563, r"H$\alpha$+[N II]"), (8600, "Ca II triplet"),
]


def jsd(p, q, eps=1e-12):
    p = p / p.sum(-1, keepdims=True)
    q = q / q.sum(-1, keepdims=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(a + eps) - np.log(b + eps))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def softmax(z):
    z = np.asarray(z, dtype="float64")
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main():
    from aion.modalities import Z

    rng = np.random.default_rng(SEED)
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device=DEVICE)
    T = P.tables()
    vals = T.index_to_value("tok_z")
    starts = list(range(1, 273 - WIN, STRIDE))     # token 0 is the normalisation token
    print(f"device={DEVICE}  N={N}  {len(starts)} windows of {WIN} tokens "
          f"(~{WIN * P.SPEC_A_PER_TOKEN:.0f} A), stride {STRIDE}")

    def zlogits(tok, mask=None):
        im = None if mask is None else {KEY: mask.to(DEVICE)}
        o = model.forward({k: v.to(DEVICE) for k, v in tok.items()},
                          target_modality=[Z], input_mask=im)
        return o["tok_z"][:, 0, :].detach().float().cpu().numpy()

    all_jsd, all_dz, all_z = [], [], []
    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), N, replace=False))
        for s in range(0, N, BATCH):
            b = r.aion_batch(rows[s:s + BATCH], with_segmap=False)
            ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            ts = A.encode_spectrum(cm, **b["spectrum"], survey="desi", device=DEVICE)
            tok = {**ti, **ts}
            ztrue = np.asarray(b["labels"]["z"], dtype="float64")

            lg0 = zlogits(tok)
            p0 = softmax(lg0)
            e0 = T.expected_value(lg0, "tok_z")
            good = np.isfinite(ztrue) & (np.abs(e0 - ztrue) < GOOD_DZ)
            if good.sum() == 0:
                continue

            nb = ts[KEY].shape[0]
            J = np.zeros((nb, len(starts)))
            D = np.zeros((nb, len(starts)))
            for wi, st in enumerate(starts):
                m = torch.zeros(nb, 273, dtype=torch.bool)
                m[:, st:st + WIN] = True
                lg = zlogits(tok, m)
                J[:, wi] = jsd(p0, softmax(lg))
                D[:, wi] = np.abs(T.expected_value(lg, "tok_z") - e0)
            all_jsd.append(J[good]); all_dz.append(D[good]); all_z.append(ztrue[good])
            print(f"  {min(s + BATCH, N)}/{N}  (kept {good.sum()}/{len(good)})", flush=True)

    J = np.concatenate(all_jsd)
    D = np.concatenate(all_dz)
    zs = np.concatenate(all_z)
    n = len(zs)
    print(f"\n{n} objects with a correct full-spectrum redshift  (median z {np.median(zs):.3f})")

    # window centre -> observed wavelength
    lam_obs = np.array([P.LATENT_LAMBDA_MIN + (st - 1 + WIN / 2) * P.SPEC_A_PER_TOKEN
                        for st in starts])

    # ------------------------------------------------------------- the stacks
    def stack(lams_per_object, weights, lo, hi, nbin=70):
        edges = np.linspace(lo, hi, nbin + 1)
        num = np.zeros(nbin); den = np.zeros(nbin)
        for i in range(len(weights)):
            idx = np.digitize(lams_per_object[i], edges) - 1
            ok = (idx >= 0) & (idx < nbin)
            np.add.at(num, idx[ok], weights[i][ok])
            np.add.at(den, idx[ok], 1.0)
        prof = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        return 0.5 * (edges[:-1] + edges[1:]), prof

    lam_rest = np.array([lam_obs / (1 + z) for z in zs])
    lam_observed = np.tile(lam_obs, (n, 1))

    # normalise each object's profile so bright objects do not dominate the stack
    Jn = J / np.maximum(J.sum(1, keepdims=True), 1e-30)

    x_rest, prof_rest = stack(lam_rest, Jn, 3000, 9000)
    x_obs, prof_obs = stack(lam_observed, Jn, 3500, 10000)

    # ESSENTIAL CONTROL: de-redshift each object by ANOTHER object's z. If the rest-frame
    # sharpness is real, this must destroy it. Without this control a rest-frame peak could
    # simply be a rebinning artifact, since over a narrow z range the rest and observed
    # stacks differ by little more than a scale factor.
    z_shuf = zs[np.random.default_rng(SEED).permutation(len(zs))]
    lam_shuf = np.array([lam_obs / (1 + z) for z in z_shuf])
    x_shuf, prof_shuf = stack(lam_shuf, Jn, 3000, 9000)

    def contrast(p):
        p = p[np.isfinite(p)]
        return float(np.nanmax(p) / np.nanmedian(p)) if len(p) else float("nan")

    c_rest, c_obs, c_shuf = contrast(prof_rest), contrast(prof_obs), contrast(prof_shuf)
    peak_rest = float(x_rest[np.nanargmax(prof_rest)])
    peak_obs = float(x_obs[np.nanargmax(prof_obs)])

    print(f"\nSTACKED ABLATION PROFILES")
    print(f"  rest frame     peak at {peak_rest:7.0f} A   contrast (max/median) = {c_rest:.2f}")
    print(f"  observed frame peak at {peak_obs:7.0f} A   contrast (max/median) = {c_obs:.2f}")
    print(f"  SHUFFLED-z control     contrast = {c_shuf:.2f}   (must be much lower than rest)")
    real = (c_rest > c_obs) and (c_rest > 1.5 * c_shuf)
    print(f"  -> {'REST-frame sharper AND survives the shuffled-z control: LINE-LOCKING' if real else 'NOT established — rest-frame sharpness does not beat its own null'}")

    # which rest-frame features carry the most causal weight
    print(f"\n  causal weight near known rest-frame features (profile value / median):")
    feats = []
    for lam, name in LINES:
        j = int(np.argmin(np.abs(x_rest - lam)))
        v = prof_rest[j] / np.nanmedian(prof_rest)
        feats.append({"lambda": lam, "name": name, "relative_weight": float(v)})
        print(f"    {name:16s} {lam:5d} A   {v:5.2f}x")

    out = {
        "n": int(n), "window_tokens": WIN, "stride": STRIDE,
        "window_A": float(WIN * P.SPEC_A_PER_TOKEN),
        "median_z": float(np.median(zs)),
        "rest_frame": {"peak_A": peak_rest, "contrast": c_rest,
                       "x": x_rest.tolist(), "profile": np.nan_to_num(prof_rest).tolist()},
        "observed_frame": {"peak_A": peak_obs, "contrast": c_obs,
                           "x": x_obs.tolist(), "profile": np.nan_to_num(prof_obs).tolist()},
        "shuffled_z_control": {"contrast": c_shuf},
        "features": feats,
        "verdict": ("line-locking (rest-frame sharper, beats shuffled-z null)" if real
                    else "not established"),
    }
    with open(RESULTS / "c1_redshift.json", "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.plot(x_rest, prof_rest / np.nanmedian(prof_rest), lw=1.6, color="C0")
    for lam, name in LINES:
        if x_rest[0] <= lam <= x_rest[-1]:
            ax.axvline(lam, color="grey", ls=":", lw=0.8)
            ax.text(lam, ax.get_ylim()[1] * 0.97, name, rotation=90, fontsize=6,
                    va="top", ha="right", color="grey")
    ax.set_xlabel("rest-frame wavelength (A)")
    ax.set_ylabel("ablation effect / median")
    ax.set_title(f"REST frame — contrast {c_rest:.2f}", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(x_obs, prof_obs / np.nanmedian(prof_obs), lw=1.6, color="C1")
    ax.set_xlabel("observed-frame wavelength (A)")
    ax.set_ylabel("ablation effect / median")
    ax.set_title(f"OBSERVED frame — contrast {c_obs:.2f}", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(x_rest, prof_rest / np.nanmedian(prof_rest), lw=1.6, label="rest frame")
    ax.plot(x_obs, prof_obs / np.nanmedian(prof_obs), lw=1.4, ls="--", label="observed frame")
    ax.plot(x_shuf, prof_shuf / np.nanmedian(prof_shuf), lw=1.2, ls=":", color="C3",
            label="shuffled-z null")
    ax.set_xlabel("wavelength (A)"); ax.set_ylabel("ablation effect / median")
    ax.set_title("Which frame is sharper?", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("C1: which parts of the spectrum does AION's redshift actually depend on?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "c1_redshift.png", dpi=150)
    print(f"\nwrote {RESULTS / 'c1_redshift.json'} and .png")


if __name__ == "__main__":
    main()
