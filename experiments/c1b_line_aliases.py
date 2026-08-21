"""
C1b — Does removing Halpha push the redshift onto a PREDICTED alias?   (tests C1's prediction)

C1 showed AION's redshift is line-locked with Halpha carrying 25x the median causal weight.
The pre-registered consequence is fragility: a line-locker should fail catastrophically and at
PREDICTABLE places when it misidentifies a line, whereas a continuum-fitter degrades smoothly.

Testing that on natural outliers is impractical -- with spectra the outlier rate is 0.27%, so
~37,000 objects would be needed for ~100 failures. Instead we INDUCE the failure causally:
ablate the ~205 A window containing rest-frame Halpha (its observed position is known from the
true redshift) and see where the redshift goes.

THE PREDICTION. If the model finds some other feature at rest wavelength L_x but interprets it
as the line it lost, the new redshift obeys

    rho = (1 + z_new) / (1 + z_true) = L_x / L_assumed

so rho must pile up at RATIOS OF LINE WAVELENGTHS, not spread smoothly. Ratios against Halpha:
[O III]/Ha = 0.763, Hb/Ha = 0.741, Mg b/Ha = 0.788, Na D/Ha = 0.898, [O II]/Ha = 0.568.

CONTROLS, both necessary:
  - a QUIET window of identical width at a rest wavelength with no strong feature. Ablating it
    should not induce failures. Without this, "ablation breaks the model" is trivial.
  - a SHUFFLED-alias null: the same rho values tested against ratios of RANDOM wavelengths.
    Line ratios are dense on [0.5, 1], so clustering must be shown to beat a random grid.

Run:  python experiments/c1b_line_aliases.py [N]
Out:  results/c1b_aliases.json, results/c1b_aliases.png
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 480
BATCH = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
KEY = "tok_spectrum_desi"
WIN = 8                       # tokens ~ 205 A, matched to C0's resolution
GOOD_DZ = 0.01                # start from objects the model gets right
FAIL_DZ = 0.02                # what counts as an induced failure

HALPHA = 6563.0
QUIET = 7400.0                # rest-frame, no strong galaxy feature nearby
# strong rest-frame features the model could latch onto instead
FEATURES = {"[O II]": 3727.0, "Ca H&K": 3950.0, "Hd": 4102.0, "Hb": 4861.0,
            "[O III]": 5007.0, "Mg b": 5175.0, "Na D": 5892.0, "Ha": 6563.0}


def token_for(lam_obs):
    """Spectrum token index whose window contains an observed wavelength."""
    return int(np.clip(P.lambda_token(lam_obs), 1, 272 - WIN))


def main():
    from aion.modalities import Z

    rng = np.random.default_rng(SEED)
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device=DEVICE)
    T = P.tables()
    print(f"device={DEVICE}  N={N}  ablating a {WIN}-token (~{WIN * P.SPEC_A_PER_TOKEN:.0f} A) window")

    def zpred(tok, mask=None):
        im = None if mask is None else {KEY: mask.to(DEVICE)}
        o = model.forward({k: v.to(DEVICE) for k, v in tok.items()},
                          target_modality=[Z], input_mask=im)
        return T.expected_value(o["tok_z"][:, 0, :].detach().float().cpu().numpy(), "tok_z")

    z_true_l, z0_l, z_ha_l, z_q_l, in_range_l = [], [], [], [], []
    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), N, replace=False))
        for s in range(0, N, BATCH):
            b = r.aion_batch(rows[s:s + BATCH], with_segmap=False)
            ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            ts = A.encode_spectrum(cm, **b["spectrum"], survey="desi", device=DEVICE)
            tok = {**ti, **ts}
            zt = np.asarray(b["labels"]["z"], dtype="float64")
            nb = ts[KEY].shape[0]

            z0 = zpred(tok)
            good = np.isfinite(zt) & (np.abs(z0 - zt) < GOOD_DZ)
            if good.sum() == 0:
                continue

            # per-object masks: the window containing observed Halpha, and a quiet control
            m_ha = torch.zeros(nb, 273, dtype=torch.bool)
            m_q = torch.zeros(nb, 273, dtype=torch.bool)
            ok_range = np.zeros(nb, dtype=bool)
            for i in range(nb):
                lam_ha = HALPHA * (1 + zt[i])
                lam_q = QUIET * (1 + zt[i])
                if not (3600 < lam_ha < 10300):
                    continue
                ok_range[i] = True
                t1 = token_for(lam_ha); m_ha[i, t1:t1 + WIN] = True
                t2 = token_for(min(lam_q, 10300)); m_q[i, t2:t2 + WIN] = True

            z_ha = zpred(tok, m_ha)
            z_q = zpred(tok, m_q)

            sel = good & ok_range
            z_true_l.append(zt[sel]); z0_l.append(z0[sel])
            z_ha_l.append(z_ha[sel]); z_q_l.append(z_q[sel])
            print(f"  {min(s + BATCH, N)}/{N}  (kept {sel.sum()})", flush=True)

    zt = np.concatenate(z_true_l); z0 = np.concatenate(z0_l)
    zha = np.concatenate(z_ha_l); zq = np.concatenate(z_q_l)
    n = len(zt)
    print(f"\n{n} objects: correct at baseline and with Halpha inside the DESI range")

    f_ha = float(np.mean(np.abs(zha - zt) > FAIL_DZ))
    f_q = float(np.mean(np.abs(zq - zt) > FAIL_DZ))
    print(f"\nINDUCED FAILURE RATE (|dz| > {FAIL_DZ})")
    print(f"  ablate the Halpha window : {100 * f_ha:5.1f}%")
    print(f"  ablate a quiet window    : {100 * f_q:5.1f}%   <- control")
    if f_q > 0:
        print(f"  ratio = {f_ha / f_q:.1f}x")
    else:
        print(f"  control induced ZERO failures, so the ratio is unbounded")

    # ------------------------------------------------------------- alias analysis
    fail = np.abs(zha - zt) > FAIL_DZ
    rho = (1 + zha[fail]) / (1 + zt[fail])
    print(f"\n{fail.sum()} induced failures; rho = (1+z_new)/(1+z_true)")

    pred_ratios = {f"{k}/Ha": v / HALPHA for k, v in FEATURES.items() if k != "Ha"}
    # also allow the reverse identification (model thinks it lost a different line)
    for k, v in FEATURES.items():
        if k != "Ha":
            pred_ratios[f"Ha/{k}"] = HALPHA / v

    hits = {}
    TOL = 0.02
    for name, r_ in sorted(pred_ratios.items(), key=lambda kv: kv[1]):
        frac = float(np.mean(np.abs(rho - r_) < TOL)) if fail.sum() else 0.0
        if frac > 0:
            hits[name] = {"ratio": r_, "frac_within_tol": frac}
    print(f"  fraction of failures within {TOL} of a predicted line-ratio alias:")
    for k, v in sorted(hits.items(), key=lambda kv: -kv[1]["frac_within_tol"])[:8]:
        print(f"    {k:14s} rho={v['ratio']:.3f}   {100 * v['frac_within_tol']:5.1f}%")
    total_on_alias = float(np.mean([np.min(np.abs(r_ - np.array(list(pred_ratios.values())))) < TOL
                                    for r_ in rho])) if fail.sum() else 0.0

    # NULL: random "lines" spanning the same range, same count, same tolerance
    rng2 = np.random.default_rng(SEED)
    nulls = []
    for _ in range(200):
        fake = rng2.uniform(3700, 6600, len(FEATURES) - 1)
        fr = np.concatenate([fake / HALPHA, HALPHA / fake])
        nulls.append(np.mean([np.min(np.abs(r_ - fr)) < TOL for r_ in rho]) if fail.sum() else 0)
    null_med = float(np.median(nulls))
    print(f"\n  ON-ALIAS FRACTION: {100 * total_on_alias:.1f}%")
    print(f"  random-wavelength null: {100 * null_med:.1f}%  (median of 200 draws)")
    print(f"  excess over null: {100 * (total_on_alias - null_med):+.1f} pp")

    out = {
        "n": int(n), "fail_rate_halpha": f_ha, "fail_rate_quiet_control": f_q,
        "ratio_to_control": (float(f_ha / f_q) if f_q > 0 else None),
        "n_failures": int(fail.sum()),
        "on_alias_fraction": total_on_alias,
        "random_null_fraction": null_med,
        "excess_pp": float(100 * (total_on_alias - null_med)),
        "per_alias": hits,
        "rho_values": rho.tolist()[:500],
    }
    strong = (f_ha > 3 * max(f_q, 1e-9)) and (total_on_alias > null_med + 0.10)
    out["verdict"] = ("fragile AND aliases are predictable" if strong
                      else ("fragile but aliases not predictable"
                            if f_ha > 3 * max(f_q, 1e-9) else "not fragile to Halpha removal"))
    print(f"\nVERDICT: {out['verdict']}")

    with open(RESULTS / "c1b_aliases.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    ax = axes[0]
    ax.bar([0, 1], [100 * f_ha, 100 * f_q], color=["C3", "C0"])
    ax.set_xticks([0, 1]); ax.set_xticklabels([r"ablate H$\alpha$", "ablate quiet\n(control)"], fontsize=9)
    ax.set_ylabel(f"% pushed to |dz| > {FAIL_DZ}")
    ax.set_title("Is one 205 A window load-bearing?", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    if fail.sum():
        ax.hist(rho, bins=60, range=(0.4, 1.6), color="C0", alpha=0.85)
        for name, v in pred_ratios.items():
            if 0.4 < v < 1.6:
                ax.axvline(v, color="C3", ls=":", lw=0.9)
        ax.axvline(np.nan, color="C3", ls=":", lw=0.9, label="predicted line-ratio aliases")
        ax.legend(fontsize=7)
    ax.set_xlabel(r"$\rho = (1+z_{\rm new})/(1+z_{\rm true})$")
    ax.set_ylabel("induced failures")
    ax.set_title("Do failures land on line ratios?", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.scatter(zt[~fail], zha[~fail], s=6, alpha=0.3, label="survived")
    ax.scatter(zt[fail], zha[fail], s=10, alpha=0.7, color="C3", label="induced failure")
    lim = [0, max(0.6, float(np.nanpercentile(zt, 99)) * 1.3)]
    ax.plot(lim, lim, "k:", lw=1)
    ax.set_xlim(lim); ax.set_ylim(0, max(1.2, lim[1]))
    ax.set_xlabel("true z"); ax.set_ylabel(r"z with H$\alpha$ ablated")
    ax.set_title("Where the redshift goes", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(r"C1b: ablating the H$\alpha$ window — is AION's redshift as fragile as line-locking predicts?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "c1b_aliases.png", dpi=150)
    print(f"wrote {RESULTS / 'c1b_aliases.json'} and .png")


if __name__ == "__main__":
    main()
