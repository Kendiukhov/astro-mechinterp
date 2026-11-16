"""
Publication figures for the manuscript, rebuilt as vector PDFs from the stored result JSONs.

Reads only results/*.json — no model inference — so the figures are reproducible from the
recorded numbers and cannot silently drift from the text.

Run:  python experiments/make_paper_figures.py
Out:  paper/figs/fig{1..6}.pdf
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    # PLOS requires Arial, Times or Symbol at 8-12 pt inside figures, and the figures are
    # rasterised to TIFF for submission, so nothing below 8 pt may survive here.
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
C = {"gate": "#c0392b", "ok": "#2c7fb8", "alt": "#31a354", "grey": "#888888",
     "warn": "#e08214"}


def load(name):
    p = R / name
    return json.load(open(p)) if p.exists() else None


# ---------------------------------------------------------------- Figure 1
def fig1():
    """The gate: readout vs mask manipulation, and the mechanism discriminator."""
    d2, d2b = load("d2_segmap.json"), load("d2b_gating.json")
    if not (d2 and d2b):
        return
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8))

    order = ["image_only", "eroded_5", "real", "dilated_10", "shifted_10",
             "shifted_25", "swapped", "erased", "placebo_ra"]
    lab = ["no mask", "eroded 5px", "TRUE mask", "dilated 10px", "shifted 10px",
           "shifted 25px", "swapped", "erased", "placebo"]
    vals = [d2["arms"][a]["median_pred"] for a in order]
    cols = [C["grey"] if a in ("image_only", "placebo_ra") else
            (C["alt"] if a == "swapped" else (C["gate"] if a == "erased" else C["ok"]))
            for a in order]
    ax[0].bar(range(len(order)), vals, color=cols)
    ax[0].axhline(d2["arms"]["real"]["median_pred"], color="k", ls=":", lw=0.8)
    ax[0].set_xticks(range(len(order)))
    ax[0].set_xticklabels(lab, rotation=55, ha="right")
    ax[0].set_ylabel(r"median $E[f_g]$  (nmgy)")
    ax[0].set_title("(a) pixels identical, mask varied")

    mech = d2.get("mechanism", {})
    names = ["light enclosed\nby mask", "central mask\ncoverage"]
    v = [mech.get("corr_frac_dpred_vs_denclosed_light_allarms", np.nan),
         mech.get("corr_frac_dpred_vs_dcentral_coverage_allarms", np.nan)]
    ax[1].bar(range(2), v, color=[C["ok"], C["gate"]], width=0.55)
    ax[1].set_xticks(range(2)); ax[1].set_xticklabels(names)
    ax[1].set_ylabel(r"within-object corr. with $\Delta$readout")
    ax[1].set_title("(b) aperture or gate?")

    reads = ["tok_flux_g", "tok_flux_z", "tok_shape_r", "tok_shape_e1", "tok_z"]
    nice = ["flux $g$", "flux $z$", "size $R$", r"ellip. $e_1$", "redshift"]
    e = d2b["effects_sigma"]
    x = np.arange(len(reads)); w = 0.27
    for i, (arm, c, lb) in enumerate([("shifted_25", C["gate"], "displaced 25px"),
                                      ("swapped", C["alt"], "swapped"),
                                      ("placebo_ra", C["grey"], "placebo")]):
        ax[2].bar(x + (i - 1) * w, [e["img"][k][arm] for k in reads], w, color=c, label=lb)
    ax[2].set_yscale("symlog", linthresh=0.01)
    ax[2].set_xticks(x); ax[2].set_xticklabels(nice, rotation=35, ha="right")
    ax[2].set_ylabel(r"effect ($\sigma$ of readout)")
    ax[2].set_title("(c) every readout is gated")
    # headroom above the tallest bar so the legend cannot sit on the data
    hi = max(e["img"][k][a] for k in reads
             for a in ("shifted_25", "swapped", "placebo_ra"))
    ax[2].set_ylim(top=hi * 400)
    ax[2].legend(frameon=False, fontsize=8, loc="upper right",
                 handlelength=1.3, handletextpad=0.5, borderaxespad=0.2,
                 labelspacing=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig1.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def fig2():
    """Real blends: the model ignores how the pipeline partitioned the light."""
    f, g = load("d2f_real_blends.json"), load("d2g_severe_blends.json")
    if not (f and g):
        return
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.8))

    i = f["integration"]
    ax[0].errorbar([0], [i["R_median"]],
                   yerr=[[i["R_median"] - i["R_lo16"]], [i["R_hi84"] - i["R_median"]]],
                   fmt="o", color=C["ok"], capsize=3, label="all blends (n=%d)" % i["n_used"])
    s = g["R_all_severe"]
    ax[0].errorbar([1], [s["median"]],
                   yerr=[[s["median"] - s["lo16"]], [s["hi84"] - s["median"]]],
                   fmt="s", color=C["gate"], capsize=3, label="severe (n=%d)" % s["n"])
    ax[0].axhline(0, color=C["grey"], ls=":", lw=1, label="ignores mask extent")
    ax[0].axhline(1, color="k", ls="--", lw=1, label="integrates the light")
    ax[0].set_xlim(-0.5, 1.5); ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["all", "severe"])
    ax[0].set_ylabel(r"$R=\Delta$readout$/\Delta$light")
    ax[0].set_title("(a) does it integrate?")
    ax[0].legend(frameon=False, loc="center right")

    by = g.get("R_by_severity", {})
    if by:
        ks = list(by)
        xs = np.arange(len(ks))
        m = [by[k]["R"] for k in ks]
        lo = [by[k]["R"] - by[k]["lo16"] for k in ks]
        hi = [by[k]["hi84"] - by[k]["R"] for k in ks]
        ax[1].bar(xs, m, yerr=[lo, hi], capsize=3, color=C["ok"], width=0.55)
        ax[1].set_xticks(xs)
        ax[1].set_xticklabels([k.replace("-", "–") for k in ks], rotation=20, ha="right")
    ax[1].axhline(0, color=C["grey"], ls=":", lw=1)
    ax[1].set_xlabel("central segment's share of the light")
    ax[1].set_ylabel(r"$R$")
    ax[1].set_title("(b) no trend with severity")
    fig.tight_layout(); fig.savefig(OUT / "fig2.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def fig3():
    """Incidence and the tomographic bias against the LSST requirement."""
    e1 = load("e1_incidence_impact.json")
    if not e1:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1),
                           gridspec_kw={"width_ratios": [1, 1.7]})
    pa = e1["part_a_incidence"]["centroid_offset_px"]
    ks = ["median", "p90", "p99"]
    ax[0].bar(range(3), [pa[k] for k in ks], color=C["ok"], width=0.55)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["median", "p90", "p99"])
    ax[0].set_ylabel("centroid offset (pixels)")
    ax[0].set_title("(a) real mask offsets\n%.2f%% have no segment at centre"
                    % (100 * e1["part_a_incidence"]["frac_no_segment_at_centre"]))

    # Panel (b) represents a detection miss by transplanting a REAL miss field rather than
    # zeroing the map. The all-zero arm is retained in grey as a sensitivity comparison: the
    # answer depends on that modelling choice by a factor of four.
    e1b = load("e1b_faithful_miss.json")
    req = e1["part_b"]["lsst_requirement"]
    src = e1b["part_b"]["by_config"] if e1b else e1["part_b"]["by_config"]
    cfgs = [("img|miss_real", "imaging, measured miss rate", C["gate"], "o-"),
            ("img|miss_real_shift", "imaging, + displacements", C["warn"], "s--"),
            ("img_spec|miss_real", "+ spectra", C["ok"], "^:"),
            ("img|miss_erased", "all-zero field (sensitivity)", C["grey"], "x:")]
    for key, lb, c, st in cfgs:
        rec = src.get(key)
        if not rec:
            continue
        b = rec["bins"]
        xs = np.arange(len(b))
        ax[1].plot(xs, [x["shift_over_1pz"] for x in b], st, color=c, ms=4, label=lb)
        ax[1].set_xticks(xs)
        ax[1].set_xticklabels([f"[{x['lo']:.2f},{x['hi']:.2f})" for x in b],
                              rotation=30, ha="right")
    # 16-84% band across the 40-draw miss-assignment ensemble (E1c), so the reader sees the
    # realisation scatter rather than one draw.
    e1c = load("e1c_correlated_miss.json")
    if e1c:
        per = {}
        for dr in e1c["draws"]["uniform"]:
            for b in dr["img"]["bins"]:
                per.setdefault((b["lo"], b["hi"]), []).append(b["shift_over_1pz"])
        keys = sorted(per)
        lo = [float(np.percentile(per[k], 16)) for k in keys]
        hi = [float(np.percentile(per[k], 84)) for k in keys]
        ax[1].fill_between(np.arange(len(keys)), lo, hi, color=C["gate"], alpha=0.16,
                           lw=0, zorder=1, label="imaging, 16--84% over draws")
    ax[1].axhline(req, color="k", ls="-", lw=1.2, label="LSST SRD requirement")
    # symlog, not log: three of the "+ spectra" Scenario-1 bins are exactly zero and must
    # sit on the axis rather than be floored to a fictitious residual bias.
    ax[1].set_yscale("symlog", linthresh=1e-5)
    ax[1].set_ylim(bottom=0)
    ax[1].set_ylabel(r"$|\Delta\langle z\rangle|/(1+z)$")
    ax[1].set_title("(b) tomographic bias")
    ax[1].legend(frameon=False, fontsize=8, loc="upper center",
                 bbox_to_anchor=(0.5, -0.34), ncol=2, handlelength=1.8,
                 columnspacing=1.0)
    fig.tight_layout(); fig.savefig(OUT / "fig3.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Figure 4
def fig4():
    """Mitigation and the metadata-preference generalisation."""
    d2d, e2 = load("d2d_mitigation.json"), load("e2_metadata_preference.json")
    if not (d2d and e2):
        return
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.8))

    cfgs = ["img", "img+seg_real", "img+seg_disc", "spec", "spec+seg_real", "spec+seg_disc"]
    lab = ["img", "img\n+mask", "img\n+disc", "spec", "spec\n+mask", "spec\n+disc"]
    q = [d2d["quality"][c]["tok_flux_g"]["median"] for c in cfgs]
    lo = [q[i] - d2d["quality"][c]["tok_flux_g"]["lo16"] for i, c in enumerate(cfgs)]
    hi = [d2d["quality"][c]["tok_flux_g"]["hi84"] - q[i] for i, c in enumerate(cfgs)]
    cols = [C["ok"] if "seg" not in c else (C["warn"] if "real" in c else C["gate"])
            for c in cfgs]
    ax[0].bar(range(len(cfgs)), q, yerr=[lo, hi], capsize=2.5, color=cols)
    ax[0].set_xticks(range(len(cfgs))); ax[0].set_xticklabels(lab)
    ax[0].set_ylabel(r"$|\Delta \mathrm{mag}|$ vs catalogue")
    ax[0].set_title("(a) withholding the mask is free")

    names = ["wrong\ncatalogue\nscalars", "wrong\nsegmap", "placebo"]
    v = [e2["comparison"]["wrong_scalars_mag"], e2["comparison"]["wrong_segmap_mag"],
         e2["comparison"]["placebo_mag"]]
    ax[1].bar(range(3), v, color=[C["gate"], C["warn"], C["grey"]], width=0.55)
    ax[1].axhline(e2["arms"]["img"]["err_vs_truth"][0], color="k", ls="--", lw=1,
                  label="error using NO metadata")
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(names)
    ax[1].set_ylabel(r"induced $|\Delta \mathrm{mag}|$")
    ax[1].set_title("(b) metadata preference is general")
    ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig4.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Figure 5
def fig5():
    """Tokeniser: codebook collapse and the brightness ladder."""
    a0b, a0c, b4 = load("a0b_code_frequency.json"), load("a0c_hsc_codebook.json"), \
        load("b4_code_semantics.json")
    if not (a0b and b4):
        return
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8))

    sp = a0b["image_split"]
    labs, vals, cols = ["image\n(source)", "image\n(sky)"], \
        [sp["source"]["perplexity"], sp["sky"]["perplexity"]], [C["gate"], C["grey"]]
    if a0c:
        labs.append("image\n(HSC)"); vals.append(a0c["perplexity"]); cols.append(C["warn"])
    labs.append("spectrum"); vals.append(a0b["modalities"]["tok_spectrum_desi"]["perplexity"])
    cols.append(C["ok"])
    ax[0].bar(range(len(vals)), vals, color=cols, width=0.6)
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(len(vals)))
    ax[0].set_xticklabels(labs, rotation=25, ha="right")
    ax[0].set_ylabel("effective codebook size")
    ax[0].set_title("(a) usable states (of 4375 / 1024)")

    codes = b4["codes"]
    fl = np.array([c["flux"] for c in codes])
    rs = np.array([c["structure"] / max(abs(c["flux"]), 1e-9) for c in codes])
    co = np.array([c["colour"] for c in codes])
    s = ax[1].scatter(fl, rs, c=co, s=16, cmap="coolwarm", edgecolor="none")
    plt.colorbar(s, ax=ax[1], label="$g-z$")
    ax[1].set_xlabel("patch mean flux")
    ax[1].set_ylabel("relative structure (rms/flux)")
    ax[1].set_title("(b) shape is slaved to brightness")

    ax[2].bar([0, 1], [b4["r2_flux_vs_relative_structure"], b4["r2_flux_vs_colour"]],
              color=[C["ok"], C["warn"]], width=0.5)
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["relative\nstructure", "colour\n$g-z$"])
    ax[2].set_ylim(0, 1)
    ax[2].set_ylabel(r"$R^2$ with flux")
    ax[2].set_title("(c) PC1 = %.0f%% of variance" % (100 * b4["pc1_var"]))
    fig.tight_layout(); fig.savefig(OUT / "fig5.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Figure 6
def fig6():
    """Redshift: line-locking, and the dictionary negative."""
    c1, d1 = load("c1_redshift.json"), load("d1_gate_dictionary.json")
    if not (c1 and d1):
        return
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.9))

    xr, pr = np.array(c1["rest_frame"]["x"]), np.array(c1["rest_frame"]["profile"])
    xo, po = np.array(c1["observed_frame"]["x"]), np.array(c1["observed_frame"]["profile"])
    ax[0].plot(xr, pr / np.median(pr[pr > 0]), lw=1.1, color=C["gate"], label="rest frame")
    ax[0].plot(xo, po / np.median(po[po > 0]), lw=1.0, ls="--", color=C["ok"],
               label="observed frame")
    lines = [(3950, "Ca H&K"), (4861, r"H$\beta$"), (5007, "[O III]"),
             (5175, "Mg b"), (5892, "Na D"), (6563, r"H$\alpha$")]
    top = ax[0].get_ylim()[1]
    for j, (lam, nm) in enumerate(lines):
        ax[0].axvline(lam, color=C["grey"], ls=":", lw=0.5)
        ax[0].text(lam, top * (0.93 if j % 2 == 0 else 0.55), nm, rotation=90,
                   fontsize=8, ha="right", va="top", color=C["grey"])
    ax[0].set_xlabel(r"wavelength ($\AA$)")
    ax[0].set_ylabel("ablation effect / median")
    ax[0].set_title("(a) redshift is line-locked")
    ax[0].legend(frameon=False)

    # Panel (b): the whole D1b sweep, not the single D1 dictionary. Every SAE configuration is
    # drawn as a thin line so the reader can see that the null is a property of the method here
    # and not of one unlucky hyperparameter choice.
    sw = load("d1b_sae_sweep.json")
    if sw:
        alb = sw["alphas_baseline"]
        for k, c, st in [("difference_in_means", C["ok"], "o-"),
                         ("random", C["grey"], "x--")]:
            ax[1].plot(alb, [100 * v for v in sw["baseline_curves"][k]], st, ms=3.5,
                       color=c, label=k.replace("_", " "), zorder=3)
        # PCA is rebuilt at each sparsity; k=32 is D1's setting and is plotted as the
        # representative curve (k=16 and 64 sit within a few points of it).
        ax[1].plot(alb, [100 * v for v in sw["pca_curves"]["32"]], "^-", ms=3.5,
                   color=C["alt"], label="pca (matched $k$)", zorder=3)
        als = sw["alphas_sweep"]
        for j, cfg in enumerate(sw["configs"]):
            ys = [100 * cfg["recovery"][f"{a}"] for a in als]
            ax[1].plot(als, ys, "-", lw=0.7, alpha=0.55, color=C["gate"], zorder=2,
                       label=f"SAE ({len(sw['configs'])} configs)" if j == 0 else None)
        # The test was pre-registered at alpha=4. Beyond it every large-norm perturbation
        # restores the readout, so the curves converge and cross; mark where the claim is made.
        astar = sw["alpha_star"]
        ax[1].axvline(astar, color="k", ls=":", lw=0.9, zorder=1)
        ax[1].plot([astar], [-6], marker="^", ms=4, color="k", clip_on=False, zorder=4)
    else:
        al = d1["alphas"]
        for k, c, st in [("difference_in_means", C["ok"], "o-"),
                         ("pca_topk", C["alt"], "^-"),
                         ("sae_gate_features", C["gate"], "s-"),
                         ("random", C["grey"], "x--")]:
            ax[1].plot(al, [100 * v for v in d1["steering_curves"][k]], st, ms=3.5,
                       color=c, label=k.replace("_", " "))
    ax[1].set_xlabel(r"steering coefficient $\alpha$ (matched norm)")
    ax[1].set_ylabel("% of collapse recovered")
    ax[1].set_title("(b) dictionaries: wide spread, no advantage")
    ax[1].legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.6,
                 labelspacing=0.3, borderaxespad=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig6.pdf"); plt.close(fig)


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6):
        try:
            fn(); print(f"  {fn.__name__} ok")
        except Exception as exc:
            print(f"  {fn.__name__} FAILED: {type(exc).__name__}: {exc}")
    print("figures in", OUT)
