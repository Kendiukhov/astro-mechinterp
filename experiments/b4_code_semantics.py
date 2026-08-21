"""
B4 — What are the ~28 states the image tokeniser actually uses?   (follows directly from A0b)

A0b measured that AION's image codec emits 8.3% of its 4375 codes, and that on patches
containing the source its effective vocabulary is ~28 codes. The obvious next question is what
those states MEAN, and specifically:

    do the codes encode MORPHOLOGY, or are they essentially a BRIGHTNESS LADDER?

If a single flux axis explains most of the variation between codes, then the image tokeniser
compresses a galaxy to a coarse surface-brightness map with almost no shape information, and
AION's imaging-vs-spectra performance gap is explained at the tokeniser without any appeal to
the transformer.

Method — decode IN CONTEXT, not in isolation. Decoding a grid filled with one code gives the
decoder's prior, not the code's meaning. Instead: take real cutouts, overwrite the token at one
central grid position with code c, decode, and difference against the unmodified decode. What
changes in those 4x4 pixels is what c contributes.

For each frequent code we measure, in the 4x4 patch it controls:
    flux        mean level across bands
    structure   RMS spatial variation within the patch, after removing the mean
    colour      g - z of the patch
then ask how much of the between-code variance one principal component explains, and how much
of it is flux.

Run:  python experiments/b4_code_semantics.py [N_CODES]
Out:  results/b4_code_semantics.json, results/b4_code_semantics.png
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

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
N_CODES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N_CONTEXT = 32               # real cutouts to average the in-context decode over
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
GRID, PATCH = 24, 4
CENTRE = GRID // 2 * GRID + GRID // 2      # flat index of a central grid cell


def main():
    from aion.modalities import LegacySurveyImage

    rng = np.random.default_rng(SEED)
    cm = A.load_codecs(DEVICE)
    freq_path = RESULTS / "a0b_code_frequency.json"
    if not freq_path.exists():
        print("run experiments/a0b_code_frequency.py first"); return

    # --------------------------------------------- pick the most-used source codes
    print(f"device={DEVICE}  collecting context cutouts and code frequencies")
    counts = np.zeros(4375, dtype="int64")
    ctx_tokens, ctx_flux, ctx_bands = [], [], None
    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), 400, replace=False))
        for s in range(0, len(rows), 16):
            b = r.aion_batch(rows[s:s + 16], with_segmap=True)
            ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            ids = ti["tok_image"].cpu().numpy()
            seg = b["segmap"]
            off = (seg.shape[-1] - 96) // 2
            m = (seg[..., off:off + 96, off:off + 96] > 0).float()
            m = (m.reshape(len(m), GRID, PATCH, GRID, PATCH).sum((2, 4)) > 0)
            m = m.reshape(len(m), -1).numpy()
            np.add.at(counts, ids[m], 1)
            if len(ctx_tokens) * 16 < N_CONTEXT:
                ctx_tokens.append(ti["tok_image"].cpu())
                ctx_flux.append(b["image"]["flux"])
                ctx_bands = b["image"]["bands"]
    codes = np.argsort(-counts)[:N_CODES]
    share = counts[codes].sum() / max(counts.sum(), 1)
    print(f"  top {N_CODES} source codes carry {100 * share:.1f}% of source-patch tokens")

    base_tok = torch.cat(ctx_tokens)[:N_CONTEXT]
    DES = ["DES-G", "DES-R", "DES-I", "DES-Z"]

    @torch.no_grad()
    def decode(tok):
        # CodecManager.decode takes token INDICES and handles the index->FSQ-latent step;
        # codec.decode() takes latents and fails on indices. Pass DES bands explicitly:
        # the HSC decode path has a broken attribute reference.
        out = cm.decode({"tok_image": tok.reshape(len(tok), -1).to(DEVICE)},
                        LegacySurveyImage, bands=DES)
        out = out.flux if hasattr(out, "flux") else out
        return out.float().cpu()

    base_img = decode(base_tok)                         # (B, C, 96, 96)
    r0, c0 = (GRID // 2) * PATCH, (GRID // 2) * PATCH

    rows_out = []
    for ci, code in enumerate(codes):
        tk = base_tok.clone().reshape(-1, GRID, GRID)
        tk[:, GRID // 2, GRID // 2] = int(code)
        img = decode(tk.reshape(len(tk), -1))
        d = (img - base_img)[..., r0:r0 + PATCH, c0:c0 + PATCH]     # (B, C, 4, 4)
        patch = img[..., r0:r0 + PATCH, c0:c0 + PATCH]
        # decoding with DES bands returns exactly those 4 channels: g, r, i, z
        g, z = patch[:, 0], patch[:, 3]
        flux = float(patch.mean())
        struct = float((patch - patch.mean((-2, -1), keepdim=True)).pow(2).mean().sqrt())
        colour = float((g.mean() - z.mean()))
        rows_out.append({"code": int(code), "freq": float(counts[code] / counts.sum()),
                         "flux": flux, "structure": struct, "colour": colour,
                         "delta_rms": float(d.pow(2).mean().sqrt())})
        if ci < 6:
            print(f"    code {code:4d}  freq {100 * counts[code] / counts.sum():5.2f}%  "
                  f"flux {flux:+.4f}  struct {struct:.4f}  g-z {colour:+.4f}")

    # A brighter patch has larger ABSOLUTE rms by construction, so raw structure co-varies
    # with flux trivially. The non-trivial question is whether the SHAPE of a patch --
    # structure divided by its own flux -- varies independently across codes.
    for r_ in rows_out:
        r_["rel_structure"] = r_["structure"] / max(abs(r_["flux"]), 1e-9)
    fl = np.array([r["flux"] for r in rows_out])
    rs = np.array([r["rel_structure"] for r in rows_out])
    co = np.array([r["colour"] for r in rows_out])
    r2_rel = float(np.corrcoef(fl, rs)[0, 1] ** 2)
    r2_col = float(np.corrcoef(fl, co)[0, 1] ** 2)
    print(f"\n  DIMENSIONLESS SHAPE TEST (removes the trivial flux scaling)")
    print(f"    relative structure (rms/flux): spread {rs.std() / rs.mean():.3f} of its mean, "
          f"R^2 with flux = {r2_rel:.3f}")
    print(f"    colour g-z:                    R^2 with flux = {r2_col:.3f}")
    print(f"    -> {'shape and colour are slaved to brightness' if (r2_rel > 0.5 and r2_col > 0.5) else 'shape/colour vary independently of brightness'}")

    F = np.array([[r["flux"], r["structure"], r["colour"]] for r in rows_out])
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-12)
    u, sv, vt = np.linalg.svd(Fz - Fz.mean(0), full_matrices=False)
    var = sv**2 / (sv**2).sum()
    pc1_load = dict(zip(["flux", "structure", "colour"], np.abs(vt[0]).round(3).tolist()))

    # how much of the between-code variation is a pure brightness ladder?
    flux_only_r2 = float(np.corrcoef(F[:, 0], u[:, 0] * sv[0])[0, 1] ** 2)
    struct_frac = float(F[:, 1].std() / (abs(F[:, 0]).std() + 1e-12))

    print(f"\nBETWEEN-CODE STRUCTURE ({N_CODES} codes)")
    print(f"  PC1 explains {100 * var[0]:.1f}% of the variance; loadings {pc1_load}")
    print(f"  R^2(flux, PC1) = {flux_only_r2:.3f}")
    print(f"  within-patch structure / flux spread = {struct_frac:.3f}")
    # A brightness ladder is PC1 dominating with all loadings comparable (one latent scalar
    # driving every measured quantity), not PC1 loading on flux alone -- with three
    # standardised variables an isotropic loading of ~1/sqrt(3)=0.577 is exactly that case.
    ladder = (var[0] > 0.8) and (flux_only_r2 > 0.9) and (r2_rel > 0.5) and (r2_col > 0.5)
    print(f"  -> {'BRIGHTNESS LADDER: one flux axis dominates' if ladder else 'codes carry more than brightness'}")

    out = {"n_codes": N_CODES, "share_of_source_tokens": float(share),
           "codes": rows_out,
           "pc1_var": float(var[0]), "pc1_loadings": pc1_load,
           "r2_flux_pc1": flux_only_r2, "structure_to_flux_ratio": struct_frac,
           "r2_flux_vs_relative_structure": r2_rel, "r2_flux_vs_colour": r2_col,
           "verdict": "brightness ladder" if ladder else "more than brightness"}
    with open(RESULTS / "b4_code_semantics.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    ax.semilogy(100 * np.array([r["freq"] for r in rows_out]), "o-", ms=4)
    ax.set_xlabel("code rank (source patches)"); ax.set_ylabel("% of source tokens")
    ax.set_title(f"top {N_CODES} codes carry {100 * share:.0f}% of source patches", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sc = ax.scatter([r["flux"] for r in rows_out], [r["structure"] for r in rows_out],
                    c=[r["colour"] for r in rows_out], s=40, cmap="coolwarm")
    plt.colorbar(sc, ax=ax, label="g − z")
    ax.set_xlabel("patch mean flux"); ax.set_ylabel("within-patch structure (RMS)")
    ax.set_title("Do codes differ in shape, or only in level?", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.bar(range(3), var[:3])
    ax.set_xticks(range(3)); ax.set_xticklabels(["PC1", "PC2", "PC3"])
    ax.set_ylabel("fraction of between-code variance")
    ax.set_title(f"PC1 = {100 * var[0]:.0f}%  (flux loading {pc1_load['flux']:.2f})", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("B4: what the image tokeniser's most-used codes encode", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "b4_code_semantics.png", dpi=150)
    print(f"wrote {RESULTS / 'b4_code_semantics.json'} and .png")


if __name__ == "__main__":
    main()
