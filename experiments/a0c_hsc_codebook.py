"""
A0c — Direct code-frequency measurement for tok_image_hsc, closing A0b's last proxy.

A0b measured Legacy image-code usage directly and found the row-norm proxy overstated it ~3x
(8.3% emitted, ~28 effective codes on source patches). `tok_image_hsc` could not be measured
because the PROVABGS file has no HSC cutouts, so its 9.6% stayed a proxy — and by A0b's
argument was probably an overestimate.

This uses real HSC PDR3 deep/ultradeep cutouts from MultimodalUniverse to settle it.

Note the two image modalities share ONE codec and ONE 4375-code vocabulary but have separate
embedding tables (established in A0), so this also asks whether the two surveys occupy the same
region of the shared codebook or different ones — the direct version of A0's finding that HSC's
live embedding rows were a strict subset of Legacy's.

Only ENCODE is used. The HSC decode path is broken upstream (`_reverse_zeropoint` vs
`reverse_zeropoint`), which does not affect tokenisation.

Run:  python experiments/a0c_hsc_codebook.py [N]
Out:  results/a0c_hsc_codebook.json, results/a0c_hsc_codebook.png
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import astron_mi as A

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
HSC = ROOT / "data" / "hsc" / "pdr3_dud_22.5" / "train-00000-of-00046.parquet"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
BATCH = 8
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
INIT_STD = 0.02
VOCAB = 4375


def main():
    if not HSC.exists():
        print(f"{HSC} not found — download a MultimodalUniverse/hsc shard first"); return
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device="cpu")

    f = pq.ParquetFile(HSC)
    n_rows = min(N, f.metadata.num_rows)
    print(f"device={DEVICE}  HSC shard: {f.metadata.num_rows} rows, using {n_rows}")

    counts = np.zeros(VOCAB, dtype="int64")
    seen = 0
    buf_flux, buf_bands = [], None

    def flush(buf, bands):
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).float()
        tok = A.encode_image(cm, x, bands, device=DEVICE)
        ids = list(tok.values())[0].reshape(-1).cpu().numpy()
        np.add.at(counts, ids, 1)

    for rg in range(f.metadata.num_row_groups):
        if seen >= n_rows:
            break
        tbl = f.read_row_group(rg, columns=["image"]).to_pydict()["image"]
        for rec in tbl:
            if seen >= n_rows:
                break
            # the parquet stores bands as 'hsc-g' etc; normalise_bands just uppercases,
            # which is all that is needed — prepending 'HSC-' double-prefixes them
            from astron_mi.encode import normalise_bands
            bands = normalise_bands(rec["band"])
            arr = np.asarray(rec["flux"], dtype="float32")
            if arr.ndim != 3:
                continue
            if buf_bands is None:
                buf_bands = bands
                print(f"  bands {bands}, cutout {arr.shape}")
            if bands != buf_bands:
                continue
            buf_flux.append(arr)
            seen += 1
            if len(buf_flux) == BATCH:
                flush(buf_flux, buf_bands); buf_flux = []
                if seen % 200 == 0:
                    print(f"  {seen}/{n_rows}", flush=True)
    flush(buf_flux, buf_bands)

    tot = int(counts.sum())
    used = int((counts > 0).sum())
    p = counts / max(tot, 1)
    nz = p[p > 0]
    perp = float(np.exp(-(nz * np.log(nz)).sum()))

    w = model.encoder_embeddings["tok_image_hsc"].token_emb.weight.detach().float().numpy()
    thr = 0.5 * INIT_STD * np.sqrt(w.shape[1])
    proxy = float((np.linalg.norm(w, axis=1) > thr).mean())

    print(f"\ntok_image_hsc, measured on {seen} real HSC cutouts ({tot} tokens)")
    print(f"  codes ever emitted     : {used}/{VOCAB} = {100 * used / VOCAB:.1f}%")
    print(f"  row-norm proxy (A0)    : {100 * proxy:.1f}%")
    print(f"  effective size (perplexity): {perp:.1f}")
    print(f"  codes for 50% of mass  : {int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.5) + 1)}")

    # overlap with the Legacy usage measured in A0b, if available
    overlap = None
    a0b = RESULTS / "a0b_code_frequency.json"
    legacy_used = None
    if a0b.exists():
        # recompute Legacy live set cheaply from its embedding table is not equivalent;
        # instead report overlap of THIS measured HSC set with Legacy's row-norm live set,
        # stating the asymmetry rather than pretending they are like-for-like.
        wl = model.encoder_embeddings["tok_image"].token_emb.weight.detach().float().numpy()
        legacy_live = np.linalg.norm(wl, axis=1) > thr
        hsc_emitted = counts > 0
        overlap = {
            "hsc_emitted": int(hsc_emitted.sum()),
            "legacy_live_rownorm": int(legacy_live.sum()),
            "intersection": int((hsc_emitted & legacy_live).sum()),
            "frac_hsc_inside_legacy": float((hsc_emitted & legacy_live).sum()
                                            / max(hsc_emitted.sum(), 1)),
        }
        print(f"\n  of the {overlap['hsc_emitted']} codes HSC actually emits, "
              f"{overlap['intersection']} ({100 * overlap['frac_hsc_inside_legacy']:.1f}%) "
              f"are also Legacy-live by row norm")

    out = {"n_cutouts": int(seen), "tokens": tot, "vocab": VOCAB,
           "codes_used": used, "frac_used": used / VOCAB,
           "rownorm_proxy_frac": proxy, "perplexity": perp,
           "overlap_with_legacy": overlap,
           "verdict": ("proxy overstates HSC usage, as it did for Legacy"
                       if used / VOCAB < proxy - 0.02 else
                       "proxy roughly agrees for HSC")}
    print(f"\nVERDICT: {out['verdict']}")
    with open(RESULTS / "a0c_hsc_codebook.json", "w") as fj:
        json.dump(out, fj, indent=2)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    srt = np.sort(p)[::-1]
    ax.semilogy(np.arange(1, VOCAB + 1), np.maximum(srt, 1e-9), lw=1.5, label="HSC (measured)")
    ax.axvline(used, color="C3", ls="--", lw=1.2, label=f"last emitted ({used})")
    ax.axvline(perp, color="C2", ls=":", lw=1.4, label=f"perplexity ({perp:.0f})")
    ax.axvline(proxy * VOCAB, color="C1", ls="-.", lw=1.2,
               label=f"row-norm proxy ({proxy * VOCAB:.0f})")
    ax.set_xlabel("code rank"); ax.set_ylabel("emission frequency")
    ax.set_title(f"tok_image_hsc on {seen} real HSC cutouts\n"
                 f"{100 * used / VOCAB:.1f}% emitted vs {100 * proxy:.1f}% by row norm", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "a0c_hsc_codebook.png", dpi=150)
    print(f"wrote {RESULTS / 'a0c_hsc_codebook.json'} and .png")


if __name__ == "__main__":
    main()
