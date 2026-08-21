"""
Build the labelled activation cache that B1, D0 and C2 all read.

For N PROVABGS galaxies this stores, per object:
  - mean-pooled encoder activations, per modality and globally, at selected layers
  - tok_z logits under an information ladder (photometry / +imaging / +spectra)
  - PROVABGS physical labels and the AION scalar inputs

Deliberately modest: pooled vectors, not per-token activations. Per-token caching is
~7 GB/layer for 100k objects and is only needed once a dictionary is actually being
trained (D1). Pooled is enough for every Week-1 experiment.

Run:  python experiments/build_cache.py [N]
Out:  results/cache_N{N}.npz
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import astron_mi as A
from astron_mi.hooks import pool_by_modality

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
BATCH = 16
SEED = 0   # rows are RANDOM, not the first N: the file is healpix-ordered, so the
           # first N rows are one contiguous ~2 deg sky patch. That would confound
           # anything involving position, dust, depth or survey systematics.
LAYERS = [5, 11]                      # mid and last encoder block
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SCALARS = [
    ("LegacySurveyFluxG", "FLUX_G"), ("LegacySurveyFluxR", "FLUX_R"),
    ("LegacySurveyFluxI", "FLUX_I"), ("LegacySurveyFluxZ", "FLUX_Z"),
    ("LegacySurveyFluxW1", "FLUX_W1"), ("LegacySurveyFluxW2", "FLUX_W2"),
    ("LegacySurveyEBV", "EBV"),
]


def encode_scalars(cm, scalars, device):
    from aion import modalities as M

    out = {}
    for cls_name, key in SCALARS:
        cls = getattr(M, cls_name, None)
        if cls is None or key not in scalars:
            continue
        v = np.asarray(scalars[key], dtype="float32")
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0)
        out.update(cm.encode(cls(value=torch.from_numpy(v).to(device))))
    return {k: (v.reshape(v.shape[0], -1) if v.dim() > 1 else v.unsqueeze(1))
            for k, v in out.items()}


def main():
    from aion.modalities import Z

    print(f"device={DEVICE}  N={N}  layers={LAYERS}")
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device=DEVICE)

    acc: dict[str, list] = {}
    t0 = time.time()
    with A.ProvabgsReader() as r:
        n_all = r.n_rows()
        total = min(N, n_all)
        rows = np.sort(np.random.default_rng(SEED).choice(n_all, total, replace=False))
        print(f"sampling {total} random rows out of {n_all}")
        for start in range(0, total, BATCH):
            idx = rows[start:start + BATCH]
            b = r.aion_batch(idx)

            ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            ts = A.encode_spectrum(cm, **b["spectrum"], survey="desi", device=DEVICE)
            tsc = encode_scalars(cm, b["scalars"], DEVICE)

            all_tokens = {**tsc, **ti, **ts}
            out = A.encode_activations(
                model, all_tokens, num_encoder_tokens=900,
                sites=("enc_resid",), layers=LAYERS,
            )

            # pooled activations: per modality, and globally over valid rows
            for tag, h in [("final", out["final"])] + [
                (f"L{i}", out["acts"][f"enc_resid.{i}"]) for i in LAYERS
            ]:
                pooled = pool_by_modality(h, out["mod_mask"])
                for k, v in pooled.items():
                    acc.setdefault(f"pool.{tag}.{k}", []).append(v.numpy().astype("float32"))
                valid = out["valid"].unsqueeze(-1).to(h.dtype)
                g = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
                acc.setdefault(f"pool.{tag}.ALL", []).append(g.numpy().astype("float32"))

            # tok_z logits under the information ladder
            ladders = {
                "phot": dict(tsc),
                "phot_img": {**tsc, **ti},
                "phot_img_spec": {**tsc, **ti, **ts},
            }
            for name, tk in ladders.items():
                tk = {k: v.to(DEVICE) for k, v in tk.items()}
                lg = model.forward(tk, target_modality=[Z])["tok_z"][:, 0, :]
                acc.setdefault(f"logits_z.{name}", []).append(
                    lg.detach().float().cpu().numpy().astype("float32")
                )

            for k, v in b["labels"].items():
                acc.setdefault(f"label.{k}", []).append(np.asarray(v))
            for k, v in b["scalars"].items():
                acc.setdefault(f"scalar.{k}", []).append(np.asarray(v))
            acc.setdefault("ra", []).append(b["ra"])
            acc.setdefault("dec", []).append(b["dec"])

            done = min(start + BATCH, total)
            if done % (BATCH * 10) == 0 or done >= total:
                el = time.time() - t0
                print(f"  {done}/{total}  {el:.0f}s  ({el / done:.2f} s/obj, "
                      f"eta {(total - done) * el / done / 60:.1f} min)")

    packed = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    path = RESULTS / f"cache_N{total}_rand.npz"
    np.savez_compressed(path, **packed)
    mb = path.stat().st_size / 1e6
    print(f"\nwrote {path} ({mb:.1f} MB)")
    print(f"keys: {len(packed)}; example shapes:")
    for k in list(packed)[:6]:
        print(f"   {k:34s} {packed[k].shape}")


if __name__ == "__main__":
    main()
