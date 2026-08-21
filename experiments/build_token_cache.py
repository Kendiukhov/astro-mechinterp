"""
Per-token activation cache for dictionary learning (D1).

Pooled vectors cannot train a dictionary, so this caches individual token rows. Two
deliberate choices:

1. PAIRED CONDITIONS. Every object is cached twice at the same layer: once with its real
   segmentation mask (gate on) and once with the mask displaced 25 px (gate off), pixels
   identical. The dictionary therefore sees the gate contrast directly, and evaluation can
   ask which features move between the two conditions rather than hunting an atlas.

2. STRATIFIED ROWS. Caching all 720 rows per object is 1.1 MB/object/condition and mostly
   redundant. We keep a fixed stratified sample per modality (image and segmap rows), which
   also prevents the 576 image rows from swamping the 144 segmap rows in the training set.

Layer 6 by default: D2c showed the gate is present and steerable there, while steering at
layer 11 recovers only 42%.

Run:  python experiments/build_token_cache.py [N]
Out:  results/tokens_L{layer}_N{N}.npz
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import astron_mi as A
from astron_mi.hooks import modality_id_map

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
BATCH = 16
LAYER = 6
SEED = 0
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ROWS_PER_MOD = {"tok_image": 64, "tok_segmap": 32}      # stratified sample per object


def main():
    from aion.modalities import LegacySurveySegmentationMap

    rng = np.random.default_rng(SEED)
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device=DEVICE)
    fwd, _ = modality_id_map()
    print(f"device={DEVICE}  N={N}  layer={LAYER}  rows/object={sum(ROWS_PER_MOD.values())}")

    def seg(binary):
        t = cm.encode(LegacySurveySegmentationMap(field=binary.clone().to(DEVICE)))
        return {k: v.reshape(v.shape[0], -1) for k, v in t.items()}

    acts_r, acts_w, modrow, objid = [], [], [], []
    t0 = time.time()
    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), N, replace=False))
        for s in range(0, N, BATCH):
            b = r.aion_batch(rows[s:s + BATCH], with_segmap=True)
            ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            binary = (b["segmap"] > 0).float()
            wrong = torch.roll(binary, shifts=(25, 25), dims=(-2, -1))

            hs = {}
            for tag, m in (("r", binary), ("w", wrong)):
                hs[tag] = A.encode_activations(
                    model, {**ti, **seg(m)}, num_encoder_tokens=900,
                    sites=("enc_resid",), layers=[LAYER],
                )
            h_r = hs["r"]["acts"][f"enc_resid.{LAYER}"]
            h_w = hs["w"]["acts"][f"enc_resid.{LAYER}"]
            mm = hs["r"]["mod_mask"]

            # the row->modality map is identical across conditions (same token counts),
            # so one stratified index set serves both
            for i in range(h_r.shape[0]):
                sel, tags = [], []
                for key, n_keep in ROWS_PER_MOD.items():
                    idx = torch.nonzero(mm[i] == fwd[key]).flatten()
                    if len(idx) == 0:
                        continue
                    take = idx[rng.choice(len(idx), min(n_keep, len(idx)), replace=False)]
                    sel.append(take)
                    tags += [key] * len(take)
                if not sel:
                    continue
                sel = torch.cat(sel)
                acts_r.append(h_r[i, sel].numpy().astype("float16"))
                acts_w.append(h_w[i, sel].numpy().astype("float16"))
                modrow.append(np.array(tags))
                objid.append(np.full(len(sel), s + i))

            done = min(s + BATCH, N)
            if done % (BATCH * 10) == 0 or done >= N:
                el = time.time() - t0
                print(f"  {done}/{N}  {el:.0f}s  eta {(N - done) * el / done / 60:.1f} min",
                      flush=True)

    out = {
        "acts_real": np.concatenate(acts_r),
        "acts_wrong": np.concatenate(acts_w),
        "modality": np.concatenate(modrow),
        "object_id": np.concatenate(objid),
        "layer": np.array([LAYER]),
    }
    path = RESULTS / f"tokens_L{LAYER}_N{N}.npz"
    np.savez(path, **out)     # uncompressed: fp16 activations barely compress and it is 10x faster
    print(f"\nwrote {path} ({path.stat().st_size / 1e9:.2f} GB)")
    print(f"  token rows: {out['acts_real'].shape}  per condition")
    for k in ROWS_PER_MOD:
        print(f"    {k}: {(out['modality'] == k).sum()}")


if __name__ == "__main__":
    main()
