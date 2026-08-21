# Data

No data is committed to this repository. The cross-match is 115 GiB and every component is
public. This document explains what is needed, where it comes from, and what the loader expects
on disk.

## What the experiments need

| # | Dataset | Size | Used by |
|---|---|---|---|
| 1 | PROVABGS × DESI × Legacy Survey DR9 cross-match | 115 GiB | all `b*`, `c*`, `d*`, `e*` experiments |
| 2 | MultimodalUniverse HSC PDR3 (one shard) | 1.9 GB | `a0c_hsc_codebook.py` only |
| 3 | AION-1 test fixtures | a few MB | regression tests in `astron_mi.data` |

Datasets 2 and 3 are optional; everything else runs from dataset 1.

## 1. The PROVABGS × DESI × Legacy cross-match

**Sources.**

- Galaxy sample and physical parameters: the
  [PROVABGS](https://changhoonhahn.github.io/provabgs/) value-added catalogue for the DESI
  Bright Galaxy Survey ([Hahn et al. 2023](https://arxiv.org/abs/2202.01809)).
- Imaging and segmentation: [DESI Legacy Imaging Surveys DR9](https://www.legacysurvey.org/dr9/)
  cutout service — `grz` plus WISE `W1–W4`, and the `legacypipe` per-object segmentation map.
- Spectra: [DESI EDR](https://data.desi.lbl.gov/doc/releases/edr/) coadded spectra.

**Layout expected by `astron_mi.data.ProvabgsReader`.** A single HDF5 file at
`data/provabgs_desi_ls.hdf5` with one row per galaxy and these datasets:

```
legacysurvey/
  image            (N, 8, 160, 160)   float32   grz + W1-W4 in nanomaggies
  object_mask      (N, 160, 160)      int32     legacypipe segmentation, 0 = no source
  bands            (N, 8)             bytes     e.g. b'DES-G'
desi/
  spectrum_flux    (N, 7781)          float32
  spectrum_ivar    (N, 7781)          float32
  spectrum_lambda  (N, 7781)          float32
provabgs/
  z                (N,)               float64   spectroscopic redshift
  logmstar         (N,)               float64
  ...                                           other PROVABGS parameters
tractor/
  flux_g, flux_r, flux_z, shape_r, shape_e1, shape_e2, ra, dec   (N,) float64
```

Sentinel values of `-999` are converted to `NaN` by `astron_mi.data.mask_sentinels()`.

**Important: row ordering.** The file is HEALPix-ordered, so the first *N* rows are a single
contiguous sky field. Sampling sequentially rather than at random changes results materially —
a redshift probe scores R² = 0.48 on the first 2000 rows against 0.91 on a random draw of the
same size. Every experiment in this repository draws rows with
`np.random.default_rng(SEED).choice(...)`. Preserve that if you add your own.

## 2. MultimodalUniverse HSC PDR3

One shard is enough:

```bash
huggingface-cli download MultimodalUniverse/hsc \
  --repo-type dataset --include "pdr3_dud_22.5/train-00000-of-00046.parquet" \
  --local-dir data/hsc
```

Band names in the parquet are stored lowercase and unprefixed (`hsc-g`);
`astron_mi.encode.normalise_bands()` maps them to the codec's expected form. Passing them
through unchanged produces `HSC-HSC-G` and a silent mis-tokenisation.

## 3. Model weights

Pulled automatically from the Hugging Face Hub on first use, and pinned in the manuscript:

| Checkpoint | Revision | Parameters |
|---|---|---|
| `polymathic-ai/aion-base` | `40541618` | 314.3 M |
| `polymathic-ai/aion-large` | `cfdb89e8` | 859.7 M |

Only `d2e_scale_replication.py` needs `aion-large`.

## Caches

Two intermediate caches are built once and reused. Both are gitignored.

```bash
python experiments/build_cache.py         # results/cache_N3000_rand.npz   (~290 MB)
python experiments/build_token_cache.py   # results/tokens_L6_N3000.npz    (~900 MB)
```

The first holds tokenised cutouts so experiments do not re-tokenise; the second holds paired
layer-6 activations under true and displaced masks, and is what `d1_gate_dictionary.py` and
`d1b_sae_sweep.py` train on.

## Disk and compute

| | |
|---|---|
| Data | ~117 GiB |
| Caches | ~1.2 GiB |
| RAM | 16 GB is enough; `d1b` at width 32768 peaks around 8 GB |
| Device | Apple MPS or CUDA; CPU works but the sweep becomes impractical |

The codecs are bit-exact between MPS and CPU and 34–56× faster on MPS. Sparse-autoencoder
training on MPS is **not** bit-reproducible across runs — the same seed gives different
dictionaries — which is why `d1b_sae_sweep.py` reports a seed ensemble rather than a single run.
