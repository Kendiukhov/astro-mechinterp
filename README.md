# astro-mechinterp

**A causal audit of AION-1, a multimodal foundation model for astronomy.**

This repository contains the code, per-experiment outputs and manuscript for a mechanistic
interpretability study of [AION-1](https://arxiv.org/abs/2510.17960) (Polymathic AI), a
40-table transformer trained on >2×10⁸ objects from five surveys.

The study asks what the model actually *uses* when it answers, and prices the consequence for a
downstream cosmological analysis. Every effect is measured against a matched placebo, and every
point estimate carries a bootstrap interval.

---

## The finding in one paragraph

AION-1 ingests the survey's **segmentation map** — the detection pipeline's own output — as an
input modality. Holding the image tokens byte-identical and editing only that map changes every
quantity the model reports (flux, size, ellipticity, redshift) by **110–4400×** a matched
placebo. The mechanism is *detection gating* — whether the mask asserts a source at the field
centre — and not masked aperture photometry. The behaviour generalises to other
catalogue-derived channels: contradicted catalogue photometry leaves the model **nine times
worse** than supplying no metadata at all.

Propagating the survey's own measured 3.68 % detection-miss rate through the model's photometric
redshifts shifts tomographic mean redshifts by a median **0.71×** the LSST DESC requirement, and
observed positional errors take that to **8.3×**. Withholding the detection channel removes the
vulnerability at no measurable cost.

## Results at a glance

| Result | Where | Script |
|---|---|---|
| Every readout is gated by the segmentation map, 110–4400× placebo | §3.1 | `d2b_gating_scope.py` |
| Mechanism is detection gating, not aperture photometry (r = 0.47 vs 0.30) | §3.2 | `d2_segmap_intervention.py` |
| Deblending decisions do not propagate (R = −0.006 over 322 real blends) | §3.3 | `d2g_severe_blends.py` |
| Metadata preference is general: wrong catalogue scalars ⇒ 9× worse than none | §3.5 | `e2_metadata_preference.py` |
| The vulnerability grows with model scale (1.62× base → large) | §3.6 | `d2e_scale_replication.py` |
| 3.68 % detection-miss rate ⇒ median 0.71× the LSST DESC requirement | §4 | `e1b_faithful_miss.py` |
| Miss rate is 20× magnitude-dependent, but that does not change the bias | §4.4 | `e1c_correlated_miss.py` |
| Withholding the mask is free; a generic disc is +220 % worse | §4.3 | `d2d_mitigation.py` |
| Image codec resolves ~28 effective states on source patches vs 934 for spectra | §5.1 | `a0b_code_frequency.py` |
| Image codes form a one-dimensional brightness ladder | §5.2 | `b4_code_semantics.py` |
| Redshift readout is quantisation-limited, not information-limited | §5.3 | `b1_calibration.py` |
| Redshift is line-locked (Hα 25×) but not line-brittle | §6.2 | `c1b_line_aliases.py` |
| Sparse dictionaries are unreliable causal handles (26–75 % across 15) | §7 | `d1b_sae_sweep.py` |

Full claim → script → output-file mapping: **[docs/claims-to-code.md](docs/claims-to-code.md)**.

---

## Install

Python 3.11, PyTorch ≥ 2.6. Apple Silicon (MPS) and CUDA are both supported; the codecs are
bit-exact against CPU.

```bash
git clone <this-repository>
cd astro-mechinterp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

> **Note for macOS users:** install the virtualenv on an APFS volume. `pip` fails on exFAT
> because of AppleDouble (`._*`) files.

Model weights are pulled from the Hugging Face Hub on first use:
`polymathic-ai/aion-base` (revision `40541618`) and `polymathic-ai/aion-large` (`cfdb89e8`).

## Reproduce

```bash
make figures        # rebuild all six paper figures from the committed results/*.json
make test           # smoke tests: package imports, JSON schema, figure pipeline
make paper          # compile the manuscript (needs a TeX distribution)
```

Every figure in the manuscript is regenerated from the committed JSON **with no live model
call**, so figures cannot drift from the numbers in the text. To re-derive the JSON itself you
need the data (see below) and a GPU-ish device; each experiment is independently runnable:

```bash
python experiments/build_cache.py            # cutout cache (needed by most experiments)
python experiments/d2b_gating_scope.py       # any single experiment
python experiments/make_paper_figures.py     # rebuild figures from results/
```

Runtimes on an M-series laptop range from ~2 minutes (`a0b_code_frequency.py`) to ~6 hours
(`d1b_sae_sweep.py`, which trains 15 sparse autoencoders). `d1b` checkpoints after every
configuration and resumes automatically.

## Data

**No data is committed** — the cross-match is 115 GiB and everything is public. See
**[docs/data.md](docs/data.md)** for how to rebuild it. In short:

| Dataset | Contents | Used for |
|---|---|---|
| PROVABGS × DESI × Legacy Survey DR9 | 115 404 galaxies: `grz`+WISE cutouts, per-object segmentation maps, DESI spectra, physical parameters | most experiments |
| MultimodalUniverse HSC PDR3 (one shard) | 1883 galaxies | HSC codebook measurement (`a0c`) |

---

## Repository layout

```
astron_mi/              library
  loading.py              model + codec loading, device placement
  encode.py               image/spectrum/scalar tokenisation, band handling
  hooks.py                activation capture, steering, safe multi-target readout
  physics.py              token index ↔ physical value decoding
  data.py                 lazy HDF5 reader, batching, sentinel masking
  sae.py                  BatchTopK sparse autoencoder + PCA baseline

experiments/            24 scripts, one per result (see docs/claims-to-code.md)
results/                per-experiment JSON + diagnostic PNGs (committed)
paper/                  LaTeX source, figures, and the build for submission
docs/                   claim mapping, data instructions, findings summary
tests/                  smoke tests
RESULTS.md              full experiment log: every number, with caveats and null results
```

`RESULTS.md` is the lab record. It contains results that did not make the manuscript, the exact
numbers behind those that did, and the caveats attached to each.

## Two things worth knowing before you use this code

Both cost us time and are easy to hit again. They are documented in the manuscript's appendix
and enforced in the library:

1. **Query one target modality per forward pass.** `AION.forward(..., target_modality=[A,B,C])`
   supplies an all-zero `decoder_attention_mask` that the adapter converts to "all positions
   masked". Harmless for one decoder token; with more, decoder self-attention becomes uniform
   and *every* readout is corrupted (median |Δz| 0.0038 → 0.833 → 2.350 for one, two, three
   targets). `astron_mi.hooks.read_targets()` loops one forward per target.

2. **Real DESI spectra need a trailing λ = 99999 sentinel.** Omitting it changes 20 % of the 273
   spectrum tokens while remaining invisible in end-to-end accuracy.
   `astron_mi.encode.pad_spectrum_sentinel()` handles this.

## Citing

See [CITATION.cff](CITATION.cff). The manuscript is in `paper/`.

## License

Code is MIT ([LICENSE](LICENSE)). The data products are governed by their respective survey
policies: [DESI](https://data.desi.lbl.gov/doc/), [Legacy
Surveys](https://www.legacysurvey.org/acknowledgment/), [HSC
SSP](https://hsc.mtk.nao.ac.jp/ssp/data-release/).

## Acknowledgement

This audit was only possible because the AION-1 authors released weights, tokenisers and
training code openly — including the components whose limitations are reported here.
