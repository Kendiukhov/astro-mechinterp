# Archive contents

Companion code and results for *Foundation models defer to their metadata channel: a causal
audit of AION-1*.

## What is here

```
astron_mi/          library: model loading, tokenisation, activation hooks, physics decoding,
                    sparse dictionary learning
experiments/        23 experiment scripts, one per result in the paper
results/            per-experiment JSON output + diagnostic PNGs
paper/              LaTeX source, figures, and make_paper_figures.py
RESULTS.md          full experiment log, including results not reported in the paper
```

Every figure in the paper is regenerated from `results/*.json` alone by
`experiments/make_paper_figures.py`. No figure is drawn from a live model call, so figures
cannot drift from the numbers in the text.

## Reproducing

Environment: Python 3.11, PyTorch 2.13, `polymathic-aion==0.0.2`. Weights are pulled from the
Hugging Face Hub: `polymathic-ai/aion-base` (revision `40541618`) and `polymathic-ai/aion-large`
(revision `cfdb89e8`). Results in the paper were produced on Apple Silicon with the MPS backend;
the codecs are bit-exact against CPU.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python experiments/build_cache.py            # cutout + token cache
python experiments/<script>.py               # any single experiment
python experiments/make_paper_figures.py     # rebuild all six figures from stored JSON
```

Scripts write to `results/<name>.json` and are independently runnable; none depends on another
script's in-memory state.

## Data

Public. The PROVABGS-DESI-Legacy cross-match is built from the PROVABGS value-added catalogue
and the DESI Legacy Imaging Surveys DR9 cutout service; the HSC shard is one file of
MultimodalUniverse HSC PDR3. `astron_mi/data.py` documents the expected on-disk layout. The
cross-match file itself (113 GiB) is not redistributed here; it is rebuilt by the loader.

## Map from paper claims to scripts

| Paper | Script | Result file |
|---|---|---|
| §3.1 gate existence, Table 1 | `d2b_gating_scope.py` | `d2b_gating.json` |
| §3.1 absolute fluxes, Fig 1a | `d2_segmap_intervention.py` | `d2_segmap.json` |
| §3.2 mechanism discriminator | `d2_segmap_intervention.py` | `d2_segmap.json` |
| §3.3 real blends, Fig 2 | `d2f_real_blends.py`, `d2g_severe_blends.py` | `d2f_real_blends.json`, `d2g_severe_blends.json` |
| §3.4 gate direction | `d2c_gate_direction.py` | `d2c_gate_direction.json` |
| §3.5 metadata preference, Table 2 | `e2_metadata_preference.py` | `e2_metadata_preference.json` |
| §3.6 scaling, Table 3 | `d2e_scale_replication.py` | `d2e_scale.json` |
| §4.1 incidence | `e1b_faithful_miss.py` (Part A) | `e1b_faithful_miss.json` |
| §4.2 tomographic bias, Table 4, Fig 3 | `e1b_faithful_miss.py` (Part B) | `e1b_faithful_miss.json` |
| §4.3 mitigation, Table 5 | `d2d_mitigation.py` | `d2d_mitigation.json` |
| §5.1 codebook, Table 6 | `a0b_code_frequency.py`, `a0c_hsc_codebook.py` | `a0b_code_frequency.json`, `a0c_hsc_codebook.json` |
| §5.2 brightness ladder, Fig 5 | `b4_code_semantics.py` | `b4_code_semantics.json` |
| §5.3, §6.1 calibration | `b1_calibration.py` | `b1_calibration.json` |
| §6.2 line-locking, Fig 6a | `c1_redshift_mechanism.py`, `c1b_line_aliases.py` | `c1_redshift.json`, `c1b_aliases.json` |
| §7 dictionary learning, Fig 6b | `d1_gate_dictionary.py`, `d1b_sae_sweep.py` | `d1_gate_dictionary.json`, `d1b_sae_sweep.json` |

`e1_incidence_and_impact.py` / `e1_incidence_impact.json` are retained: they are the superseded
version of §4.2, kept so the correction documented in Appendix B can be checked.
