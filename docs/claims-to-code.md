# Claim → code → output

Every numbered claim in the manuscript maps to one script and one JSON file. Section numbers
refer to `paper/main.pdf`.

| § | Claim | Script (`experiments/`) | Output (`results/`) | Key path |
|---|---|---|---|---|
| 3.1 | Every readout is gated; Table 1 | `d2b_gating_scope.py` | `d2b_gating.json` | `effects_sigma.img.<readout>.{shifted_25,swapped,placebo_ra}` |
| 3.1 | Absolute flux collapse 6.71 -> 1.55 nmgy; Fig 1a | `d2_segmap_intervention.py` | `d2_segmap.json` | `arms.<arm>.median_pred` |
| 3.2 | Mechanism discriminator r=0.474 vs 0.298; Fig 1b | `d2_segmap_intervention.py` | `d2_segmap.json` | `mechanism.corr_frac_dpred_vs_{dcentral_coverage,denclosed_light}_allarms` |
| 3.3 | Real blends R=-0.012; Fig 2a | `d2f_real_blends.py` | `d2f_real_blends.json` | `integration.{R_median,R_lo16,R_hi84,n_used}` |
| 3.3 | Severe blends R=-0.0064; Fig 2b | `d2g_severe_blends.py` | `d2g_severe_blends.json` | `R_all_severe.{median,lo16,hi84,n}` |
| 3.4 | Gate direction 46% at alpha=4 | `d2c_gate_direction.py` | `d2c_gate_direction.json` | `steering.{gate_direction,random_matched_norm}` |
| 3.5 | Metadata preference; Table 2, Fig 4b | `e2_metadata_preference.py` | `e2_metadata_preference.json` | `arms.<arm>.err_vs_truth, comparison.*` |
| 3.6 | Scaling; Table 3 | `d2e_scale_replication.py` | `d2e_scale.json` | `models.{base,large}.intervention.*` |
| 4.1 | Incidence 3.68%; Fig 3a | `e1b_faithful_miss.py` | `e1b_faithful_miss.json` | `part_a.*` |
| 4.2 | Tomographic bias; Table 4, Fig 3b | `e1b_faithful_miss.py` | `e1b_faithful_miss.json` | `part_b.by_config['img|miss_real'].bins` |
| 4.4 | Correlated-miss sensitivity; Appendix C | `e1c_correlated_miss.py` | `e1c_correlated_miss.json` | `part_a.p_miss_by_mag_bin_raw`, `draws.{uniform,correlated}` |
| 4.3 | Mitigation; Table 5, Fig 4a | `d2d_mitigation.py` | `d2d_mitigation.json` | `quality.<config>.<readout>` |
| 5.1 | Codebook utilisation; Table 6, Fig 5a | `a0b_code_frequency.py` | `a0b_code_frequency.json` | `modalities.<modality>.{codes_emitted,perplexity}` |
| 5.1 | HSC codebook containment | `a0c_hsc_codebook.py` | `a0c_hsc_codebook.json` | `{codes_emitted,perplexity,contained_in_legacy}` |
| 5.2 | Brightness ladder; Fig 5b,c | `b4_code_semantics.py` | `b4_code_semantics.json` | `codes, pc1_explained, r2_flux_pc1` |
| 5.3 | Redshift quantisation floor | `b1_calibration.py` | `b1_calibration.json` | `tok_z.*` |
| 6.1 | Posterior calibration | `b1_calibration.py` | `b1_calibration.json` | `coverage.*` |
| 6.2 | Line-locking; Fig 6a | `c1_redshift_mechanism.py` | `c1_redshift.json` | `{rest_frame,observed_frame}.{x,profile}, contrast` |
| 6.2 | Line aliases falsified | `c1b_line_aliases.py` | `c1b_aliases.json` | `fail_rate_halpha, on_alias_fraction, random_null_fraction` |
| 7 | Dictionary sweep; Fig 6b, Table 7 | `d1b_sae_sweep.py` | `d1b_sae_sweep.json` | `configs[], baseline_curves, pca_curves` |
| 7 | Single-dictionary FVU vs PCA | `d1_gate_dictionary.py` | `d1_gate_dictionary.json` | `fvu_sae, fvu_pca` |

## Superseded and support scripts

| Script | Purpose |
|---|---|
| `build_cache.py` | builds the cutout cache most experiments read |
| `build_token_cache.py` | builds the paired activation cache used by `d1`/`d1b` |
| `make_paper_figures.py` | rebuilds all six figures from `results/*.json`, no model call |
| `a0_provenance_embeddings.py` | embedding-table provenance, reported in the log only |
| `b3_ordinal_geometry.py` | scalar-token ordinal geometry, reported in the log only |
| `c0_spectral_receptive_field.py` | codec effective resolution (~280 Å), used by §6.2 |
| `d0_dilution.py` | metadata dilution scan, superseded by `e2` |
| `e1_incidence_and_impact.py` | superseded by `e1b_faithful_miss.py`; see ../RESULTS.md |

## Reproducing a single number

```bash
python -c "import json; d=json.load(open('results/d2b_gating.json')); \
  print(d['effects_sigma']['img']['tok_flux_g'])"
```

To re-derive it from the model, run the corresponding script; it will overwrite the JSON.
