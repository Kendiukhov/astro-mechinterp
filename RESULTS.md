# Results log

Environment: venv at `~/.venvs/astron-mi` (the project drive is exFAT — no symlinks, and
AppleDouble `._*` files break pip's distribution finder, so the venv must live on the
internal disk). `polymathic-aion` 0.0.2, torch 2.13, MPS available. `aion-base` = 314.3M
params, 40 modality embedding tables.

---

## A0 — Shared-codebook two-table provenance test ✅ done

`experiments/a0_provenance_embeddings.py` → `results/a0_provenance.json`,
`results/a0_provenance.png`, `results/a0_usage.png`. Weights only, no data, ~2 min.

### A0.1 (unplanned) — AION's image codebook is mostly dead

Row norm is a usage proxy: a code that is never emitted gets no gradient and is pulled
below its initialisation norm (`init_std=0.02` → expected init norm `0.02·√768 = 0.554`)
by weight decay. Observed dead rows sit at **0.19**, i.e. *below* init — decayed, not
merely untrained.

| modality | codec | codes in use |
|---|---|---|
| `tok_ebv`, `tok_shape_r`, `tok_parallax`, `tok_ra`, `tok_xp_bp` | empirical-CDF | **100.0%** |
| `tok_flux_g` | empirical-CDF | 99.5% |
| `tok_segmap` (FSQ 1000), `tok_spectrum_sdss/desi` (LFQ 1024) | — | **100.0%** |
| `tok_z` | **linear** grid over z | **73.1%** |
| `tok_image` (Legacy) | FSQ 4375 | **23.2%** (1014) |
| `tok_image_hsc` (HSC) | FSQ 4375 | **9.6%** (422) |

This has a built-in positive control that works: an empirical-CDF codec makes every code
equiprobable *by construction*, so those modalities must be fully used — and they are, to
the row. The single scalar that is **not** CDF-quantised is `tok_z` (a `GridScalarCodec` on
a linear grid over z, found earlier by reading the source), and it is the only scalar with
dead codes, at exactly the rate you would expect from a linear grid over a
heavily-skewed redshift distribution. The control predicts which modality should break,
and only that one breaks.

So: **AION's image tokenizer uses 23% of its FSQ codebook, and HSC exercises only 9.6%.**
Further, HSC's live set is *exactly contained* in Legacy's (intersection = 422 = all of
HSC's live codes; Jaccard 0.416).

Consequences: B4 has ~1000 image codes to interpret, not 4375; anyone adapting AION to a
new imaging survey has only ~400 "HSC-shaped" codes to work with; and this is a candidate
partial explanation for image-only performance lagging spectra.

*Caveat — now measured directly, and the proxy was WRONG in the generous direction. See
A0b below: only 8.3% of image codes are ever emitted, not 23.2%, and the effective vocabulary
on source patches is ~28 codes.*

### A0.2 — The actual test: is provenance an additive tag?

Restricted to codes live in both tables. Null for the cosine is the index-shuffled
median, which is **not** zero because the embeddings are anisotropic.

| | spectra (SDSS↔DESI) | images (Legacy↔HSC) |
|---|---|---|
| live codes | 1024 / 1024 (100%) | 422 / 4375 (9.6%) |
| per-code cosine, matched | **+0.732** | +0.498 |
| per-code cosine, shuffled null | +0.023 | +0.284 |
| **excess over null** | **+0.709** | **+0.214** |
| additive-tag fraction | 0.029 | 0.155 |
| Δ rank-1 fraction | 0.059 | 0.318 |
| dims for 90% of ‖Δ‖² | **254** | 41 |
| deflated Δ ~ FSQ lattice, held-out R² | n/a | **+0.616** |

**Verdict: provenance is NOT an additive tag, in either pair.**

- **Spectra** are the well-conditioned instance (100% live, 4× smaller vocabulary). A given
  LFQ code means substantially the same thing to the model whether it arrived from SDSS or
  DESI — cosine 0.73 against a 0.02 null is a strong index-specific correspondence. But the
  *difference* between the tables is spread over 254 dimensions and only 2.9% of it is a
  single constant offset. Shared semantics plus a high-rank survey-specific modulation.
- **Images** show weaker index correspondence (excess +0.21), and critically the survey
  difference is substantially a **smooth function of the codebook's own lattice geometry**:
  held-out R² = 0.62 regressing the deflated difference on the FSQ lattice coordinates.
  That is the signature of survey-conditioned *visual semantics* — the difference depends
  on what the code looks like, not just on which telescope produced it.

This is the second branch of the pre-registered decision rule in AGENDA.md §5A. The naive
version of the A3 deliverable — "project out one provenance direction at the embedding
layer" — **does not exist**.

### What this does and does not license

It does **not** kill A2. LEACE operates on the *residual stream*, a different object from
the input embedding tables: survey identity could still be linearly decodable and cleanly
removable mid-network even though the input tables differ in a high-rank, content-dependent
way. What A0 changes is the prior — A2 should expect to erase a multi-dimensional subspace,
and the risk that erasure also removes physical information is now the thing to measure,
not a formality. Run A2 with the trade-off curve as the headline, not as a check.

*Weakly determined:* the "unexplained after ridge linear map" numbers (0.59 spectra, 0.29
images) come from fits with n_train < 768, so they are regularisation-dependent. Do not
quote them. (The first pass of this experiment used in-sample least squares and reported
0.06 for spectra — pure overfitting; the held-out split fixed it.)

### Next

- Direct code-frequency measurement to replace the row-norm proxy (needs cutouts).

---

## B3-lite — Ordinal geometry of the scalar codebooks ✅ done

`experiments/b3_ordinal_geometry.py` → `results/b3_ordinal.json`, `results/b3_ordinal.png`.
Weights only, ~3 min.

### B3.1 — AION learned the ordinal structure of a free lookup table

Nothing in the architecture tells the model that code 500 lies between 499 and 501: the
embedding is an unconstrained 1024×768 lookup. It learns the ordering anyway. Median
consecutive-step distance ÷ median random-pair distance (1.0 = no ordering):

| modality | encoder table | index-shuffled control | Spearman(dist, \|i−j\|) |
|---|---|---|---|
| `tok_flux_g` | **0.253** | 0.995 | 0.855 |
| `tok_ra` | 0.294 | 0.975 | 0.341 |
| `tok_dec` | 0.305 | 0.985 | 0.576 |
| `tok_parallax` | 0.352 | 1.004 | **0.879** |
| `tok_shape_r` | 0.389 | 0.986 | 0.864 |
| `tok_z` | 0.405 | 0.999 | 0.230 |
| `tok_ebv` | 0.729 | 1.005 | 0.634 |

Every codebook is a smooth 1-D curve traversed in index order. The strength of the
monotone ordering tracks how physically load-bearing that ordering is: strong for
parallax / flux / radius (0.85–0.88), weak for sky position and redshift.

### B3.2 — AION learned that right ascension is periodic

The sharpest result so far, and it has ground truth. `tok_ra` and `tok_dec` are handled
identically — same codec class (`ScalarCodec` + `ScalarReservoirQuantizer`), same
architecture path, same free 1024-row table. But decoding the codebooks to physical units:

| | code 0 | code 1023 | true separation on the sky |
|---|---|---|---|
| `tok_ra` | 0.0042° | 359.9968° | **0.008° — the same point** |
| `tok_dec` | −89.50° | +88.87° | **178° — opposite poles** |

So a model that recovered sky topology must place RA's extreme codes together and Dec's
apart. It does. Distance between the first m and last m codes, ÷ median pair distance:

| modality | m=1 | m=5 | m=10 | m=25 |
|---|---|---|---|---|
| **`tok_ra`** | **0.454** | **0.486** | **0.564** | **0.726** |
| `tok_dec` | 1.313 | 1.084 | 1.082 | 1.164 |
| `tok_parallax` | 1.625 | 1.601 | 1.515 | 1.357 |
| `tok_flux_g` | 1.660 | 1.521 | 1.396 | 1.307 |
| `tok_shape_r` | 1.906 | 1.568 | 1.536 | 1.668 |
| `tok_ebv` | 3.217 | 2.862 | 2.569 | 2.191 |

RA is the only modality below 1.0, and it is below across every block size. Independently,
the distance-vs-separation curve (fig, bottom left) **turns down** at \|i−j\|→K for RA alone;
every non-periodic scalar rises or saturates. Periodicity is supplied by neither the codec
nor the architecture — it is learned from data.

*A caution I initially got wrong.* The obvious global statistic — Spearman with circular
index distance minus Spearman with linear distance — is nearly flat for RA (+0.056) and I
first read that as refuting the result. It does not: wraparound only affects pairs at
near-maximal index separation, a few percent of random pairs, so a global rank correlation
is insensitive to it by construction. The statistic is still worth reporting because it
behaves correctly in the negative direction — genuinely monotone scalars show clearly
negative gains (`tok_ebv` −0.242, `tok_shape_r` −0.252, `tok_flux_g` −0.185), confirming
the test can detect non-periodicity.

### B3.3 — Ordinal structure lives where the modality is used

Comparing the encoder embedding with the decoder unembedding (`to_logits`), Spearman(dist,
index): every modality is more ordinal on the **encoder** side — `tok_parallax` 0.879→0.325,
`tok_shape_r` 0.864→−0.015, `tok_flux_g` 0.855→0.415 — with exactly one reversal:

- **`tok_z`: 0.230 (encoder) → 0.846 (unembedding)**, with top-3 PCs holding 74% of variance.

Redshift is predominantly a prediction *target* and rarely an input; photometry and shape
are predominantly inputs. The ordinal geometry sits on whichever side the modality is
actually used, and reading the tables tells you the training role. `tok_shape_r`'s
unembedding has no ordinal structure at all (−0.015), i.e. it is essentially never predicted.

### B3.4 — Metric confirms the codec family split

Local step size vs position (fig, bottom right) is flat for CDF-quantised scalars
(`tok_ra`, `tok_parallax`) — uniform spacing in *quantile*. `tok_z`, the one scalar on a
linear grid, is strongly non-uniform (2.1× at low z falling to 0.7× at high z), i.e. the
model re-weights a physical grid toward where objects actually are.

### Incidental: `tok_z`'s 1025th code is a special token

`tok_z` has vocab 1025 while its quantizer holds exactly 1024 value buckets
(`q.decode(arange(1024))` is the full range 0→6; index 1024 is out of bounds). So the extra
code is a sentinel, not a value. This closes one of the open questions in AGENDA.md §5B1.
Confirmed too: `Z` is `GridScalarCodec` + `ScalarLinearQuantizer`, linear on [0, 6],
Δz = 6/1023 = 0.005865, exactly as predicted from source.

---

## Data pipeline ✅ built

Package `astron_mi/`: `loading` (model/codecs), `encode` (safe raw→tokens), `hooks`
(activations), `physics` (token↔physical units), `data` (fixtures + HDF5). Verified end to
end: HDF5 rows → modalities → tokens → activations → labelled feature matrix.

### Data sources

- **`data/provabgs_desi_ls.hdf5` — 113 GiB, 115,404 galaxies.** The AION tutorial's loader is
  broken for it: it assumes flat keys and would materialise the whole file. The real layout is
  three nested groups (`desi`, `legacysurvey`, `provabgs_desi_edr_sv3`, 70 datasets).
  `ProvabgsReader` reads lazily by row.
  Contents: DESI spectra (7781 px), Legacy imaging `(4, 160, 160)` + ivar + **`object_mask`, the
  pipeline segmentation map**, all 8 Legacy fluxes + E(B−V), and PROVABGS physical labels
  (redshift, log M⋆, log Z_MW, age, log sSFR, SFR, dust, burst fraction, plus SFH parameters).
  The segmentation map matters: it is the input to D2, the highest-discovery-potential project.
- **The AION authors' test fixtures** (`aion-repo/tests/test_data/`) turned out to be more
  valuable than the 113 GiB file for method development: 64 real Legacy stamps, 128 real DESI
  spectra, and reference encoded/decoded tensors for ~37 modalities. They double as a
  regression suite and need no download.
- The two Gaia files the AION notebooks reference (`~lparker/gaia_{desi,apogee}.fits`) now
  return **HTTP 403**. Stellar work needs another route.

### Five traps found and neutralised

1. **Spectrum sentinel padding — the dangerous one.** Real DESI spectra have 7781 pixels; the
   codec expects a trailing point at `wavelength=99999`. Omitting it changes **20% of the 273
   tokens**, at positions spread across the whole grid, because the sentinel changes
   `interp1d`'s out-of-range test and hence the codec's global GRN statistic. It is invisible
   end to end (E[z] agrees to 3e-4) and shows up only in token-resolved work — exactly what
   this project does. `encode_spectrum` enforces it; `check_regression()` guards it against the
   authors' reference tokens (currently 0/273 differ).
2. **Version skew.** Installed `polymathic-aion==0.0.2` lacks repo HEAD's `pad_spectrum`.
   0.0.2 reproduces the reference tokens bit-exactly, so it is pinned.
3. **`NormAttention`, not `Attention`.** `aion-base` has `qk_norm=True` and instantiates
   `NormAttention`/`NormCrossAttention`, which do **not** subclass `Attention`. Patching
   `Attention.forward` for ablations is a silent no-op. (This corrects an earlier note here.)
4. **In-place mutation is real on this data.** PROVABGS cutouts reach 9.5 nmgy, above the
   DES-G clamp of 6, so re-encoding an unclonedtensor gives different tokens. All encode paths clone.
5. **`-999` missing-value sentinels** in six PROVABGS columns; unmasked they silently poison
   every probe. `mask_sentinels` converts them to NaN.

Also: `CodecManager(device='mps')` does **not** move your inputs — pass them yourself and
codecs run **34–56× faster, bit-exact** (spectrum 1604 → 29 ms/obj). Catalog codec ids need
`+204` before entering the model or it raises IndexError; `catalog_to_model_ids` handles it.

### Measured: the norm-heterogeneity hazard is mild

The agenda warned that dictionaries would waste capacity on whichever modality has the largest
activation norm. Measured ratio (image vs spectrum rows), real data:

| layer | 0–8 | 9 | 10 | 11 | final |
|---|---|---|---|---|---|
| ratio | 1.00–1.06 | 1.16 | 1.18 | 1.27 | 1.00 |

Mild through layer 8, 1.27 at the last block, and zero post-LayerNorm. Per-type
standardisation is worth an ablation, not a redesign.

---

## C0 — Spectral receptive field ⚠️ done, kill criterion FIRED

`experiments/c0_spectral_receptive_field.py` → `results/c0_receptive_field.{json,png}`.
48 real DESI spectra, both directions.

**The λ↔token map is confirmed exactly.** Injecting a line at a known wavelength moves the
predicted token in **99–100%** of trials (100% within ±1). So token *p* ↔
[3500 + (p−1)·25.6, +25.6) Å holds, and token 0 is the log-flux normalisation token.

**But the effective receptive field is far wider than one token:**

| injected line EW | tokens changed | flip FWHM |
|---|---|---|
| 5 Å | 13.1 | 9 tokens (230 Å) |
| 15 Å | 20.8 | 11 tokens (282 Å) |
| 30 Å | 37.8 | 12 tokens (307 Å) |

Reverse direction: flipping one token moves decoded flux with a tight **43 Å core** (1.7
tokens) but a low-level tail at ~1% of peak spanning **6962 Å** — the whole spectrum. That is
the GRN sequence-axis reduction made visible.

**Verdict: effective FWHM = 11 tokens ≈ 282 Å, which fails the agenda's <6-token criterion.**
Per the pre-registered rule, every wavelength-resolved claim must now be stated in terms of
**spectral regions of ~300 Å**, not lines. Concretely:

- [O III] 4959/5007 (48 Å apart) — **not separable**
- [N II] 6548/6583 from Hα 6563 — **not separable** (sub-token)
- Hα vs Hβ (1700 Å apart) — separable
- the 4000 Å break vs the Hα complex — separable

This directly rewrites C1: the redshift-mechanism experiment must test *break-fitting versus
line-complex-locking over ~300 Å regions*, and cannot claim single-line attribution. The
earlier synthetic-spectrum pass reported 3–5 tokens; that used a different statistic (influence
magnitude, not flip probability) **and** the unpadded tokenisation, so it is superseded.

The asymmetry is itself the useful result: reading *out* of a token is local (43 Å), attributing
*into* tokens is delocalised (~280 Å). Interpreting what a token means is on firmer ground than
asking which token carries a given line.

---

## Activation cache

`experiments/build_cache.py` → `results/cache_N3000_rand.npz` (277 MB, 3000 galaxies,
mean-pooled activations per modality at layers 5/11/final, `tok_z` logits under a three-step
information ladder, PROVABGS labels, AION scalar inputs). ~0.37 s/object on MPS.

**Sampling matters, badly.** The first build used the first 2000 rows. The file is
healpix-ordered, so that is a single contiguous **2.3° × 2.5° sky patch** — and it distorted
the results: redshift probe R² was 0.48 there versus **0.91** on a random draw. Anything
touching position, dust, depth or survey systematics would have been confounded. The builder
now samples random rows. (Random access to a 113 GiB file is ~4× slower than sequential;
worth it.)

---

## D0 — Token dilution and cross-modal directions ✅ done

`experiments/d0_dilution.py` → `results/d0_dilution.{json,png}`. n = 3000, ridge probes with
5-fold out-of-fold R², all from a single forward pass.

| target | all rows (mean) | image rows | spectrum rows | img+spec | **baseline: input fluxes+colours** |
|---|---|---|---|---|---|
| z | 0.907 | 0.887 | **0.923** | 0.917 | 0.597 |
| log M⋆ | 0.916 | 0.898 | 0.922 | **0.927** | 0.620 |
| log Z_MW | 0.502 | 0.494 | 0.512 | **0.518** | 0.316 |
| t_age_MW | 0.430 | 0.415 | 0.421 | **0.431** | 0.168 |
| log sSFR | 0.685 | 0.684 | **0.688** | 0.688 | 0.474 |

**Question 1 — dilution: no.** Probing from spectrum rows alone beats the all-rows mean by only
**+0.016** R² on redshift. Token dilution does not explain a large pooling collapse in this
setup, so the paper's Table 6 effect needs another explanation (or is specific to their
attentive-pooling head). Filed as unresolved rather than explained.

**Question 2 — shared direction: yes, D1 is worth running.** Probes fitted on image rows and on
spectrum rows have weight-vector cosine 0.16–0.31, and cross-applying them transfers well:
the spectrum-trained z probe scores **R² = 0.70** applied to image rows, and the image-trained
one scores 0.57 on spectrum rows. A partially modality-independent physical direction exists,
which is what a cross-modal dictionary would be looking for.

AION beats its own scalar inputs on every target (z: 0.91 vs 0.60; age: 0.43 vs 0.17), so the
representation carries real information beyond the catalogue numbers it was fed.

---

## B1 — Calibration of the native single-token posterior ✅ done, and it has a headline

`experiments/b1_calibration.py` → `results/b1_calibration.{json,png}`. n = 3000.

| ladder | KS(PIT) | median \|Δz\| (mean / median / mode) | posterior sd | 50% | 68% | 90% | 95% | 99% |
|---|---|---|---|---|---|---|---|---|
| photometry | 0.259 | 0.236 / **0.050** / 0.047 | 0.567 | 0.513 | 0.662 | 0.932 | 0.993 | 0.999 |
| + imaging | 0.134 | 0.098 / **0.015** / 0.014 | 0.383 | 0.667 | 0.833 | 0.970 | 0.988 | 0.999 |
| + spectra | 0.408 | 0.003 / **0.003** / 0.003 | 0.032 | 0.546 | 0.568 | 0.623 | 0.653 | **0.751** |

Three findings.

**1. Never use the posterior mean.** With photometry only, median |Δz| is **0.236 for the mean
versus 0.050 for the median** — a 4.7× penalty. The linear z ∈ [0, 6] grid drags the mean
upward whenever the posterior is broad. Use the median or mode. This is a one-line change that
anyone reading `tok_z` should make.

**2. Photometry-only posteriors are well calibrated; adding imaging makes them too wide.**
Coverage 0.513/0.662 at nominal 0.50/0.68 for photometry is close to ideal. With imaging added
the intervals become conservative (0.667 at nominal 0.50).

**3. With spectra the intervals are far too narrow in the tails — nominal 99% covers 75%.**
This survives restriction to the BGS design range z ≤ 0.6 (n = 2987), where the outlier rate is
only 0.27% and median |Δz| = 0.0029. So it is not driven by a high-z tail.

**But the honest reading is quantisation, not overconfidence.** `tok_z`'s linear grid has
dz = 0.005865, and:

- median |Δz| = 0.002917 = **0.497 × the grid spacing**
- **80.8%** of objects have >90% of posterior mass in a *single* bucket

The error is pure discretisation. With spectra, AION's redshift readout is
**quantisation-limited, not information-limited**: it cannot express a posterior narrower than
one bucket while its residual error sits at exactly that scale, so the 90–99% intervals
under-cover almost by construction. DESI spectroscopic redshifts are good to ~1e-5–1e-4, i.e.
**59–587× finer than the token grid**. For spectroscopic-precision work the tokeniser, not the
model, is the bottleneck — and no amount of scaling the transformer fixes it.

This is the answer B1 was for: the output class the authors endorse is usable for photometric
redshift with the median as the estimator, and its uncertainties are trustworthy in the
photometry regime but not in the spectroscopic one.

*Bug caught in passing:* the first version of the coverage computation used bucket centres, so
a posterior concentrated in one bucket got a zero-width interval and coverage collapsed to ~6%
— which would have read as catastrophic miscalibration. It must use bucket edges.

---

## D2 — Does the flux estimate follow the detection mask or the light? ✅ done

`experiments/d2_segmap_intervention.py` → `results/d2_segmap.{json,png}`. n = 240 random
galaxies. **Image pixels byte-identical in every arm; only the segmentation map changes.**

AION takes the Legacy Survey segmentation map as an input modality (`tok_segmap`), which
makes it — as far as I know — the only astronomical foundation model with an editable
*detection* channel. ~60% of LSST sources will be blended, so whether a model reads the sky
or the pipeline's opinion of the sky is a live deployment question.

| arm (pixels identical) | median E[flux_g] | change vs real mask |
|---|---|---|
| **image_only** (no mask supplied) | 6.58 | 2.1% |
| **real** mask (reference) | 6.71 | — |
| dilated 2 px | 7.04 | 4.4% |
| eroded 2 px | 6.06 | 5.3% |
| dilated 10 px | 5.31 | 21.3% |
| **swapped — another galaxy's REAL mask** | **4.93** | **13.0%** |
| eroded 5 px | 2.34 | 59.7% |
| shifted 10 px | 2.63 | 57.7% |
| shifted 25 px | 1.55 | 78.8% |
| erased (off-manifold, discounted) | 1.41 | 81.0% |
| **placebo** — RA moved 250 codebook steps | 6.70 | **0.0%** |

**The mask wins over the pixels.** Displace the mask by 25 px and the flux estimate falls by
79%, though every photon is still in the image. The decisive arm is `swapped`: it uses another
galaxy's genuine Legacy Survey mask, so nothing is out of distribution, and it still moves the
estimate by 13.0% against a **0.0%** placebo (RA tokens verified to change in 100% of objects).

**The sharpest framing: AION does not need the mask, but it will believe a wrong one over the
image.** Supplying no mask at all costs only 2.1% (and 6.4% error against the true flux). The
damage comes from supplying an *incorrect* mask.

### Mechanism: detection gating, not aperture photometry

The obvious hypothesis is that the model integrates light inside the supplied mask. It does
not. Within object, correlating the fractional change in prediction against the change in each
candidate driver across all 9 intervention arms:

| driver | correlation |
|---|---|
| light enclosed by the mask | +0.298 |
| **central mask coverage** (does the mask assert a source at the centre?) | **+0.474** |

Central coverage wins, and the graded series confirms it: eroding 5 px collapses the estimate
(2.34) while dilating 10 px barely does (5.31), which is backwards for an aperture and exactly
right for a detection gate. Every "collapsed" arm lands on the same low floor of ~1.4–2.6 nmgy
— the model's "there is nothing here" answer.

So the mechanism is: **the segmentation map acts as a detection prior on whether a source is
present at the centre; when it says no, the model reports near-zero flux regardless of the
photons.** Neither driver dominates (0.30 vs 0.47), so this is a leading explanation, not a
complete one.

*A statistic that would have been wrong:* the naive pooled correlation between prediction and
enclosed light is 0.97 and looks like decisive evidence for aperture photometry. It is entirely
driven by object-to-object brightness — bright galaxies are bright under every mask — and the
swap arm shows the same 0.97. Only the within-object delta is informative, and it is 0.09.

### Why this matters

This is a systematics finding, not a physics one, and it is the class of result the agenda
predicted was most likely to pay off. For any survey where detection and deblending are
uncertain — which is Rubin's central problem — a foundation model that defers to the detection
pipeline will launder that pipeline's errors as model predictions, with no signal that it has
done so. The 2.1% cost of simply not supplying the mask suggests an easy mitigation worth
testing: withhold the detection channel when it is untrustworthy.

---

## ⚠️ A bug that invalidated one experiment — read this first

`AION.forward(tokens, target_modality=[A, B, C])` **corrupts every readout.** It supplies an
all-zero `decoder_attention_mask` (model.py:99) which `adapt_decoder_attention_mask` turns
into "every position masked". With one decoder token that is harmless; with more than one the
decoder self-attention goes uniform over all decoder tokens and `decoder_sep_mask` cannot save
it. Measured on redshift with image+spectrum input:

| targets in one call | median \|Δz\| | max posterior |
|---|---|---|
| `[Z]` | **0.0038** | 0.895 |
| `[FluxG, Z]` | 0.833 | 0.413 |
| `[FluxG, FluxZ, Z]` | 2.350 | 0.148 |

My earlier note said the bug was "harmless for 1-token targets". That is true per target and
**false when several 1-token targets are queried together** — which is exactly what batching
does. D2b's first run batched five targets and its numbers were invalid; it has been re-run.
D2, D2c, D1 and B1 all used a single target and are unaffected.

`astron_mi.read_targets()` now loops one forward per target so this cannot recur.

---

## D2b — How general is the gate? ✅ done (re-run after the batching bug)

`experiments/d2b_gating_scope.py` → `results/d2b_gating.{json,png}`. n = 480. Effects in
robust σ of each readout's own spread. **These numbers are from the corrected re-run**; the
first version batched five targets into one forward and was invalid (see the bug note above).

**1. Every readout is gated — a global object-presence switch, not a photometry quirk.**
Imaging only, against a placebo that is now essentially exactly zero (0.000–0.005 σ):

| readout | shifted 25 px | swapped mask | placebo |
|---|---|---|---|
| flux g | 0.630 | 0.110 | 0.000 |
| **flux z** | **20.55** | **9.83** | 0.005 |
| size R | 1.313 | 0.718 | 0.003 |
| ellipticity e1 | 0.435 | 0.931 | 0.004 |
| redshift | 0.580 | 0.312 | 0.003 |

flux z is gated enormously — 20σ under a displaced mask — consistent with D2d, where a swapped
mask moved flux z by 2.9 magnitudes. A **displaced** mask is generally far more damaging than a
**swapped** one (0.630 vs 0.110 for flux g): wrong *location* hurts more than wrong *shape*.

**2. Independent evidence overrides the gate almost completely, where it applies.**

| readout | imaging only | + spectra |
|---|---|---|
| **redshift** | 0.312 | **0.003 (100× smaller)** |
| flux z | 9.83 | 3.04 |
| flux g | 0.110 | 0.074 |
| size R | 0.718 | 0.674 |
| **ellipticity e1** | 0.931 | **0.895 (unchanged)** |

The corrected numbers make this much sharper than the first pass suggested: spectra all but
eliminate the redshift gate (100×, not 2.4×), substantially reduce it for flux, and do nothing
for morphology — which is exactly right, since a spectrum informs redshift and colour and says
nothing about ellipticity. The model weighs evidence; the gate survives only where no
counter-evidence exists.

**3. Still NOT concentrated in the ambiguous regime — the honest negative stands.** With the
scale computed *within* stratum:

| stratum | n | flux g | flux z | size R | e1 | redshift |
|---|---|---|---|---|---|---|
| single source | 296 | 0.106 | 11.11 | 0.742 | 0.904 | 0.360 |
| blended (≥2) | 184 | 0.129 | 8.46 | 0.699 | 0.860 | 0.277 |
| bright half | 240 | 0.204 | 16.67 | 0.683 | 0.911 | 0.426 |
| faint half | 240 | 0.257 | 4.24 | 0.985 | 0.933 | 0.412 |
| faint AND blended | 94 | 0.204 | 3.71 | 0.813 | 0.941 | 0.344 |

Blendedness barely matters. Faint objects are slightly more gated on flux g and size, less on
flux z. **No consistent dependence** — the gate is broad and roughly uniform, so the "worst
exactly where LSST hurts" framing is not supported. It applies to every object instead.

## D2c — Is the gate a single direction? ✅ done — partially

`experiments/d2c_gate_direction.py` → `results/d2c_gate_direction.{json,png}`. n = 256.
The cheap decisive test before committing GPU-days to a dictionary.

Take activations under a real mask and a displaced mask (pixels identical), difference in
means → a candidate "there is a source here" direction. Then run with the *displaced* mask and
add α·d back into the residual stream at layer 6.

| α | gate direction recovers | random matched-norm control |
|---|---|---|
| 3 | 23% | −3% |
| **4** | **46%** | **5%** |
| 6 | 84% | 39% |
| 8 | 89% | 66% |

**A single 768-d direction carries roughly half the gate, specifically.** At α = 4 it recovers
46% of the flux collapse against 5% for a random vector of the same norm — a ~9× separation.

**But peak recovery is not the statistic to quote.** At α = 8 the gate direction recovers 89%,
which looks like a complete explanation until you notice the random control reaches 66% at the
same α: large-norm perturbations inflate the flux estimate non-specifically. I briefly read the
α = 8 number as the result before the matched-norm control corrected it. (My first pass made the
opposite error — sweeping only to α = 3 gave 18% and read as a flat negative. The response peaks
and then over-steers, so the sweep has to bracket the peak *and* carry the control at every α.)

Supporting structure:
- **Direction consistency 0.797** — objects move largely the same way, so a shared direction is
  a reasonable description.
- **22 dimensions for 90%** of the object-to-object variation in the difference, rank-1 fraction
  0.27 — real residual structure beyond the mean direction.
- Steering at layer 11 alone reaches only 42% even at α = 16, so **the gate is computed upstream
  of the final blocks**; layer 6 is an effective intervention point.

### What this settles about the dictionary question

The gate is **partially** low-dimensional: about half of it is one direction, and the rest is
spread over ~20 dimensions with genuine per-object structure. That is the outcome that actually
justifies a dictionary rather than making it redundant —

- if the gate had been *fully* one direction, difference-in-means already gave it and an SAE
  would add nothing;
- if it had been fully smeared, a clean feature was never going to exist.

So D1 goes ahead, but **targeted, with a pre-specified causal success criterion**: does a small
set of dictionary features beat this difference-in-means baseline at restoring the readout,
at matched intervention norm? That is a falsifiable bar the usual "the features look
interpretable" framing never has. The difference-in-means direction is now the baseline any
dictionary has to beat — alongside PCA at matched sparsity.

---

## D1 — A dictionary targeted at the gate ❌ pre-registered test FAILED

`experiments/build_token_cache.py`, `astron_mi/sae.py`, `experiments/d1_gate_dictionary.py`
→ `results/d1_gate_dictionary.{json,png}`.

Infrastructure: per-token cache of **288,000 token rows per condition** (3000 objects ×
96 stratified rows, layer 6), cached *paired* — same objects under a real and a displaced
mask — so the dictionary sees the gate contrast directly. BatchTopK SAE, 8192 features,
k = 32, 4000 steps.

**Pre-registered criterion, fixed before running:** a small set of dictionary features, used
as a steering vector *at matched intervention norm*, must recover more of the flux collapse
than the difference-in-means direction.

### The dictionary is good at what dictionaries are for

- Reconstruction **FVU 0.031 vs PCA's 0.242** at matched k = 32 — **7.7× better**.
- **7347 / 8192 features live** (90%), so this is a healthy dictionary, not a collapsed one.
- It finds clear gate-selective features: #5048 fires on 19.3% of rows under a real mask and
  6.0% under a displaced one, and there are dozens like it.

### But it loses the causal test to the simplest possible baseline

Recovery of the flux collapse, all vectors at identical norm:

| α | SAE gate features | difference-in-means | PCA top-k | random |
|---|---|---|---|---|
| 3 | 3.0% | 35.5% | **43.9%** | 2.4% |
| 4 | 15.7% | 64.3% | **72.4%** | 13.4% |
| 6 | 63.7% | **91.8%** | 89.5% | 48.0% |
| 8 | 84.0% | 89.5% | 84.4% | 64.2% |

**Margin vs difference-in-means: −7.8 pp. The dictionary does not beat the simple baseline.**

It is worse than that, and the low-α rows are where to look, because that is where the random
control is still near zero. At α = 4 the SAE vector recovers **15.7% against random's 13.4%** —
essentially indistinguishable from a random vector of the same norm — while difference-in-means
gets 64.3% and PCA 72.4%. The SAE curve tracks the random curve almost exactly up to α = 6.
cos(SAE vector, difference-in-means) = 0.393, so it is pointing somewhere substantially
different, and that somewhere does not work.

The test had power: two other methods pass it comfortably on the same objects, same norms,
same code path. This is a real negative, not a broken harness.

### What it means

**Reconstruction quality and causal relevance are different things, and here they come apart
sharply.** A dictionary trained to reconstruct beat PCA 7.7× on reconstruction and still lost
to PCA on the intervention. Nothing in the SAE objective asks its features to align with the
directions the model actually uses, and on this task they do not.

This is consistent with the precedents the agenda flagged — Walrus SAE features were only
"piecewise consistent", and SAE downstream utility is weak in general — but it is a sharper
version, because here there is an unambiguous causal metric and a baseline that passes it.

**Practical upshot for the project: for the detection gate, use difference-in-means.** It is
two lines of numpy, it beats a 4000-step dictionary, and D2c already established its
specificity against a matched-norm control.

### Caveats, stated so the negative is not overclaimed

- Features were selected by **mean activation gap**, and combined weighted by that gap. A
  different selection rule (e.g. ranking features by measured causal effect) might do better.
  That would be post-hoc selection, which the pre-registration deliberately excluded — it is a
  legitimate *separate, exploratory* question, not a rescue of this one.
- One seed, one layer (6), one k (32), one width (8192). Multi-seed subspace stability was not
  run; on a failed test it would only tell us how reliably it fails.
- The gate may simply be an unusually *easy* target — a large, near-linear population-level
  shift is exactly what difference-in-means is best at. A dictionary might still earn its place
  on genuinely multi-modal or per-object structure. That is the honest steelman, and it is
  testable: the residual 22 dimensions from D2c are the place to look.

### Next

- **C1** (redshift mechanism): C0 caps resolution at ~300 Å regions, B1 says read the median.
- If revisiting dictionaries: select features by causal effect, not activation gap, and label
  clearly as exploratory.
- The cross-modal crosscoder (image ↔ spectrum features) remains the genuinely novel methods
  claim and is untouched by this negative — D0 measured a real shared direction (cross-applied
  R² = 0.70) and that is a different question from the gate.

---

## D2d — Is withholding the mask a viable mitigation? ✅ YES

`experiments/d2d_mitigation.py` → `results/d2d_mitigation.{json,png}`. n = 480, bootstrap
16–84% intervals over objects. The model is never given the scalar it is asked to predict.

This converts the finding from "here is a worrying property" into a recommendation.

**Quality against catalogue truth** (flux in |Δmag|, redshift in |Δz|):

| config | flux g | flux z | redshift |
|---|---|---|---|
| imaging only | **0.0665** [.0616,.0717] | 2.179 | **0.1169** [.1143,.1200] |
| imaging + real mask | 0.0693 [.0638,.0749] | **1.907** | 0.1273 [.1253,.1314] |
| imaging + generic disc | 0.2215 | 6.551 | 0.1566 |
| +spectra | **0.0534** [.0506,.0574] | 2.636 | **0.0034** [.0033,.0035] |
| +spectra + real mask | 0.0575 [.0544,.0609] | **2.527** | 0.0038 [.0036,.0039] |
| +spectra + generic disc | 0.1205 | 6.355 | 0.0044 |

**Vulnerability** (median change when the mask is swapped for another galaxy's real mask):

| config | flux g | flux z | redshift |
|---|---|---|---|
| imaging + real mask | **0.132 mag** | 2.91 mag | **0.0308** |
| +spectra + real mask | 0.086 mag | 1.78 mag | 0.0004 |
| any config without a mask | 0 by construction | 0 | 0 |

### Three findings

**1. Withholding the mask is free.** Redshift improves with non-overlapping CIs in both bases
(0.1169 vs 0.1273, and 0.0034 vs 0.0038). Flux g improves by 4–7% here — but D2e, on an
independent sample, found it 0.005 mag *worse* without the mask, so **flux g is best described
as quality-neutral**, with the difference either way inside the intervals. Only flux z
genuinely benefits from the mask (1.91 vs 2.18 mag), and both numbers there are poor, so it is
a choice between two bad estimates.

**2. It removes the entire vulnerability.** 0.13 mag on flux g and 0.031 in redshift, gone,
at no quality cost. For a survey where detection is uncertain, that is an unambiguous trade.

**3. Do NOT substitute a generic mask — that is much worse than either option.** A fixed
centred disc, carrying "there is a source here" and nothing else, degrades flux g by **220%**
and flux z to 6.6 mag. So the channel is *not* merely a presence flag: the model has learned
to read object-specific structure from it, and a plausible-but-generic mask actively misleads.
This kills the obvious "just feed a neutral mask" fix and leaves withholding as the mitigation.

Note also that spectra shrink the redshift vulnerability to 0.0004 — consistent with D2b:
independent evidence protects whatever it informs.

**Recommendation for anyone deploying AION:** if you do not trust your detection mask, do not
supply one. It costs nothing measurable on flux and redshift and eliminates a systematic that
is otherwise invisible in the output.

---

## D2e — Does it replicate at 2.7× scale? ✅ YES, and it gets WORSE

`experiments/d2e_scale_replication.py` → `results/d2e_scale.{json,png}`. n = 240 identical
objects through both models. aion-base (314M, 12/12, d=768) vs aion-large (860M, 24/24,
d=1024) — same architecture family, same codecs, same 40 modality tables.

This was the last thing standing between "aion-base has a quirk" and "models given a
detection channel defer to it".

**The gate persists and grows with scale.** Vulnerability = median change when the mask is
swapped for another galaxy's real mask:

| readout | aion-base | aion-large | ratio |
|---|---|---|---|
| flux g | 0.1465 [.1274,.1655] mag | **0.2367** [.1704,.2667] mag | **1.62×** |
| redshift | 0.0327 [.0266,.0361] | **0.0441** [.0377,.0518] | **1.35×** (CIs disjoint) |

**The 2.7×-larger model defers *more* to the detection pipeline, not less.** Scaling does not
fix this, and on current evidence makes it worse — which is the strongest version of the
finding for a paper, and the most consequential one for surveys planning to use bigger models.

**The mitigation still holds at scale.** Dropping the mask costs +0.005 mag on flux g (both
models, inside the intervals) and *improves* redshift on both — by 0.005 on base and **0.016
on large**, i.e. the benefit of withholding grows with scale too.

**Codebook utilisation is scale-invariant** (weights only, A0's proxy):

| modality | base | large |
|---|---|---|
| `tok_image` | 23.2% | 22.6% |
| `tok_image_hsc` | 9.6% | 9.6% |
| `tok_spectrum_desi`, `tok_ebv` | 100% | 100% |
| `tok_z` | 73.1% | 73.2% |

Nearly identical at 2.7× the parameters, which localises A0's dead-codebook result to the
**tokeniser and the data**, not to model capacity. Training a bigger transformer on the same
tokens will not revive those codes.

---

## D2f — Real blends: NO synthetic edits ✅ done, and it reframes the finding

`experiments/d2f_real_blends.py` → `results/d2f_real_blends.{json,png}`. 240 genuinely
blended cutouts (≥2 pipeline segments with a real source at the centre), found by scanning
random rows. **No synthetic geometry at all** — the only choice is which of the pipeline's own
segments the mask includes:

- `central_only` — mask = the segment containing the centre pixel (what FLUX_G refers to)
- `all_segments` — mask = every segment in the cutout (an under-deblended target)

Both are genuine pipeline output; the difference between them *is* the deblending decision.

**Control first.** The two masks tokenise to **28.5% different segmap tokens** (median), with
**zero** objects tokenising identically. The model definitely sees two different inputs.

**The quantitative test.** If the model integrates the light the mask assigns it, then
R = Δ(AION flux) / Δ(true light in the added segments) ≈ 1. Measured on 231 objects with real
neighbour light:

| quantity | value |
|---|---|
| median true light added by neighbour segments | **+1.053 nmgy** |
| median change in AION's flux | **−0.016 nmgy** |
| **R** | **−0.012** [−0.014, −0.008] |
| Spearman(Δlight, Δflux) | +0.028 |

**R ≈ 0. AION completely ignores which segments the mask includes.** A full nanomaggy of real
neighbour light is added to the object's mask and the readout does not move. Error against
catalogue truth is likewise unchanged (0.083 vs 0.079 mag, overlapping intervals).

### Why this is a *reframing*, not a contradiction

It looks like it conflicts with D2, where mask changes moved flux by up to 79%. It does not —
it is the sharpest confirmation yet of the D2c mechanism. Every intervention that produced a
large effect (**displace**, **swap**, **erase**) removes or moves the mask *at the centre*.
The D2f arms both keep the centre covered and differ only in the periphery. Detection gating
predicts exactly this: the mask is read as "is there a source here?", and adding neighbour
segments does not change that answer.

So the correct statement of the whole result is narrower and more precise than the LSST-blending
framing I had been using:

> **AION defers to the detection pipeline about whether a source is present at the centre, and
> ignores everything else the mask says — including how the pipeline split the light.**

### What that changes in practice

- **Deblending errors do NOT propagate through this channel.** Which pixels get assigned to
  which source is invisible to the model. The blending worry, as I had framed it, is not
  supported — and this experiment is the one that would have found it.
- **The dangerous failure mode is detection and position**, not segmentation: a missed source,
  a spurious one, or a mask centred in the wrong place. Those are the cases where the readout
  collapses (up to 79%).
- The mitigation from D2d is unaffected: withholding the mask removes the positional
  vulnerability at no quality cost.

This is why Tier 2 was worth doing before writing anything up. The synthetic experiments alone
would have supported a claim about blending that the real-blend data does not.

*Caveat, now closed by D2g:* this file's segmentation is Legacy Survey's, on 0.262″/px imaging
of relatively bright BGS galaxies, where the central segment holds a median 86% of the cutout
light. Severe, nearly-equal blends are rare here, so the null could have been an artifact of
mild blending. D2g selects on severity and finds it is not.

---

## D2g — Does the null survive SEVERE blends? ✅ YES

`experiments/d2g_severe_blends.py` → `results/d2g_severe_blends.{json,png}`.

Rather than taking whatever blends come, this scans 4000 rows and keeps only cutouts where the
central segment holds **≤70%** of the light — the closest thing to a crowded field this dataset
contains. **322 severe blends** (8.1% of cutouts), median central fraction 0.523, minimum
**0.012** (some cutouts are almost entirely neighbour light). Same tokenisation control: 38.9%
of segmap tokens differ, zero objects identical.

| central segment's share of the light | n | R = Δflux / Δlight |
|---|---|---|
| **0–40%** (extreme) | 94 | **−0.0046** [−0.0060, −0.0035] |
| 40–55% | 80 | −0.0059 [−0.0088, −0.0030] |
| 55–70% | 148 | −0.0103 [−0.0188, −0.0061] |
| **all severe** | 322 | **−0.0064** [−0.0075, −0.0046] |
| *(D2f, mild blends)* | 231 | *−0.0117* |

R is zero in every bin, **including the extreme one where the central segment holds under 40%
of the light**. A median of 2.93 nmgy of genuine neighbour light gets added to the mask and the
readout moves by −0.02 nmgy.

So the D2f null is a property of the model, not of a mild-blend sample. Within the range this
dataset spans — down to cutouts that are 99% neighbour light — **AION never tracks how the
pipeline divided the light.** The remaining untested regime is genuinely crowded imaging
(deeper, finer pixels, HSC or Rubin), which this file does not contain.
- Still missing for the mandatory systematics control: per-object `PSFSIZE_*`/`PSFDEPTH_*`/
  `GALDEPTH_*` via a Legacy Survey sweep-catalogue join on (ra, dec). `TSNR2_BGS` and `f_fiber`
  are in-file as weak proxies.


---

## C1 — Lines or continuum? ✅ LINE-LOCKING, dominated by Hα

`experiments/c1_redshift_mechanism.py` → `results/c1_redshift.{json,png}`. 185 objects the
model already gets right (|Δz| < 0.01 with the full spectrum), median z 0.246.

Causal ablation, not attention: mask a window of 8 spectrum tokens (~205 Å, matched to C0's
measured ~11-token resolution — ablating finer than the codec resolves would only blur the
answer), measure how far the `tok_z` posterior moves (Jensen–Shannon), then **stack the
273-vector in the rest frame** using the known redshift.

The ablation mechanism was validated first: masking all 273 tokens reproduces the image-only
redshift error exactly (0.1207 vs 0.1207); masking none reproduces the full result.

**The discriminator, with its null:**

| stack | contrast (max/median) |
|---|---|
| **rest frame** | **32.54** |
| observed frame | 4.36 |
| **shuffled-z control** | **2.95** |

The rest-frame profile is 7× sharper than observed-frame and **11× sharper than its own null**
(de-redshifting each object by another object's z). The peak sits at **6643 Å — Hα (6563 Å),
within one bin.**

**Causal weight by rest-frame feature** (profile value ÷ median):

| feature | weight |
|---|---|
| **Hα + [N II] 6563** | **25.0×** |
| Mg b 5175 | 15.4× |
| Na D 5892 | 12.8× |
| Ca H&K / 4000 Å break | 8.1× |
| [O III] 5007 | 6.8× |
| Hβ 4861 | 4.3× |
| [O II] 3727 | 2.6× |
| Ca II triplet 8600 | **0.10×** |

The Ca II triplet result is an internal consistency check that passes: at median z = 0.246 it
sits at 10 720 Å, past the codec's 10 462 Å limit, so it *should* carry no weight — and it
carries essentially none.

**Verdict: AION's redshift is line-locked, and Hα carries the most causal weight of any
feature by a factor of ~1.6 over the next.** It is not fitting continuum shape; the 4000 Å
break earns only 8×, a third of Hα.

### Why this matters, and what it predicts

Per the pre-registered reading in AGENDA.md §5C, line-locking is the **fragile** branch. A
continuum-fitter degrades gracefully; a line-locker fails *catastrophically and
unpredictably* when it misidentifies a line — and it is right almost always in between.

This gives a mechanistic explanation for **B1's tail miscalibration**: the nominal 99% interval
covers only 75%, i.e. the posterior core is excellent (median |Δz| = 0.003) but the tails are
far too thin. A model that locks onto Hα is right to be sharp when it has found Hα, and has no
representation of the possibility that it found the wrong line. Two independent experiments,
one mechanism.

**Falsifiable prediction, not yet tested:** the catastrophic outliers should cluster at
*redshift aliases* corresponding to line confusions, not scatter randomly. Confusing Hα (6563)
with [O III] (5007) puts the error at z_wrong = 1.311(1+z_true) − 1; with Hβ (4861) at
1.350(1+z_true) − 1. Checking whether AION's outliers land on those loci is the natural next
experiment and needs no new data.

*Caveats:* one sample (DESI BGS, median z 0.246), so Hα is comfortably inside the wavelength
range for nearly every object — its dominance is partly an availability effect and would not
transfer to a higher-z sample where Hα redshifts out. Windows are 205 Å, so "Hα" here means
the Hα + [N II] complex, which C0 showed cannot be separated.

---

## C1b — Testing C1's fragility prediction ⚠️ PREDICTION FALSIFIED (and the reason is instructive)

`experiments/c1b_line_aliases.py` → `results/c1b_aliases.{json,png}`. n = 2865 objects that are
baseline-correct with Hα inside the DESI range.

C1 predicted that a line-locked model should fail *catastrophically and at predictable places*.
Natural outliers are too rare to test (0.27% with spectra), so failures were induced causally:
ablate the ~205 Å window containing rest-frame Hα, whose observed position is known from the
true redshift, and see where the redshift goes.

**Part 1 — one window is load-bearing.** Ablating the Hα window pushes 6.1% of previously
correct redshifts past |Δz| > 0.02. An **identical-width window at a quiet rest wavelength
induces zero failures in 2865 objects.** So Hα really is doing the work, confirming C1 by an
independent causal route.

**Part 2 — the aliases are NOT predictable. Prediction falsified.** Of 174 induced failures,
only 5.7% land within 0.02 of a line-ratio alias, against a **43.4%** random-wavelength null.
The failures actively avoid the predicted loci.

**Part 3 — why, and it revises Part 1 too.** All 174 failures push the redshift **upward**
(100%, median ρ = 1.036, i.e. Δz ≈ +0.04). That is not a misidentification signature; it is the
signature of a broadening posterior dragging the *mean* up the linear z ∈ [0, 6] grid. Testing
the three estimators directly (n = 612):

| estimator | baseline \|Δz\| | Hα ablated \|Δz\| | failure rate |
|---|---|---|---|
| mean | 0.00324 | 0.00383 | **5.7%** |
| **median** | 0.00285 | **0.00293** | **1.1%** |
| mode | 0.00285 | 0.00291 | 1.3% |

Posterior entropy rises **2.37×** (0.136 → 0.321) when Hα is removed.

**The corrected conclusion: AION responds to losing Hα approximately correctly.** It broadens
its posterior by the right kind of amount and its median stays accurate to within 3%. Most of
the apparent "fragility" was my own readout choice — the posterior mean, which B1 had already
shown to be a bad estimator on a linear grid. A residual 1.1% genuine failure rate remains
(against 0% for the quiet control), so the dependence is real but modest.

**What this changes upstream.** C1's measurement stands — Hα carries 25× the median causal
weight, and that is measured on posterior *divergence*, not on a point estimate. But C1's
*interpretation* — that line-locking implies catastrophic fragility — is not supported for this
model. AION is line-dependent without being line-brittle.

It also means B1's tail miscalibration needs a different explanation than "unanticipated line
confusions". Since the model broadens correctly when information is removed, the under-covering
99% interval is more likely the quantisation floor B1 itself identified (median error = 0.497 ×
grid spacing) than a failure to represent uncertainty.

*Method note:* this is the third time in this project that the posterior mean has produced a
misleading result on `tok_z` (B1's point-estimate comparison, D2d's flux framing, and here).
The linear z ∈ [0, 6] grid makes the mean systematically wrong whenever the posterior broadens.
Use the median.

---

## A0b — Direct code-frequency measurement ✅ done, and it corrects A0

`experiments/a0b_code_frequency.py` → `results/a0b_code_frequency.{json,png}`. 2000 real
cutouts, **1.15M image tokens**.

A0 inferred codebook utilisation from embedding row norms. That is a proxy, and the caveat has
been carried in every write-up since. Measuring directly — encode real cutouts, count which
codes the codec emits — changes the number substantially.

| | row-norm proxy (A0) | **directly measured** |
|---|---|---|
| `tok_image` codes ever emitted | 23.2% | **8.3%** |
| `tok_spectrum_desi` | 100% | **100%** ✓ |

**The proxy overstated image usage by ~3×.** The reason is straightforward in hindsight: a code
emitted rarely still receives some gradient and keeps a large embedding norm, so the norm
threshold measures "ever meaningfully trained", not "actually used". The spectrum control
matches exactly, which is why the error was invisible.

### Effective vocabulary, and the sky confound

The raw histogram gives perplexity 7 for images — but a 96×96 cutout is mostly blank sky, and
that number is dominated by whatever code means "empty". Splitting tokens by whether the 4×4
patch overlaps the pipeline segmentation map:

| | tokens | codes emitted | **perplexity** | 50% of mass in |
|---|---|---|---|---|
| sky patches | 1,050,032 | 356 | **5.2** | **1 code** |
| **source patches** | 101,968 | 237 | **28.2** | 5 codes |
| spectra (all) | — | 1024 | **934 / 1024** | 355 codes |

So the meaningful comparison is **~28 effective codes for galaxy morphology versus ~934 for
spectra** — a 33× difference in effective capacity, out of nominal vocabularies of 4375 and
1024 respectively. The image tokeniser is not merely under-using its codebook; on the patches
that carry the object it has collapsed to a few dozen distinct states.

This strengthens A0's downstream implications rather than weakening them, and it is a
candidate explanation for the imaging/spectra performance gap that does not require any
appeal to the transformer.

*Still a proxy:* `tok_image_hsc` cannot be measured here — this dataset contains no HSC
cutouts — so its 9.6% remains row-norm-based and, by the same argument, is probably an
overestimate.

*Method note:* I nearly reported the raw perplexity of 7 as "the image codebook has collapsed
to seven codes". The sky split was the check that made the number honest.

---

## B4 — What the image codes encode ✅ a brightness ladder

`experiments/b4_code_semantics.py` → `results/b4_code_semantics.{json,png}`. Follows directly
from A0b: if only ~28 codes effectively carry the source, what are those states?

Decoded **in context** (overwrite one central grid token in real cutouts and difference against
the unmodified decode) — decoding a uniform grid would give the decoder's prior, not the code's
meaning. The 40 most-used source codes carry **93.3%** of source-patch tokens.

| statistic | value |
|---|---|
| PC1 of (flux, structure, colour) across codes | **96.3%** of variance |
| PC1 loadings | flux 0.588, structure 0.573, colour 0.571 |
| R²(flux, PC1) | **0.999** |
| R²(flux, *relative* structure) | 0.604 |
| R²(flux, colour g−z) | **0.937** |
| spread of relative structure | 0.087 of its mean |

**The codes lie on a one-dimensional manifold, and that dimension is brightness.** PC1's
loadings are near-isotropic (≈1/√3), which is the signature of a single latent scalar driving
every measured quantity rather than three independent axes — and R²(flux, PC1) = 0.999
identifies that scalar as flux.

The triviality is ruled out: absolute RMS structure *must* scale with brightness, so the test
that matters is the dimensionless one. Relative structure (rms ÷ flux) varies by only **8.7%**
across the 40 codes and is still 60% predicted by flux; colour is **94%** predicted by flux.
Shape and colour are slaved to brightness rather than free.

**Consequence.** AION's image tokeniser compresses each 4×4 patch (≈1″ at Legacy's pixel scale)
to a position on a brightness ladder with almost no independent shape or colour information.
Morphology can only be represented in the *spatial arrangement* of ladder positions across the
24×24 grid, not within a token. Together with A0b's ~28 effective states, this is a concrete,
tokeniser-level account of why AION's imaging results trail its spectroscopic ones — and it
implies that scaling the transformer cannot close that gap, consistent with D2e's finding that
codebook utilisation is scale-invariant.

*Caveats:* three summary statistics per code (flux, within-patch RMS, g−z), not a full
characterisation — a patch could carry structure these miss. In-context decoding uses 32 real
cutouts, so "what a code contributes" is averaged over those contexts. And a 4×4 patch is small
enough that limited internal structure is unsurprising; the finding is about how little
*independent* variation the codes carry, not that patches should look complicated.

*Method note:* my first verdict rule required PC1 to load on flux above 0.6 and so reported
"more than brightness" — exactly backwards. With standardised inputs, near-equal loadings are
what a single-factor structure looks like.

---

## E1 — Does the gate fire in practice, and does it cost a cosmology analysis? ✅ YES to both

`experiments/e1_incidence_and_impact.py` → `results/e1_incidence_impact.{json,png}`.
The two objections a referee would raise about D2–D2g, answered together.

### Part A — incidence, measured from the survey's own masks (5000 real cutouts)

| quantity | value |
|---|---|
| **no segment covering the centre pixel** | **3.68%** |
| centroid offset of the central segment | median 1.58 px, p90 12.3 px, p99 35.1 px |
| offset > 5 px / > 10 px | 16.2% / 11.7% |

The 3.68% is an unambiguous pipeline failure: the Legacy Survey segmentation contains no source
at the target position. D2 showed this is exactly the configuration that collapses the readout.

### Part B — downstream cost, on tomographic mean redshift

Photo-z requirements are stated on the **bin mean**, not per-object scatter: LSST SRD requires
|Δ⟨z⟩|/(1+z) < 0.003. Redshifts read with the **median** (per B1). Two scenarios, deliberately
separated because they rest on different evidence:

**Scenario 1 — only the measured 3.68% miss rate.** Fully defensible; nothing is assumed.

| bin | imaging only (LSST case) | imaging + spectra |
|---|---|---|
| [0.00, 0.15) | 0.00266 | 0.00000 |
| **[0.15, 0.25)** | **0.00816 — 2.7× req** | 0.00002 |
| **[0.25, 0.35)** | **0.00735 — 2.5× req** | 0.00000 |
| [0.35, 1.00) | 0.00213 | 0.00000 |

**Scenario 2 — misses plus displacements of plausible magnitude** (drawn from the observed
segment-centroid offset distribution). Every imaging-only bin exceeds, worst **0.0235 = 7.8×**.

### The finding

**Used for photometric redshifts — the actual LSST use case — AION's deference to the detection
pipeline biases tomographic mean redshifts past the LSST requirement, from the measured
detection-miss rate alone.** Two of four bins exceed at 2.5–2.7×, before assuming anything about
displacement magnitudes. That is a systematic which would have to be budgeted, and it is
invisible in the model's output.

**Spectroscopy removes it entirely** (worst bin 0.00002, i.e. 0.007× the requirement) —
consistent with D2b's finding that spectra suppress the redshift gate ~100×. So the systematic
is specific to photometric-only inference, which is precisely the regime LSST operates in and
where a foundation model is most attractive.

Together with D2d's mitigation, the practical statement is complete: **if you are doing
photometric redshifts and you cannot vouch for your detection masks, withhold the channel** —
it costs nothing measurable and removes a bias of several times the survey requirement.

*Caveats, stated because the two scenarios are not equally strong:*
- The 3.68% miss rate is measured. The displacement magnitudes are **not** a validated error
  distribution: the observed segment-centroid offsets mix genuine astrometric/detection error
  with real morphological asymmetry, so Scenario 2 is a plausible magnitude, not a measured one.
  The headline rests on Scenario 1.
- BGS-bright galaxies at Legacy depth; a fainter or deeper sample will have a different miss rate.
- Tomographic bins are defined on true z, not on the model's own estimate as a real analysis
  would; that choice isolates the mask effect from binning-induced selection.
- 960 galaxies, so bin means carry a few × 10⁻³ of statistical noise — comparable to the
  requirement itself, which is why only the larger exceedances should be quoted.

---

## E2 — Is the detection gate one case of a general metadata preference? ✅ YES, and worse

`experiments/e2_metadata_preference.py` → `results/e2_metadata_preference.{json,png}`. n = 480.

The obvious generalisation question — "is this AION, or any model given a detection channel?" —
**cannot be tested directly**: AION is the only public astronomical foundation model that takes
a segmentation map as an input modality, so there is no second model with that channel. Training
one is out of scope. The tractable substitute is to ask whether the behaviour generalises across
*channels* within AION.

Pixels identical; the model is never given the band it must predict (g). Readout in magnitudes:

| input configuration | \|dmag\| vs catalogue truth | shift vs its clean arm |
|---|---|---|
| imaging only | 0.0559 | — |
| imaging + true Tractor r,i,z | **0.0339** | — |
| **imaging + WRONG r,i,z** | **0.6210** | **0.6121 mag** |
| imaging + true segmap | 0.0693 | — |
| imaging + wrong segmap | 0.1682 | 0.1322 mag |
| placebo (RA moved 250 codes) | 0.0350 | 0.0005 mag |

**The catalogue-scalar channel is followed ~4.6× more strongly than the segmentation map**
(0.61 vs 0.13 mag), against a placebo of 0.0005.

### The framing that makes this a finding rather than a triviality

Corrupting r, i, z *should* move a g-band estimate — galaxy colours are correlated, so those
scalars carry genuine information about g. The pathology is not that the model responds; it is
**how much, relative to ignoring the channel entirely**:

- imaging alone: **0.056 mag**
- imaging + contradicted metadata: **0.621 mag — 11× worse than having no metadata at all**

A model weighing evidence would, when metadata contradicts the pixels, fall back *toward* the
image-only answer. AION instead follows the metadata off a cliff. The same pattern holds for the
segmentation map (0.168 vs 0.056, 3× worse than no channel).

So the detection gate is a comparatively *mild* instance of a general policy: **when a
catalogue-derived channel that is highly predictive in training is available, AION leans on it
in preference to the raw pixels, and does not discount it when it conflicts with them.** The
paper's architectural claim can be made about the design pattern — supplying redundant,
highly-predictive metadata alongside raw data — rather than about one channel.

*Caveats:* single model family (the point of this experiment is that a cross-model test is not
currently possible); the corrupted-scalar arm is out of distribution in the sense that real
catalogue fluxes are never wrong by a whole galaxy, so it bounds the behaviour rather than
estimating a field rate — the field-rate version is E1, which used a measured failure rate.

---

## A0c — Direct HSC codebook measurement ✅ last proxy closed

`experiments/a0c_hsc_codebook.py` → `results/a0c_hsc_codebook.{json,png}`. 1500 real HSC PDR3
deep/ultradeep cutouts from MultimodalUniverse (one 1.9 GB shard), 864,000 tokens.

| | row-norm proxy (A0) | **measured** |
|---|---|---|
| `tok_image_hsc` codes emitted | 9.6% | **7.1%** (312 / 4375) |
| effective size (perplexity, all patches) | — | **2.5** |
| codes holding 50% of the mass | — | **1** |

The proxy overstated HSC usage too, exactly as it did for Legacy — so A0b's correction was not
a one-off, and the row-norm heuristic is systematically generous. Every one of the 312 codes HSC
emits is also Legacy-live, the direct confirmation of A0's embedding-table containment result.

**Comparison must be like-for-like:** 2.5 is the *all-patch* perplexity, and the honest
comparison is against Legacy's all-patch value of 7, not Legacy's source-patch 28. HSC has no
segmentation map in this dataset so the source/sky split cannot be repeated. On that matched
basis HSC's image tokenisation is ~3× more collapsed than Legacy's, which is consistent with
A0's finding that HSC exercises a strict subset of Legacy's codes.
---

## E1b — Faithful detection-miss representation (supersedes E1 Part B)

E1 propagated the measured 3.68% detection-miss rate by **zeroing the whole segmentation
field**. That tensor is identical in construction to D2's `erased` arm, which D2 itself labels
"OFF-MANIFOLD, reported but discounted" and which is the most damaging of its ten arms.

Re-scanning 5000 cutouts (same seed; reproduces 184 misses = 3.68% exactly):

| | misses (n=184) | detected |
|---|---|---|
| completely blank field | **0** | — |
| central 12x12 coverage (median) | 0.146 | 0.788 |
| labelled area px (median) | 84 | 324 |

An all-zero field occurs in **0 of 5000 real cutouts**. E1b instead transplants a segmentation
field drawn from those 184 measured misses, holding seed, galaxies, flagged objects and
displacement draws fixed, so only the miss representation changes.

| scenario (imaging only) | worst bin | vs requirement | bins exceeding |
|---|---|---|---|
| all-zero field (E1, superseded) | 0.00818 | 2.73x | 2/4 |
| **real miss field (E1b)** | **0.00204** | **0.68x** | **0/4** |
| real miss + displacements | 0.02479 | 8.26x | 3/4 |

**The all-zero representation overstates the bias by exactly 4.0x.** The corrected headline: the
measured miss rate alone consumes two-thirds of the LSST error budget without breaching it;
positional error is what breaches it. That ordering matches D2 (displacement > swap) and D2f/g
(deblending does not propagate).

Also corrected in this pass: the pooled prediction/enclosed-light correlation is **0.384**, not
0.97 — the 0.97 came from an a-priori code comment in `d2_segmap_intervention.py:209`, not from
any run. The quiet-window control in C1b broke **one** object, not zero
(`fail_rate_quiet_control * 2865 = 1.0` exactly). E2's imaging-only baseline is **0.0665**, not
0.0559, making the metadata-preference ratio **9.3x**, not 11x.

---

## D1b — SAE hyperparameter sweep (supersedes D1's negative result)

D1 tested ONE dictionary (8192 features, k=32, seed 0) and reported that it failed as a causal
handle. The referee objection — one point in hyperparameter space is not evidence about a method
— turned out to be correct against us.

**First attempt was invalid** and was discarded: it trained dictionaries on `Xr` alone, whereas
D1 trains on `concat([Xr, Xw])`. A dictionary that never sees the gated state cannot be asked
which of its features represent it. Detected because the sweep's PCA baseline returned 21.3% at
alpha=4 where D1 published 72.4%. Five further deviations (normaliser, PCA fit set, feature
ranking, liveness test, PCA sparsity matching) were corrected at the same time. The corrected run
reproduces D1's baselines exactly: diff-in-means 64.3%, PCA k=32 72.4%.

**Corrected sweep, 15 dictionaries** (3 widths x 3 k at seed 0, plus 2 extra seeds at k=32 per
width), all trained on 576000 pooled rows, all steered at matched norm, evaluated on the same 192
held-out objects:

| | recovery @ alpha=4 |
|---|---|
| random control | 13.4% |
| difference-in-means | 64.3% |
| PCA at matched k | 68.1 / 72.4 / 73.4% (k=16/32/64) |
| **SAE, 15 dictionaries** | **26.0 - 74.7%, median 52.2%** |

- **15/15 beat random**; **2/15 beat difference-in-means**; **1/15 beats PCA at matched k**.
- Best: w=32768, k=16 -> 74.7%, despite only 10.5% of features alive and FVU 0.067.
- Best reconstructor (w=8192, k=64, FVU 0.023) recovers only 53.5%.
- **Seed spread up to 18.4pp** at fixed width and k (62.8 / 63.9 / 45.5 at w=2048, k=32).
- **D1's published 15.7% falls OUTSIDE the 3-seed range [26.0, 34.2] of its own configuration.**

**The negative result does not replicate.** The defensible claim is about reliability, not
capability: most dictionaries lose to difference-in-means and to PCA, a minority beat both, and
which is which is not predictable from reconstruction quality. A single-seed causal evaluation of
a dictionary is not evidence about dictionaries.

Note: dictionary training on MPS is not bit-reproducible — the same config re-run gave alive
100.0% vs 53.9% across attempts — so the quoted seed spread includes run-to-run nondeterminism.
