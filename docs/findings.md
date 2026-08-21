# Findings

A plain-language summary of what the audit measured, for readers who want the result without the
manuscript. Section numbers refer to `paper/main.pdf`; every number here maps to a script and a
JSON file in [claims-to-code.md](claims-to-code.md).

## 1. The model prefers catalogue metadata to the pixels

AION-1 takes the survey's **segmentation map** — the detection pipeline's own output, saying
which pixels belong to which source — as one of its input modalities. That makes the pipeline's
decisions directly editable, which is the basis of most of this study.

Holding the image tokens byte-identical and changing only that map moves every quantity the
model reports:

| Readout | Mask displaced 25 px | Mask swapped | Placebo |
|---|---|---|---|
| flux *g* | 0.630 σ | 0.110 σ | 0.0004 σ |
| flux *z* | 20.55 σ | 9.83 σ | 0.0047 σ |
| size *R* | 1.313 σ | 0.718 σ | 0.0025 σ |
| ellipticity *e₁* | 0.435 σ | 0.931 σ | 0.0036 σ |
| redshift | 0.580 σ | 0.312 σ | 0.0028 σ |

The placebo displaces an unrelated token (right ascension) by 250 codebook steps, so it is a
matched intervention that should do nothing — and does.

In absolute terms, displacing the mask by 25 px drops the median *g*-band flux estimate from
6.71 to 1.55 nmgy, a 79 % per-object reduction **with every photon still in the image**.

## 2. The mechanism is detection gating, not aperture photometry

The obvious hypothesis is that the model integrates light inside the mask, like an aperture. It
does not. Across nine intervention arms, the change in prediction tracks **whether the mask
asserts a source at the field centre** (r = 0.47) more strongly than **the light the mask
encloses** (r = 0.30). Eroding the mask by 5 px collapses the estimate; dilating it by 10 px
barely moves it — the reverse of aperture behaviour.

The sharpest test uses no synthetic geometry at all. In a blended cutout the pipeline has
already split the light into segments; comparing "central segment only" against "all segments"
changes *only the deblending decision*. If the model integrated its assigned light, the ratio
Δreadout/Δlight would be ≈ 1. Measured over 322 severely blended systems it is **−0.006**.

**Deblending errors do not propagate through this channel. Detection and positional errors do.**

## 3. The preference is not specific to the detection channel

Corrupting the catalogue *photometry* instead moves the *g*-band readout by 0.61 mag — 4.6×
more than corrupting the segmentation map. Some response is legitimate, since galaxy colours are
correlated. The pathology is the comparison against ignoring the channel: contradicted metadata
leaves the model **nine times worse than supplying no metadata at all** (0.62 vs 0.067 mag).

A model weighing evidence would fall back toward the pixels. This one does not.

## 4. What it costs a cosmological analysis

Photometric-redshift requirements constrain the **mean** redshift of a tomographic bin, so a
small coherent shift matters far more than a large random one.

The Legacy Survey pipeline leaves **3.68 %** of targets with no segment covering their position —
measured directly over 5000 real cutouts, not modelled. Propagating that measured rate through
AION-1's photometric redshifts, and representing a miss by the segmentation fields the pipeline
actually returns when it misses:

| Scenario | Worst bin vs LSST DESC requirement |
|---|---|
| Measured miss rate only | 0.68× |
| Measured misses + observed positional errors | 8.3× |
| Either, with spectra supplied | ≤ 0.37× |

One routine pipeline failure consumes two-thirds of the entire error budget for a bin mean.
Positional error is what actually breaches the requirement — consistent with §2, where wrong
location cost more than wrong shape.

**How a failure is represented matters by a factor of four.** Zeroing the whole segmentation
field — the intuitive choice — gives 2.7× instead of 0.68×, and occurs in 0 of 5000 real
cutouts. This is worth knowing for anyone pricing a learned model's failure modes by intervening
on its inputs.

## 5. Mitigation

Withholding the detection channel entirely costs nothing measurable — redshift error improves
with non-overlapping intervals — and removes the vulnerability completely. Substituting a
generic centred disc is **220 % worse** on flux *g* than either supplying the real mask or
supplying none, so "give it a neutral mask" is not a fix.

**Recommendation: withhold the detection channel when the detection cannot be vouched for.**

## 6. Two limits live in the tokeniser, not the transformer

- Measured over 1.15 × 10⁶ real tokens, the image codec emits 8.3 % of its 4375 codes, and on
  patches containing the source its **effective vocabulary is 28 codes** against 934 of 1024 for
  the spectrum codec. Those 28 states lie on a one-dimensional **brightness ladder**: PC1
  explains 96.3 % of between-code variance, and R²(flux, PC1) = 0.999.
- `tok_z` is quantised on a linear grid with Δz = 0.005865, and the median redshift error with
  spectra is 0.497× the grid spacing. The readout is **quantisation-limited**, not
  information-limited.

Codebook utilisation is unchanged at 2.7× the parameter count, so **neither is addressable by
scaling the transformer**. Practitioners comparing AION-1's imaging and spectroscopic
performance should look at the codec before the transformer.

## 7. Sparse dictionaries are unreliable causal handles

A BatchTopK sparse autoencoder reconstructs 7.6× better than PCA at matched sparsity and finds
clearly gate-selective features. Whether that buys causal control depends on which dictionary
you happen to train.

Across 15 dictionaries (widths 2048–32768, k ∈ {16,32,64}, three seeds), recovery of the gated
readout spans **26–75 %**. All beat a matched-norm random control; only two beat
difference-in-means; only one beats PCA at matched sparsity. Holding width and sparsity fixed
and changing **only the random seed** moves recovery by up to **18.4 points** — larger than the
gap between the median dictionary and the baseline.

Reconstruction quality does not predict causal utility: the best reconstructor (FVU 0.023)
recovers 53.5 %, while the best steerer reconstructs *worse* than average (FVU 0.067) with only
10.5 % of its features alive.

**A causal evaluation of a dictionary that reports a single seed is uninformative about
dictionaries**, because the seed-to-seed spread exceeds the effect being claimed.

## What this does not establish

- All measurements are on one model family, one survey (DESI BGS-bright at Legacy depth) and one
  depth. The miss rate, and the blending null in particular, may differ in deeper or more crowded
  imaging.
- The scaling evidence is two checkpoints.
- Tomographic bins are defined on true spectroscopic redshift, isolating the mask effect from
  binning-induced selection. A survey-realised systematic would need source-injection frameworks.
- The dictionary sweep stays at one layer, with features chosen by activation gap.
