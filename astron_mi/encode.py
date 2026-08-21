"""
Safe raw-data -> token encoding.

Three hazards this module exists to remove, all verified by execution:

1. SPECTRUM SENTINEL PADDING (the dangerous one).
   Real DESI spectra have 7781 real pixels; the authors' arrays carry 19 trailing points
   at wavelength=99999. Those sentinels are NOT decoration — dropping them changes 20% of
   the 273 tokens, at positions spread across the whole grid (min 5, max 271), because the
   sentinel changes interp1d's out-of-range test and hence the codec's global GRN
   statistic. One trailing sentinel is necessary and sufficient.
   The corruption is invisible end-to-end: E[z] agrees to 3e-4 with or without it. It only
   shows up in token-resolved work (C0, C1, B4) — exactly the work this project does.
   `pad_spectrum_sentinel` enforces the convention.

2. IN-PLACE MUTATION. `Clamp` and `RescaleToLegacySurvey` write into caller tensors.
   Real PROVABGS cutouts reach 9.5 nmgy, above the DES-G clamp of 6, so this fires on
   actual data: re-encoding the same tensor twice gives different tokens for HSC. Always clone.

3. DEVICE. `CodecManager(device='mps')` does NOT move your inputs; passing CPU tensors
   raises. Move them yourself and codecs are 34-56x faster and bit-exact vs CPU
   (image 610 -> 18 ms/obj, spectrum 1604 -> 29 ms/obj; tokens identical).

BAND ORDER: the contract is `bands[i]` describes channel i of your tensor; any permutation
is valid as long as the pair is consistent. The real hazard is a MISMATCHED pair, which
fails silently when the channel count is smaller than the band list. `check_bands` lints it.
"""

from __future__ import annotations

import torch

SENTINEL_LAMBDA = 99999.0

LEGACY_BANDS = ["DES-G", "DES-R", "DES-I", "DES-Z"]
HSC_BANDS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]


def pad_spectrum_sentinel(flux, ivar, mask, wavelength, n_sentinel: int = 1):
    """
    Append sentinel points at wavelength=99999 if not already present.

    Idempotent: arrays that already carry sentinels (e.g. the authors' fixtures, which
    have 19) are returned unchanged.
    """
    if bool((wavelength >= SENTINEL_LAMBDA).any()):
        return flux, ivar, mask, wavelength

    b = flux.shape[0]
    pad = lambda t, v: torch.cat(  # noqa: E731
        [t, torch.full((b, n_sentinel), v, dtype=t.dtype, device=t.device)], dim=1
    )
    return (
        pad(flux, 0.0),
        pad(ivar, 0.0),
        pad(mask, True) if mask.dtype == torch.bool else pad(mask, 1),
        pad(wavelength, SENTINEL_LAMBDA),
    )


def normalise_bands(bands) -> list[str]:
    """
    Canonicalise band names. The PROVABGS file stores them as lowercase bytes
    (b'des-g'); aion's Clamp table is keyed on 'DES-G' and raises KeyError otherwise.
    """
    out = []
    for b in bands:
        if isinstance(b, bytes):
            b = b.decode()
        out.append(str(b).strip().upper())
    return out


def check_bands(flux: torch.Tensor, bands: list[str]) -> None:
    """
    Guard the silent-failure mode: a 4-channel tensor with a 2-band list produces tokens
    with no error at all. (The reverse, too few channels, raises IndexError.)

    Band ORDER is free — the contract is bands[i] describes channel i, and the codec's
    scatter handles any permutation. What must never happen is a mismatched pair.
    """
    if flux.shape[1] != len(bands):
        raise ValueError(
            f"channel/band mismatch: tensor has {flux.shape[1]} channels but "
            f"{len(bands)} bands were given ({bands}). bands[i] must describe channel i. "
            "A longer band list than channel count fails SILENTLY in aion."
        )
    known = set(LEGACY_BANDS) | set(HSC_BANDS)
    unknown = [b for b in bands if b not in known]
    if unknown:
        raise ValueError(f"unknown band names {unknown}; expected from {sorted(known)}")


def encode_image(cm, flux: torch.Tensor, bands: list[str], device: str | None = None):
    """Encode imaging to tokens. Clones, lints bands, handles device."""
    from aion.modalities import HSCImage, LegacySurveyImage

    bands = normalise_bands(bands)
    check_bands(flux, bands)
    x = flux.clone()
    if device is not None:
        x = x.to(device)
    cls = HSCImage if bands[0].startswith("HSC") else LegacySurveyImage
    out = cm.encode(cls(flux=x, bands=list(bands)))
    return {k: v.reshape(v.shape[0], -1) for k, v in out.items()}


def encode_spectrum(cm, flux, ivar, mask, wavelength, survey: str = "desi",
                    device: str | None = None):
    """Encode a spectrum to tokens. Clones, enforces the sentinel, handles device."""
    from aion.modalities import DESISpectrum, SDSSSpectrum

    f, i, m, w = (t.clone() for t in (flux, ivar, mask, wavelength))
    f, i, m, w = pad_spectrum_sentinel(f, i, m, w)
    if device is not None:
        f, i, m, w = (t.to(device) for t in (f, i, m, w))
    cls = DESISpectrum if survey.lower() == "desi" else SDSSSpectrum
    out = cm.encode(cls(flux=f, ivar=i, mask=m.bool(), wavelength=w))
    return {k: v.reshape(v.shape[0], -1) for k, v in out.items()}


CATALOG_VOCAB_OFFSET = 204
CATALOG_MASK_VALUE = 9999


def catalog_to_model_ids(codec_ids: torch.Tensor):
    """
    Convert `CodecManager.encode(LegacySurveyCatalog(...))['catalog']` into model input.

    AION does not do this for you: the codec emits raw codes 0..3263 plus 9999 for absent
    neighbours, while the model's embedding table has 3468 rows with the first 204
    reserved. Feeding codec output straight in raises IndexError.

    Returns (ids, input_mask) for model.embed_inputs(..., mask={'catalog': input_mask}).
    """
    ids = codec_ids.clone().long()
    pad = ids == CATALOG_MASK_VALUE
    ids[~pad] += CATALOG_VOCAB_OFFSET
    ids[pad] = 0
    return ids, pad
