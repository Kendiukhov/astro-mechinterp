"""aion_physics -- verified token <-> physical-units layer for AION-1.

Everything here is derived from `aion` source + loaded codec state, never assumed.

Core API
--------
    index_to_value(key)                 -> np.ndarray, shape (K,)  or (55, K) for Gaia XP
    bin_edges(key)                      -> np.ndarray, shape (K+1,)
    bin_widths(key)                     -> np.ndarray, shape (K,)
    expected_value(logits, key)         -> softmax(logits) @ index_to_value(key)
    density(logits, key)                -> probability density in physical units
    pit(logits, truth, key)             -> probability integral transform
    token_lambda(p, survey)             -> (lam_lo, lam_hi) for spectrum token p
    image_token_pixels(k)               -> (row_slice, col_slice) in the 96x96 cutout
    xp_token_order(p)                   -> Gauss-Hermite coefficient index for XP token p
    catalog_layout()                    -> full spec of the `catalog` sequence
"""

from __future__ import annotations

import numpy as np
import torch

from aion import modalities as M
from aion.codecs.manager import CodecManager
from aion.codecs.config import MODALITY_CODEC_MAPPING
from aion.fourm.modality_info import MODALITY_INFO

# --------------------------------------------------------------------------- #
# token_key -> modality dataclass
# --------------------------------------------------------------------------- #
KEY_TO_MODALITY = {
    m.token_key: m
    for m in MODALITY_CODEC_MAPPING
    if getattr(m, "token_key", "")
}
assert len(KEY_TO_MODALITY) == 40, len(KEY_TO_MODALITY)

# Model-side vocab size (may exceed the codec codebook: tok_z has a sentinel,
# `catalog` has 204 reserved special ids in front of the codec vocabulary).
MODEL_VOCAB = {k: MODALITY_INFO[k]["vocab_size"] for k in KEY_TO_MODALITY}

# Physical units of each scalar, read off the source field names.
UNITS = {
    "tok_ra": "deg", "tok_dec": "deg", "tok_z": "dimensionless (redshift)",
    "tok_ebv": "mag  E(B-V), SFD",
    "tok_flux_g": "nanomaggy", "tok_flux_r": "nanomaggy", "tok_flux_i": "nanomaggy",
    "tok_flux_z": "nanomaggy", "tok_flux_w1": "nanomaggy", "tok_flux_w2": "nanomaggy",
    "tok_flux_w3": "nanomaggy", "tok_flux_w4": "nanomaggy",
    "tok_shape_r": "arcsec (half-light radius)",
    "tok_shape_e1": "dimensionless (ellipticity e1)",
    "tok_shape_e2": "dimensionless (ellipticity e2)",
    "tok_a_g": "mag", "tok_a_r": "mag", "tok_a_i": "mag", "tok_a_z": "mag",
    "tok_a_y": "mag",
    "tok_mag_g": "AB mag", "tok_mag_r": "AB mag", "tok_mag_i": "AB mag",
    "tok_mag_z": "AB mag", "tok_mag_y": "AB mag",
    "tok_shape11": "arcsec^2 (2nd moment Ixx)",
    "tok_shape22": "arcsec^2 (2nd moment Iyy)",
    "tok_shape12": "arcsec^2 (2nd moment Ixy)",
    "tok_flux_g_gaia": "e-/s", "tok_flux_bp_gaia": "e-/s", "tok_flux_rp_gaia": "e-/s",
    "tok_parallax": "mas",
    "tok_xp_bp": "e-/s (Gauss-Hermite coefficient)",
    "tok_xp_rp": "e-/s (Gauss-Hermite coefficient)",
    "tok_image": "nanomaggy (FSQ latent, no scalar value table)",
    "tok_image_hsc": "nanomaggy (FSQ latent, no scalar value table)",
    "tok_segmap": "dimensionless in [0,1] (FSQ latent, no scalar value table)",
    "tok_spectrum_desi": "1e-17 erg/s/cm2/A (LFQ latent; token 0 is log10 normalisation)",
    "tok_spectrum_sdss": "1e-17 erg/s/cm2/A (LFQ latent; token 0 is log10 normalisation)",
    "catalog": "mixed, see catalog_layout()",
}

# Which keys actually have a scalar value table.
SCALAR_KEYS = [k for k, m in KEY_TO_MODALITY.items()
               if MODALITY_CODEC_MAPPING[m].__name__ in
               ("ScalarCodec", "LogScalarCodec", "GridScalarCodec")]
XP_KEYS = ["tok_xp_bp", "tok_xp_rp"]
LATENT_KEYS = ["tok_image", "tok_image_hsc", "tok_segmap",
               "tok_spectrum_desi", "tok_spectrum_sdss"]


class Physics:
    """Cached token<->physics layer. One instance per process is enough."""

    def __init__(self, device="cpu"):
        self.cm = CodecManager(device=device)
        self._tab: dict[str, np.ndarray] = {}
        self._codec: dict[str, object] = {}

    # ------------------------------------------------------------------ codecs
    def codec(self, key: str):
        if key not in self._codec:
            self._codec[key] = self.cm._load_codec(KEY_TO_MODALITY[key])
        return self._codec[key]

    # ------------------------------------------------------- value tables
    def index_to_value(self, key: str) -> np.ndarray:
        """Physical value of every codec index.

        Returns (K,) for a plain scalar, (55, K) for Gaia XP, (3264,) for
        `catalog` (concatenated per-field tables with the ComposedScalarQuantizer
        offsets already applied).  Raises for latent modalities that have no
        scalar interpretation.
        """
        if key in self._tab:
            return self._tab[key]
        if key in LATENT_KEYS:
            raise ValueError(
                f"{key} is a learned-latent modality (FSQ/LFQ); its codes index a "
                f"vector in latent space, not a scalar. Use codec.quantizer.decode() "
                f"for the latent vector, or decode the whole field."
            )
        q = self.codec(key).quantizer
        K = int(q.codebook_size)
        if key in XP_KEYS:
            v = np.stack([q.quantizers[i].decode(torch.arange(K)).numpy()
                          for i in range(q.num_quantizers)])
        elif key == "catalog":
            parts = []
            for off, (name, sq) in zip(q.offsets, q.quantizers.items()):
                n = int(sq.codebook_size)
                if type(sq).__name__ == "IdentityQuantizer":
                    parts.append(np.arange(n, dtype="float64"))   # pixel index 0..95
                else:
                    parts.append(sq.decode(torch.arange(n)).numpy().astype("float64"))
            v = np.concatenate(parts)
        else:
            # NB: quantizer.decode, NOT quantizer.codebook -- the latter is
            # pre-exp for ScalarLogReservoirQuantizer.
            v = q.decode(torch.arange(K)).numpy().astype("float64")
        self._tab[key] = v
        return v

    def n_finite(self, key: str) -> int:
        return int(np.isfinite(self.index_to_value(key)).sum())

    # ------------------------------------------------------------ bin geometry
    def bin_edges(self, key: str, sub: int | None = None) -> np.ndarray:
        """Midpoint edges of the value table, extrapolated at both ends.

        Length K+1.  For `catalog` pass sub=field-index; for Gaia XP pass
        sub=coefficient-index.
        """
        v = self.index_to_value(key)
        if v.ndim == 2:
            if sub is None:
                raise ValueError(f"{key}: pass sub=<coefficient index 0..54>")
            v = v[sub]
        elif key == "catalog":
            if sub is None:
                raise ValueError("catalog: pass sub=<field index 0..4>")
            lay = catalog_layout()
            o, n = lay["offsets"][sub], lay["sizes"][sub]
            v = v[o:o + n]
        mid = 0.5 * (v[:-1] + v[1:])
        first = v[0] - (mid[0] - v[0])
        last = v[-1] + (v[-1] - mid[-1])
        return np.concatenate([[first], mid, [last]])

    def bin_widths(self, key: str, sub: int | None = None) -> np.ndarray:
        e = self.bin_edges(key, sub)
        return np.diff(e)

    # --------------------------------------------------------------- readouts
    @staticmethod
    def _probs(logits) -> np.ndarray:
        z = np.asarray(logits, dtype="float64")
        z = z - z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=-1, keepdims=True)

    def _aligned_table(self, key, n_logits, sub=None):
        """Return (values, probs-slice) handling model-vocab != codec-vocab."""
        v = self.index_to_value(key)
        if v.ndim == 2:
            if sub is None:
                raise ValueError(f"{key}: pass sub=<coefficient index 0..54>")
            v = v[sub]
        elif key == "catalog":
            lay = catalog_layout()
            if sub is None:
                raise ValueError("catalog: pass sub=<field index 0..4>")
            o, n = lay["offsets"][sub], lay["sizes"][sub]
            v = v[o:o + n]
            # model ids are codec code + CATALOG_VOCAB_OFFSET
            return v, slice(CATALOG_VOCAB_OFFSET + o, CATALOG_VOCAB_OFFSET + o + n)
        K = len(v)
        if n_logits == K:
            return v, slice(0, K)
        if n_logits == K + 1 and key == "tok_z":
            # 1025th code is a sentinel with no value bucket -- drop it and
            # renormalise (the model virtually never puts mass there; check with
            # sentinel_mass()).
            return v, slice(0, K)
        raise ValueError(f"{key}: got {n_logits} logits, table has {K}")

    def sentinel_mass(self, logits, key="tok_z") -> np.ndarray:
        """Probability mass the model puts on codes with no value bucket."""
        p = self._probs(logits)
        v, sl = self._aligned_table(key, p.shape[-1])
        keep = np.zeros(p.shape[-1], bool); keep[sl] = True
        return p[..., ~keep].sum(-1)

    def expected_value(self, logits, key: str, sub: int | None = None) -> np.ndarray:
        p = self._probs(logits)
        v, sl = self._aligned_table(key, p.shape[-1], sub)
        pv = p[..., sl]
        pv = pv / pv.sum(-1, keepdims=True)
        ok = np.isfinite(v)
        if not ok.all():                       # NaN-poisoned tables, see report
            pv = np.where(ok, pv, 0.0)
            pv = pv / pv.sum(-1, keepdims=True)
            v = np.where(ok, v, 0.0)
        return (pv * v).sum(-1)

    def density(self, logits, key: str, sub: int | None = None) -> np.ndarray:
        """Probability DENSITY per unit of the physical variable, per bin.

        Bins of zero physical width exist (repeated table values where the
        empirical CDF saturates -- e.g. tok_flux_g's exp(-3) floor); those
        return +inf.  `bin_widths(key) == 0` tells you which.
        """
        p = self._probs(logits)
        v, sl = self._aligned_table(key, p.shape[-1], sub)
        pv = p[..., sl]
        pv = pv / pv.sum(-1, keepdims=True)
        w = self.bin_widths(key, sub)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = pv / w
        return d

    def pit(self, logits, truth, key: str, sub: int | None = None,
            randomized: bool = False, rng=None) -> np.ndarray:
        """Probability integral transform: F(truth) under the token pmf.

        Uses the *cumulative* mass strictly below the bin containing `truth`
        plus a linear share of that bin (or a uniform random share when
        randomized=True, which is what makes PIT exactly uniform for a
        discrete-but-calibrated predictor).
        """
        p = self._probs(logits)
        v, sl = self._aligned_table(key, p.shape[-1], sub)
        pv = p[..., sl]
        pv = pv / pv.sum(-1, keepdims=True)
        pv2 = pv.reshape(-1, pv.shape[-1])                 # (N, K)
        edges = self.bin_edges(key, sub)
        t = np.atleast_1d(np.asarray(truth, dtype="float64")).ravel()
        if pv2.shape[0] == 1 and t.size > 1:
            pv2 = np.repeat(pv2, t.size, axis=0)
        if t.size == 1 and pv2.shape[0] > 1:
            t = np.repeat(t, pv2.shape[0])
        j = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(v) - 1)
        cdf = np.cumsum(pv2, axis=-1)
        rows = np.arange(pv2.shape[0])
        below = cdf[rows, j] - pv2[rows, j]
        pj = pv2[rows, j]
        if randomized:
            rng = rng if rng is not None else np.random.default_rng()
            frac = rng.random(below.shape)
        else:
            lo, hi = edges[j], edges[j + 1]
            frac = np.clip((t - lo) / np.where(hi > lo, hi - lo, 1.0), 0, 1)
        return below + frac * pj


# --------------------------------------------------------------------------- #
# index -> physical coordinate maps (all constants verified against source)
# --------------------------------------------------------------------------- #

# --- spectra --------------------------------------------------------------- #
LATENT_LAMBDA_MIN = 3500.0        # AutoencoderSpectrumCodec(lambda_min=...)
LATENT_DLAMBDA = 0.8              # ...resolution=... (A/pixel)
LATENT_NPIX = 8704                # ...num_pixels=...
SPEC_STRIDE = 32                  # 4 (stem) x 2 x 2 x 2 downsamples
N_SPEC_LATENT = LATENT_NPIX // SPEC_STRIDE          # 272
N_SPEC_TOKENS = N_SPEC_LATENT + 1                   # 273: token 0 = normalisation
SPEC_A_PER_TOKEN = SPEC_STRIDE * LATENT_DLAMBDA     # 25.6 A


def token_lambda(p: int):
    """Wavelength window of spectrum token p (p = 0..272).

    p == 0 is the NORMALISATION token: it is not a wavelength at all, it is a
    ScalarLinearQuantizer code for log10(<flux> + 1) over [-1, 5].
    p >= 1 maps onto latent pixels [(p-1)*32, (p-1)*32+31] of the 8704-point
    3500 -> 10462.4 A grid.
    """
    if p == 0:
        return None
    if not 1 <= p <= N_SPEC_LATENT:
        raise IndexError(p)
    i0 = (p - 1) * SPEC_STRIDE
    lo = LATENT_LAMBDA_MIN + LATENT_DLAMBDA * i0
    hi = LATENT_LAMBDA_MIN + LATENT_DLAMBDA * (i0 + SPEC_STRIDE - 1)
    return lo, hi


def lambda_token(lam: float) -> int:
    """Inverse of token_lambda (returns the token whose window contains lam)."""
    i = int(np.floor((lam - LATENT_LAMBDA_MIN) / LATENT_DLAMBDA))
    return i // SPEC_STRIDE + 1


# --- images ---------------------------------------------------------------- #
IMAGE_CROP = 96          # CenterCrop(crop_size=96) in AutoencoderImageCodec
IMAGE_PATCH = 4          # MODALITY_INFO['tok_image']['patch_size'], = 2 downsamples
IMAGE_GRID = IMAGE_CROP // IMAGE_PATCH                 # 24
N_IMAGE_TOKENS = IMAGE_GRID ** 2                       # 576
LS_PIXSCALE = 0.262      # arcsec/pix, Legacy Survey DR10 (notebooks/Tutorial.ipynb)
HSC_PIXSCALE = 0.168     # arcsec/pix, HSC


def image_token_pixels(k: int):
    """Which pixels of the *centre-cropped* 96x96 frame token k covers.

    Row-major: k = row * 24 + col   (h.reshape(B, C, -1) over a (H, W) map).
    """
    if not 0 <= k < N_IMAGE_TOKENS:
        raise IndexError(k)
    r, c = divmod(k, IMAGE_GRID)
    return (slice(r * IMAGE_PATCH, (r + 1) * IMAGE_PATCH),
            slice(c * IMAGE_PATCH, (c + 1) * IMAGE_PATCH))


def image_token_arcsec(k: int, pixscale: float = LS_PIXSCALE):
    """Offset of token k's centre from the cutout centre, in arcsec (dx, dy)."""
    r, c = divmod(k, IMAGE_GRID)
    cy = (r + 0.5) * IMAGE_PATCH - IMAGE_CROP / 2
    cx = (c + 0.5) * IMAGE_PATCH - IMAGE_CROP / 2
    return cx * pixscale, cy * pixscale


SEGMAP_PATCH = 8                                       # 3 downsamples
SEGMAP_GRID = IMAGE_CROP // SEGMAP_PATCH               # 12
N_SEGMAP_TOKENS = SEGMAP_GRID ** 2                     # 144


def segmap_token_pixels(k: int):
    r, c = divmod(k, SEGMAP_GRID)
    return (slice(r * SEGMAP_PATCH, (r + 1) * SEGMAP_PATCH),
            slice(c * SEGMAP_PATCH, (c + 1) * SEGMAP_PATCH))


# --- Gaia XP --------------------------------------------------------------- #
N_XP = 55


def xp_token_order(p: int) -> int:
    """Gaia XP token p IS Gauss-Hermite basis-function index p (0..54).

    MultiScalarCompressedReservoirQuantizer.encode does
    `stack([q.encode(z[:, i]) for i, q in enumerate(self.quantizers)], dim=1)`,
    i.e. channel i of the input coefficient vector -> token i, each with its own
    1024-entry table.  Gaia DR3 XP continuous spectra carry exactly 55
    bp_coefficients / rp_coefficients in ascending basis order, so token p is
    coefficient p.  Compression is arcsinh(arcsinh(.)).
    """
    if not 0 <= p < N_XP:
        raise IndexError(p)
    return p


# --- catalog --------------------------------------------------------------- #
CATALOG_FIELDS = ["X", "Y", "SHAPE_E1", "SHAPE_E2", "SHAPE_R"]
CATALOG_SIZES = [96, 96, 1024, 1024, 1024]
CATALOG_OFFSETS = [0, 96, 192, 1216, 2240]      # ComposedScalarQuantizer.offsets
CATALOG_CODEC_VOCAB = 3264
CATALOG_VOCAB_OFFSET = 204                       # MODALITY_INFO['catalog']
CATALOG_MODEL_VOCAB = 3468
CATALOG_MASK_VALUE = 9999                        # CatalogCodec(mask_value=...)
CATALOG_MAX_TOKENS = 50                          # -> 10 neighbours x 5 fields


def catalog_layout() -> dict:
    return dict(
        fields=CATALOG_FIELDS, sizes=CATALOG_SIZES, offsets=CATALOG_OFFSETS,
        codec_vocab=CATALOG_CODEC_VOCAB, vocab_offset=CATALOG_VOCAB_OFFSET,
        model_vocab=CATALOG_MODEL_VOCAB, mask_value=CATALOG_MASK_VALUE,
        max_tokens=CATALOG_MAX_TOKENS,
        flat_order="neighbour-major: [n0.X, n0.Y, n0.E1, n0.E2, n0.R, n1.X, ...]",
    )


def catalog_slot(neighbour: int, field: str) -> int:
    """Position in the flat catalog token sequence."""
    return neighbour * len(CATALOG_FIELDS) + CATALOG_FIELDS.index(field)


# --------------------------------------------------------------------------- #
# catalog: codec ids -> model ids, and editing
# --------------------------------------------------------------------------- #
def catalog_to_model_ids(codec_ids: torch.Tensor):
    """`CodecManager.encode(LegacySurveyCatalog(...))['catalog']` -> model input.

    The released AION code does NOT do this for you: the codec emits raw codes
    0..3263 plus 9999 for absent neighbours, while the transformer's embedding
    table is 3468 rows with the first 204 reserved.  Feeding the codec output
    straight in raises `IndexError: index out of range in self`.

    Returns (ids, input_mask) ready for
    `model.encode({'catalog': ids}, input_mask={'catalog': input_mask})`.
    """
    ids = codec_ids.clone().long()
    pad = ids == CATALOG_MASK_VALUE
    ids[~pad] += CATALOG_VOCAB_OFFSET
    ids[pad] = 0                      # padding_idx of SequenceEncoderEmbedding
    return ids, pad


def catalog_set(ids: torch.Tensor, neighbour: int, field: str, code: int,
                already_offset: bool = True):
    """Overwrite one (neighbour, field) slot in a catalog token sequence."""
    slot = catalog_slot(neighbour, field)
    fi = CATALOG_FIELDS.index(field)
    if not 0 <= code < CATALOG_SIZES[fi]:
        raise ValueError(f"{field} code must be in [0,{CATALOG_SIZES[fi]})")
    val = CATALOG_OFFSETS[fi] + code + (CATALOG_VOCAB_OFFSET if already_offset else 0)
    ids = ids.clone()
    ids[..., slot] = val
    return ids


CIRCULAR_KEYS = {"tok_ra"}          # RA wraps at 360 deg; Dec does not


def circular_expected_value(P: "Physics", logits, key="tok_ra", period=360.0):
    """Circular mean. `expected_value` is WRONG for tok_ra near the wrap
    (equal mass at 0.004 deg and 359.997 deg gives 180.0 deg linearly)."""
    p = P._probs(logits)
    v, sl = P._aligned_table(key, p.shape[-1])
    pv = p[..., sl]; pv = pv / pv.sum(-1, keepdims=True)
    th = 2 * np.pi * v / period
    return (np.degrees(np.arctan2((pv * np.sin(th)).sum(-1),
                                  (pv * np.cos(th)).sum(-1))) % 360.0) * (period / 360.0)


def image_token_pixels_original(k: int, orig_h: int, orig_w: int):
    """Same as image_token_pixels but in the coordinates of the *uncropped*
    input array (CenterCrop takes rows [(H-96)//2, +96), cols [(W-96)//2, +96))."""
    r0, c0 = (orig_h - IMAGE_CROP) // 2, (orig_w - IMAGE_CROP) // 2
    rs, cs = image_token_pixels(k)
    return (slice(r0 + rs.start, r0 + rs.stop), slice(c0 + cs.start, c0 + cs.stop))


# --------------------------------------------------------------------------- #
# package-level shims
# --------------------------------------------------------------------------- #
PhysicsTables = Physics          # canonical name used by astron_mi.__init__

_DEFAULT: "Physics | None" = None


def tables(device: str = "cpu") -> "Physics":
    """Process-wide singleton; codec loading is the slow part."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Physics(device=device)
    return _DEFAULT


def index_to_value(key: str, **kw):
    return tables().index_to_value(key, **kw)


def expected_value(logits, key: str, sub: int | None = None):
    """Expected physical value under the token pmf.

    For tok_ra use circular_expected_value instead: RA's codebook wraps (see
    RESULTS.md B3.2), so the linear mean of mass split across 0/360 deg is 180 deg.
    """
    return tables().expected_value(logits, key, sub)


def density(logits, key: str, sub: int | None = None):
    return tables().density(logits, key, sub)


def pit(logits, truth, key: str, sub: int | None = None, randomized: bool = False, rng=None):
    return tables().pit(logits, truth, key, sub, randomized=randomized, rng=rng)
