"""
astron_mi — mechanistic interpretability tooling for AION-1.

Design notes (all verified against the checkpoint, see RESULTS.md):

- Never use `AION.encode()`. It returns the decoder-projected tensor with position and
  modality embeddings re-added, and it throws away `mod_mask`, the only record of which
  modality each row came from. Use `hooks.encode_activations` instead.
- The installed package is `polymathic-aion==0.0.2`. Repo HEAD adds spectrum padding
  (`pad_spectrum`, `pad_length`) that 0.0.2 lacks, and the two produce different spectrum
  tokens. 0.0.2 reproduces the authors' own reference token fixtures bit-exactly, so it is
  what we pin. `data.check_regression()` enforces this.
- `aion-base` uses `NormAttention`/`NormCrossAttention` (qk_norm=True), NOT the
  `Attention`/`CrossAttention` classes. Monkeypatching the latter silently does nothing.
- Codec preprocessing mutates inputs in place, so everything here clones first.
"""

from .loading import load_model, load_codecs, device_for
from .physics import PhysicsTables, expected_value, density, pit
from .hooks import (
    ActivationCapture,
    encode_activations,
    row_index,
    modality_id_map,
    read_targets,
    Steer,
)
from .encode import (
    encode_image,
    encode_spectrum,
    pad_spectrum_sentinel,
    catalog_to_model_ids,
    check_bands,
)
from .data import (
    FIXTURES,
    load_fixture_images,
    load_fixture_spectra,
    check_regression,
    ProvabgsReader,
)

__all__ = [
    "load_model",
    "load_codecs",
    "device_for",
    "PhysicsTables",
    "expected_value",
    "density",
    "pit",
    "ActivationCapture",
    "encode_activations",
    "row_index",
    "modality_id_map",
    "read_targets",
    "Steer",
    "FIXTURES",
    "load_fixture_images",
    "load_fixture_spectra",
    "check_regression",
    "ProvabgsReader",
    "encode_image",
    "encode_spectrum",
    "pad_spectrum_sentinel",
    "catalog_to_model_ids",
    "check_bands",
]
