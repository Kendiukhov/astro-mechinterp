"""Model and codec loading, with device handling."""

from __future__ import annotations

import functools
import os

import torch

DEFAULT_MODEL = "polymathic-ai/aion-base"
_SIZES = {
    "base": "polymathic-ai/aion-base",
    "large": "polymathic-ai/aion-large",
    "xlarge": "polymathic-ai/aion-xlarge",
}


def device_for(prefer: str | None = None) -> torch.device:
    """
    Pick a device. MPS is ~3x faster than CPU here but historically has had reduction
    bugs; set ASTRON_MI_DEVICE=cpu to force CPU. Any experiment making a numerical claim
    should be spot-checked on CPU (see hooks.compare_devices).
    """
    if prefer:
        return torch.device(prefer)
    env = os.environ.get("ASTRON_MI_DEVICE")
    if env:
        return torch.device(env)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@functools.lru_cache(maxsize=3)
def _load_model_cached(repo: str, device_str: str):
    from aion import AION

    model = AION.from_pretrained(repo).eval().to(torch.device(device_str))
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_model(size: str = "base", device: str | torch.device | None = None):
    """
    Load an AION checkpoint, eval mode, gradients off.

    Gradients are disabled on parameters because every method here is either
    weights-only or activation-based; if you need input gradients, re-enable
    selectively and bypass CodecManager (its encode/decode are @torch.no_grad).
    """
    repo = _SIZES.get(size, size)
    dev = torch.device(device) if device is not None else device_for()
    return _load_model_cached(repo, str(dev))


@functools.lru_cache(maxsize=2)
def load_codecs(device: str = "cpu"):
    """
    CodecManager. Keep this on CPU: the codecs are small, they run once per object during
    caching, and keeping them off the accelerator avoids dtype surprises.
    """
    from aion.codecs.manager import CodecManager

    return CodecManager(device=device)


def load_single_codec(modality_cls, device: str = "cpu"):
    """
    Load one codec by modality dataclass (e.g. aion.modalities.Ra).

    NOTE the dataclass names are not the token keys: Ra/Dec/Z/GaiaParallax, not
    tok_ra/tok_dec/tok_z/tok_parallax. Several modalities share one codec because the
    HF subfolder is keyed on `modality.name` — Image and HSCImage are both "image",
    DESISpectrum and SDSSSpectrum are both "spectrum".
    """
    return load_codecs(device)._load_codec(modality_cls)
