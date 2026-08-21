"""
Activation extraction for AION-1.

The canonical path (NOT `model.encode()`, which is the decoder-projected tensor with
pos/mod embeddings re-added and no `mod_mask`):

    enc_tok, enc_emb, enc_mask, mod_mask = model.embed_inputs(tokens, num_encoder_tokens=N)
    h = model.forward_encoder(enc_tok + enc_emb, encoder_mask=enc_mask)   # post encoder_norm

Verified on aion-base: `forward_encoder` returns (B, N, 768) with row norms ~27.0 = sqrt(768)
and std 0.09, i.e. it is post-LayerNorm. `mod_mask` is int16 carrying
MODALITY_INFO[key]['id'] (a unique uint15 hash), with -1 for padding rows.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Iterable, Sequence

import numpy as np
import torch


# --------------------------------------------------------------------- modality ids
@functools.lru_cache(maxsize=1)
def modality_id_map() -> tuple[dict[str, int], dict[int, str]]:
    """(key -> id, id -> key) from MODALITY_INFO. Ids are unique; verified for all 40."""
    from aion.fourm.modality_info import MODALITY_INFO

    fwd = {k: int(v["id"]) for k, v in MODALITY_INFO.items()}
    if len(set(fwd.values())) != len(fwd):
        raise RuntimeError("modality ids are not unique; row attribution would be wrong")
    return fwd, {v: k for k, v in fwd.items()}


def row_index(mod_mask: torch.Tensor) -> list[list[tuple[str, int]]]:
    """
    Map each encoder row to (modality_key, position_within_modality).

    Ordering guarantee, from fm.py:441-465: tokens are concatenated in `input_dict`
    insertion order, then sorted by `input_mask + arange*1e-6`. With an all-zero input
    mask (the default) that sort is the identity, so rows appear in original order and
    the position within a modality is simply a running counter. If you pass a non-trivial
    input_mask, masked rows sort to the end and their original positions are NOT
    recoverable from mod_mask alone — track them yourself.

    Padding rows (mod_mask == -1) are returned as ("<pad>", -1).
    """
    _, inv = modality_id_map()
    out = []
    for row in mod_mask.detach().cpu().numpy():
        counter: dict[str, int] = {}
        seq = []
        for mid in row:
            mid = int(mid)
            if mid < 0:
                seq.append(("<pad>", -1))
                continue
            key = inv.get(mid, f"<unknown:{mid}>")
            pos = counter.get(key, 0)
            counter[key] = pos + 1
            seq.append((key, pos))
        out.append(seq)
    return out


def modality_row_mask(mod_mask: torch.Tensor, key: str) -> torch.Tensor:
    """Boolean mask selecting the rows belonging to one modality. (B, N)."""
    fwd, _ = modality_id_map()
    if key not in fwd:
        raise KeyError(f"unknown modality {key!r}")
    return mod_mask == fwd[key]


# ------------------------------------------------------------------------- capture
class ActivationCapture:
    """
    Context manager capturing intermediate tensors by forward hook.

    Sites (aion-base uses NormAttention / NormCrossAttention because qk_norm=True; the
    plain Attention/CrossAttention classes exist but are never instantiated, so patching
    them silently does nothing):

        enc_resid.{i}   output of encoder block i          (B, N, D)
        enc_attn.{i}    attention sublayer output          (B, N, D)
        enc_mlp.{i}     MLP sublayer output                (B, N, D)
        enc_attnmap.{i} post-softmax attention             (B, H, N, N)
        dec_resid.{i}, dec_selfmap.{i}, dec_crossmap.{i}   decoder equivalents

    Attention maps come free from the p=0 `attn_drop` Dropout, whose forward still fires
    on the post-softmax tensor. Rows sum to 1 (assert_attention_normalised checks this).

    Attention maps are large: N=600, 12 heads, 12 layers, fp32 is ~200 MB per batch
    element. Pass `layers=` to restrict.
    """

    def __init__(
        self,
        model,
        sites: Sequence[str] = ("enc_resid",),
        layers: Iterable[int] | None = None,
        store_dtype: torch.dtype = torch.float32,
        cpu: bool = True,
    ):
        self.model = model
        self.sites = set(sites)
        self.layers = None if layers is None else set(layers)
        self.store_dtype = store_dtype
        self.cpu = cpu
        self.acts: dict[str, torch.Tensor] = {}
        self._handles: list = []

    def _keep(self, i: int) -> bool:
        return self.layers is None or i in self.layers

    def _save(self, name: str):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.detach().to(self.store_dtype)
            self.acts[name] = t.cpu() if self.cpu else t

        return hook

    def __enter__(self):
        self.acts.clear()
        for i, blk in enumerate(self.model.encoder):
            if not self._keep(i):
                continue
            if "enc_resid" in self.sites:
                self._handles.append(blk.register_forward_hook(self._save(f"enc_resid.{i}")))
            if "enc_attn" in self.sites:
                self._handles.append(blk.attn.register_forward_hook(self._save(f"enc_attn.{i}")))
            if "enc_mlp" in self.sites:
                self._handles.append(blk.mlp.register_forward_hook(self._save(f"enc_mlp.{i}")))
            if "enc_attnmap" in self.sites:
                self._handles.append(
                    blk.attn.attn_drop.register_forward_hook(self._save(f"enc_attnmap.{i}"))
                )
        for i, blk in enumerate(getattr(self.model, "decoder", [])):
            if not self._keep(i):
                continue
            if "dec_resid" in self.sites:
                self._handles.append(blk.register_forward_hook(self._save(f"dec_resid.{i}")))
            if "dec_selfmap" in self.sites:
                self._handles.append(
                    blk.self_attn.attn_drop.register_forward_hook(self._save(f"dec_selfmap.{i}"))
                )
            if "dec_crossmap" in self.sites:
                self._handles.append(
                    blk.cross_attn.attn_drop.register_forward_hook(self._save(f"dec_crossmap.{i}"))
                )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def read_targets(model, tokens, target_modalities, physics_tables, device=None,
                 steer=None, alpha=0.0, layers=(6,), seed: int = 0):
    """
    Read several scalar targets SAFELY — one forward pass per target.

    Do NOT pass a list of targets to `AION.forward`. `AION.forward` supplies an all-zero
    `decoder_attention_mask` (model.py:99), which `adapt_decoder_attention_mask` cumsums to
    zero and turns into `arange >= 0` — i.e. every position masked. With a single decoder
    token that is harmless. With more than one it makes the decoder self-attention uniform
    over all decoder tokens, and `decoder_sep_mask` cannot save it. Measured on redshift
    with image+spectrum input:

        target_modality=[Z]                        median |dz| = 0.0038, max posterior 0.90
        target_modality=[FluxG, Z]                 median |dz| = 0.83,   max posterior 0.41
        target_modality=[FluxG, FluxZ, Z]          median |dz| = 2.35,   max posterior 0.15

    Batching targets is therefore not an optimisation, it is a correctness bug. This helper
    loops instead, which costs one forward per target and is always right.
    """
    import random as _random

    dev = device or next(model.parameters()).device
    tk = {k: v.to(dev) for k, v in tokens.items()}
    out = {}
    for mod in target_modalities:
        _random.seed(seed)
        torch.manual_seed(seed)
        if steer is None:
            o = model.forward(tk, target_modality=[mod])
        else:
            with Steer(model, steer, alpha=alpha, layers=layers):
                o = model.forward(tk, target_modality=[mod])
        for key, v in o.items():
            out[key] = physics_tables.expected_value(
                v[:, 0, :].detach().float().cpu().numpy(), key
            )
    return out


class Steer:
    """
    Add a fixed vector to the residual stream at chosen encoder blocks during a forward.

    Used to test whether a candidate direction is causally responsible for a behaviour:
    find the direction by difference-in-means between two conditions, then add it back
    under the condition where it is absent and see whether the behaviour returns.

        with Steer(model, direction=d, alpha=1.0, layers=[6]):
            out = model.forward(tokens, target_modality=[LegacySurveyFluxG])

    `direction` is (D,) and is added to every row. Always compare against a random
    direction of matched norm — adding any large vector perturbs the output.
    """

    def __init__(self, model, direction: torch.Tensor, alpha: float = 1.0,
                 layers: Iterable[int] = (6,)):
        self.model = model
        self.d = direction
        self.alpha = alpha
        self.layers = list(layers)
        self._handles: list = []

    def __enter__(self):
        dev = next(self.model.parameters()).device
        vec = self.d.to(dev).to(torch.float32).reshape(1, 1, -1) * self.alpha

        def hook(_m, _i, out):
            if isinstance(out, tuple):
                return (out[0] + vec,) + out[1:]
            return out + vec

        for i in self.layers:
            self._handles.append(self.model.encoder[i].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def assert_attention_normalised(attn: torch.Tensor, tol: float = 1e-3) -> None:
    """Sanity check that a captured map really is post-softmax."""
    s = attn.sum(-1)
    if not torch.allclose(s, torch.ones_like(s), atol=tol):
        raise AssertionError(
            f"attention rows do not sum to 1 (min {float(s.min()):.4f}, max {float(s.max()):.4f}); "
            "the hook is probably on the wrong module"
        )


# ----------------------------------------------------------------------- main entry
@torch.no_grad()
def encode_activations(
    model,
    tokens: dict[str, torch.Tensor],
    num_encoder_tokens: int = 600,
    sites: Sequence[str] = ("enc_resid",),
    layers: Iterable[int] | None = None,
    return_rows: bool = False,
):
    """
    Run the encoder and return activations plus the row->modality map.

    Returns a dict with:
        final     (B, N, D)  post-encoder_norm output
        mod_mask  (B, N)     int16 modality id per row, -1 for padding
        valid     (B, N)     bool, True for real (non-padding) rows
        acts      {site: tensor}
        rows      list of [(modality_key, pos), ...] if return_rows
    """
    device = next(model.parameters()).device
    toks = {k: v.to(device) for k, v in tokens.items()}

    enc_tok, enc_emb, enc_mask, mod_mask = model.embed_inputs(
        toks, num_encoder_tokens=num_encoder_tokens
    )
    with ActivationCapture(model, sites=sites, layers=layers) as cap:
        final = model.forward_encoder(enc_tok + enc_emb, encoder_mask=enc_mask)

    out = {
        "final": final.detach().cpu(),
        "mod_mask": mod_mask.detach().cpu(),
        "valid": (mod_mask.detach().cpu() >= 0),
        "acts": dict(cap.acts),
    }
    if return_rows:
        out["rows"] = row_index(out["mod_mask"])
    return out


def pool_by_modality(
    h: torch.Tensor, mod_mask: torch.Tensor, keys: Sequence[str] | None = None
) -> dict[str, torch.Tensor]:
    """
    Mean-pool encoder rows separately per modality.

    This is the primitive behind agenda experiment D0: probing from all rows versus from
    spectrum rows only tests whether the paper's mean-pooling collapse is token dilution.
    Modalities absent from a given batch element yield NaN for that element.
    """
    fwd, inv = modality_id_map()
    keys = keys or sorted({inv[int(m)] for m in mod_mask.unique() if int(m) >= 0})
    out = {}
    for k in keys:
        sel = (mod_mask == fwd[k]).unsqueeze(-1).to(h.dtype)
        n = sel.sum(1)
        pooled = (h * sel).sum(1) / n.clamp(min=1)
        pooled[n.squeeze(-1) == 0] = float("nan")
        out[k] = pooled
    return out


def compare_devices(model_cpu, model_acc, tokens, num_encoder_tokens: int = 600) -> float:
    """Max abs difference between CPU and accelerator encoder outputs. Run before
    trusting any accelerator-only numerical result."""
    a = encode_activations(model_cpu, tokens, num_encoder_tokens)["final"]
    b = encode_activations(model_acc, tokens, num_encoder_tokens)["final"]
    return float((a - b).abs().max())
