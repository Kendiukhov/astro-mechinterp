"""
BatchTopK sparse autoencoder, and the baselines any result has to beat.

BatchTopK (Bussmann et al. 2024) rather than L1: the sparsity level is exact and set by k,
there is no L1 coefficient to tune, and there is no shrinkage bias on the active features.
The top-k is taken over the whole BATCH rather than per example, so different examples can
use different numbers of features while the average is exactly k.

Deliberately small and readable — this is a research instrument, not a library. The point
is not the SAE, it is whether the SAE beats:

  - PCA at matched sparsity (project onto the top-k PCs)          -> `PCABaseline`
  - difference-in-means, the D2c gate direction                   -> supplied externally

Standing requirement from AGENDA.md 11: report the delta against those, not the SAE alone.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchTopKSAE(nn.Module):
    """
    x -> encode -> BatchTopK -> decode -> x_hat

    Pre-encoder bias subtraction and unit-norm decoder columns are the standard setup;
    without the norm constraint the model can cheat sparsity by scaling features up.
    """

    def __init__(self, d_in: int, n_features: int, k: int = 32):
        super().__init__()
        self.d_in, self.n_features, self.k = d_in, n_features, k
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_in, n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(n_features, d_in))
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T.clone())
            self._normalise_decoder()

    @torch.no_grad()
    def _normalise_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True) + 1e-8)

    def encode_pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x, k: int | None = None):
        """BatchTopK: keep the k*B largest activations across the whole batch."""
        pre = F.relu(self.encode_pre(x))
        k = self.k if k is None else k
        n_keep = max(1, k * x.shape[0])
        flat = pre.flatten()
        if n_keep < flat.numel():
            thresh = torch.topk(flat, n_keep, sorted=False).values.min()
            pre = torch.where(pre >= thresh, pre, torch.zeros_like(pre))
        return pre

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


def train_sae(
    X: np.ndarray,
    n_features: int = 8192,
    k: int = 32,
    steps: int = 4000,
    batch: int = 4096,
    lr: float = 3e-4,
    device: str = "cpu",
    seed: int = 0,
    log_every: int = 500,
    dead_after: int = 400,
    verbose: bool = True,
):
    """
    Train on rows of X. Returns (model, info).

    Inputs are standardised to unit mean-norm, which matters here because image and segmap
    rows have somewhat different scales (measured 1.00-1.27 across layers, so mild, but
    free to correct).

    Dead features are resampled to the worst-reconstructed inputs, the standard fix; the
    fraction that die is reported because a dictionary where most features are dead is not
    the dictionary you think you trained.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    Xt = torch.from_numpy(np.asarray(X, dtype="float32"))
    scale = Xt.norm(dim=1).mean()
    Xt = Xt / scale
    Xt = Xt.to(dev)

    sae = BatchTopKSAE(Xt.shape[1], n_features, k).to(dev)
    with torch.no_grad():                       # init decoder bias at the data mean
        sae.b_dec.copy_(Xt.mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed)

    last_fired = torch.zeros(n_features, dtype=torch.long, device=dev)
    info = {"loss": [], "fvu": [], "dead_frac": []}

    for step in range(steps):
        idx = torch.randint(0, Xt.shape[0], (batch,), generator=g).to(dev)
        x = Xt[idx]
        xh, f = sae(x)
        loss = (xh - x).pow(2).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # remove the gradient component along each decoder row before the step, so the
        # unit-norm projection afterwards does not fight the optimiser
        with torch.no_grad():
            proj = (sae.W_dec.grad * sae.W_dec).sum(1, keepdim=True) * sae.W_dec
            sae.W_dec.grad -= proj
        opt.step()
        sae._normalise_decoder()

        fired = (f > 0).any(0)
        last_fired = torch.where(fired, torch.zeros_like(last_fired), last_fired + 1)

        if (step + 1) % dead_after == 0:
            dead = last_fired > dead_after
            if dead.any():
                with torch.no_grad():
                    # Resample dead features onto the WORST-RECONSTRUCTED inputs. The
                    # reconstruction and the residual must come from the SAME batch --
                    # using two different random batches makes `resid` meaningless and
                    # blows the loss up for hundreds of steps afterwards.
                    ridx = torch.randint(0, Xt.shape[0], (batch,), generator=g).to(dev)
                    xb = Xt[ridx]
                    xh_b, _ = sae(xb)
                    resid = xb - xh_b
                    n = min(int(dead.sum()), batch)
                    worst = resid[resid.pow(2).sum(1).topk(n).indices]
                    w = worst / (worst.norm(dim=1, keepdim=True) + 1e-8)
                    di = torch.nonzero(dead).flatten()[:n]
                    sae.W_dec[di] = w
                    # small encoder init so a resampled feature does not immediately
                    # dominate the batch top-k
                    sae.W_enc[:, di] = w.T * 0.1
                    sae.b_enc[di] = 0.0
                    last_fired[di] = 0

        if verbose and ((step + 1) % log_every == 0 or step == 0):
            with torch.no_grad():
                sub = Xt[torch.randint(0, Xt.shape[0], (8192,), generator=g).to(dev)]
                xh2, f2 = sae(sub)
                fvu = ((xh2 - sub).pow(2).sum() / (sub - sub.mean(0)).pow(2).sum()).item()
                dfrac = float((last_fired > dead_after).float().mean())
            info["loss"].append(float(loss.detach())); info["fvu"].append(fvu)
            info["dead_frac"].append(dfrac)
            print(f"    step {step + 1:5d}  loss {float(loss):.4f}  FVU {fvu:.4f}  "
                  f"dead {dfrac:.1%}", flush=True)

    sae.scale = float(scale)
    return sae, info


class PCABaseline:
    """
    The mandatory co-equal arm: project onto the top-k principal components.

    Matched sparsity means k components against the SAE's k active features. If the
    dictionary cannot beat this, it has not earned its GPU-hours.
    """

    def __init__(self, X: np.ndarray, n_components: int = 256):
        self.mean = X.mean(0)
        Xc = X - self.mean
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        self.components = vt[:n_components]
        self.sv = s[:n_components]

    def encode(self, X: np.ndarray, k: int) -> np.ndarray:
        """Top-k coefficients by magnitude, everything else zeroed."""
        c = (X - self.mean) @ self.components.T
        if k < c.shape[1]:
            thresh = np.partition(np.abs(c), -k, axis=1)[:, -k][:, None]
            c = np.where(np.abs(c) >= thresh, c, 0.0)
        return c

    def reconstruct(self, X: np.ndarray, k: int) -> np.ndarray:
        return self.encode(X, k) @ self.components + self.mean

    def fvu(self, X: np.ndarray, k: int) -> float:
        r = self.reconstruct(X, k)
        return float(((X - r) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())
