from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAEOutput(NamedTuple):
    x_hat: torch.Tensor   # [B, d_model] reconstructed input
    z: torch.Tensor       # [B, d_sae] sparse activations (post-threshold)
    z_pre: torch.Tensor   # [B, d_sae] pre-threshold activations (post-ReLU)
    l0: torch.Tensor      # scalar, mean number of active features per token


class BatchTopKSAE(nn.Module):
    """
    BatchTopK Sparse Autoencoder following the InterProt paper.

    Standard TopK enforces exactly k active features per token. BatchTopK
    instead enforces exactly k*B active features across the entire batch of B
    tokens, so each token gets a variable number of active features with mean k.
    This avoids forcing simple and complex tokens into the same sparsity budget.

    Architecture:
        encode:  z_pre = ReLU((x - b_dec) @ W_enc + b_enc)
        threshold: keep top k*B values across batch (zero rest)
        decode:  x_hat = z @ W_dec + b_dec

    W_dec columns are constrained to unit norm throughout training.
    """

    def __init__(self, d_model: int, d_sae: int, k: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        nn.init.kaiming_uniform_(self.W_enc)
        # Initialize decoder as normalized transpose of encoder so features
        # start geometrically aligned — common practice for SAEs.
        with torch.no_grad():
            self.W_dec.data = self.W_enc.data.T.clone()
            self._normalize_decoder()

    # ------------------------------------------------------------------
    # Core forward components
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, d_model] → [B, d_sae] pre-threshold activations."""
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, d_sae] → [B, d_model] reconstructed activations."""
        return z @ self.W_dec + self.b_dec

    def _apply_batch_topk(self, z_pre: torch.Tensor) -> torch.Tensor:
        """Zero out all but the top k*B values across the batch."""
        B = z_pre.shape[0]
        k_total = min(self.k * B, z_pre.numel())
        # topk on the flattened tensor; .values[-1] is the smallest kept value.
        threshold = torch.topk(z_pre.reshape(-1), k_total, sorted=True).values[-1]
        return z_pre * (z_pre >= threshold)

    def forward(self, x: torch.Tensor) -> SAEOutput:
        z_pre = self.encode(x)
        z = self._apply_batch_topk(z_pre)
        x_hat = self.decode(z)
        l0 = (z > 0).float().sum(dim=-1).mean()
        return SAEOutput(x_hat=x_hat, z=z, z_pre=z_pre, l0=l0)

    # ------------------------------------------------------------------
    # Unit-norm decoder constraint
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        """Renormalize each decoder feature vector to unit norm."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @torch.no_grad()
    def remove_parallel_component_of_grads(self) -> None:
        """Project decoder gradients onto the tangent plane of the unit sphere.

        Called before optimizer.step() so the update stays near unit norm
        and the subsequent renormalization is a small correction rather than
        a large projection.
        """
        if self.W_dec.grad is not None:
            parallel = (self.W_dec.grad * self.W_dec.data).sum(dim=1, keepdim=True) * self.W_dec.data
            self.W_dec.grad -= parallel

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "d_model": self.d_model,
                "d_sae": self.d_sae,
                "k": self.k,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "BatchTopKSAE":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = cls(d_model=ckpt["d_model"], d_sae=ckpt["d_sae"], k=ckpt["k"])
        model.load_state_dict(ckpt["state_dict"])
        return model
