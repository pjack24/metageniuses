from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from metageniuses.extraction.contracts import iter_layer_batches
from .config import SAEConfig
from .model import BatchTopKSAE, SAEOutput


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_activations(cfg: SAEConfig) -> tuple[torch.Tensor, float]:
    """Stream all token activations for the target layer into a CPU tensor.

    Returns (x_all, mean_norm) where mean_norm is the mean L2 norm of the
    activation vectors (used for optional input normalization).
    """
    print(f"Loading activations: {cfg.artifact_root}  layer={cfg.transformer_layer}")
    chunks: list[torch.Tensor] = []
    for batch_vecs, _ in iter_layer_batches(
        cfg.artifact_root, cfg.transformer_layer, batch_size=8192
    ):
        chunks.append(torch.tensor(batch_vecs, dtype=torch.float32))

    x_all = torch.cat(chunks, dim=0)  # [N, d_model]
    mean_norm = x_all.norm(dim=-1).mean().item()
    print(f"  {x_all.shape[0]:,} tokens  d_model={x_all.shape[1]}  mean_norm={mean_norm:.3f}")
    return x_all, mean_norm


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def auxiliary_loss(
    sae: BatchTopKSAE,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    z_pre: torch.Tensor,
    dead_mask: torch.Tensor,
) -> torch.Tensor:
    """Ghost-grads auxiliary loss to prevent dead features.

    Dead features' pre-activations are used to produce a "ghost" reconstruction
    of the current residual. This gives dead features a gradient signal even
    though they were not selected by BatchTopK.
    """
    if not dead_mask.any():
        return x.new_tensor(0.0)

    residual = (x - x_hat).detach()                          # [B, d_model]
    z_ghost = z_pre * dead_mask.float().unsqueeze(0)          # [B, d_sae], dead only
    x_ghost = (z_ghost @ sae.W_dec)                           # [B, d_model], no bias

    # Scale ghost output to match residual magnitude per token
    res_norm = residual.norm(dim=-1, keepdim=True)            # [B, 1]
    ghost_norm = x_ghost.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_ghost = x_ghost * (res_norm / ghost_norm)

    return F.mse_loss(x_ghost, residual)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: SAEConfig) -> None:
    cfg.validate()
    device = cfg.device
    torch_dtype = torch.float32 if cfg.dtype == "float32" else torch.bfloat16

    # ---- Load activations ------------------------------------------------
    x_all_cpu, mean_norm = load_activations(cfg)
    activation_scale = mean_norm if cfg.normalize_activations else 1.0
    x_all_cpu = x_all_cpu / activation_scale
    x_all = x_all_cpu.to(device=device, dtype=torch_dtype)
    n_tokens = x_all.shape[0]
    del x_all_cpu

    # ---- Build model -----------------------------------------------------
    sae = BatchTopKSAE(d_model=cfg.d_model, d_sae=cfg.d_sae, k=cfg.k)

    # Initialize b_dec to the mean of the training data so the encoder starts
    # from a centered representation.
    with torch.no_grad():
        sae.b_dec.data = x_all.mean(dim=0).float()

    sae = sae.to(device=device, dtype=torch_dtype)

    n_params = sum(p.numel() for p in sae.parameters())
    print(
        f"SAE: d_model={cfg.d_model}  d_sae={cfg.d_sae}  k={cfg.k}  "
        f"params={n_params:,}  dtype={cfg.dtype}"
    )

    # ---- Optimizer -------------------------------------------------------
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

    # ---- Output dir ------------------------------------------------------
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    # Save activation scale so checkpoints can decode back to original space
    (out_dir / "activation_scale.json").write_text(
        json.dumps({"activation_scale": activation_scale, "layer": cfg.transformer_layer})
    )

    # ---- Dead-feature tracker --------------------------------------------
    steps_since_fired = torch.zeros(cfg.d_sae, device=device)

    # ---- Training --------------------------------------------------------
    global_step = 0
    t0 = time.time()

    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(n_tokens, device=device)
        x_shuffled = x_all[perm]

        for batch_start in range(0, n_tokens, cfg.batch_size):
            x = x_shuffled[batch_start : batch_start + cfg.batch_size]
            if x.shape[0] < 2:
                continue

            # Forward
            out: SAEOutput = sae(x)
            dead_mask = steps_since_fired >= cfg.dead_steps_threshold

            loss_recon = reconstruction_loss(x, out.x_hat)
            loss_aux = auxiliary_loss(sae, x, out.x_hat, out.z_pre, dead_mask)
            loss = loss_recon + cfg.aux_loss_coeff * loss_aux

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Project decoder grads before step to stay near unit-norm manifold
            sae.remove_parallel_component_of_grads()

            optimizer.step()

            # Hard renormalize decoder columns after each step
            sae._normalize_decoder()

            # Update dead-feature tracker
            fired = (out.z.detach() > 0).any(dim=0)  # [d_sae]
            steps_since_fired = torch.where(
                fired,
                torch.zeros_like(steps_since_fired),
                steps_since_fired + 1,
            )

            # Logging
            if global_step % cfg.log_every == 0:
                dead_pct = (steps_since_fired >= cfg.dead_steps_threshold).float().mean().item() * 100
                elapsed = time.time() - t0
                print(
                    f"step={global_step:6d}  epoch={epoch}  "
                    f"loss={loss.item():.5f}  recon={loss_recon.item():.5f}  "
                    f"aux={loss_aux.item():.5f}  l0={out.l0.item():.1f}  "
                    f"dead={dead_pct:.1f}%  t={elapsed:.0f}s"
                )

            # Checkpoint
            if global_step > 0 and global_step % cfg.checkpoint_every == 0:
                ckpt_path = out_dir / f"sae_step_{global_step:07d}.pt"
                sae.save(ckpt_path)
                print(f"  checkpoint → {ckpt_path}")

            global_step += 1

    # Final save
    final_path = out_dir / "sae_final.pt"
    sae.save(final_path)
    print(f"\nDone. {global_step} steps  →  {final_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BatchTopK SAE on METAGENE-1 activations")
    p.add_argument("--config", type=str, help="Path to SAE config JSON (optional)")
    p.add_argument("--artifact_root", type=str)
    p.add_argument("--layer", type=int, dest="transformer_layer")
    p.add_argument("--k", type=int)
    p.add_argument("--expansion_factor", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--n_epochs", type=int)
    p.add_argument("--output_dir", type=str)
    p.add_argument("--device", type=str)
    p.add_argument("--dtype", type=str)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = SAEConfig.from_json(args.config) if args.config else SAEConfig()

    # Apply CLI overrides on top of config file
    overrides = {
        k: v
        for k, v in vars(args).items()
        if k != "config" and v is not None
    }
    if overrides:
        cfg = SAEConfig(**{**cfg.to_dict(), **overrides})

    train(cfg)


if __name__ == "__main__":
    main()
