from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SAEConfig:
    # Architecture
    d_model: int = 4096          # METAGENE-1 hidden size
    expansion_factor: int = 8    # d_sae = expansion_factor * d_model → 32768
    k: int = 64                  # average active features per token (BatchTopK)

    # Data
    artifact_root: str = "data/activations"
    transformer_layer: int = 32  # which extracted layer to train on

    # Training
    lr: float = 2e-4
    batch_size: int = 4096       # token-level, not sequence-level
    n_epochs: int = 10
    aux_loss_coeff: float = 0.03125   # 1/32, weight on ghost-grads aux loss
    dead_steps_threshold: int = 200   # steps without firing → feature considered dead
    normalize_activations: bool = True  # divide inputs by their mean norm

    # Output
    output_dir: str = "data/sae"
    checkpoint_every: int = 500
    log_every: int = 50

    # Device / dtype
    device: str = "cuda"
    dtype: str = "float32"   # float32 or bfloat16

    @property
    def d_sae(self) -> int:
        return self.d_model * self.expansion_factor

    def validate(self) -> None:
        if self.d_model < 1:
            raise ValueError("d_model must be >= 1")
        if self.expansion_factor < 1:
            raise ValueError("expansion_factor must be >= 1")
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SAEConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in valid_keys})

    @classmethod
    def from_json(cls, path: str | Path) -> "SAEConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
