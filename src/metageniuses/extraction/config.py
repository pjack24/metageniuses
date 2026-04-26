from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputConfig:
    path: str
    format: str = "jsonl"
    sequence_key: str = "sequence"
    id_key: str = "sequence_id"
    metadata_keys: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.format not in {"jsonl", "fasta"}:
            raise ValueError(f"Unsupported input format: {self.format}")
        if not self.path:
            raise ValueError("Input path is required.")


@dataclass(frozen=True)
class PreprocessConfig:
    uppercase: bool = True
    allowed_chars: str = "ACGTUN"
    replace_invalid_with: str = "N"
    max_invalid_fraction: float = 0.05
    min_length: int = 1
    max_length: int = 512
    strip_whitespace: bool = True

    def validate(self) -> None:
        if self.min_length < 1:
            raise ValueError("min_length must be >= 1")
        if self.max_length < self.min_length:
            raise ValueError("max_length must be >= min_length")
        if self.replace_invalid_with not in self.allowed_chars:
            raise ValueError("replace_invalid_with must be in allowed_chars")
        if not (0.0 <= self.max_invalid_fraction <= 1.0):
            raise ValueError("max_invalid_fraction must be in [0, 1]")


@dataclass(frozen=True)
class LayerSelectionConfig:
    layers: list[int] | None = None
    last_n_layers: int | None = None

    def validate(self) -> None:
        if self.layers and self.last_n_layers:
            raise ValueError("Set either layers or last_n_layers, not both.")
        if self.layers is None and self.last_n_layers is None:
            raise ValueError("Set layers or last_n_layers.")
        if self.layers is not None:
            if not self.layers:
                raise ValueError("layers cannot be empty.")
            for layer in self.layers:
                if layer < 1:
                    raise ValueError("layers must use transformer-layer indices starting at 1.")
        if self.last_n_layers is not None and self.last_n_layers < 1:
            raise ValueError("last_n_layers must be >= 1")

    def resolve(self, num_transformer_layers: int) -> list[int]:
        self.validate()
        if self.layers is not None:
            resolved = sorted(set(self.layers))
        else:
            assert self.last_n_layers is not None
            start = max(1, num_transformer_layers - self.last_n_layers + 1)
            resolved = list(range(start, num_transformer_layers + 1))

        for layer in resolved:
            if layer > num_transformer_layers:
                raise ValueError(
                    f"Requested layer {layer}, but model only has {num_transformer_layers} layers."
                )
        return resolved


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    tokenizer_id: str | None = None
    revision: str | None = None
    local_files_only: bool = False
    trust_remote_code: bool = False
    device: str = "auto"
    dtype: str = "auto"
    stop_after_last_requested_layer: bool = True

    def validate(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required.")


@dataclass(frozen=True)
class RuntimeConfig:
    output_root: str = "results/extraction"
    run_id: str | None = None
    batch_size: int = 4
    max_batch_size: int | None = None
    initial_max_batch_size: int | None = None
    release_to_max_after_sequences: int | None = None
    batch_growth_success_batches: int = 2
    batch_growth_step: int | None = None
    reduce_batch_on_oom: bool = True
    defer_token_index: bool = True
    async_write: bool = True
    async_queue_max_batches: int | None = None
    max_rows_per_shard: int = 100000
    max_reads: int | None = None
    resume: bool = False
    flush_every_sequences: int = 100
    progress_every_sequences: int = 100

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        configured_max_batch_size = self.max_batch_size or self.batch_size
        if configured_max_batch_size < self.batch_size:
            raise ValueError("max_batch_size must be >= batch_size when set")
        if self.initial_max_batch_size is not None:
            if self.initial_max_batch_size < self.batch_size:
                raise ValueError("initial_max_batch_size must be >= batch_size when set")
            if self.initial_max_batch_size > configured_max_batch_size:
                raise ValueError(
                    "initial_max_batch_size must be <= max_batch_size when max_batch_size is set"
                )
            if (
                self.initial_max_batch_size < configured_max_batch_size
                and self.release_to_max_after_sequences is None
            ):
                raise ValueError(
                    "release_to_max_after_sequences is required when initial_max_batch_size "
                    "is lower than max_batch_size"
                )
        if self.initial_max_batch_size is None and self.release_to_max_after_sequences is not None:
            raise ValueError(
                "release_to_max_after_sequences requires initial_max_batch_size to also be set"
            )
        if (
            self.release_to_max_after_sequences is not None
            and self.release_to_max_after_sequences < 1
        ):
            raise ValueError("release_to_max_after_sequences must be >= 1 when set")
        if self.batch_growth_success_batches < 1:
            raise ValueError("batch_growth_success_batches must be >= 1")
        if self.batch_growth_step is not None and self.batch_growth_step < 1:
            raise ValueError("batch_growth_step must be >= 1 when set")
        if self.async_queue_max_batches is not None and self.async_queue_max_batches < 1:
            raise ValueError("async_queue_max_batches must be >= 1 when set")
        if self.max_rows_per_shard < 1:
            raise ValueError("max_rows_per_shard must be >= 1")
        if self.max_reads is not None and self.max_reads < 1:
            raise ValueError("max_reads must be >= 1 when set")
        if self.flush_every_sequences < 1:
            raise ValueError("flush_every_sequences must be >= 1")
        if self.progress_every_sequences < 1:
            raise ValueError("progress_every_sequences must be >= 1")


@dataclass(frozen=True)
class ExtractionConfig:
    input: InputConfig
    preprocess: PreprocessConfig
    model: ModelConfig
    layer_selection: LayerSelectionConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExtractionConfig":
        return cls(
            input=InputConfig(**payload["input"]),
            preprocess=PreprocessConfig(**payload.get("preprocess", {})),
            model=ModelConfig(**payload["model"]),
            layer_selection=LayerSelectionConfig(**payload["layer_selection"]),
            runtime=RuntimeConfig(**payload.get("runtime", {})),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ExtractionConfig":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    def validate(self) -> None:
        self.input.validate()
        self.preprocess.validate()
        self.model.validate()
        self.layer_selection.validate()
        self.runtime.validate()
