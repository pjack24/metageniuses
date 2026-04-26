from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import ExtractionConfig
from .input_io import iter_sequence_records
from .model_adapter import ModelAdapter, TransformersModelAdapter
from .preprocess import preprocess_record
from .schemas import ExtractionStats, ModelDescription, RunManifest, SCHEMA_VERSION, utc_now_iso
from .storage import ActivationStore


class ResidualExtractionPipeline:
    """Extract token-level hidden states for selected transformer layers."""

    def run(
        self,
        cfg: ExtractionConfig,
        adapter: ModelAdapter | None = None,
    ) -> Path:
        cfg.validate()
        self._validate_input_path(cfg)
        self._validate_output_root(cfg)

        run_id = cfg.runtime.run_id or self._make_run_id()
        artifact_root = Path(cfg.runtime.output_root) / run_id
        self._validate_run_directory(artifact_root=artifact_root, resume=cfg.runtime.resume)
        prior_progress = self._read_progress(artifact_root=artifact_root) if cfg.runtime.resume else {}

        model_adapter = adapter or TransformersModelAdapter(cfg.model)
        desc = model_adapter.describe()
        selected_layers = cfg.layer_selection.resolve(desc.num_transformer_layers)
        async_queue_batches = self._resolve_async_queue_batches(
            cfg=cfg,
            d_model=desc.d_model,
            selected_layer_count=len(selected_layers),
        )

        store = ActivationStore(
            artifact_root=artifact_root,
            selected_layers=selected_layers,
            d_model=desc.d_model,
            max_rows_per_shard=cfg.runtime.max_rows_per_shard,
            resume=cfg.runtime.resume,
            defer_token_index=cfg.runtime.defer_token_index,
            async_write=cfg.runtime.async_write,
            async_queue_max_batches=async_queue_batches,
        )

        stats = ExtractionStats(
            total_sequences_seen=max(
                int(prior_progress.get("total_sequences_seen", 0)),
                store.existing_sequences_kept,
            ),
            total_sequences_kept=store.existing_sequences_kept,
            total_sequences_skipped=int(prior_progress.get("total_sequences_skipped", 0)),
            total_tokens=store.existing_tokens,
            total_rows_written=store.existing_rows_written,
        )

        records = iter_sequence_records(cfg.input)
        skip_count = stats.total_sequences_seen
        pending_records = []
        current_batch_size = cfg.runtime.batch_size
        configured_max_batch_size = cfg.runtime.max_batch_size or cfg.runtime.batch_size
        adaptive_max_batch_size = configured_max_batch_size
        if cfg.runtime.initial_max_batch_size is not None:
            adaptive_max_batch_size = min(adaptive_max_batch_size, cfg.runtime.initial_max_batch_size)
        max_cap_released = adaptive_max_batch_size >= configured_max_batch_size
        max_cap_release_blocked = False
        successful_batches = 0
        stop_requested = False

        for record in records:
            if skip_count > 0:
                skip_count -= 1
                continue
            pending_records.append(record)
            while len(pending_records) >= current_batch_size:
                process_size = current_batch_size
                try:
                    stats = self._process_batch(
                        batch_records=pending_records[:process_size],
                        cfg=cfg,
                        stats=stats,
                        store=store,
                        model_adapter=model_adapter,
                        selected_layers=selected_layers,
                        artifact_root=artifact_root,
                    )
                except RuntimeError as exc:
                    can_retry_smaller = cfg.runtime.reduce_batch_on_oom and self._is_oom_error(exc)
                    if not can_retry_smaller or current_batch_size <= 1:
                        raise
                    adaptive_max_batch_size = min(adaptive_max_batch_size, max(1, current_batch_size - 1))
                    current_batch_size = min(max(1, current_batch_size // 2), adaptive_max_batch_size)
                    max_cap_release_blocked = True
                    successful_batches = 0
                    self._clear_cuda_cache(model_adapter)
                    continue

                pending_records = pending_records[process_size:]
                successful_batches += 1
                adaptive_max_batch_size, max_cap_released = self._maybe_release_max_batch_cap(
                    cfg=cfg,
                    stats=stats,
                    adaptive_max_batch_size=adaptive_max_batch_size,
                    configured_max_batch_size=configured_max_batch_size,
                    max_cap_released=max_cap_released,
                    max_cap_release_blocked=max_cap_release_blocked,
                )
                if (
                    current_batch_size < adaptive_max_batch_size
                    and successful_batches >= cfg.runtime.batch_growth_success_batches
                ):
                    current_batch_size = self._grow_batch_size(
                        current=current_batch_size,
                        max_allowed=adaptive_max_batch_size,
                        cfg=cfg,
                    )
                    successful_batches = 0
                if cfg.runtime.max_reads is not None and stats.total_sequences_seen >= cfg.runtime.max_reads:
                    stop_requested = True
                    break
            if stop_requested:
                break

        while pending_records and (
            cfg.runtime.max_reads is None or stats.total_sequences_seen < cfg.runtime.max_reads
        ):
            process_size = min(current_batch_size, len(pending_records))
            try:
                stats = self._process_batch(
                    batch_records=pending_records[:process_size],
                    cfg=cfg,
                    stats=stats,
                    store=store,
                    model_adapter=model_adapter,
                    selected_layers=selected_layers,
                    artifact_root=artifact_root,
                )
            except RuntimeError as exc:
                can_retry_smaller = cfg.runtime.reduce_batch_on_oom and self._is_oom_error(exc)
                if not can_retry_smaller or process_size <= 1:
                    raise
                adaptive_max_batch_size = min(adaptive_max_batch_size, max(1, process_size - 1))
                current_batch_size = min(max(1, process_size // 2), adaptive_max_batch_size)
                max_cap_release_blocked = True
                successful_batches = 0
                self._clear_cuda_cache(model_adapter)
                continue

            pending_records = pending_records[process_size:]
            successful_batches += 1
            adaptive_max_batch_size, max_cap_released = self._maybe_release_max_batch_cap(
                cfg=cfg,
                stats=stats,
                adaptive_max_batch_size=adaptive_max_batch_size,
                configured_max_batch_size=configured_max_batch_size,
                max_cap_released=max_cap_released,
                max_cap_release_blocked=max_cap_release_blocked,
            )
            if (
                current_batch_size < adaptive_max_batch_size
                and successful_batches >= cfg.runtime.batch_growth_success_batches
            ):
                current_batch_size = self._grow_batch_size(
                    current=current_batch_size,
                    max_allowed=adaptive_max_batch_size,
                    cfg=cfg,
                )
                successful_batches = 0

        layers_payload = store.finalize()
        self._write_manifest(
            cfg=cfg,
            desc=desc,
            run_id=run_id,
            selected_layers=selected_layers,
            stats=stats,
            layers_payload=layers_payload,
            artifact_root=artifact_root,
        )
        self._write_progress(artifact_root, stats)
        return artifact_root

    def _process_batch(
        self,
        batch_records: list,
        cfg: ExtractionConfig,
        stats: ExtractionStats,
        store: ActivationStore,
        model_adapter: ModelAdapter,
        selected_layers: list[int],
        artifact_root: Path,
    ) -> ExtractionStats:
        processed_records = []
        preprocess_metadata = []

        for record in batch_records:
            if cfg.runtime.max_reads is not None and stats.total_sequences_seen >= cfg.runtime.max_reads:
                break

            stats = ExtractionStats(
                total_sequences_seen=stats.total_sequences_seen + 1,
                total_sequences_kept=stats.total_sequences_kept,
                total_sequences_skipped=stats.total_sequences_skipped,
                total_tokens=stats.total_tokens,
                total_rows_written=stats.total_rows_written,
            )

            # If a sequence was fully committed before a disconnect, skip safely.
            if record.sequence_id in store.completed_sequence_ids:
                stats = ExtractionStats(
                    total_sequences_seen=stats.total_sequences_seen,
                    total_sequences_kept=stats.total_sequences_kept,
                    total_sequences_skipped=stats.total_sequences_skipped + 1,
                    total_tokens=stats.total_tokens,
                    total_rows_written=stats.total_rows_written,
                )
                self._write_progress(artifact_root, stats)
                continue

            prep = preprocess_record(record, cfg.preprocess)
            if prep.record is None:
                stats = ExtractionStats(
                    total_sequences_seen=stats.total_sequences_seen,
                    total_sequences_kept=stats.total_sequences_kept,
                    total_sequences_skipped=stats.total_sequences_skipped + 1,
                    total_tokens=stats.total_tokens,
                    total_rows_written=stats.total_rows_written,
                )
                self._write_progress(artifact_root, stats)
                continue

            processed_records.append(prep.record)
            preprocess_metadata.append(
                {
                    "invalid_char_count": prep.invalid_char_count,
                    "invalid_fraction": prep.invalid_fraction,
                }
            )

        if not processed_records:
            return stats

        sequences = [record.sequence for record in processed_records]
        batch = model_adapter.extract_batch(
            sequences=sequences,
            transformer_layers=selected_layers,
            max_length=cfg.preprocess.max_length,
        )

        sequence_rows = []
        total_tokens_in_batch = 0
        for seq_idx, record in enumerate(processed_records):
            token_ids = batch.token_ids[seq_idx]
            total_tokens_in_batch += len(token_ids)
            sequence_row = {
                "sequence_id": record.sequence_id,
                "sequence": record.sequence,
                "sequence_length": len(record.sequence),
                "token_count": len(token_ids),
                "metadata": record.metadata,
                "preprocess": preprocess_metadata[seq_idx],
            }
            if cfg.runtime.defer_token_index:
                sequence_row["token_ids"] = token_ids
            sequence_rows.append(sequence_row)

        store.append_batch(
            sequence_rows=sequence_rows,
            token_ids_batch=batch.token_ids,
            hidden_states_by_layer=batch.hidden_states_by_layer,
            selected_layers=selected_layers,
        )

        stats = ExtractionStats(
            total_sequences_seen=stats.total_sequences_seen,
            total_sequences_kept=stats.total_sequences_kept + len(sequence_rows),
            total_sequences_skipped=stats.total_sequences_skipped,
            total_tokens=stats.total_tokens + total_tokens_in_batch,
            total_rows_written=stats.total_rows_written + (total_tokens_in_batch * len(selected_layers)),
        )

        if (
            stats.total_sequences_seen % cfg.runtime.flush_every_sequences == 0
            or stats.total_sequences_seen % cfg.runtime.progress_every_sequences == 0
        ):
            store.flush()
            self._write_progress(artifact_root, stats)

        return stats

    def _slice_sequence_matrix(self, layer_values: Any, seq_idx: int, token_count: int) -> Any:
        if hasattr(layer_values, "shape") and len(layer_values.shape) == 3:
            return layer_values[seq_idx, :token_count, :]
        return layer_values[seq_idx][:token_count]

    def _grow_batch_size(self, current: int, max_allowed: int, cfg: ExtractionConfig) -> int:
        if current >= max_allowed:
            return current
        if cfg.runtime.batch_growth_step is not None:
            return min(max_allowed, current + cfg.runtime.batch_growth_step)
        return min(max_allowed, current * 2)

    def _maybe_release_max_batch_cap(
        self,
        cfg: ExtractionConfig,
        stats: ExtractionStats,
        adaptive_max_batch_size: int,
        configured_max_batch_size: int,
        max_cap_released: bool,
        max_cap_release_blocked: bool,
    ) -> tuple[int, bool]:
        if max_cap_released:
            return adaptive_max_batch_size, True
        if max_cap_release_blocked:
            return adaptive_max_batch_size, False
        release_after = cfg.runtime.release_to_max_after_sequences
        if release_after is None:
            return adaptive_max_batch_size, max_cap_released
        if stats.total_sequences_seen < release_after:
            return adaptive_max_batch_size, max_cap_released
        return configured_max_batch_size, True

    def _resolve_async_queue_batches(
        self,
        cfg: ExtractionConfig,
        d_model: int,
        selected_layer_count: int,
    ) -> int:
        if not cfg.runtime.async_write:
            return 1
        if cfg.runtime.async_queue_max_batches is not None:
            return cfg.runtime.async_queue_max_batches

        total_mem_bytes = self._get_total_memory_bytes()
        memory_budget_bytes = min(1 * 1024**3, max(256 * 1024**2, int(total_mem_bytes * 0.08)))
        max_batch = cfg.runtime.max_batch_size or cfg.runtime.batch_size
        approx_batch_bytes = max_batch * cfg.preprocess.max_length * selected_layer_count * d_model * 4
        if cfg.runtime.defer_token_index:
            approx_batch_bytes += max_batch * cfg.preprocess.max_length * 4
        approx_batch_bytes = max(1, approx_batch_bytes)
        auto_batches = memory_budget_bytes // approx_batch_bytes
        return max(1, min(4, int(auto_batches)))

    def _get_total_memory_bytes(self) -> int:
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and phys_pages > 0:
                return page_size * phys_pages
        except Exception:
            pass
        return 16 * 1024**3

    def _is_oom_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "out of memory" in message or "cuda out of memory" in message

    def _clear_cuda_cache(self, model_adapter: ModelAdapter) -> None:
        torch_mod = getattr(model_adapter, "_torch", None)
        if torch_mod is None:
            return
        cuda_mod = getattr(torch_mod, "cuda", None)
        if cuda_mod is None:
            return
        if cuda_mod.is_available():
            cuda_mod.empty_cache()

    def _validate_run_directory(self, artifact_root: Path, resume: bool) -> None:
        if resume:
            return
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError(
                f"Run directory already exists and is not empty: {artifact_root}. "
                "Use a new run_id or set runtime.resume=true."
            )

    def _validate_input_path(self, cfg: ExtractionConfig) -> None:
        input_path = Path(cfg.input.path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input sequence file does not exist: {input_path}")
        if not input_path.is_file():
            raise ValueError(f"Input path must be a file: {input_path}")

    def _read_progress(self, artifact_root: Path) -> dict:
        progress_path = artifact_root / "_progress.json"
        if not progress_path.exists():
            return {}
        try:
            return json.loads(progress_path.read_text())
        except json.JSONDecodeError:
            return {}

    def _write_progress(self, artifact_root: Path, stats: ExtractionStats) -> None:
        artifact_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": utc_now_iso(),
            "total_sequences_seen": stats.total_sequences_seen,
            "total_sequences_kept": stats.total_sequences_kept,
            "total_sequences_skipped": stats.total_sequences_skipped,
            "total_tokens": stats.total_tokens,
            "total_rows_written": stats.total_rows_written,
        }
        tmp_path = artifact_root / "_progress.json.tmp"
        final_path = artifact_root / "_progress.json"
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(final_path)

    def _validate_output_root(self, cfg: ExtractionConfig) -> None:
        output_root = Path(cfg.runtime.output_root).resolve()
        blocked_roots = [
            Path("data/raw_sources").resolve(),
            Path("data/curated_sequences").resolve(),
        ]
        for blocked in blocked_roots:
            try:
                output_root.relative_to(blocked)
            except ValueError:
                continue
            raise ValueError(
                "runtime.output_root must not be inside input data directories. "
                f"Got {output_root}, which is under blocked path {blocked}."
            )

    def _make_run_id(self) -> str:
        return f"extract_{utc_now_iso().replace(':', '-').replace('+00:00', 'Z')}_{uuid4().hex[:8]}"

    def _write_manifest(
        self,
        cfg: ExtractionConfig,
        desc: ModelDescription,
        run_id: str,
        selected_layers: list[int],
        stats: ExtractionStats,
        layers_payload: dict[str, list[dict]],
        artifact_root: Path,
    ) -> None:
        manifest = RunManifest(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            created_at=utc_now_iso(),
            input_path=cfg.input.path,
            input_format=cfg.input.format,
            model={
                "model_id": desc.model_id,
                "tokenizer_id": desc.tokenizer_id,
                "revision": desc.revision,
                "num_transformer_layers": desc.num_transformer_layers,
                "d_model": desc.d_model,
            },
            layer_selection={
                "selected_transformer_layers": selected_layers,
            },
            preprocess=asdict(cfg.preprocess),
            runtime=asdict(cfg.runtime),
            stats=stats.to_dict(),
            layers=layers_payload,
        )
        tmp_path = artifact_root / "manifest.json.tmp"
        final_path = artifact_root / "manifest.json"
        tmp_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        tmp_path.replace(final_path)
