from __future__ import annotations

import json
import os
import queue
import re
import threading
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO


@dataclass(frozen=True)
class ShardDescriptor:
    shard_id: int
    rows: int
    data_file: str
    index_file: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "rows": self.rows,
            "data_file": self.data_file,
            "index_file": self.index_file,
        }


class _LayerWriter:
    SHARD_RE = re.compile(r"^shard_(\d{5})\.(f32|jsonl)$")

    def __init__(
        self,
        root: Path,
        layer: int,
        d_model: int,
        max_rows_per_shard: int,
        resume: bool = False,
        defer_token_index: bool = False,
    ) -> None:
        self._root = root
        self._layer = layer
        self._d_model = d_model
        self._max_rows_per_shard = max_rows_per_shard
        self._defer_token_index = defer_token_index
        self._layer_dir = self._root / f"layer_{layer:02d}"
        self._layer_dir.mkdir(parents=True, exist_ok=True)

        self._existing_shards: list[ShardDescriptor] = []
        self._current_shard_id = -1
        self._rows_in_shard = 0
        self._total_rows = 0
        self._data_fp: Any | None = None
        self._index_fp: TextIO | None = None
        self._shards: list[ShardDescriptor] = []
        if resume:
            self._existing_shards = self._scan_existing_shards()
            if self._existing_shards:
                self._current_shard_id = max(shard.shard_id for shard in self._existing_shards)
                self._total_rows = sum(shard.rows for shard in self._existing_shards)
        self._open_next_shard()

    @property
    def existing_rows(self) -> int:
        return sum(shard.rows for shard in self._existing_shards)

    def _count_valid_json_lines(self, path: Path) -> tuple[int, int]:
        valid_lines = 0
        valid_offset = 0
        with path.open("rb") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                next_offset = handle.tell()
                try:
                    json.loads(line.decode("utf-8"))
                except Exception:
                    break
                valid_lines += 1
                valid_offset = next_offset
        return valid_lines, valid_offset

    def _truncate_json_lines(self, path: Path, keep_lines: int) -> None:
        line_count = 0
        keep_offset = 0
        with path.open("rb") as handle:
            while line_count < keep_lines:
                line = handle.readline()
                if not line:
                    break
                keep_offset = handle.tell()
                line_count += 1
        with path.open("r+b") as handle:
            handle.truncate(keep_offset)

    def _scan_existing_shards(self) -> list[ShardDescriptor]:
        shard_ids: set[int] = set()
        for path in self._layer_dir.glob("shard_*.*"):
            match = self.SHARD_RE.match(path.name)
            if match is None:
                continue
            shard_ids.add(int(match.group(1)))

        descriptors: list[ShardDescriptor] = []
        row_size_bytes = self._d_model * 4
        for shard_id in sorted(shard_ids):
            data_path = self._layer_dir / f"shard_{shard_id:05d}.f32"
            index_path = self._layer_dir / f"shard_{shard_id:05d}.jsonl"
            if not data_path.exists():
                continue

            if self._defer_token_index:
                data_rows = data_path.stat().st_size // row_size_bytes
                if data_rows <= 0:
                    continue
                expected_data_size = data_rows * row_size_bytes
                if data_path.stat().st_size != expected_data_size:
                    with data_path.open("r+b") as data_fp:
                        data_fp.truncate(expected_data_size)
                descriptors.append(
                    ShardDescriptor(
                        shard_id=shard_id,
                        rows=int(data_rows),
                        data_file=f"layer_{self._layer:02d}/shard_{shard_id:05d}.f32",
                        index_file=f"layer_{self._layer:02d}/shard_{shard_id:05d}.jsonl",
                    )
                )
                continue

            if not index_path.exists():
                continue

            data_rows = data_path.stat().st_size // row_size_bytes
            json_rows, json_valid_offset = self._count_valid_json_lines(index_path)
            valid_rows = int(min(data_rows, json_rows))
            if valid_rows <= 0:
                continue

            expected_data_size = valid_rows * row_size_bytes
            if data_path.stat().st_size != expected_data_size:
                with data_path.open("r+b") as data_fp:
                    data_fp.truncate(expected_data_size)

            if json_rows != valid_rows:
                self._truncate_json_lines(index_path, keep_lines=valid_rows)
            elif json_valid_offset != index_path.stat().st_size:
                with index_path.open("r+b") as index_fp:
                    index_fp.truncate(json_valid_offset)

            descriptors.append(
                ShardDescriptor(
                    shard_id=shard_id,
                    rows=valid_rows,
                    data_file=f"layer_{self._layer:02d}/shard_{shard_id:05d}.f32",
                    index_file=f"layer_{self._layer:02d}/shard_{shard_id:05d}.jsonl",
                )
            )
        return descriptors

    def _close_current_shard(self) -> None:
        if self._data_fp is None or self._current_shard_id < 0:
            return
        self._data_fp.flush()
        os.fsync(self._data_fp.fileno())
        if self._index_fp is not None:
            self._index_fp.flush()
            os.fsync(self._index_fp.fileno())
        self._data_fp.close()
        if self._index_fp is not None:
            self._index_fp.close()
        if self._rows_in_shard > 0:
            self._shards.append(
                ShardDescriptor(
                    shard_id=self._current_shard_id,
                    rows=self._rows_in_shard,
                    data_file=f"layer_{self._layer:02d}/shard_{self._current_shard_id:05d}.f32",
                    index_file=f"layer_{self._layer:02d}/shard_{self._current_shard_id:05d}.jsonl",
                )
            )
        self._data_fp = None
        self._index_fp = None

    def _open_next_shard(self) -> None:
        self._close_current_shard()
        self._current_shard_id += 1
        self._rows_in_shard = 0
        data_path = self._layer_dir / f"shard_{self._current_shard_id:05d}.f32"
        index_path = self._layer_dir / f"shard_{self._current_shard_id:05d}.jsonl"
        self._data_fp = data_path.open("wb")
        self._index_fp = None
        if not self._defer_token_index:
            self._index_fp = index_path.open("w", encoding="utf-8")

    def _write_matrix(self, matrix: Any, rows: int) -> None:
        assert self._data_fp is not None
        if hasattr(matrix, "detach"):
            tensor = matrix.detach()
            matrix = tensor.to(dtype=tensor.new_empty(()).float().dtype, device="cpu")
            matrix = matrix.contiguous().numpy()
        if hasattr(matrix, "astype") and hasattr(matrix, "shape"):
            if len(matrix.shape) != 2:
                raise ValueError(f"Layer {self._layer} expected a 2D matrix, got shape {matrix.shape}")
            if int(matrix.shape[0]) != rows or int(matrix.shape[1]) != self._d_model:
                raise ValueError(
                    f"Layer {self._layer} expected matrix shape ({rows}, {self._d_model}), "
                    f"got {tuple(matrix.shape)}"
                )
            matrix = matrix.astype("float32", copy=False)
            self._data_fp.write(matrix.tobytes(order="C"))
            return

        flat = array("f")
        for vector in matrix:
            if len(vector) != self._d_model:
                raise ValueError(
                    f"Layer {self._layer} expected vector length {self._d_model}, got {len(vector)}"
                )
            flat.extend(vector)
        if len(flat) != rows * self._d_model:
            raise ValueError(
                f"Layer {self._layer} expected {rows * self._d_model} floats, got {len(flat)}"
            )
        flat.tofile(self._data_fp)

    def append(self, vector: Sequence[float], metadata: dict[str, Any]) -> None:
        if len(vector) != self._d_model:
            raise ValueError(
                f"Layer {self._layer} expected vector length {self._d_model}, got {len(vector)}"
            )
        assert self._data_fp is not None and self._index_fp is not None
        array("f", vector).tofile(self._data_fp)
        record = dict(metadata)
        record["row_in_shard"] = self._rows_in_shard
        record["row_global"] = self._total_rows
        self._index_fp.write(json.dumps(record, sort_keys=True) + "\n")
        self._rows_in_shard += 1
        self._total_rows += 1

        if self._rows_in_shard >= self._max_rows_per_shard:
            self._open_next_shard()

    def append_many(
        self,
        matrix: Any,
        sequence_id: str,
        token_ids: Sequence[int],
    ) -> None:
        total_rows = len(token_ids)
        start = 0
        while start < total_rows:
            capacity = self._max_rows_per_shard - self._rows_in_shard
            stop = min(total_rows, start + capacity)
            chunk_rows = stop - start
            self._write_matrix(matrix[start:stop], rows=chunk_rows)
            if not self._defer_token_index:
                assert self._index_fp is not None
                lines = []
                for token_index in range(start, stop):
                    record = {
                        "layer": self._layer,
                        "row_global": self._total_rows + (token_index - start),
                        "row_in_shard": self._rows_in_shard + (token_index - start),
                        "sequence_id": sequence_id,
                        "token_id": int(token_ids[token_index]),
                        "token_index": token_index,
                    }
                    lines.append(json.dumps(record, separators=(",", ":")))
                self._index_fp.write("\n".join(lines) + "\n")
            self._rows_in_shard += chunk_rows
            self._total_rows += chunk_rows
            start = stop
            if self._rows_in_shard >= self._max_rows_per_shard:
                self._open_next_shard()

    def flush(self) -> None:
        if self._data_fp is not None:
            self._data_fp.flush()
            os.fsync(self._data_fp.fileno())
        if self._index_fp is not None:
            self._index_fp.flush()
            os.fsync(self._index_fp.fileno())

    def finalize(self) -> dict[str, Any]:
        self._close_current_shard()
        return {
            "rows": self._total_rows,
            "shards": [
                descriptor.to_dict()
                for descriptor in (self._existing_shards + self._shards)
                if descriptor.rows > 0
            ],
        }


class ActivationStore:
    def __init__(
        self,
        artifact_root: Path,
        selected_layers: list[int],
        d_model: int,
        max_rows_per_shard: int,
        resume: bool = False,
        defer_token_index: bool = False,
        async_write: bool = False,
        async_queue_max_batches: int = 1,
    ) -> None:
        self.artifact_root = artifact_root
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._d_model = d_model
        self._selected_layers = list(selected_layers)
        self._defer_token_index = defer_token_index
        self._sequences_path = self.artifact_root / "sequences.jsonl"
        self._index_state_path = self.artifact_root / "_index_state.json"
        self.completed_sequence_ids: set[str] = set()
        self.existing_sequences_kept = 0
        self.existing_tokens = 0
        self._async_write = async_write
        self._async_queue_max_batches = max(1, async_queue_max_batches)
        self._write_queue: queue.Queue | None = None
        self._writer_thread: threading.Thread | None = None
        self._writer_error: Exception | None = None
        if resume and self._sequences_path.exists():
            self.existing_sequences_kept, self.existing_tokens, self.completed_sequence_ids = (
                self._load_existing_sequences()
            )

        self._writers = {
            layer: _LayerWriter(
                root=self.artifact_root / "activations",
                layer=layer,
                d_model=d_model,
                max_rows_per_shard=max_rows_per_shard,
                resume=resume,
                defer_token_index=defer_token_index,
            )
            for layer in selected_layers
        }
        self.existing_rows_written = sum(writer.existing_rows for writer in self._writers.values())
        mode = "a" if resume and self._sequences_path.exists() else "w"
        self._sequences_fp = self._sequences_path.open(mode, encoding="utf-8")
        if self._async_write:
            self._write_queue = queue.Queue(maxsize=self._async_queue_max_batches)
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="metageniuses-activation-writer",
                daemon=True,
            )
            self._writer_thread.start()

    def _load_existing_sequences(self) -> tuple[int, int, set[str]]:
        valid_count = 0
        total_tokens = 0
        ids: set[str] = set()
        valid_offset = 0
        with self._sequences_path.open("rb") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                next_offset = handle.tell()
                try:
                    row = json.loads(line.decode("utf-8"))
                except Exception:
                    break
                valid_count += 1
                total_tokens += int(row.get("token_count", 0))
                seq_id = row.get("sequence_id")
                if seq_id is not None:
                    ids.add(str(seq_id))
                valid_offset = next_offset
        if self._sequences_path.stat().st_size != valid_offset:
            with self._sequences_path.open("r+b") as handle:
                handle.truncate(valid_offset)
        return valid_count, total_tokens, ids

    def append_sequence(self, row: dict[str, Any]) -> None:
        self._sequences_fp.write(json.dumps(row, separators=(",", ":")) + "\n")
        seq_id = row.get("sequence_id")
        if seq_id is not None and not self._async_write:
            self.completed_sequence_ids.add(str(seq_id))

    def append_activation(self, layer: int, vector: list[float], row: dict[str, Any]) -> None:
        self._writers[layer].append(vector=vector, metadata=row)

    def append_sequence_activations(
        self,
        layer: int,
        matrix: Any,
        sequence_id: str,
        token_ids: Sequence[int],
    ) -> None:
        self._writers[layer].append_many(
            matrix=matrix,
            sequence_id=sequence_id,
            token_ids=token_ids,
        )

    def _slice_sequence_matrix(self, layer_values: Any, seq_idx: int, token_count: int) -> Any:
        if hasattr(layer_values, "shape") and len(layer_values.shape) == 3:
            return layer_values[seq_idx, :token_count, :]
        return layer_values[seq_idx][:token_count]

    def _write_batch_sync(
        self,
        sequence_rows: list[dict[str, Any]],
        token_ids_batch: list[list[int]],
        hidden_states_by_layer: dict[int, Any],
        selected_layers: list[int],
    ) -> None:
        if len(sequence_rows) != len(token_ids_batch):
            raise ValueError(
                f"Batch row/token count mismatch: {len(sequence_rows)} rows vs {len(token_ids_batch)} token lists."
            )
        for seq_idx, row in enumerate(sequence_rows):
            token_ids = token_ids_batch[seq_idx]
            self.append_sequence(row)
            for layer in selected_layers:
                layer_values = hidden_states_by_layer.get(layer)
                if layer_values is None:
                    raise ValueError(f"Missing hidden states for layer {layer} in batch payload.")
                self.append_sequence_activations(
                    layer=layer,
                    matrix=self._slice_sequence_matrix(layer_values, seq_idx=seq_idx, token_count=len(token_ids)),
                    sequence_id=str(row["sequence_id"]),
                    token_ids=token_ids,
                )

    def _writer_loop(self) -> None:
        assert self._write_queue is not None
        while True:
            payload = self._write_queue.get()
            try:
                if payload is None:
                    return
                if self._writer_error is not None:
                    continue
                sequence_rows, token_ids_batch, hidden_states_by_layer, selected_layers = payload
                self._write_batch_sync(
                    sequence_rows=sequence_rows,
                    token_ids_batch=token_ids_batch,
                    hidden_states_by_layer=hidden_states_by_layer,
                    selected_layers=selected_layers,
                )
            except Exception as exc:
                self._writer_error = exc
            finally:
                self._write_queue.task_done()

    def _ensure_writer_healthy(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError("Asynchronous activation writer failed.") from self._writer_error

    def append_batch(
        self,
        sequence_rows: list[dict[str, Any]],
        token_ids_batch: list[list[int]],
        hidden_states_by_layer: dict[int, Any],
        selected_layers: list[int],
    ) -> None:
        if not self._async_write:
            self._write_batch_sync(
                sequence_rows=sequence_rows,
                token_ids_batch=token_ids_batch,
                hidden_states_by_layer=hidden_states_by_layer,
                selected_layers=selected_layers,
            )
            return
        for row in sequence_rows:
            seq_id = row.get("sequence_id")
            if seq_id is not None:
                self.completed_sequence_ids.add(str(seq_id))
        self._ensure_writer_healthy()
        assert self._write_queue is not None
        self._write_queue.put(
            (sequence_rows, token_ids_batch, hidden_states_by_layer, list(selected_layers)),
            block=True,
        )
        self._ensure_writer_healthy()

    def flush(self) -> None:
        if self._async_write:
            assert self._write_queue is not None
            self._write_queue.join()
            self._ensure_writer_healthy()
        self._sequences_fp.flush()
        os.fsync(self._sequences_fp.fileno())
        for writer in self._writers.values():
            writer.flush()

    def _write_index_state(self, status: str, details: dict[str, Any] | None = None) -> None:
        payload = {"status": status}
        if details:
            payload.update(details)
        tmp_path = self._index_state_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(self._index_state_path)

    def _iter_sequence_token_metadata(self) -> Iterator[tuple[str, list[int]]]:
        with self._sequences_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                token_ids = payload.get("token_ids")
                if token_ids is None:
                    raise ValueError(
                        "Deferred token index build requires token_ids in sequences.jsonl rows."
                    )
                yield str(payload["sequence_id"]), [int(token_id) for token_id in token_ids]

    def _build_deferred_index_for_layer(self, layer: int, layer_payload: dict[str, Any]) -> None:
        shards = sorted(layer_payload["shards"], key=lambda item: int(item["shard_id"]))
        if not shards:
            return

        activation_root = self.artifact_root / "activations"
        expected_rows = int(layer_payload["rows"])
        row_global = 0
        shard_idx = 0
        row_in_shard = 0
        current_rows_in_shard = int(shards[shard_idx]["rows"])

        def open_tmp_fp(index: int) -> tuple[TextIO, Path, Path]:
            final_path = activation_root / shards[index]["index_file"]
            final_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
            return tmp_path.open("w", encoding="utf-8"), tmp_path, final_path

        fp, tmp_path, final_path = open_tmp_fp(shard_idx)

        def close_and_commit(handle: TextIO, tmp: Path, final: Path) -> None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            tmp.replace(final)

        for sequence_id, token_ids in self._iter_sequence_token_metadata():
            for token_index, token_id in enumerate(token_ids):
                while row_in_shard >= current_rows_in_shard:
                    close_and_commit(fp, tmp_path, final_path)
                    shard_idx += 1
                    if shard_idx >= len(shards):
                        raise ValueError(
                            f"Deferred index build overflow for layer {layer}: "
                            f"row_global={row_global}, expected_rows={expected_rows}"
                        )
                    row_in_shard = 0
                    current_rows_in_shard = int(shards[shard_idx]["rows"])
                    fp, tmp_path, final_path = open_tmp_fp(shard_idx)

                record = {
                    "layer": layer,
                    "row_global": row_global,
                    "row_in_shard": row_in_shard,
                    "sequence_id": sequence_id,
                    "token_id": int(token_id),
                    "token_index": token_index,
                }
                fp.write(json.dumps(record, separators=(",", ":")) + "\n")
                row_global += 1
                row_in_shard += 1

        if row_global != expected_rows:
            fp.close()
            raise ValueError(
                f"Deferred index rows mismatch for layer {layer}: "
                f"expected {expected_rows}, wrote {row_global}"
            )

        if row_in_shard != current_rows_in_shard:
            fp.close()
            raise ValueError(
                f"Deferred shard rows mismatch for layer {layer}, shard {shards[shard_idx]['shard_id']}: "
                f"expected {current_rows_in_shard}, wrote {row_in_shard}"
            )
        close_and_commit(fp, tmp_path, final_path)

        if shard_idx != len(shards) - 1:
            raise ValueError(
                f"Deferred index ended early for layer {layer}: "
                f"expected {len(shards)} shards, wrote {shard_idx + 1}"
            )

    def _build_deferred_indexes(self, layers_payload: dict[str, Any]) -> None:
        if not layers_payload:
            return
        self._write_index_state(status="building")
        for layer_key, layer_payload in layers_payload.items():
            self._build_deferred_index_for_layer(layer=int(layer_key), layer_payload=layer_payload)
        self._write_index_state(
            status="complete",
            details={"layers_indexed": sorted(int(layer_key) for layer_key in layers_payload.keys())},
        )

    def finalize(self) -> dict[str, Any]:
        if self._async_write:
            self.flush()
            assert self._write_queue is not None
            assert self._writer_thread is not None
            self._write_queue.put(None)
            self._write_queue.join()
            self._writer_thread.join(timeout=30)
            if self._writer_thread.is_alive():
                raise RuntimeError("Asynchronous activation writer did not shut down cleanly.")
            self._ensure_writer_healthy()
        self._sequences_fp.flush()
        os.fsync(self._sequences_fp.fileno())
        self._sequences_fp.close()
        layers_payload: dict[str, Any] = {}
        for layer, writer in self._writers.items():
            layers_payload[str(layer)] = writer.finalize()
        if self._defer_token_index:
            self._build_deferred_indexes(layers_payload=layers_payload)
        return layers_payload
