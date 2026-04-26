from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .config import ModelConfig
from .schemas import ModelDescription


class _EarlyStopForward(RuntimeError):
    """Internal control-flow exception used to stop forward after target layer."""


@dataclass(frozen=True)
class BatchExtraction:
    token_ids: list[list[int]]
    hidden_states_by_layer: dict[int, Any]


class ModelAdapter(ABC):
    @abstractmethod
    def describe(self) -> ModelDescription:
        raise NotImplementedError

    @abstractmethod
    def extract_batch(
        self,
        sequences: list[str],
        transformer_layers: list[int],
        max_length: int,
    ) -> BatchExtraction:
        raise NotImplementedError


class FakeModelAdapter(ModelAdapter):
    """Deterministic adapter for local tests without external model downloads."""

    def __init__(
        self,
        model_id: str = "fake/metagene-tiny",
        num_transformer_layers: int = 8,
        d_model: int = 16,
    ) -> None:
        self._desc = ModelDescription(
            model_id=model_id,
            tokenizer_id=f"{model_id}-tokenizer",
            revision=None,
            num_transformer_layers=num_transformer_layers,
            d_model=d_model,
        )
        self._tok = {
            "A": 10,
            "C": 11,
            "G": 12,
            "T": 13,
            "U": 14,
            "N": 15,
        }
        self._bos = 1
        self._unk = 2

    def describe(self) -> ModelDescription:
        return self._desc

    def _tokenize(self, seq: str, max_length: int) -> list[int]:
        token_ids = [self._bos]
        for ch in seq:
            token_ids.append(self._tok.get(ch, self._unk))
            if len(token_ids) >= max_length:
                break
        return token_ids

    def _vector(self, token_id: int, layer: int, token_index: int) -> list[float]:
        d_model = self._desc.d_model
        base = token_id + (layer * 17) + token_index
        return [((base + dim) % 257) / 257.0 for dim in range(d_model)]

    def extract_batch(
        self,
        sequences: list[str],
        transformer_layers: list[int],
        max_length: int,
    ) -> BatchExtraction:
        token_ids_batch = [self._tokenize(seq, max_length=max_length) for seq in sequences]
        hidden_states_by_layer: dict[int, Any] = {}
        for layer in transformer_layers:
            per_sequence: list[list[list[float]]] = []
            for seq_tokens in token_ids_batch:
                seq_vectors: list[list[float]] = []
                for token_index, token_id in enumerate(seq_tokens):
                    seq_vectors.append(self._vector(token_id, layer=layer, token_index=token_index))
                per_sequence.append(seq_vectors)
            hidden_states_by_layer[layer] = per_sequence
        return BatchExtraction(token_ids=token_ids_batch, hidden_states_by_layer=hidden_states_by_layer)


class TransformersModelAdapter(ModelAdapter):
    """Hugging Face adapter. Imports torch/transformers lazily."""

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._torch, self._transformers = self._import_dependencies()
        self._tokenizer = self._transformers.AutoTokenizer.from_pretrained(
            cfg.tokenizer_id or cfg.model_id,
            revision=cfg.revision,
            local_files_only=cfg.local_files_only,
            trust_remote_code=cfg.trust_remote_code,
        )
        model_dtype = self._resolve_dtype(cfg.dtype)
        model_kwargs = {
            "revision": cfg.revision,
            "local_files_only": cfg.local_files_only,
            "trust_remote_code": cfg.trust_remote_code,
            "torch_dtype": model_dtype,
            "low_cpu_mem_usage": True,
        }
        try:
            # The base model skips the language-model head/logits, which are unnecessary
            # when we only need residual stream hidden states.
            self._model = self._transformers.AutoModel.from_pretrained(cfg.model_id, **model_kwargs)
        except Exception:
            self._model = self._transformers.AutoModelForCausalLM.from_pretrained(
                cfg.model_id, **model_kwargs
            )
        self._device = self._resolve_device(cfg.device)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._desc = ModelDescription(
            model_id=cfg.model_id,
            tokenizer_id=cfg.tokenizer_id or cfg.model_id,
            revision=cfg.revision,
            num_transformer_layers=int(getattr(self._model.config, "num_hidden_layers")),
            d_model=int(getattr(self._model.config, "hidden_size")),
        )

    def _import_dependencies(self) -> tuple[Any, Any]:
        try:
            import torch
            import transformers
        except Exception as exc:  # pragma: no cover - environment-specific.
            raise RuntimeError(
                "TransformersModelAdapter requires torch and transformers. "
                "Install them first or use FakeModelAdapter for tests."
            ) from exc
        return torch, transformers

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self._torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_dtype(self, requested: str):
        if requested == "auto":
            if self._torch.cuda.is_available():
                return self._torch.bfloat16
            return self._torch.float32
        requested = requested.lower()
        if requested in {"bf16", "bfloat16"}:
            return self._torch.bfloat16
        if requested in {"fp16", "float16"}:
            return self._torch.float16
        if requested in {"fp32", "float32"}:
            return self._torch.float32
        raise ValueError(f"Unsupported dtype value: {requested}")

    def describe(self) -> ModelDescription:
        return self._desc

    def _resolve_transformer_blocks(self) -> list[Any]:
        candidate_paths = [
            ("model", "layers"),
            ("model", "decoder", "layers"),
            ("transformer", "h"),
            ("transformer", "blocks"),
            ("gpt_neox", "layers"),
            ("decoder", "layers"),
            ("layers",),
        ]
        for path in candidate_paths:
            node = self._model
            try:
                for name in path:
                    node = getattr(node, name)
            except AttributeError:
                continue
            try:
                blocks = list(node)
            except TypeError:
                continue
            if blocks:
                return blocks
        raise RuntimeError(
            "Unable to resolve transformer block modules for selected-layer extraction. "
            "Model architecture is unsupported by this adapter."
        )

    def _extract_block_tensor(self, block_output: Any):
        if isinstance(block_output, (tuple, list)):
            if not block_output:
                raise RuntimeError("Transformer block output tuple was empty.")
            block_output = block_output[0]
        if not hasattr(block_output, "shape"):
            raise RuntimeError("Transformer block output is not a tensor.")
        if len(block_output.shape) != 3:
            raise RuntimeError(
                f"Expected transformer block output shape [batch, seq, hidden], got {tuple(block_output.shape)}"
            )
        return block_output

    def extract_batch(
        self,
        sequences: list[str],
        transformer_layers: list[int],
        max_length: int,
    ) -> BatchExtraction:
        encoded = self._tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        blocks = self._resolve_transformer_blocks()
        max_layer = max(transformer_layers)
        if max_layer > len(blocks):
            raise ValueError(
                f"Requested layer {max_layer}, but resolved transformer has only {len(blocks)} blocks."
            )

        captured_layers: dict[int, Any] = {}
        hook_handles = []

        stop_after_target = bool(self._cfg.stop_after_last_requested_layer)

        def make_hook(layer_number: int):
            def _hook(_module, _inputs, output):
                tensor = self._extract_block_tensor(output)
                captured_layers[layer_number] = tensor.detach().to(dtype=self._torch.float32, device="cpu")
                if stop_after_target and layer_number == max_layer:
                    raise _EarlyStopForward()

            return _hook

        for layer in transformer_layers:
            handle = blocks[layer - 1].register_forward_hook(make_hook(layer))
            hook_handles.append(handle)

        with self._torch.inference_mode():
            try:
                self._model(
                    **encoded,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
            except _EarlyStopForward:
                # Expected fast path: we have all requested layers and intentionally
                # stop before executing unnecessary deeper blocks.
                pass
            finally:
                for handle in hook_handles:
                    handle.remove()

        input_ids = encoded["input_ids"].detach().cpu()
        attention_mask = encoded["attention_mask"].detach().cpu()

        token_ids_batch: list[list[int]] = []
        for batch_idx in range(input_ids.shape[0]):
            valid_tokens = int(attention_mask[batch_idx].sum().item())
            token_ids_batch.append(input_ids[batch_idx, :valid_tokens].tolist())

        hidden_states_by_layer: dict[int, Any] = {}
        for layer in transformer_layers:
            layer_tensor = captured_layers.get(layer)
            if layer_tensor is None:
                raise RuntimeError(
                    f"Missing captured hidden states for layer {layer}. "
                    "Forward hook did not fire as expected."
                )
            hidden_states_by_layer[layer] = layer_tensor

        return BatchExtraction(
            token_ids=token_ids_batch,
            hidden_states_by_layer=hidden_states_by_layer,
        )
