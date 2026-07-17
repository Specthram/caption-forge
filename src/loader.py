"""Model loading and unloading for the three caption backends.

Supports local safetensors (transformers), local GGUF + mmproj (llama-cpp
vision) and local GGUF without mmproj (llama-cpp text-only). The device is
enforced from settings ("cuda" by default): there is no silent CPU fallback —
if "cuda" is selected but unavailable, loading raises.
"""

# This module is a server-side singleton: the loaded model and its metadata
# live in mutable module globals, so the loader functions legitimately use
# ``global`` and the globals are intentionally lowercase (not constants).
# pylint: disable=global-statement,invalid-name

import os

# torch (Intel OpenMP) and llama-cpp (LLVM OpenMP) each link an OpenMP
# runtime; loading both aborts with "OMP Error #15" unless duplicates
# are allowed.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# pylint: disable=wrong-import-position
import gc
import importlib.util
import shutil
from pathlib import Path

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from src import hf_assets
from src.constants import CONFIG_CACHE_DIR
from src.settings import get_device, get_gguf_n_ctx

try:
    from huggingface_hub import snapshot_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
# pylint: enable=wrong-import-position


def _bootstrap_cuda_runtime():
    """Copy CUDA runtime DLLs next to ``ggml-cuda.dll`` for llama-cpp.

    The CUDA-enabled llama-cpp wheel ships ``ggml-cuda.dll`` but not the CUDA
    12.x runtime DLLs it depends on; torch (cu128) bundles them. Idempotent and
    runs before importing llama-cpp.
    """
    spec = importlib.util.find_spec("llama_cpp")
    if spec is None or spec.origin is None:
        return
    llama_lib = Path(spec.origin).parent / "lib"
    torch_lib = Path(torch.__file__).parent / "lib"
    if not (llama_lib.exists() and (llama_lib / "ggml-cuda.dll").exists()):
        return
    for dll in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"):
        src, dst = torch_lib / dll, llama_lib / dll
        if src.exists() and not dst.exists():
            try:
                shutil.copy2(src, dst)
            except OSError:
                pass


try:
    _bootstrap_cuda_runtime()
    from llama_cpp import Llama
    from llama_cpp import llama_chat_format as _lcf

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Global (server-singleton) model state.
model = None
processor = None
current_model_type = None  # family: qwen3 / gemma3 / llava / mistral3 / …
current_format = None  # "transformers" | "gguf-text" | "gguf-vision"
last_status = "Ready."  # last status line, survives page reloads
loaded_name = None  # filename of the currently loaded model

# GGUF vision chat handlers, keyed by family. Resolved via getattr so missing
# handlers (older llama-cpp) simply drop out instead of breaking the import.
if LLAMA_CPP_AVAILABLE:
    _HANDLER_NAMES = {
        "qwen3": "Qwen3VLChatHandler",
        "qwen2.5": "Qwen25VLChatHandler",
        "gemma3": "Gemma3ChatHandler",
        "gemma3n": "Gemma3ChatHandler",
        "gemma4": "Gemma4ChatHandler",
        "llava": "Llava16ChatHandler",
        # Mistral Small 3.2 / Pixtral and Qwen3.6: the generic mtmd handler
        # reads the model's chat template straight from the GGUF metadata.
        "mistral3": "GenericMTMDChatHandler",
        "qwen3.6": "GenericMTMDChatHandler",
    }
    _GGUF_VISION_HANDLERS = {
        fam: getattr(_lcf, name)
        for fam, name in _HANDLER_NAMES.items()
        if hasattr(_lcf, name)
    }
else:
    _GGUF_VISION_HANDLERS = {}


def is_model_loaded() -> bool:
    """Return whether a model is currently loaded."""
    return model is not None


# --- Device enforcement ---


def _resolve_device() -> str:
    """Return the configured device, raising if 'cuda' is unavailable."""
    device = get_device()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Device is set to 'cuda' but PyTorch sees no CUDA GPU. "
            "Install a CUDA build of torch, or switch the device "
            "to 'cpu' in Settings."
        )
    return device


def _has_cuda_backend() -> bool:
    """Return whether the installed llama-cpp ships the CUDA backend DLL."""
    spec = importlib.util.find_spec("llama_cpp")
    if spec is None or spec.origin is None:
        return False
    return (Path(spec.origin).parent / "lib" / "ggml-cuda.dll").exists()


def _check_gguf_gpu(device: str):
    """Raise when cuda is requested but the CUDA backend DLL is absent.

    ``llama_supports_gpu_offload()`` is unreliable on recent builds (returns
    False even when CUDA loads), so we check for the backend DLL instead.
    ``torch.cuda`` availability is already enforced by ``_resolve_device``.
    """
    if device == "cuda" and not _has_cuda_backend():
        raise RuntimeError(
            "GGUF on GPU requested but this llama-cpp build has no CUDA "
            "backend (ggml-cuda.dll missing). Install a CUDA wheel, or "
            "switch device to 'cpu'."
        )


# --- Config source (local first, download fallback; never weights) ---

# Files never copied into the per-weight load dir: the shard index (it points
# at sharded weights we don't have — the single local file is symlinked in as
# ``model.safetensors`` instead) and repo bookkeeping.
_ASSEMBLY_SKIP = frozenset(
    {"model.safetensors.index.json", ".gitattributes", "README.md"}
)


def _ensure_hf_config(hf_config_id: str) -> Path:
    """Return the local HF config folder, downloading metadata if missing.

    The user may pre-populate ``hf_config/<repo>/`` to avoid any download; when
    required files are missing, the config/processor (never the weights) is
    fetched into that same folder so it is self-healed for next time.
    """
    config_src = hf_assets.hf_config_dir(hf_config_id)
    if not hf_assets.missing_hf_files(hf_config_id):
        return config_src
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface-hub is required to fetch the model config "
            f"({hf_config_id!r}); install it, or place the config + "
            f"processor files (no weights) in '{config_src}'."
        )
    config_src.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hf_config_id,
        local_dir=str(config_src),
        ignore_patterns=list(hf_assets.WEIGHT_IGNORE_PATTERNS),
    )
    return config_src


def _prepare_model_dir(hf_config_id: str, local_weights: Path) -> Path:
    """Assemble a load dir: the HF config/processor + the local weights.

    The config/processor come from the user's ``hf_config/<repo>/`` folder
    (fetched there on demand if absent); the weights are symlinked in as a
    single ``model.safetensors``.
    """
    config_src = _ensure_hf_config(hf_config_id)

    safe_id = hf_config_id.replace("/", "--")
    model_dir = CONFIG_CACHE_DIR / f"{safe_id}--{local_weights.stem}"
    model_dir.mkdir(parents=True, exist_ok=True)

    for item in config_src.iterdir():
        if not item.is_file() or item.name in _ASSEMBLY_SKIP:
            continue
        dst = model_dir / item.name
        if not dst.exists():
            shutil.copy2(item, dst)

    weights_link = model_dir / "model.safetensors"
    if weights_link.exists() or weights_link.is_symlink():
        weights_link.unlink()
    weights_link.symlink_to(local_weights.resolve())
    return model_dir


# --- safetensors (transformers) ---


def _load_local(local_path: Path, hf_config_id: str, model_type: str):
    """Load a local safetensors checkpoint via transformers."""
    global model, processor, current_model_type, current_format

    device = _resolve_device()
    # model_dir = HF config/processor + a 'model.safetensors' symlink to the
    # local weights, so from_pretrained loads them with the HF config.
    model_dir = _prepare_model_dir(hf_config_id, local_path)

    # local_files_only: the config/processor are guaranteed present (assembled
    # by _prepare_model_dir), so transformers must not reach the network here.
    use_fast = model_type == "llava"
    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        use_fast=use_fast,
        local_files_only=True,
    )

    # AutoModelForImageTextToText resolves the concrete architecture from the
    # config, so every image-text-to-text family (and any future one the
    # installed transformers knows, or that ships its own modeling code) loads
    # through this single call — no per-family class table to maintain.
    model, info = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        output_loading_info=True,
        local_files_only=True,
        trust_remote_code=True,
    )

    # ComfyUI text-encoder safetensors use a different key layout and often
    # lack the vision tower, so almost nothing matches and the model loads
    # with random weights. Fail loudly.
    n_missing = len(info.get("missing_keys", []))
    n_params = sum(1 for _ in model.state_dict())
    if n_params and n_missing / n_params > 0.5:
        del model
        raise RuntimeError(
            "This safetensors file is not a compatible Hugging Face "
            f"checkpoint ({n_missing}/{n_params} weights missing — "
            "likely a ComfyUI text-encoder or a text-only/quantized "
            "file). Use the GGUF version of this model for captioning "
            "instead."
        )

    model = model.to(device).eval()
    current_model_type = model_type
    current_format = "transformers"


# --- GGUF (llama-cpp) ---


def _load_gguf_vision(
    local_path: Path,
    mmproj_path: Path,
    model_type: str,
    n_ctx: int | None = None,
):
    """Load a local GGUF vision model (weights + mmproj) via llama-cpp.

    ``n_ctx`` overrides the global context-size setting (model profiles);
    ``None`` keeps the Settings value.
    """
    global model, processor, current_model_type, current_format

    if not LLAMA_CPP_AVAILABLE:
        raise ImportError(
            "llama-cpp-python required: pip install llama-cpp-python"
        )

    device = _resolve_device()
    _check_gguf_gpu(device)

    if mmproj_path is None:
        raise ValueError(
            "No mmproj (vision projector) GGUF found next to this model. "
            "Vision GGUF needs an 'mmproj-*.gguf' file in the same folder."
        )

    handler_cls = _GGUF_VISION_HANDLERS.get(model_type)
    if handler_cls is None:
        raise ValueError(
            f"GGUF vision is not supported for '{model_type}' in this "
            "llama-cpp version (no chat handler available). Use the "
            ".safetensors version instead."
        )
    # The generic mtmd handler (Mistral 3.2 / Pixtral) takes an explicit
    # ``chat_format`` (None → read the template from the GGUF metadata) and a
    # required ``mmproj_path``; the family handlers fix their own template and
    # accept the legacy ``clip_model_path`` alias instead.
    if handler_cls.__name__ == "GenericMTMDChatHandler":
        chat_handler = handler_cls(
            chat_format=None,
            mmproj_path=str(mmproj_path),
            verbose=False,
        )
    else:
        chat_handler = handler_cls(
            clip_model_path=str(mmproj_path), verbose=False
        )

    try:
        model = Llama(
            model_path=str(local_path),
            chat_handler=chat_handler,
            n_ctx=n_ctx if n_ctx is not None else get_gguf_n_ctx(),
            n_gpu_layers=-1 if device == "cuda" else 0,
            n_batch=512,
            verbose=False,
            use_mmap=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"llama-cpp could not load this GGUF ({exc}). The bundled "
            "llama.cpp may not support this model's architecture yet "
            "(e.g. 'qwen3vl' needs a newer llama-cpp-python). Use the "
            ".safetensors version of this model instead."
        ) from exc
    processor = None
    current_model_type = model_type
    current_format = "gguf-vision"


def _load_gguf_text(
    weights_path: str,
    processor_repo: str,
    model_type: str,
    n_ctx: int | None = None,
):
    """Load a local text-only GGUF model via llama-cpp.

    ``n_ctx`` overrides the global context-size setting (model profiles).
    """
    global model, processor, current_model_type, current_format

    if not LLAMA_CPP_AVAILABLE:
        raise ImportError(
            "llama-cpp-python required: pip install llama-cpp-python"
        )

    device = _resolve_device()
    _check_gguf_gpu(device)

    model = Llama(
        model_path=weights_path,
        n_ctx=n_ctx if n_ctx is not None else get_gguf_n_ctx(),
        n_gpu_layers=-1 if device == "cuda" else 0,
        n_batch=512,
        verbose=False,
        use_mmap=True,
    )

    # Best-effort, offline-only processor (just for its chat template). Never
    # downloads: if the config is not present locally, the captioner falls back
    # to a built-in template.
    processor = _load_local_processor(processor_repo)

    # Preserve the detected family so the UI shows this model's own prompts and
    # generation defaults — not a hardcoded one. Its chat template comes from
    # the processor above (built-in fallback when absent).
    current_model_type = model_type
    current_format = "gguf-text"


def _load_local_processor(hf_config_id: str):
    """Load a processor from local HF config files only, or None.

    Used by the text-only GGUF path for its chat template. Never reaches the
    network: when the config is absent locally, returns None. A profile with
    no known HF repo (text-only fallback) passes None and gets the built-in
    template.
    """
    if not hf_config_id or hf_assets.missing_hf_files(hf_config_id):
        return None
    config_src = str(hf_assets.hf_config_dir(hf_config_id))
    try:
        return AutoProcessor.from_pretrained(
            config_src,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        try:
            return AutoTokenizer.from_pretrained(
                config_src, trust_remote_code=True, local_files_only=True
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return None


# --- Public API ---


def _track(gen):
    """Forward (status, is_loaded) pairs, remembering the last status."""
    global last_status
    for status, loaded in gen:
        last_status = status
        yield status, loaded


def load_model(model_cfg: dict):
    """Load a model, yielding (status, is_loaded) and tracking last_status."""
    yield from _track(_load_model(model_cfg))


def unload_model():
    """Unload the model, yielding (status, is_loaded) and tracking status."""
    yield from _track(_unload_model())


def _load_model(model_cfg: dict):
    """Load a local model from ``model_cfg``, yielding (status, is_loaded).

    ``model_cfg`` keys (from :func:`scanner.scan_local_models`): ``format``
    (``"safetensors"`` | ``"gguf"``), ``type`` (model family), ``local_path``
    (Path to the weights), ``mmproj_path`` (Path | None, gguf vision projector)
    and ``hf_config`` (HF repo id for config/processor metadata only).
    """
    global model, processor, current_model_type, current_format, loaded_name

    if model is not None:
        yield "✅ Model already loaded.", True
        return

    fmt = model_cfg["format"]
    model_type = model_cfg["type"]
    name = Path(model_cfg["local_path"]).name

    yield (
        f"⏳ Loading {model_cfg['local_path'].name!r} on device "
        f"'{get_device()}'...\n",
        False,
    )

    try:
        if fmt == "safetensors":
            yield (
                f"📦 Fetching config metadata for "
                f"{model_cfg['hf_config']!r} (first time only)...\n",
                False,
            )
            _load_local(
                Path(model_cfg["local_path"]),
                model_cfg["hf_config"],
                model_type,
            )
            loaded_name = name
            yield (
                "✅ Model loaded (local safetensors, vision enabled).\n",
                True,
            )

        elif fmt == "gguf":
            if model_cfg.get("mmproj_path"):
                _load_gguf_vision(
                    Path(model_cfg["local_path"]),
                    Path(model_cfg["mmproj_path"]),
                    model_type,
                    n_ctx=model_cfg.get("n_ctx"),
                )
                loaded_name = name
                yield (
                    "✅ Model loaded (local GGUF, vision enabled "
                    "via mmproj).\n",
                    True,
                )
            else:
                _load_gguf_text(
                    str(model_cfg["local_path"]),
                    model_cfg["hf_config"],
                    model_type,
                    n_ctx=model_cfg.get("n_ctx"),
                )
                loaded_name = name
                yield (
                    "✅ Model loaded (local GGUF, ⚠️ text-only — no "
                    "mmproj found, image is NOT used).\n",
                    True,
                )

        else:
            raise ValueError(f"Unknown model format: {fmt!r}")

    except Exception as exc:  # pylint: disable=broad-exception-caught
        model = None
        processor = None
        current_model_type = None
        current_format = None
        loaded_name = None
        yield f"❌ Error: {exc}\n", False


def _release_model():
    """Drop every reference to the loaded model and reclaim GPU memory."""
    global model, processor, current_model_type, current_format, loaded_name

    if hasattr(model, "close"):
        model.close()
    del model
    del processor
    model = None
    processor = None
    current_model_type = None
    current_format = None
    loaded_name = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def _unload_model():
    """Unload the model, yielding (status, is_loaded) updates."""
    if model is None:
        yield "⚠️ No model loaded.", False
        return

    yield "♻️ Unloading model...\n", True
    try:
        _release_model()
        yield "✅ Model unloaded.\n", False
    except Exception as exc:  # pylint: disable=broad-exception-caught
        yield f"❌ Error unloading: {exc}\n", True
