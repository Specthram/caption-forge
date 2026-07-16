"""Caption generation for images and videos across the model backends.

Dispatches on the loaded model's format/family (set by :mod:`src.loader`):
transformers (safetensors), GGUF vision (llama-cpp + mmproj) and GGUF text.
Reasoning ("thinking") output is stripped unless the caller asks to keep it.
"""

import base64
import gc
import io
import logging
import re
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from transformers.video_utils import VideoMetadata

from src import loader
from src.constants import GEMMA_MAX_IMAGE_SIZE
from src.settings import (
    estimate_video_frames,
    get_caption_image_size,
    get_model_max_new_tokens,
    get_video_fps,
    get_video_max_seconds,
    get_video_prompt,
    get_video_resolution,
)

logger = logging.getLogger(__name__)


class _TransformersNoiseFilter(logging.Filter):
    # pylint: disable=too-few-public-methods
    """Drop repetitive transformers deprecation warnings from the console.

    Recent transformers emit, on every processor/model call, lines such as
    "Kwargs passed to `processor.__call__` have to be in `processor_kwargs`
    dict…" and "The `use_fast` parameter is deprecated…". A single caption
    review fires ~20 judge inferences, so these flood the console. This
    filter silences only those known-noisy messages (matched by substring),
    leaving every other transformers log intact.
    """

    _NEEDLES = ("processor_kwargs", "use_fast")

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False (drop) for a known-noisy deprecation message."""
        message = record.getMessage()
        return not any(needle in message for needle in self._NEEDLES)


def _silence_transformers_noise() -> None:
    """Silence the repetitive transformers deprecation warnings.

    They reach the console through two channels: the transformers *logger*
    (the ``processor_kwargs`` line) and Python's ``warnings`` (the
    ``use_fast`` deprecation), so both are muted for those messages only.
    """
    tf_logger = logging.getLogger("transformers")
    filt = _TransformersNoiseFilter()
    tf_logger.addFilter(filt)
    for handler in tf_logger.handlers:
        handler.addFilter(filt)
    for needle in ("use_fast", "processor_kwargs"):
        warnings.filterwarnings("ignore", message=f".*{needle}.*")


_silence_transformers_noise()

# Ceiling on generated caption length (tokens); generation still stops on EOS.
# 1024 truncated the exhaustive multi-section prompts some families use
# (Gemma 4); 2048 gives headroom. GGUF is 2048 too — images have ample n_ctx,
# video just stops when the context fills, so the ceiling never hurts.
MAX_NEW_TOKENS = 2048
MAX_NEW_TOKENS_GGUF = 2048

# Per-call token-budget override, set by :func:`generate_caption` and reset in
# its ``finally``. Safe as a module global (single-threaded worker). Claim
# decomposition sets a larger budget so a reasoning model can finish
# ``<think>`` *and* emit the JSON answer without being cut off mid-thought.
_budget_override = None  # pylint: disable=invalid-name


def _budget(default: int) -> int:
    """Return the active token budget: the per-call override, else default."""
    return _budget_override or default


class CaptionError(Exception):
    """Raised when captioning a single media fails.

    Carries a human-readable reason so callers (the gallery) can report it
    without persisting it as if it were a real caption.
    """


def _resize_to(img: Image.Image, max_size: int) -> Image.Image:
    """Downscale so the longest side is at most ``max_size`` (ratio kept)."""
    width, height = img.size
    if max(width, height) <= max_size:
        return img
    scale = max_size / max(width, height)
    return img.resize(
        (int(width * scale), int(height * scale)), Image.Resampling.LANCZOS
    )


def _resize(img: Image.Image) -> Image.Image:
    """Downscale an image to the configured caption size (Gemma capped)."""
    max_size = get_caption_image_size()
    # Gemma's vision encoder is fixed at 896px — never upscale past it.
    if loader.current_model_type in {"gemma3", "gemma3n"}:
        max_size = min(max_size, GEMMA_MAX_IMAGE_SIZE)
    return _resize_to(img, max_size)


# Reasoning ("thinking") blocks some models emit before the answer. Two shapes:
#   1. <think>...</think>                    (DeepSeek-R1, Qwen3-Thinking, ...)
#   2. Harmony-style channels as plain text:
#        <|channel>thought ... <|channel>final<|message|> <answer>
#      (baked into some fine-tunes — keep only the final channel).
_THINK_TAG_RE = re.compile(r"(?is)<think>.*?</think>")
# Thinking that overruns the budget opens ``<think>`` and is cut off before
# ``</think>`` — all reasoning, no answer. Everything from a dangling open tag
# to the end is dropped (after the closed-pair removal, so only unclosed left).
_THINK_OPEN_RE = re.compile(r"(?is)<think>.*$")
_CHANNEL_RE = re.compile(
    r"(?i)<\|?\s*channel\s*\|?>\s*([a-z_]+)?\s*(?:<\|?\s*message\s*\|?>)?"
)
_LEFTOVER_MARKER_RE = re.compile(
    r"(?i)<\|?\s*(?:start|end|return|message|channel|assistant|user|system)"
    r"\s*\|?>"
)
_FINAL_CHANNELS = {"final", "answer", "response", "output", "message"}


def _strip_reasoning(text: str) -> str:
    """Remove <think> blocks and harmony channels, keeping the answer."""
    if not text:
        return text
    text = _THINK_TAG_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)

    # Channel/harmony format: keep only the final channel's content. With no
    # explicit 'final' channel, fall back to what follows the last marker.
    matches = list(_CHANNEL_RE.finditer(text))
    if matches:
        final = next(
            (
                m
                for m in reversed(matches)
                if (m.group(1) or "").lower() in _FINAL_CHANNELS
            ),
            None,
        )
        chosen = final or matches[-1]
        text = text[chosen.end() :]

    text = _LEFTOVER_MARKER_RE.sub("", text)
    return text.strip()


def _finalize(text: str, think_mode: str) -> str:
    """Clean model output; 'show' keeps the reasoning, others strip it."""
    text = (text or "").strip()
    return text if think_mode == "show" else _strip_reasoning(text)


def _thinking_off(think_mode: str) -> bool:
    """Return whether reasoning should be actively suppressed."""
    return think_mode == "off"


def _image_to_data_uri(image: Image.Image) -> str:
    """Encode an image as a base64 ``data:image/jpeg`` URI."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# --- transformers backends ---


def _chat_template_kwargs(think_mode: str) -> dict:
    """Return extra kwargs for ``apply_chat_template``.

    ``'off'`` asks thinking-aware templates to suppress reasoning; templates
    that ignore the flag simply drop it (no error).
    """
    return {"enable_thinking": False} if _thinking_off(think_mode) else {}


def _caption_llava(
    image: Image.Image,
    prompt: str,
    temperature: float,
    seed: int,
    think_mode: str,
) -> str:
    """Caption one image with a Llava (joycaption) transformers model."""
    # seed is part of the shared captioner signature but unused by this path.
    # pylint: disable=unused-argument
    messages = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user", "content": prompt},
    ]
    text_prompt = loader.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **_chat_template_kwargs(think_mode),
    )
    inputs = loader.processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        resample=Image.Resampling.BICUBIC,
    ).to(loader.model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    ids = loader.model.generate(
        **inputs,
        max_new_tokens=_budget(MAX_NEW_TOKENS),
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        suppress_tokens=None,
        use_cache=True,
        top_k=None,
    )[0][inputs["input_ids"].shape[1] :]

    decoded = loader.processor.tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return _finalize(decoded, think_mode)


def _run_transformers(
    content: list, images: list, temperature: float, seed: int, think_mode: str
) -> str:
    """Run transformers generation for one user message + its image(s)."""
    messages = [{"role": "user", "content": content}]
    text_prompt = loader.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **_chat_template_kwargs(think_mode),
    )
    inputs = loader.processor(
        text=[text_prompt], images=images, return_tensors="pt", truncation=True
    ).to(loader.model.device)

    model_dtype = next(iter(loader.model.parameters())).dtype
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)

    kwargs = {"max_new_tokens": _budget(MAX_NEW_TOKENS), "do_sample": False}
    if temperature > 0.0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
    if seed != -1 and kwargs["do_sample"]:
        torch.manual_seed(int(seed))

    generated = loader.model.generate(**inputs, **kwargs)
    generated = [
        out[len(ins) :] for ins, out in zip(inputs.input_ids, generated)
    ]
    decoded = loader.processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return _finalize(decoded, think_mode)


def _caption_transformers(
    image: Image.Image,
    prompt: str,
    temperature: float,
    seed: int,
    think_mode: str,
) -> str:
    """Caption one image (qwen/gemma transformers share this format)."""
    content = [{"type": "image"}, {"type": "text", "text": prompt}]
    return _run_transformers(content, [image], temperature, seed, think_mode)


def _run_transformers_text(
    prompt: str, temperature: float, seed: int, think_mode: str
) -> str:
    """Run a transformers model on a text-only prompt (no image tokens).

    The judge's text pass (see :mod:`src.caption_judge`): a VL processor
    accepts a chat with no image, so the caption is fed as plain text and no
    pixel values are built. Same decode path as :func:`_run_transformers`.
    """
    messages = [{"role": "user", "content": prompt}]
    text_prompt = loader.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **_chat_template_kwargs(think_mode),
    )
    inputs = loader.processor(
        text=[text_prompt], return_tensors="pt", truncation=True
    ).to(loader.model.device)

    kwargs = {"max_new_tokens": _budget(MAX_NEW_TOKENS), "do_sample": False}
    if temperature > 0.0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
    if seed != -1 and kwargs["do_sample"]:
        torch.manual_seed(int(seed))

    generated = loader.model.generate(**inputs, **kwargs)
    generated = [
        out[len(ins) :] for ins, out in zip(inputs.input_ids, generated)
    ]
    decoded = loader.processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return _finalize(decoded, think_mode)


def _run_gguf_text_chat(
    prompt: str, temperature: float, seed: int, think_mode: str
) -> str:
    """Run a GGUF chat model on a text-only prompt (no image).

    Used by :func:`generate_text` when the loaded judge is a GGUF vision
    model: the same ``create_chat_completion`` call as
    :func:`_run_gguf_vision`, with a plain-text message and no image part.
    """
    out = loader.model.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_budget(MAX_NEW_TOKENS_GGUF),
        temperature=temperature if temperature > 0.0 else 0.7,
        top_p=0.9,
        seed=int(seed) if seed != -1 else None,
    )
    return _finalize(out["choices"][0]["message"]["content"], think_mode)


def _run_transformers_video(
    video,
    metadata: VideoMetadata,
    prompt: str,
    temperature: float,
    seed: int,
    think_mode: str,
) -> str:
    """Run native video generation (Qwen3-VL / Qwen2.5-VL / Gemma 4).

    The pre-sampled frames are handed to the processor as a single video so it
    inserts the trained timestamp tokens (``do_sample_frames`` off — sampling
    already happened in :func:`_sample_video_frames`). ``video`` is a
    ``(num_frames, H, W, 3)`` RGB array; ``metadata`` the source timing (fps +
    sampled indices) used to label each frame with its real timestamp.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = loader.processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        processor_kwargs={
            "video_metadata": [metadata],
            "do_sample_frames": False,
        },
        **_chat_template_kwargs(think_mode),
    ).to(loader.model.device)

    model_dtype = next(iter(loader.model.parameters())).dtype
    for key in ("pixel_values_videos", "pixel_values"):
        if key in inputs:
            inputs[key] = inputs[key].to(dtype=model_dtype)

    kwargs = {"max_new_tokens": _budget(MAX_NEW_TOKENS), "do_sample": False}
    if temperature > 0.0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
    if seed != -1 and kwargs["do_sample"]:
        torch.manual_seed(int(seed))

    generated = loader.model.generate(**inputs, **kwargs)
    trimmed = [
        out[len(ins) :] for ins, out in zip(inputs.input_ids, generated)
    ]
    decoded = loader.processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return _finalize(decoded, think_mode)


def _caption_video_transformers(
    frames: list,
    timestamps: list,
    prompt: str,
    temperature: float,
    seed: int,
    think_mode: str,
) -> str:
    """Caption a video as interleaved (timestamp + image) pairs (Gemma 3/3n).

    Labeling each frame with its real time lets the model gauge motion speed.
    Image order is preserved, so the separate ``frames`` list still lines up.
    """
    content = []
    for timestamp in timestamps:
        content.append({"type": "text", "text": f"[{timestamp:.2f}s]"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    return _run_transformers(content, frames, temperature, seed, think_mode)


# --- GGUF backends (llama-cpp) ---


def _run_gguf_vision(
    content: list, temperature: float, seed: int, think_mode: str
) -> str:
    """Run GGUF vision generation for one user message (image(s) + text)."""
    out = loader.model.create_chat_completion(
        messages=[{"role": "user", "content": content}],
        max_tokens=_budget(MAX_NEW_TOKENS_GGUF),
        temperature=temperature if temperature > 0.0 else 0.7,
        top_p=0.9,
        seed=int(seed) if seed != -1 else None,
    )
    return _finalize(out["choices"][0]["message"]["content"], think_mode)


def _caption_gguf_vision(
    image: Image.Image,
    prompt: str,
    temperature: float,
    seed: int,
    think_mode: str,
) -> str:
    """Caption one image via a GGUF + mmproj chat handler (base64 data URI)."""
    content = [
        {"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}},
        {"type": "text", "text": prompt},
    ]
    return _run_gguf_vision(content, temperature, seed, think_mode)


def _caption_video_gguf(
    frames: list,
    timestamps: list,
    prompt: str,
    temperature: float,
    seed: int,
    think_mode: str,
) -> str:
    """Caption a video as interleaved (timestamp + image) pairs (Gemma GGUF).

    Each frame is sent as a base64 data URI prefixed with its real time so the
    model can judge motion speed.
    """
    content = []
    for timestamp, frame in zip(timestamps, frames):
        content.append({"type": "text", "text": f"[{timestamp:.2f}s]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_uri(frame)},
            }
        )
    content.append({"type": "text", "text": prompt})
    return _run_gguf_vision(content, temperature, seed, think_mode)


def _caption_gguf_text(
    image: Image.Image,
    prompt: str,
    temperature: float,
    seed: int,
    think_mode: str,
) -> str:
    """Caption via a text-only GGUF model (the image is NOT passed in)."""
    # image is part of the shared captioner signature but unused here.
    # pylint: disable=unused-argument
    fallback = (
        f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    processor = loader.processor
    if hasattr(processor, "apply_chat_template"):
        try:
            text_prompt = processor.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **_chat_template_kwargs(think_mode),
            )
        except Exception:  # pylint: disable=broad-exception-caught
            text_prompt = fallback
    else:
        text_prompt = fallback

    kwargs = {
        "max_tokens": _budget(MAX_NEW_TOKENS_GGUF),
        "temperature": temperature if temperature > 0.0 else 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }
    if seed != -1:
        kwargs["seed"] = int(seed)

    output = loader.model(text_prompt, **kwargs)
    response = (
        output.get("choices", [{}])[0].get("text", "")
        if isinstance(output, dict)
        else str(output)
    )
    return _finalize(response.replace("<end_of_turn>", ""), think_mode)


# Dispatch on the loaded format (set by the loader).
_CAPTIONERS = {
    "transformers": None,  # resolved per-family below
    "gguf-vision": _caption_gguf_vision,
    "gguf-text": _caption_gguf_text,
}

_TRANSFORMERS_BY_FAMILY = {
    "llava": _caption_llava,
    "qwen2.5": _caption_transformers,
    "qwen3": _caption_transformers,
    "gemma3": _caption_transformers,
    "gemma3n": _caption_transformers,
    "gemma4": _caption_transformers,
    "mistral3": _caption_transformers,
    "qwen3.6": _caption_transformers,
}


def _pick_captioner():
    """Return the captioner callable for the loaded format/family, or None."""
    if loader.current_format == "transformers":
        return _TRANSFORMERS_BY_FAMILY.get(loader.current_model_type)
    return _CAPTIONERS.get(loader.current_format)


def generate_caption(
    image_source,
    prompt: str,
    temperature: float,
    seed: int,
    think_mode: str = "auto",
    max_new_tokens: int | None = None,
) -> str:
    """Caption a single image and return the caption text.

    ``image_source`` is a file path or a loaded image; ``temperature`` ``0`` is
    greedy where supported; ``seed`` ``-1`` is random; ``think_mode``
    ``auto``/``off`` strip reasoning, ``show`` keeps it; ``max_new_tokens``
    overrides the budget for this call (``None`` falls back to the loaded model
    type's configured ceiling, e.g. JoyCaption's 512). Raises ``CaptionError``
    when no captioner matches the loaded model or generation fails.
    """
    global _budget_override  # pylint: disable=global-statement
    if max_new_tokens is None and loader.current_model_type:
        max_new_tokens = get_model_max_new_tokens(loader.current_model_type)
    _budget_override = max_new_tokens
    try:
        image = (
            Image.open(image_source).convert("RGB")
            if isinstance(image_source, str)
            else image_source
        )
        image = _resize(image)

        captioner = _pick_captioner()
        if captioner is None:
            raise CaptionError(
                f"no captioner for format '{loader.current_format}' / "
                f"type '{loader.current_model_type}'"
            )

        return captioner(image, prompt, temperature, seed, think_mode)

    except CaptionError:
        raise
    except Exception as exc:
        raise CaptionError(str(exc)) from exc
    finally:
        _budget_override = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# Text-only inference by loaded format (the judge's text pass). Transformers
# and GGUF-text feed the prompt with no image; a GGUF vision model answers a
# plain-text chat with the same handler it uses for captions.
_TEXT_RUNNERS = {
    "transformers": _run_transformers_text,
    "gguf-vision": _run_gguf_text_chat,
    "gguf-text": lambda prompt, temperature, seed, think_mode: (
        _caption_gguf_text(None, prompt, temperature, seed, think_mode)
    ),
}


def generate_text(
    prompt: str,
    temperature: float = 0.0,
    seed: int = -1,
    think_mode: str = "off",
    max_new_tokens: int | None = None,
) -> str:
    """Run the loaded model on a text-only prompt and return its output.

    The judge's text pass (see :mod:`src.caption_judge`): no image is loaded,
    so a text-only rule is checked without paying for pixel tokens. Dispatches
    on the loaded format exactly like :func:`generate_caption`. Raises
    ``CaptionError`` when the loaded format has no text runner or generation
    fails.
    """
    global _budget_override  # pylint: disable=global-statement
    if max_new_tokens is None and loader.current_model_type:
        max_new_tokens = get_model_max_new_tokens(loader.current_model_type)
    _budget_override = max_new_tokens
    try:
        runner = _TEXT_RUNNERS.get(loader.current_format)
        if runner is None:
            raise CaptionError(
                f"no text runner for format '{loader.current_format}'"
            )
        return runner(prompt, temperature, seed, think_mode)
    except CaptionError:
        raise
    except Exception as exc:
        raise CaptionError(str(exc)) from exc
    finally:
        _budget_override = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# --- Video captioning ---

# Native-video families: their transformers processor accepts a real video and
# inserts the timestamp tokens the model was trained on. Every other family
# (Gemma 3 / 3n, all GGUF) reads the sampled frames as still images instead.
_NATIVE_VIDEO_FAMILIES = ("qwen3", "qwen2.5", "gemma4")


def _video_timing(timestamps: list) -> str:
    """Return a sentence describing the clip's span and pace, or ""."""
    if timestamps and len(timestamps) > 1:
        span = timestamps[-1] - timestamps[0]
        interval = span / (len(timestamps) - 1)
        return (
            f"The clip spans about {span:.1f}s of footage, roughly one "
            f"sample every {interval:.2f}s. "
        )
    return ""


def _compose_video_prompt(
    kind: str, n_frames: int, timestamps: list, body: str
) -> str:
    """Prepend the path-specific preamble and timing hint to the body.

    ``kind`` is ``"native"`` (the model sees a real video) or ``"frames"`` (the
    model sees still images). ``body`` is the user's editable video prompt.
    """
    timing = _video_timing(timestamps)
    if kind == "native":
        preamble = (
            "You are shown a short video clip. "
            f"{timing}Watch it as one continuous moving shot and "
            "describe the video as a whole.\n\n"
        )
    else:
        preamble = (
            f"You are shown {n_frames} still frames extracted in "
            "chronological order from a SINGLE video clip — not "
            f"separate images. {timing}Read them together as one "
            "continuous moving shot and describe the video as a "
            "whole.\n\n"
        )
    return preamble + body


def _sample_video_frames(video_path: str, fps: float, max_seconds: float):
    """Sample frames at ``fps`` over the first ``max_seconds`` of the clip.

    Count = ``fps * seconds + 1`` (t=0 included), bounded by the clip's frames.
    Returns ``(frames, timestamps, metadata)`` — ``frames`` PIL images in
    chronological order, ``timestamps`` their seconds, ``metadata`` a dict
    (``total``/``src_fps``/``indices``/``width``/``height``/``duration``) the
    native path uses to rebuild the trained timestamps. All empty when
    unreadable.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], {}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0:
        cap.release()
        return [], [], {}

    duration = (total / src_fps) if src_fps and src_fps > 0 else None
    span = min(duration, max_seconds) if duration else max_seconds

    count = min(estimate_video_frames(fps, span), total)

    last_index = (
        min(total - 1, int(span * src_fps))
        if (duration and src_fps)
        else total - 1
    )
    wanted = [
        int(round(i)) for i in torch.linspace(0, last_index, count).tolist()
    ]

    def _timestamp(idx: int) -> float:
        # Real time of this frame. With unknown source FPS, fall back to a
        # linear estimate across `span` so label spacing stays meaningful.
        if src_fps and src_fps > 0:
            return idx / src_fps
        return span * (idx / last_index) if last_index > 0 else 0.0

    frames, timestamps, indices, seen = [], [], [], set()
    for idx in wanted:
        if idx in seen:
            continue
        seen.add(idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            )
            timestamps.append(_timestamp(idx))
            indices.append(idx)
    cap.release()

    if not frames:
        return [], [], {}
    width, height = frames[0].size
    metadata = {
        "total": total,
        "src_fps": float(src_fps) if src_fps and src_fps > 0 else None,
        "indices": indices,
        "width": width,
        "height": height,
        "duration": duration,
    }
    return frames, timestamps, metadata


def _sampled_resized(video_path: str):
    """Sample then resize the clip's frames, or raise when none can be read."""
    frames, timestamps, metadata = _sample_video_frames(
        video_path,
        get_video_fps(),
        get_video_max_seconds(),
    )
    if not frames:
        raise CaptionError("could not read any frame from the video.")
    resolution = get_video_resolution()
    frames = [_resize_to(f, resolution) for f in frames]
    return frames, timestamps, metadata


def generate_captions_for_video(
    video_path: str,
    prompt: str,
    temperature: float,
    seed: int,
    think_mode: str = "auto",
) -> str:
    """Caption a video, using native video input where the model supports it.

    The UI ``prompt`` is ignored: videos use the dedicated, editable video
    prompt from Settings (the framing preamble and timing hint are added
    automatically).

    Raises
    ------
    CaptionError
        If the video cannot be read or generation fails.
    """
    # prompt is part of the shared caption signature but unused for video.
    # pylint: disable=unused-argument
    body = get_video_prompt()
    fmt = loader.current_format
    family = loader.current_model_type
    try:
        if fmt == "transformers" and family in _NATIVE_VIDEO_FAMILIES:
            return _caption_video_native(
                video_path, body, temperature, seed, think_mode
            )
        if fmt in {"transformers", "gguf-vision"}:
            return _caption_video_frames(
                video_path, body, temperature, seed, think_mode
            )
        return _caption_video_per_frame(
            video_path, body, temperature, seed, think_mode
        )
    except CaptionError:
        raise
    except Exception as exc:
        raise CaptionError(str(exc)) from exc
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _caption_video_native(
    video_path: str,
    body: str,
    temperature: float,
    seed: int,
    think_mode: str,
) -> str:
    """Caption a clip as a native video (the processor adds the timestamps)."""
    frames, timestamps, metadata = _sampled_resized(video_path)
    prompt = _compose_video_prompt("native", len(frames), timestamps, body)

    video = np.stack([np.asarray(f) for f in frames])
    height, width = int(video.shape[1]), int(video.shape[2])
    src_fps = metadata.get("src_fps")
    duration = metadata.get("duration") or (
        timestamps[-1] if timestamps else 0.0
    )
    vmeta = VideoMetadata(
        total_num_frames=metadata.get("total", len(frames)),
        fps=src_fps if src_fps else float(len(frames)),
        width=width,
        height=height,
        duration=duration,
        video_backend="opencv",
        frames_indices=metadata.get("indices") or list(range(len(frames))),
    )
    return _run_transformers_video(
        video, vmeta, prompt, temperature, seed, think_mode
    )


def _caption_video_frames(
    video_path: str,
    body: str,
    temperature: float,
    seed: int,
    think_mode: str,
) -> str:
    """Caption a clip as still frames (Gemma 3/3n transformers or GGUF)."""
    frames, timestamps, _ = _sampled_resized(video_path)
    prompt = _compose_video_prompt("frames", len(frames), timestamps, body)
    if loader.current_format == "gguf-vision":
        return _caption_video_gguf(
            frames, timestamps, prompt, temperature, seed, think_mode
        )
    return _caption_video_transformers(
        frames, timestamps, prompt, temperature, seed, think_mode
    )


def _caption_video_per_frame(
    video_path: str,
    body: str,
    temperature: float,
    seed: int,
    think_mode: str,
) -> str:
    """Caption each sampled frame independently, prefixed with its time.

    Fallback for single-image models (Llava) and text-only GGUF.

    Raises
    ------
    CaptionError
        If the video cannot be read or a frame fails to caption.
    """
    frames, timestamps, _ = _sampled_resized(video_path)
    captions = []
    for frame, seconds in zip(frames, timestamps):
        caption = generate_caption(frame, body, temperature, seed, think_mode)
        captions.append(
            f"[{int(seconds // 60):02d}:{int(seconds % 60):02d}] {caption}"
        )
    return "\n".join(captions)
