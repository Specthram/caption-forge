"""Tests for :mod:`src.model_registry` filename-based detection."""

from src.model_registry import detect_model, find_mmproj


class TestDetectModel:
    """Tests for :func:`detect_model`."""

    def test_qwen3_vl_safetensors_4b(self):
        """A Qwen3-VL 4B safetensors maps to the 4B HF config."""
        meta = detect_model("Qwen3-VL-4B-Instruct.safetensors")
        assert meta == {
            "type": "qwen3",
            "format": "safetensors",
            "hf_config": "Qwen/Qwen3-VL-4B-Instruct",
        }

    def test_qwen3_vl_gguf_8b(self):
        """The 8B size is read from the filename for a GGUF model."""
        meta = detect_model("Qwen3-VL-8B-Instruct-Q8_0.gguf")
        assert meta["type"] == "qwen3"
        assert meta["format"] == "gguf"
        assert meta["hf_config"] == "Qwen/Qwen3-VL-8B-Instruct"

    def test_qwen3_vl_unknown_size_falls_back_to_default(self):
        """An unrecognized size falls back to the default HF config."""
        meta = detect_model("qwen3-vl-2b.gguf")
        assert meta["hf_config"] == "Qwen/Qwen3-VL-4B-Instruct"

    def test_gemma3n_takes_precedence_over_gemma3(self):
        """A Gemma 3n file is detected as gemma3n, not gemma3."""
        assert detect_model("gemma-3n-E4B-it.gguf")["type"] == "gemma3n"

    def test_gemma4_is_its_own_type(self):
        """A Gemma 4 file maps to the dedicated gemma4 type, not gemma3n."""
        meta = detect_model("gemma-4-E4B-it-Q8_0.gguf")
        assert meta["type"] == "gemma4"
        assert meta["hf_config"] == "unsloth/gemma-4-E4B-it"

    def test_gemma4_takes_precedence_over_gemma3(self):
        """A 'gemma4' file is detected as gemma4, never gemma3."""
        assert detect_model("gemma4-vision.safetensors")["type"] == "gemma4"

    def test_gemma3_vision(self):
        """A plain Gemma 3 file is detected as gemma3."""
        assert detect_model("gemma-3-12b-it.safetensors")["type"] == "gemma3"

    def test_joycaption_gguf_maps_to_llava(self):
        """A JoyCaption GGUF maps to the llava type (loader handles it)."""
        meta = detect_model("Llama-JoyCaption-Beta-One-Hf-Llava-Q4_K_M.gguf")
        assert meta["type"] == "llava"
        assert meta["format"] == "gguf"
        assert meta["hf_config"] == (
            "fancyfeast/llama-joycaption-beta-one-hf-llava"
        )

    def test_joycaption_safetensors_maps_to_llava(self):
        """A JoyCaption safetensors file is detected as llava."""
        meta = detect_model("llama-joycaption-beta-one-hf-llava.safetensors")
        assert meta["type"] == "llava"
        assert meta["format"] == "safetensors"

    def test_mistral_small_3_2_gguf_maps_to_mistral3(self):
        """A Mistral Small 3.2 GGUF maps to the mistral3 type."""
        meta = detect_model("Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf")
        assert meta["type"] == "mistral3"
        assert meta["format"] == "gguf"
        assert meta["hf_config"] == (
            "unsloth/Mistral-Small-3.2-24B-Instruct-2506"
        )

    def test_pixtral_maps_to_mistral3(self):
        """A Pixtral file is detected as mistral3."""
        assert detect_model("pixtral-12b-Q8_0.gguf")["type"] == "mistral3"

    def test_text_only_mistral_is_ignored(self):
        """A plain text Mistral (no 3.2 / pixtral) is not a vision model."""
        assert detect_model("Mistral-7B-Instruct-v0.3-Q4_K_M.gguf") is None

    def test_skips_text_encoders(self):
        """CLIP / T5 text-encoder files are not vision models."""
        assert detect_model("clip_l.safetensors") is None
        assert detect_model("t5xxl_fp16.safetensors") is None

    def test_skips_mmproj_file(self):
        """An mmproj projector is never itself a captioning model."""
        assert detect_model("mmproj-qwen3vl-f16.gguf") is None

    def test_qwen3_text_only_without_vl_is_ignored(self):
        """Qwen3 without 'vl' is the text encoder, not a vision model."""
        assert detect_model("qwen3-8b.safetensors") is None

    def test_unknown_extension_returns_none(self):
        """A non-weight extension is never a model."""
        assert detect_model("notes.txt") is None


class TestFindMmproj:
    """Tests for :func:`find_mmproj`."""

    def test_matches_on_shared_tokens(self):
        """The projector sharing the most distinctive tokens is chosen."""
        files = [
            "Qwen3-VL-8B-Q8_0.gguf",
            "mmproj-Qwen3-VL-8B-f16.gguf",
            "mmproj-gemma3-f16.gguf",
        ]
        assert (
            find_mmproj("Qwen3-VL-8B-Q8_0.gguf", files)
            == "mmproj-Qwen3-VL-8B-f16.gguf"
        )

    def test_joycaption_pairs_its_llava_projector(self):
        """A JoyCaption model pairs with its own mmproj (llava family)."""
        model = "Llama-JoyCaption-Beta-One-Hf-Llava-Q4_K_M.gguf"
        files = [
            model,
            "mmproj-Llama-JoyCaption-Beta-One-Llava-F16.gguf",
            "mmproj-Qwen3-VL-8B-f16.gguf",
        ]
        assert (
            find_mmproj(model, files, model_family="llava")
            == "mmproj-Llama-JoyCaption-Beta-One-Llava-F16.gguf"
        )

    def test_no_mmproj_present_returns_none(self):
        """No projector file means no match."""
        assert find_mmproj("Qwen3-VL-8B.gguf", ["Qwen3-VL-8B.gguf"]) is None

    def test_no_shared_token_returns_none(self):
        """A projector with no distinctive token in common is rejected."""
        files = ["qwen3vl.gguf", "mmproj-gemma.gguf"]
        assert find_mmproj("qwen3vl.gguf", files) is None

    def test_rejects_cross_family_projector(self):
        """A gemma projector is never paired with a qwen model.

        Reproduces the reported bug: a Qwen3-VL abliterated finetune shares
        the generic tokens 'it'/'abliterated' with a gemma-4 projector, which
        used to win the token score and crash llama-cpp's mtmd loader.
        """
        model = "Qwen3-VL-8B-it-abliterated-Q5_K_M.gguf"
        files = [
            model,
            "mmproj-fp16-gemma-4-31B-it-abliterated-Q5_K_M.gguf",
        ]
        assert find_mmproj(model, files, model_family="qwen3") is None

    def test_picks_same_family_over_cross_family(self):
        """With both a matching and a foreign projector, the match wins."""
        model = "Qwen3-VL-8B-Instruct.gguf"
        files = [
            model,
            "mmproj-Qwen3-VL-8B-f16.gguf",
            "mmproj-gemma-4-it.gguf",
        ]
        assert (
            find_mmproj(model, files, model_family="qwen3")
            == "mmproj-Qwen3-VL-8B-f16.gguf"
        )

    def test_generic_projector_still_matches_with_family(self):
        """A family-less projector name still pairs via token overlap."""
        model = "Qwen3-VL-8B-Instruct.gguf"
        files = [model, "mmproj-Qwen3-VL-8B.gguf"]
        assert (
            find_mmproj(model, files, model_family="qwen3")
            == "mmproj-Qwen3-VL-8B.gguf"
        )

    def test_rejects_size_mismatched_projector(self):
        """An E4B model is never paired with a 31B projector.

        Reproduces the reported crash: same gemma-4 family and shared
        'it'/'abliterated' tokens, but different vision towers — a 31B mmproj
        fails to load into an E4B model.
        """
        model = "gemma-4-E4B-it-abliterated-Q8_0.gguf"
        files = [
            model,
            "mmproj-fp16-gemma-4-31B-it-abliterated-Q5_K_M.gguf",
        ]
        assert find_mmproj(model, files, model_family="gemma4") is None

    def test_picks_matching_size_projector(self):
        """With both a 31B and an E4B projector, the E4B model gets the E4B."""
        model = "gemma-4-E4B-it-abliterated-Q8_0.gguf"
        files = [
            model,
            "mmproj-fp16-gemma-4-31B-it-abliterated-Q5_K_M.gguf",
            "mmproj-fp16-gemma-4-E4B-it-abliterated-Q8_0.gguf",
        ]
        assert (
            find_mmproj(model, files, model_family="gemma4")
            == "mmproj-fp16-gemma-4-E4B-it-abliterated-Q8_0.gguf"
        )
