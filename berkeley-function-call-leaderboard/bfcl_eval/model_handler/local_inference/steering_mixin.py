"""steering_mixin.py

Mixin that adds activation-steering inference to any BFCL OSS handler.

Usage:
    class QwenFCSteeringHandler(SteeringMixin, QwenFCHandler):
        pass

The MRO ensures SteeringMixin._query_prompting is called first. When no
steering config is loaded it delegates to super() (vLLM path). When a
steering config is loaded it runs direct HF inference with forward hooks.
"""

from __future__ import annotations

import gc
import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Fake OpenAI Completions response shim
# ---------------------------------------------------------------------------

@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Choice:
    text: str


class _FakeCompletionResponse:
    """Duck-types openai.types.Completion for _parse_query_response_prompting."""

    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.choices = [_Choice(text=text)]
        self.usage = _Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


# ---------------------------------------------------------------------------
# SteeringMixin
# ---------------------------------------------------------------------------

class SteeringMixin:
    """Mixin providing HF-direct inference with optional activation steering.

    Designed to be the first base class so its _query_prompting takes priority
    in the MRO:
        class MySteeringHandler(SteeringMixin, MyBaseHandler): pass

    Attributes set by load_hf_model():
        _hf_model       — loaded AutoModelForCausalLM (None until loaded)
        _hf_tokenizer   — loaded AutoTokenizer (None until loaded)
        _steering_sets  — list of runtime steering set dicts, or None
        _thinking_start — int token ID for <think>
        _thinking_end   — int token ID for </think>
        _generation_kwargs — dict passed to model.generate()
    """

    # Flag read by generate_results() to skip vLLM server startup
    is_steering_handler: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hf_model = None
        self._hf_tokenizer = None
        self._steering_sets: Optional[List[Dict[str, Any]]] = None
        self._thinking_start: int = 151667   # Qwen3 default; overridden by config
        self._thinking_end: int = 151668
        self._generation_kwargs: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public setup / teardown (called from generate_results)
    # ------------------------------------------------------------------

    def load_hf_model(
        self,
        model_path_or_id: str,
        steering_config: Optional[Dict[str, Any]],
        dtype: str = "bfloat16",
    ) -> None:
        """Load HuggingFace model + tokenizer and optionally parse steering config.

        Args:
            model_path_or_id: HF repo ID or local path (e.g. 'Qwen/Qwen3-8B')
            steering_config: parsed YAML dict, or None for unsteered HF inference
            dtype: torch dtype string ('bfloat16', 'float16', 'float32')
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        is_local = Path(model_path_or_id).is_dir()
        load_kwargs: Dict[str, Any] = {
            "pretrained_model_name_or_path": model_path_or_id,
            "trust_remote_code": True,
        }
        if is_local:
            load_kwargs["local_files_only"] = True

        print(f"[steering] Loading tokenizer from '{model_path_or_id}'")
        self._hf_tokenizer = AutoTokenizer.from_pretrained(**load_kwargs)
        if self._hf_tokenizer.pad_token is None:
            self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token

        config = AutoConfig.from_pretrained(**load_kwargs)
        # Share tokenizer with OSSHandler so _query_prompting token-count works
        self.tokenizer = self._hf_tokenizer
        self.model_path_or_id = model_path_or_id

        if hasattr(config, "max_position_embeddings"):
            self.max_context_length = config.max_position_embeddings
        elif self._hf_tokenizer.model_max_length:
            self.max_context_length = self._hf_tokenizer.model_max_length

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)
        if not torch.cuda.is_available():
            torch_dtype = torch.float32

        # Use FA2 if available (requires flash_attn package).
        # Otherwise we must tell transformers flash_attn is unavailable BEFORE
        # calling from_pretrained: modeling_qwen3 (and similar) unconditionally
        # import from modeling_flash_attention_utils at module level, which calls
        # is_flash_attn_2_available() and tries to import the broken .so —
        # crashing before attn_implementation kwarg is ever read.
        fa2_kwargs: Dict[str, Any] = {}
        if torch.cuda.is_available():
            try:
                # Test the actual CUDA extension, not just the package presence.
                import flash_attn.flash_attn_interface  # noqa: F401
                import flash_attn  # noqa: F401
                fa2_kwargs["attn_implementation"] = "flash_attention_2"
                print(f"[steering] Flash Attention 2 enabled (flash_attn {flash_attn.__version__}).")
            except Exception as _fa2_err:
                print(f"[steering] flash_attn import failed ({type(_fa2_err).__name__}: {_fa2_err}) — forcing eager attention.")
                # Monkey-patch transformers' availability check so it won't try to
                # import flash_attn at module-load time inside from_pretrained.
                try:
                    import transformers.utils.import_utils as _tu
                    if hasattr(_tu, "_flash_attn_2_available"):
                        _tu._flash_attn_2_available = False
                    _tu.is_flash_attn_2_available = lambda: False
                except Exception:
                    pass
                fa2_kwargs["attn_implementation"] = "eager"

        print(f"[steering] Loading model '{model_path_or_id}' (dtype={torch_dtype})")
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            model_path_or_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **fa2_kwargs,
            **({"local_files_only": True} if is_local else {}),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._hf_model = self._hf_model.to(device)
        self._hf_model.eval()
        print(f"[steering] Model loaded on {device}")

        # Auto-detect thinking tokens (overridden by config if present)
        self._thinking_start, self._thinking_end = self._detect_thinking_tokens(
            self._hf_tokenizer
        )

        # Default generation kwargs
        temperature = getattr(self, "temperature", 0.001)
        self._generation_kwargs = {
            "max_new_tokens": 4096,
            "temperature": temperature,
            "do_sample": temperature > 0.01,
            "return_dict_in_generate": True,
        }

        if steering_config is not None:
            self._load_steering_config(steering_config)

    def teardown_hf_model(self) -> None:
        """Free GPU memory after inference."""
        if self._hf_model is not None:
            del self._hf_model
            self._hf_model = None
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
        print("[steering] Model unloaded and GPU memory cleared.")

    # ------------------------------------------------------------------
    # _query_prompting override
    # ------------------------------------------------------------------

    def _query_prompting(self, inference_data: dict):
        """Route to steered HF inference or fall back to super() (vLLM)."""
        if self._hf_model is None:
            # Steering not active — delegate to OSSHandler (vLLM path)
            return super()._query_prompting(inference_data)
        return self._steered_generate(inference_data)

    # ------------------------------------------------------------------
    # Steered generation
    # ------------------------------------------------------------------

    def _steered_generate(self, inference_data: dict):
        """Run model.generate() with optional steering hooks and return a
        fake OpenAI-compatible response object."""
        from bfcl_eval.model_handler.local_inference.steering_primitives import (
            run_generation,
        )

        function: list = inference_data["function"]
        message: list = inference_data["message"]

        formatted_prompt: str = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        tokenizer = self._hf_tokenizer
        model = self._hf_model

        encoded = tokenizer(formatted_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        base_inputs = {k: v.to(device) for k, v in encoded.items()}

        steering = {"sets": self._steering_sets} if self._steering_sets else None

        start_time = time.time()
        with torch.inference_mode():
            result = run_generation(
                model=model,
                tokenizer=tokenizer,
                base_inputs=base_inputs,
                generation_kwargs=self._generation_kwargs,
                thinking_start_id=self._thinking_start,
                thinking_end_id=self._thinking_end,
                steering=steering,
            )
        elapsed = time.time() - start_time

        # Reconstruct raw output text that _parse_query_response_prompting expects:
        # QwenFCHandler splits on </think> to separate reasoning from answer.
        reasoning = result["reasoning"]
        answer = result["answer"]
        if reasoning:
            raw_text = f"<think>\n{reasoning}\n</think>\n\n{answer}"
        else:
            raw_text = answer

        prompt_tokens = result["token_count_prompt"]
        output_tokens = result["token_count_total"] - result["token_count_prompt"]

        return _FakeCompletionResponse(
            text=raw_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=output_tokens,
        ), elapsed

    def batch_steered_generate(
        self, batch_of_inference_data: List[Dict[str, Any]]
    ) -> List[tuple]:
        """Run a batch of samples through model.generate() in one GPU call.

        Args:
            batch_of_inference_data: list of inference_data dicts (each with
                "function" and "message" keys, same as _steered_generate expects)

        Returns:
            List of (_FakeCompletionResponse, elapsed_per_sample) tuples,
            one per input item.
        """
        from bfcl_eval.model_handler.local_inference.steering_primitives import (
            run_generation,
        )

        print(f"[steering] batch_steered_generate: {len(batch_of_inference_data)} samples")
        tokenizer = self._hf_tokenizer
        model = self._hf_model
        device = next(model.parameters()).device

        # Format all prompts
        prompts = []
        for inference_data in batch_of_inference_data:
            prompt = self._format_prompt(
                inference_data["message"], inference_data["function"]
            )
            inference_data["inference_input_log"] = {"formatted_prompt": prompt}
            prompts.append(prompt)

        # Tokenize with left-padding so all sequences align on the right
        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        tokenizer.padding_side = orig_padding_side

        base_inputs = {k: v.to(device) for k, v in encoded.items()}
        steering = {"sets": self._steering_sets} if self._steering_sets else None

        start_time = time.time()
        with torch.inference_mode():
            results = run_generation(
                model=model,
                tokenizer=tokenizer,
                base_inputs=base_inputs,
                generation_kwargs=self._generation_kwargs,
                thinking_start_id=self._thinking_start,
                thinking_end_id=self._thinking_end,
                steering=steering,
            )
        total_elapsed = time.time() - start_time
        elapsed_per_sample = total_elapsed / len(batch_of_inference_data)

        # run_generation returns list when batch > 1, single dict when batch == 1
        if isinstance(results, dict):
            results = [results]

        responses = []
        for result in results:
            reasoning = result["reasoning"]
            answer = result["answer"]
            raw_text = f"<think>\n{reasoning}\n</think>\n\n{answer}" if reasoning else answer
            responses.append((
                _FakeCompletionResponse(
                    text=raw_text,
                    prompt_tokens=result["token_count_prompt"],
                    completion_tokens=result["token_count_total"] - result["token_count_prompt"],
                ),
                elapsed_per_sample,
            ))
        return responses

    def inference_batch(
        self,
        test_cases: List[Dict[str, Any]],
        include_input_log: bool,
        exclude_state_log: bool,
    ) -> List[Dict[str, Any]]:
        """Process a batch of single-turn test cases through the GPU together.

        Replicates the logic of inference_single_turn_FC / inference_single_turn_prompting
        but batches the heavy GPU computation into one model.generate() call.

        Only supports single-turn test cases. Multi-turn cases should use the
        standard per-sample inference path.

        Returns:
            List of result dicts with keys: id, result, input_token_count,
            output_token_count, latency, (reasoning_content if present).
        """
        import traceback
        from tqdm import tqdm

        print(f"[steering] inference_batch called with {len(test_cases)} test cases")
        # Determine the inference path by checking whether _pre_query_processing_FC
        # is actually overridden. "FC" in registry_name is not reliable — some handlers
        # (e.g. QwenFCHandler) have "FC" in the name but use the prompting path.
        from bfcl_eval.model_handler.base_handler import BaseHandler
        is_fc = (
            type(self)._pre_query_processing_FC
            is not BaseHandler._pre_query_processing_FC
        )

        # Build inference_data for each test case (pre-processing, no GPU yet)
        all_inference_data = []
        valid_test_cases = []
        errors = []

        for test_case in test_cases:
            try:
                if is_fc:
                    inference_data: Dict[str, Any] = {}
                    inference_data = self._pre_query_processing_FC(inference_data, test_case)
                    inference_data = self._compile_tools(inference_data, test_case)
                    inference_data = self.add_first_turn_message_FC(
                        inference_data, test_case["question"][0]
                    )
                else:
                    inference_data = self._pre_query_processing_prompting(test_case)
                    inference_data = self.add_first_turn_message_prompting(
                        inference_data, test_case["question"][0]
                    )
                all_inference_data.append(inference_data)
                valid_test_cases.append(test_case)
            except Exception as e:
                errors.append((test_case, str(e), traceback.format_exc()))

        result_dicts = []

        # Report pre-processing errors immediately
        for test_case, err_msg, tb in errors:
            tqdm.write(
                f"❗️ Pre-processing error for {test_case['id']}: {err_msg}\n{tb}"
            )
            result_dicts.append({
                "id": test_case["id"],
                "result": f"Error during inference: {err_msg}",
                "traceback": tb,
            })

        if not all_inference_data:
            return result_dicts

        # Batch GPU inference
        try:
            responses_and_latencies = self.batch_steered_generate(all_inference_data)
            print(f"[steering] batch_steered_generate returned {len(responses_and_latencies)} results")
        except Exception as e:
            tb = traceback.format_exc()
            tqdm.write(f"❗️ Batch GPU error: {e}\n{tb}")
            # Free GPU memory before falling back to per-sample
            gc.collect()
            torch.cuda.empty_cache()
            # Fall back to per-sample
            for test_case, inference_data in zip(valid_test_cases, all_inference_data):
                try:
                    api_response, latency = self._steered_generate(inference_data)
                    responses_and_latencies_fallback = [(api_response, latency)]
                    self._append_result(
                        result_dicts, [test_case], [inference_data],
                        responses_and_latencies_fallback, is_fc, include_input_log,
                    )
                except Exception as e2:
                    result_dicts.append({
                        "id": test_case["id"],
                        "result": f"Error during inference: {e2}",
                        "traceback": traceback.format_exc(),
                    })
            return result_dicts

        self._append_result(
            result_dicts, valid_test_cases, all_inference_data,
            responses_and_latencies, is_fc, include_input_log,
        )
        return result_dicts

    def _append_result(
        self,
        result_dicts: List[Dict[str, Any]],
        test_cases: List[Dict[str, Any]],
        all_inference_data: List[Dict[str, Any]],
        responses_and_latencies: List[tuple],
        is_fc: bool,
        include_input_log: bool,
    ) -> None:
        """Parse responses and append formatted result dicts."""
        for test_case, inference_data, (api_response, latency) in zip(
            test_cases, all_inference_data, responses_and_latencies
        ):
            try:
                if is_fc:
                    model_response_data = self._parse_query_response_FC(api_response)
                else:
                    model_response_data = self._parse_query_response_prompting(api_response)

                metadata: Dict[str, Any] = {}
                if include_input_log:
                    metadata["inference_log"] = [{
                        "role": "inference_input",
                        "content": inference_data.get("inference_input_log", ""),
                    }]
                metadata["input_token_count"] = model_response_data["input_token"]
                metadata["output_token_count"] = model_response_data["output_token"]
                metadata["latency"] = latency
                if (
                    "reasoning_content" in model_response_data
                    and model_response_data["reasoning_content"] != ""
                ):
                    metadata["reasoning_content"] = model_response_data["reasoning_content"]

                result_dicts.append({
                    "id": test_case["id"],
                    "result": model_response_data["model_responses"],
                    **metadata,
                })
            except Exception as e:
                import traceback as tb_mod
                result_dicts.append({
                    "id": test_case["id"],
                    "result": f"Error during inference: {e}",
                    "traceback": tb_mod.format_exc(),
                })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_thinking_tokens(tokenizer) -> tuple[int, int]:
        """Auto-detect <think>/</ think> token IDs from the tokenizer.

        Falls back to Qwen3 defaults (151667/151668) if not found.
        Returns sentinel values (-1/-2) for models with no thinking tokens,
        which will never match any real token ID.
        """
        unk = getattr(tokenizer, "unk_token_id", None)

        think_start = tokenizer.convert_tokens_to_ids("<think>")
        think_end = tokenizer.convert_tokens_to_ids("</think>")

        start = think_start if think_start != unk else None
        end = think_end if think_end != unk else None

        if start is not None and end is not None:
            print(f"[steering] Thinking tokens detected: start={start}, end={end}")
            return start, end

        print(
            "[steering] No thinking tokens found in tokenizer. "
            "Phase tracking will treat all generated tokens as 'answer' phase. "
            "Use apply_target='all' or 'answer' in your steering config."
        )
        return -1, -2  # sentinel: will never match any token

    def _load_steering_config(self, cfg: Dict[str, Any]) -> None:
        """Parse steering YAML dict and prepare runtime steering sets."""
        from bfcl_eval.model_handler.local_inference.steering_primitives import (
            APPLY_TARGET_CHOICES,
            DEFAULT_THINKING_END,
            DEFAULT_THINKING_START,
        )

        sampling = cfg.get("sampling") or {}

        # Thinking tokens from config (override auto-detection if present)
        tokens_conf = sampling.get("thinking_tokens") or {}
        raw_start = tokens_conf.get("start")
        raw_end = tokens_conf.get("end")
        if raw_start is not None and raw_end is not None:
            self._thinking_start = int(raw_start)
            self._thinking_end = int(raw_end)
            print(
                f"[steering] Thinking tokens from config: "
                f"start={self._thinking_start}, end={self._thinking_end}"
            )
        elif raw_start is None and raw_end is None:
            # Explicit null in config → disable phase tracking
            self._thinking_start = -1
            self._thinking_end = -2
            print("[steering] Thinking tokens explicitly disabled in config.")

        # Generation kwargs from config
        for key in ("max_new_tokens", "temperature", "top_p", "top_k",
                    "repetition_penalty", "do_sample"):
            val = sampling.get(key)
            if val is not None:
                self._generation_kwargs[key] = val
        # Always force return_dict_in_generate
        self._generation_kwargs["return_dict_in_generate"] = True

        # Steering sets
        steering_conf = cfg.get("steering") or {}
        sets_conf = steering_conf.get("steering_sets") or []
        if not sets_conf:
            print("[steering] No steering_sets in config — running unsteered HF inference.")
            self._steering_sets = None
            return

        runtime_sets: List[Dict[str, Any]] = []
        for idx, set_spec in enumerate(sets_conf):
            span_id: Optional[str] = set_spec.get("token_span")
            source_conf = set_spec.get("source") or steering_conf.get("source")
            if not span_id or not source_conf:
                raise ValueError(
                    f"steering_sets[{idx}] requires 'token_span' and 'source'."
                )

            vectors = self._load_steering_vectors(source_conf, span_id)

            layers_conf = set_spec.get("layers")
            if layers_conf:
                layers = [l for l in layers_conf if l in vectors]
                missing = [l for l in layers_conf if l not in vectors]
                if missing:
                    print(
                        f"[steering] set[{idx}]: layers not found in vectors and skipped: {missing}"
                    )
            else:
                layers = sorted(vectors.keys())

            coef = float(set_spec.get("coef", 1.0))
            apply_target = set_spec.get("apply_target", "reasoning_answer")
            if apply_target not in APPLY_TARGET_CHOICES:
                raise ValueError(
                    f"steering_sets[{idx}]: invalid apply_target '{apply_target}'. "
                    f"Choose from: {APPLY_TARGET_CHOICES}"
                )
            max_tokens = set_spec.get("max_tokens")

            runtime_sets.append({
                "vectors": vectors,
                "layers": layers,
                "coef": coef,
                "apply_target": apply_target,
                "max_tokens": max_tokens,
            })
            print(
                f"[steering] Set[{idx}]: span='{span_id}', layers={layers}, "
                f"coef={coef}, apply_target='{apply_target}'"
            )

        self._steering_sets = runtime_sets if runtime_sets else None

    @staticmethod
    def _load_steering_vectors(
        source_conf: Dict[str, Any], span_id: str
    ) -> Dict[str, torch.Tensor]:
        """Load steering vectors for a given token span from a source config.

        Supports:
          source:
            type: path
            path: /absolute/path/to/dir_containing_steering_vectors.pt
          source:
            type: file
            path: /absolute/path/to/steering_vectors.pt
        """
        source_type = source_conf.get("type", "path")
        raw_path = source_conf.get("path")
        if not raw_path:
            raise ValueError("Steering source config requires a 'path' field.")

        p = Path(raw_path)
        if source_type == "file":
            vectors_path = p
        else:
            # type: path — directory containing steering_vectors.pt
            vectors_path = p / "steering_vectors.pt"

        if not vectors_path.exists():
            raise FileNotFoundError(
                f"Steering vectors file not found: {vectors_path}"
            )

        data: Dict[str, Dict[str, torch.Tensor]] = torch.load(
            vectors_path, map_location="cpu"
        )
        if span_id not in data:
            available = ", ".join(sorted(data.keys()))
            raise KeyError(
                f"Token span '{span_id}' not found in {vectors_path}. "
                f"Available spans: {available}"
            )
        print(f"[steering] Loaded vectors for span '{span_id}' from {vectors_path}")
        return data[span_id]
