"""steering_primitives.py

Pure PyTorch/transformers functions for activation steering.
Extracted from activation_analysis/steering/inference_with_steering.py and utils.py.
No cross-project imports required.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSSIBLE_LAYER_ATTRS: Sequence[str] = (
    "model.layers",       # Qwen, Llama family
    "transformer.h",      # GPT-2, GPT-3 family
    "encoder.layer",      # BERT family
    "gpt_neox.layers",    # GPT-NeoX
    "block",              # T5 family
)

APPLY_TARGET_CHOICES: Sequence[str] = (
    "all",
    "prompt",
    "reasoning",
    "answer",
    "reasoning_answer",
)

DEFAULT_THINKING_START = 151667  # Qwen3 <think>
DEFAULT_THINKING_END = 151668    # Qwen3 </think>


# ---------------------------------------------------------------------------
# Layer utilities
# ---------------------------------------------------------------------------

def locate_layer_list(model: torch.nn.Module):
    """Find the transformer layer list on the model."""
    for path in POSSIBLE_LAYER_ATTRS:
        current = model
        for part in path.split("."):
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                break
        else:
            if hasattr(current, "__getitem__"):
                return current, path
    raise ValueError(
        "Could not locate transformer layers on the model. "
        f"Tried: {POSSIBLE_LAYER_ATTRS}. "
        "Add the model's layer attribute to POSSIBLE_LAYER_ATTRS in steering_primitives.py."
    )


def layer_name_to_index(layer_name: str) -> Optional[int]:
    """Convert 'layerXX' string to integer index."""
    if layer_name.startswith("layer"):
        try:
            return int(layer_name[len("layer"):])
        except ValueError:
            return None
    return None


def remove_hooks(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
    """Remove all registered forward hooks."""
    for handle in handles:
        handle.remove()
    handles.clear()


# ---------------------------------------------------------------------------
# Phase tracker
# ---------------------------------------------------------------------------

class ReasoningPhaseTracker(StoppingCriteria):
    """Detects end-of-thinking token and updates shared phase state.

    Supports batch_size >= 1. Each sample in the batch has its own phase
    tracked in state["phases"] (list). state["phase"] is kept as phases[0]
    for backward compatibility with single-sample code.
    """

    def __init__(
        self,
        state: Dict[str, Any],
        prompt_length: int,
        thinking_end_id: int,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.state = state
        self.prompt_length = prompt_length
        self.thinking_end_id = thinking_end_id
        self.batch_size = batch_size

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        generated_count = max(seq_len - self.prompt_length, 0)
        self.state["step_index"] = generated_count

        if seq_len <= self.prompt_length:
            self.state["phases"] = ["reasoning"] * batch_size
            self.state["phase"] = "reasoning"
            return False

        phases: List[str] = list(self.state.get("phases", ["reasoning"] * batch_size))
        if len(phases) != batch_size:
            phases = ["reasoning"] * batch_size

        for i in range(batch_size):
            # Once a sample enters "answer" phase it stays there
            if phases[i] == "reasoning":
                last_token = input_ids[i, -1].item()
                if last_token == self.thinking_end_id:
                    phases[i] = "answer"

        self.state["phases"] = phases
        self.state["phase"] = phases[0]  # backward compat
        return False


# ---------------------------------------------------------------------------
# Hook construction
# ---------------------------------------------------------------------------

def build_hook(
    steering_vector: torch.Tensor,
    coeff: float,
    apply_target: str,
    state: Dict[str, Any],
    max_tokens: Optional[int] = None,
    hook_id: str = "",
) -> callable:
    """Build a forward hook that adds a scaled steering vector to layer output.

    Supports batched generation: at each decode step (seq_len == 1), the vector
    is applied independently per row based on that sample's current phase tracked
    in state["phases"]. Falls back to state["phase"] (scalar) for single-sample
    backward compatibility.
    """

    steering_vector = steering_vector.clone()
    applied_count_key = f"applied_count_{hook_id}" if hook_id else "applied_count"

    def _phase_ok(phase: str) -> bool:
        if apply_target in ("all", "reasoning_answer"):
            return True
        if apply_target == "reasoning":
            return phase == "reasoning"
        if apply_target == "answer":
            return phase == "answer"
        return False

    def _max_tokens_ok(sample_idx: int) -> bool:
        if max_tokens is None:
            return True
        key = f"{applied_count_key}_s{sample_idx}"
        return state.get(key, 0) < max_tokens

    def _should_apply(phase: str, sample_idx: int) -> bool:
        return _phase_ok(phase) and _max_tokens_ok(sample_idx)

    # ── prompt-prefill apply (seq_len > 1, all rows same treatment) ──────────
    def apply_prefill(t: torch.Tensor) -> torch.Tensor:
        """Applied during prompt processing (seq_len > 1)."""
        vec = (coeff * steering_vector).to(device=t.device, dtype=t.dtype)
        if apply_target in ("all", "prompt"):
            return t + vec
        return t

    # ── decode-step apply (seq_len == 1, per-sample phase) ───────────────────
    def apply_decode(t: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Applied during token-by-token decoding (seq_len == 1).
        Returns (modified_tensor, any_applied)."""
        phases: List[str] = state.get("phases") or [state.get("phase", "reasoning")]
        vec = (coeff * steering_vector).to(device=t.device, dtype=t.dtype)
        result = t.clone()
        any_applied = False
        for i, phase in enumerate(phases):
            if i >= t.shape[0]:
                break
            if _should_apply(phase, i):
                result[i, -1, :] += vec
                any_applied = True
                count_key = f"{applied_count_key}_s{i}"
                state[count_key] = state.get(count_key, 0) + 1
        return result, any_applied

    def record_step(was_applied: bool) -> None:
        step_index = int(state.get("step_index") or 0)
        if state.get("last_logged_step") == step_index:
            return
        current_phase = (state.get("phases") or [state.get("phase", "reasoning")])[0]
        if current_phase == "answer":
            next_token_index = step_index + 1
            state["answer_token_count"] = state.get("answer_token_count", 0) + 1
            state.setdefault("answer_first_token_index", next_token_index)
            state["answer_last_token_index"] = next_token_index
            state["hook_calls_after_activation"] = (
                state.get("hook_calls_after_activation", 0) + 1
            )
        if was_applied:
            next_token_index = int(state.get("step_index") or 0) + 1
            state.setdefault("applied_tokens", []).append(
                {"token_index": next_token_index, "phase": current_phase}
            )
            state[applied_count_key] = state.get(applied_count_key, 0) + 1
        state["last_logged_step"] = step_index

    def _apply_tensor(t: torch.Tensor) -> Tuple[torch.Tensor, Optional[bool]]:
        """Return (modified, applied_flag_or_None). None = prefill, no record needed."""
        if t.dim() < 2:
            return t, None
        if t.shape[1] == 1:
            new_t, applied = apply_decode(t)
            return new_t, applied
        return apply_prefill(t), None

    def hook(module, inputs, output):
        if torch.is_tensor(output):
            new_out, flag = _apply_tensor(output)
            if flag is not None:
                record_step(flag)
            return new_out

        if isinstance(output, tuple) and output:
            head = output[0]
            if torch.is_tensor(head):
                new_head, flag = _apply_tensor(head)
                if flag is not None:
                    record_step(flag)
                return (new_head, *output[1:])
            return output

        if isinstance(output, list) and output:
            head = output[0]
            if torch.is_tensor(head):
                new_head, flag = _apply_tensor(head)
                if flag is not None:
                    record_step(flag)
                new_output = list(output)
                new_output[0] = new_head
                return type(output)(new_output)
            return output

        return output

    return hook


# ---------------------------------------------------------------------------
# Hook attachment
# ---------------------------------------------------------------------------

def attach_steering_hooks(
    model: torch.nn.Module,
    steering_vectors: Dict[str, torch.Tensor],
    layers: Iterable[str],
    coeff: float,
    apply_target: str,
    state: Dict[str, Any],
    max_tokens: Optional[int] = None,
    hook_id: str = "",
) -> List[torch.utils.hooks.RemovableHandle]:
    """Attach steering hooks to the specified transformer layers."""

    handles: List[torch.utils.hooks.RemovableHandle] = []
    layer_list, layer_path = locate_layer_list(model)
    param_dtype = next(model.parameters()).dtype
    param_device = next(model.parameters()).device

    for layer_name in layers:
        if layer_name == "embedding":
            print(f"[steering] Skipping unsupported layer '{layer_name}'")
            continue
        idx = layer_name_to_index(layer_name)
        if idx is None:
            print(f"[steering] Cannot parse layer name '{layer_name}', skipping.")
            continue
        if not (-len(layer_list) <= idx < len(layer_list)):
            print(
                f"[steering] Layer index {idx} out of range for {layer_path}, skipping."
            )
            continue
        vector = steering_vectors.get(layer_name)
        if vector is None:
            print(f"[steering] No vector for layer '{layer_name}', skipping.")
            continue
        vector = vector.to(device=param_device, dtype=param_dtype)
        module = layer_list[idx]
        handle = module.register_forward_hook(
            build_hook(vector, coeff, apply_target, state, max_tokens, hook_id)
        )
        handles.append(handle)
        max_tokens_info = f", max_tokens={max_tokens}" if max_tokens else ""
        print(
            f"[steering] Hooked {layer_path}[{idx}] ('{layer_name}'){max_tokens_info}"
        )
    return handles


# ---------------------------------------------------------------------------
# Token mask utilities (from utils.py)
# ---------------------------------------------------------------------------

def build_masks(
    total_length: int,
    prompt_length: int,
    output_ids: Iterable[int],
    thinking_start_id: int,
    thinking_end_id: int,
) -> Tuple[Dict[str, torch.Tensor], Optional[int], Optional[int]]:
    prompt_mask = torch.zeros(total_length, dtype=torch.bool)
    prompt_mask[:prompt_length] = True

    reasoning_mask = torch.zeros(total_length, dtype=torch.bool)
    output_ids_list = list(output_ids)
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None

    for idx, token_id in enumerate(output_ids_list):
        seq_idx = prompt_length + idx
        if token_id == thinking_start_id and start_idx is None:
            start_idx = seq_idx
        elif token_id == thinking_end_id and start_idx is not None:
            end_idx = seq_idx
            break

    if start_idx is not None:
        final_idx = end_idx if end_idx is not None else total_length - 1
        reasoning_mask[start_idx: final_idx + 1] = True

    answer_mask = torch.zeros(total_length, dtype=torch.bool)
    answer_mask[prompt_length:] = True
    answer_mask[reasoning_mask] = False

    masks = {
        "prompt_mask": prompt_mask,
        "reasoning_mask": reasoning_mask,
        "answer_mask": answer_mask,
    }
    return masks, start_idx, end_idx


def split_reasoning_and_answer(
    sequences: torch.Tensor,
    prompt_length: int,
    start_idx: Optional[int],
    end_idx: Optional[int],
    tokenizer,
) -> Tuple[str, str]:
    total_tokens = sequences.tolist()
    if start_idx is not None:
        reasoning_start = start_idx + 1
        reasoning_end = end_idx if end_idx is not None else len(total_tokens)
        reasoning_tokens = total_tokens[reasoning_start:reasoning_end]
        answer_start = (end_idx + 1) if end_idx is not None else len(total_tokens)
    else:
        reasoning_tokens = []
        answer_start = prompt_length
    answer_tokens = total_tokens[answer_start:]
    reasoning_text = tokenizer.decode(
        reasoning_tokens, skip_special_tokens=True
    ).strip()
    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    return reasoning_text, answer_text


def decode_token_texts(tokenizer, token_ids: torch.Tensor) -> List[str]:
    incremental_texts: List[str] = []
    prev_text = ""
    ids_list = token_ids.tolist()
    for end in range(1, len(ids_list) + 1):
        current_text = tokenizer.decode(
            ids_list[:end],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        incremental_texts.append(current_text[len(prev_text):])
        prev_text = current_text
    return incremental_texts


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def run_generation(
    model,
    tokenizer,
    base_inputs: Dict[str, torch.Tensor],
    generation_kwargs: Dict[str, Any],
    thinking_start_id: int,
    thinking_end_id: int,
    steering: Optional[Dict[str, Any]] = None,
) -> Any:
    """Run one generation pass with optional steering hooks.

    Supports batch_size >= 1. When batch_size == 1 returns a single result dict
    (backward compatible). When batch_size > 1 returns a list of result dicts.

    Args:
        model: HuggingFace AutoModelForCausalLM
        tokenizer: corresponding tokenizer
        base_inputs: tokenized inputs (input_ids, attention_mask), shape [B, seq_len]
        generation_kwargs: kwargs forwarded to model.generate()
        thinking_start_id: token ID for <think> (use -1 if not applicable)
        thinking_end_id: token ID for </think> (use -2 if not applicable)
        steering: dict with key "sets" → list of steering set dicts, or None

    Returns:
        Single result dict when batch_size == 1, list of result dicts otherwise.
        Each dict has: reasoning, answer, token_count_prompt, token_count_total,
        thinking_start_index, thinking_end_index, steering_stats (if steered).
    """
    device = next(model.parameters()).device
    model_inputs = {k: v.clone().to(device) for k, v in base_inputs.items()}
    batch_size = model_inputs["input_ids"].shape[0]
    padded_prompt_length = model_inputs["input_ids"].shape[1]

    state: Dict[str, Any] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []
    stopping_criteria = None

    if steering is not None:
        state["phases"] = ["reasoning"] * batch_size
        state["phase"] = "reasoning"
        state["hook_calls_after_activation"] = 0
        state["answer_token_count"] = 0
        steering_sets = steering.get("sets") or []
        for set_idx, set_spec in enumerate(steering_sets):
            set_handles = attach_steering_hooks(
                model,
                set_spec["vectors"],
                set_spec["layers"],
                coeff=set_spec["coef"],
                apply_target=set_spec["apply_target"],
                state=state,
                max_tokens=set_spec.get("max_tokens"),
                hook_id=f"set{set_idx}",
            )
            handles.extend(set_handles)
        if not handles:
            raise RuntimeError(
                "No steering hooks were installed. "
                "Check that the specified layers exist in the steering vectors file."
            )
        stopping_criteria = StoppingCriteriaList(
            [
                ReasoningPhaseTracker(
                    state,
                    prompt_length=padded_prompt_length,
                    thinking_end_id=thinking_end_id,
                    batch_size=batch_size,
                )
            ]
        )

    gen_kwargs = dict(generation_kwargs)
    gen_kwargs["return_dict_in_generate"] = True

    try:
        with torch.no_grad():
            generated = model.generate(
                **model_inputs,
                stopping_criteria=stopping_criteria,
                **gen_kwargs,
            )
    finally:
        if handles:
            remove_hooks(handles)

    # attention_mask tells us the actual (non-padded) prompt length per sample
    attention_mask = model_inputs.get("attention_mask")

    results: List[Dict[str, Any]] = []
    for i in range(batch_size):
        seq = generated.sequences[i].detach().cpu()
        seq = seq[:-1]  # strip final EOS / <|im_end|>
        total_length = seq.shape[0]

        # Actual prompt token count (excludes left-padding)
        if attention_mask is not None:
            actual_prompt_length = int(attention_mask[i].sum().item())
        else:
            actual_prompt_length = padded_prompt_length

        output_ids = seq[padded_prompt_length:]

        masks, reasoning_start, reasoning_end = build_masks(
            total_length,
            padded_prompt_length,
            output_ids.tolist(),
            thinking_start_id,
            thinking_end_id,
        )

        reasoning_text, answer_text = split_reasoning_and_answer(
            seq,
            padded_prompt_length,
            reasoning_start,
            reasoning_end,
            tokenizer,
        )

        result: Dict[str, Any] = {
            "reasoning": reasoning_text,
            "answer": answer_text,
            "token_count_prompt": actual_prompt_length,
            "token_count_total": actual_prompt_length + len(output_ids),
            "thinking_start_index": reasoning_start,
            "thinking_end_index": reasoning_end,
        }

        if steering is not None:
            raw_applied = state.get("applied_tokens", [])
            result["steering_stats"] = {
                "answer_token_count": int(state.get("answer_token_count") or 0),
                "hook_calls_after_activation": int(
                    state.get("hook_calls_after_activation") or 0
                ),
                "applied_tokens_count": len(raw_applied),
            }

        results.append(result)

    return results[0] if batch_size == 1 else results
