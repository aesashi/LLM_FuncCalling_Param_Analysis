"""qwen_steering.py

Steering-enabled handler for Qwen prompt-mode models.
Combines SteeringMixin (HF direct inference + hooks) with QwenHandler
(Qwen prompt formatting and tool-call decoding).
"""

from bfcl_eval.model_handler.local_inference.steering_mixin import SteeringMixin
from bfcl_eval.model_handler.local_inference.qwen import QwenHandler


class QwenSteeringHandler(SteeringMixin, QwenHandler):
    """Qwen prompt-mode handler with activation steering support.

    Pass --steering-config /path/to/steering.yaml to bfcl generate to enable
    steered inference. Without the flag, runs standard vLLM inference.
    """
    pass
