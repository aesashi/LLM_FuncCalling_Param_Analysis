"""qwen_fc_steering.py

Steering-enabled handler for Qwen FC models.
Combines SteeringMixin (HF direct inference + hooks) with QwenFCHandler
(Qwen prompt formatting and tool-call decoding).
"""

from bfcl_eval.model_handler.local_inference.steering_mixin import SteeringMixin
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler


class QwenFCSteeringHandler(SteeringMixin, QwenFCHandler):
    """Qwen FC handler with activation steering support.

    Pass --steering-config /path/to/steering.yaml to bfcl generate to enable
    steered inference. Without the flag, runs standard vLLM inference.
    """
    pass
