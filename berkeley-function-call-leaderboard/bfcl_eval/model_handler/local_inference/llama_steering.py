"""llama_steering.py

Steering-enabled handler for Llama 3.1+ FC models.
Combines SteeringMixin (HF direct inference + hooks) with LlamaHandler_3_1
(Llama prompt formatting and tool-call decoding).
"""

from bfcl_eval.model_handler.local_inference.steering_mixin import SteeringMixin
from bfcl_eval.model_handler.local_inference.llama_3_1 import LlamaHandler_3_1


class LlamaSteeringHandler(SteeringMixin, LlamaHandler_3_1):
    """Llama 3.1+ FC handler with activation steering support.

    Pass --steering-config /path/to/steering.yaml to bfcl generate to enable
    steered inference. Without the flag, runs standard vLLM inference.

    Note: Llama models do not have thinking tokens. Use apply_target='all'
    or 'answer' in your steering config.
    """
    pass
