"""mistral_fc_steering.py

Steering-enabled handler for Mistral FC models.
Combines SteeringMixin (HF direct inference + hooks) with MistralFCHandler
(Mistral prompt formatting and tool-call decoding).
"""

from bfcl_eval.model_handler.local_inference.steering_mixin import SteeringMixin
from bfcl_eval.model_handler.local_inference.mistral_fc import MistralFCHandler


class MistralFCSteeringHandler(SteeringMixin, MistralFCHandler):
    """Mistral FC handler with activation steering support.

    Pass --steering-config /path/to/steering.yaml to bfcl generate to enable
    steered inference. Without the flag, runs standard vLLM inference.

    Note: Mistral models do not have thinking tokens. Use apply_target='all'
    or 'answer' in your steering config.
    """
    pass
