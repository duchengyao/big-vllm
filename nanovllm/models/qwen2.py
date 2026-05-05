from transformers import Qwen2Config

from nanovllm.models.qwen3 import Qwen3ForCausalLM


class Qwen2ForCausalLM(Qwen3ForCausalLM):

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__(config)
