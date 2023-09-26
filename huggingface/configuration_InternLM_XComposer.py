from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

INTERNLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class InternLMXComposerConfig(PretrainedConfig):

    model_type = "InternLMXComposer"
    _auto_class = "AutoConfig"

    def __init__(
        self,
        vocab_size=103168,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=-1,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        bias=True,
        num_query_token=32,
        num_quant=32,
        intern_converted_llm=True,
        kqvo_bias=True,
        device='cuda',
        internlm_lora=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.bias = bias
        self.num_query_token = num_query_token
        self.num_quant = num_quant
        self.internlm_lora = internlm_lora
        self.kqvo_bias = kqvo_bias
        self.intern_converted_llm = intern_converted_llm
        self.device = device
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
