""" CrystalCoder configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class CrystalCoderConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`CrystalCoderModel`]. It is used to instantiate a CrystalCoder
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the CrystalCoder model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`CrystalCoderModel`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new", "swiglu"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.
        position_embedding_type (`str`, *optional*, defaults to `"learned"`):
            Positional embedding can be either `"alibi"`, `"learned"`, or `"learned"`.
        rotary_dim (`int`, *optional*, defaults to `n_embd / n_head`):
            The dimension along which to apply rope.
        mup_width_scale (`float`, *optional*, defaults to 1.0):
            muP parameter to scale learning rate and initializers. Calculated as (`d_model,0 / d_model`), where
            `d_model` is the model's width and `d_model,0` is the proxy model's width.
        mup_embeddings_scale (`float`, *optional*, defaults to 1.0):
            muP parameter to scale token and position embeddings.
        mup_output_alpha (`float`, *optional*, defaults to 1.0):
            muP parameter to scale output logits (`output_logits_scale = mup_output_alpha * mup_width_scale`).
        mup_scale_qk_dot_by_d (`bool`, *optional*, defaults to `False`):
            Scale attention weights by dividing by hidden_size instead of sqrt(hidden_size). Need to set
            scale_attn_weights to `True` as well.

    Example:

    ```python
    >>> from transformers import CrystalCoderConfig, CrystalCoderModel

    >>> # Initializing a CrystalCoder configuration
    >>> configuration = CrystalCoderConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = CrystalCoderModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "crystalcoder"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=32032,
        n_positions=2048,
        n_embd=4096,
        n_layer=32,
        n_head=32,
        n_inner=None,
        activation_function="swiglu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        position_embedding_type="rotary",
        rotary_dim=None,
        mup_width_scale=1.0,
        mup_embeddings_scale=1.0,
        mup_output_alpha=1.0,
        mup_scale_qk_dot_by_d=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.position_embedding_type = position_embedding_type
        self.rotary_dim = rotary_dim
        self.mup_width_scale = mup_width_scale
        self.mup_embeddings_scale = mup_embeddings_scale
        self.mup_output_alpha = mup_output_alpha
        self.mup_scale_qk_dot_by_d = mup_scale_qk_dot_by_d

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
