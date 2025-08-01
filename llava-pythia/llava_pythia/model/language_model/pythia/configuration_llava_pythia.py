import os
from typing import Union
from transformers import PretrainedConfig, GPTNeoXConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LlavaPythiaVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPVisionModel`]. It is used to instantiate a
    CLIP vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        mm_vision_select_feature (`str`, *optional*, defaults to `"patch"`):
            The feature to select from the vision encoder output. Can be one of `"patch"` or `"cls_patch"`.
        mm_vision_select_layer (`int`, *optional*, defaults to `-2`):
            The layer to select from the vision encoder output.

    Example:

    ```python
    >>> from transformers import CLIPVisionConfig, CLIPVisionModel

    >>> # Initializing a CLIPVisionConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPVisionConfig()

    >>> # Initializing a CLIPVisionModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llava_pythia_clip_vision_model"

    def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            projection_dim=512,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=32,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            mm_vision_select_feature="patch",
            mm_vision_select_layer=-2,
            vision_model_name_or_path="clip",
            concat="None",
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_vision_select_layer = mm_vision_select_layer
        self.vision_model_name_or_path = vision_model_name_or_path
        self.concat = concat

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "llava_pythia":
            config_dict = config_dict["vision_config"]["vision_tower"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ProjectorConfig(PretrainedConfig):
    model_type = "llava_pythia_projector"

    def __init__(
            self,
            mm_projector_type="linear",
            mm_hidden_size=768,
            hidden_size=2560,
            **kwargs
    ):
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "llava_pythia":
            config_dict = config_dict["vision_config"]["mm_projector"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)



from typing import List

# for initialize act head 

DEFAULT_VISUAL_CONFIG = {
    "vision_tower": LlavaPythiaVisionConfig().to_dict(),
    "mm_projector": ProjectorConfig().to_dict(),
}

# print(DEFAULT_ACT_CONFIG['act'])

class LlavaPythiaConfig(GPTNeoXConfig):
    model_type = "llava_pythia"

    def __init__(self, vision_config=None, **kwargs):
        if vision_config is None:
            self.vision_config = DEFAULT_VISUAL_CONFIG
        else:
            self.vision_config = vision_config
            
        # Action head configurations
        self.action_head = kwargs.pop("action_head", "act")
        self.action_dim = kwargs.pop("action_dim", 7)  # Default to 7 for Franka arm
        self.state_dim = kwargs.pop("state_dim", 14)   # Default state dimension
        self.chunk_size = kwargs.pop("chunk_size", 6)  # Default chunk size for diffusion
        
        # ACT head specific config
        act_config = kwargs.pop("act", {})
        self.act = {
            "act": {
                "hidden_dim": act_config.get("hidden_dim", 256),
                "ff_feedforward_dim": act_config.get("ff_feedforward_dim", 256 * 4),
                "num_layers": act_config.get("num_layers", 4),
                "num_queries": act_config.get("num_queries", 6),
                "kl_weight": act_config.get("kl_weight", 10),
                "hidden_dropout": act_config.get("hidden_dropout", 0.1),
                "attention_dropout": act_config.get("attention_dropout", 0.1),
                "dropout": act_config.get("dropout", 0.1),
                "num_heads": act_config.get("num_heads", 8),
                "nheads": act_config.get("num_heads", 8),  # Alias for nheads
                "activation_dropout": act_config.get("activation_dropout", 0.0),
                "backbone": "resnet18",  # Default backbone
                "enc_layers": 4,  # Number of encoder layers
                "dec_layers": 4,  # Number of decoder layers
                "dim_feedforward": 2048,  # Dimension of feedforward layers
                "pre_norm": False,  # Whether to use pre-norm
                "vq_class": 16,  # Number of VQ classes
                "vq_dim": 256,  # Dimension of VQ embeddings
                "vq_commit": 0.25,  # VQ commit loss weight
                "shared_codebook": True,  # Whether to share codebook across layers
                "shared_encoder": True,  # Whether to share encoder across layers
                "vq_kl_weight": 0.0,  # KL weight for VQ
                "vq_decay": 0.99,  # Decay for VQ
                "vq_ema": True,  # Whether to use EMA for VQ
                "vq_ema_decay": 0.99,  # EMA decay for VQ
                "vq_ema_eps": 1e-5,  # Epsilon for EMA
                "chunk_size": 6,  # Number of action chunks
                "horizon": 6,  # Planning horizon
                "num_queries": 6,  # Number of action queries
                "num_queries_per_policy": 1,  # Queries per policy
                "num_policies": 1,  # Number of policies
                "temperature": 1.0,  # Temperature for sampling
                "use_goal": False,  # Whether to use goal conditioning
                "goal_conditioned_training": False,  # Whether to use goal conditioning during training
                "use_state": True,  # Whether to use state conditioning
                "state_dim": 14,  # Dimension of state space
                "action_dim": 7,  # Dimension of action space
                "hidden_dim": 256,  # Hidden dimension
                "n_embd": 256,  # Embedding dimension
                "n_layer": 4,  # Number of layers
                "n_head": 8,  # Number of attention heads
                "n_inner": 2048,  # Inner dimension of feedforward layers
                "activation_function": "gelu_new",  # Activation function
                "n_positions": 2048,  # Maximum sequence length
                "resid_pdrop": 0.1,  # Dropout probability for residual connections
                "embd_pdrop": 0.1,  # Dropout probability for embeddings
                "attn_pdrop": 0.1,  # Dropout probability for attention
                "layer_norm_epsilon": 1e-5,  # Epsilon for layer normalization
                "initializer_range": 0.02,  # Range for parameter initialization
                "scale_attn_weights": True,  # Whether to scale attention weights
                "use_cache": True,  # Whether to use caching for generation
                "bos_token_id": 0,  # Beginning of sequence token ID
                "eos_token_id": 2,  # End of sequence token ID
                "tie_word_embeddings": False,  # Whether to tie input and output embeddings
                "camera_names": ["agentview"],  # List of camera names
                "camera_feature_dim": 512,  # Dimension of camera features
                "camera_num_layers": 2,  # Number of layers for camera feature processing
                "camera_num_heads": 8,  # Number of attention heads for camera features
                "camera_dropout": 0.1,  # Dropout for camera feature processing
                "use_camera_embeddings": True,  # Whether to use camera embeddings
                "camera_embedding_dim": 64,  # Dimension of camera embeddings
                "camera_embedding_type": "learned",  # Type of camera embeddings
                "use_camera_transformer": True,  # Whether to use transformer for camera features
                "camera_transformer_layers": 2,  # Number of transformer layers for camera features
                "camera_transformer_heads": 8,  # Number of attention heads for camera transformer
                "camera_transformer_dim_feedforward": 1024,  # Feedforward dimension for camera transformer
            }
        }
        
        # Diffusion model config
        self.diffusion = kwargs.pop("diffusion", {
            "num_train_timesteps": 100,
            "beta_schedule": "squaredcos_cap_v2",
            "clip_sample": True,
            "set_alpha_to_one": True,
            "steps_offset": 0,
            "prediction_type": "epsilon"
        })
        
        self.concat = "None"
        super().__init__(**kwargs)
        
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = super().to_dict()
        output.update({
            "action_head": self.action_head,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "chunk_size": self.chunk_size,
            "act": self.act,
            "diffusion": self.diffusion,
        })
        return output
        


if __name__ == "__main__":
    print(LlavaPythiaVisionConfig())
