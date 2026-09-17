from .._models import BaseModel
from .optimizer import AdamOptimizerConfig, OptimizerConfig

__all__ = ["WeightsInfoResponse"]


class WeightsInfoResponse(BaseModel):
    """Minimal information for loading public checkpoints."""

    base_model: str
    optimizer_config: OptimizerConfig = AdamOptimizerConfig()

    is_lora: bool

    lora_rank: int | None = None

    train_unembed: bool | None = None

    train_mlp: bool | None = None

    train_attn: bool | None = None
