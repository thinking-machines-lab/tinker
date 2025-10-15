from typing_extensions import Literal, TypeAlias

__all__ = ["LossFnType"]

LossFnType: TypeAlias = Literal["cross_entropy", "importance_sampling", "ppo"]
