from .._models import BaseModel

__all__ = ["GetSamplerResponse"]


class GetSamplerResponse(BaseModel):
    # The sampler ID (sampling_session_id)
    sampler_id: str

    # The base model name
    base_model: str

    # Optional model path
    model_path: str | None = None
