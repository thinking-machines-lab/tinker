"""
Mock API server for the Tinker Python SDK.

This server provides mock implementations of all Tinker API endpoints for testing purposes.
"""

import random
import uuid
import traceback
import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

# Import types from tinker
from tinker import types

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tinker Mock API Server")

# Custom exception handler to log stack traces
@app.exception_handler(Exception)
async def log_exceptions(request: Request, exc: Exception):
    """Log all exceptions with full stack traces."""
    logger.error(f"Unhandled exception in {request.method} {request.url}")
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Exception message: {str(exc)}")
    logger.error(f"Full traceback:\n{traceback.format_exc()}")

    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# Handler for validation errors (422)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log validation errors with details."""
    logger.error(f"Validation error in {request.method} {request.url}")
    logger.error(f"Validation errors: {exc.errors()}")
    logger.error(f"Request body: {exc.body}")
    logger.error(f"Full traceback:\n{traceback.format_exc()}")

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body,
            "debug_info": {
                "url": str(request.url),
                "method": request.method,
                "errors": exc.errors()
            }
        }
    )

# Handler for Pydantic validation errors
@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """Log Pydantic validation errors with details."""
    logger.error(f"Pydantic validation error in {request.method} {request.url}")
    logger.error(f"Validation errors: {exc.errors()}")
    logger.error(f"Full traceback:\n{traceback.format_exc()}")

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "debug_info": {
                "url": str(request.url),
                "method": request.method,
                "errors": exc.errors()
            }
        }
    )
# In-memory storage for futures and their results
futures_store: Dict[str, Any] = {}

# In-memory storage for LoRA adapters
lora_adapters: Dict[str, Dict[str, Any]] = {}

# Mock model configurations
SUPPORTED_MODELS = [
    {"model_id": "llama-3-8b", "model_name": "meta-llama/Meta-Llama-3-8B", "arch": "llama"},
    {"model_id": "llama-3-70b", "model_name": "meta-llama/Meta-Llama-3-70B", "arch": "llama"},
    {"model_id": "qwen2-72b", "model_name": "Qwen/Qwen2-72B", "arch": "qwen2"},
]


def generate_future_id() -> str:
    """Generate a unique future ID."""
    return f"future_{uuid.uuid4().hex[:8]}"


@app.get("/healthz", response_model=types.HealthResponse)
async def health_check():
    """Health check endpoint."""
    return types.HealthResponse(status="ok")


@app.get("/get_server_capabilities", response_model=types.GetServerCapabilitiesResponse)
async def get_server_capabilities():
    """Get server capabilities including supported models."""
    supported_models = [
        {"model_id": model["model_id"], "model_name": model["model_name"], "arch": model["arch"]}
        for model in SUPPORTED_MODELS
    ]
    return types.GetServerCapabilitiesResponse(supported_models=supported_models)

def generate_mock_logprobs(seq_len: int) -> List[float]:
    """Generate mock log probabilities for a sequence."""
    return [random.uniform(-4.0, 0.0) for _ in range(seq_len)]

def generate_mock_loss() -> float:
    """Generate mock loss."""
    return random.uniform(0.5, 3.0)


def chunk_length(chunk: types.ModelInputChunkParam) -> int:
    match chunk["type"]:
        case "encoded_text":
            return len(chunk["tokens"])
        case "image_asset_pointer":
            return chunk["tokens"]
        case _:
            raise ValueError(f"Unknown chunk type: {chunk['type']}")

def sequence_length(model_input: types.ModelInputParam) -> int:
    return sum(chunk_length(chunk) for chunk in model_input["chunks"])


@app.post("/fwd", response_model=types.UntypedAPIFuture)
async def forward(params: types.TrainingForwardParams):
    """Perform forward pass."""
    future_id = generate_future_id()

    result = types.FwdBwdOutput(
        loss_fn_outputs={
            "logprobs": [generate_mock_logprobs(sequence_length(datum.input_sequence)) for datum in params.fwdbwd_input.data]
        },
        metrics={
            "loss": generate_mock_loss(),
            "perplexity": generate_mock_loss(),
        }
    )

    # Store the result for future retrieval
    futures_store[future_id] = {
        "result": result,
        "status": "completed",
        "created_at": datetime.now().isoformat(),
    }

    return types.UntypedAPIFuture(request_id=future_id, model_id=params.get("model_id"))


@app.post("/fwdbwd", response_model=types.UntypedAPIFuture)
async def forward_backward(params: types.TrainingForwardBackwardParams):
    """Perform forward and backward pass."""
    # Since the mock implementation is identical, we can reuse the forward logic
    # In a real implementation, forward_backward would also compute gradients
    return await forward(params)


@app.post("/optim_step", response_model=types.UntypedAPIFuture)
async def optim_step(params: types.TrainingOptimStepParams):
    """Perform optimization step."""
    future_id = generate_future_id()

    # Mock optimization step result (OptimStepResponse is just a Dict[str, Union[float, str]])
    result = {
        "grad_norm": random.uniform(0.1, 10.0),
        "weight_norm": random.uniform(10.0, 100.0),
        "update_norm": random.uniform(0.001, 0.1),
    }

    # Store the result for future retrieval
    futures_store[future_id] = {
        "result": result,
        "status": "completed",
        "created_at": datetime.now().isoformat(),
    }

    return UntypedAPIFuture(request_id=future_id, model_id=params.get("model_id"))

@app.post("/retrieve_future")
async def retrieve_future(params: types.FutureRetrieveParams):
    """Retrieve the result of a future."""
    future_id = params["request_id"]

    if future_id not in futures_store:
        raise HTTPException(status_code=404, detail=f"Future {future_id} not found")

    future_data = futures_store[future_id]
    result = future_data["result"]

    # Handle different result types explicitly
    if isinstance(result, (types.FwdBwdOutput, types.AddLoraResponse, types.UnloadModelResponse,
                          types.LoadWeightsResponse, types.SaveWeightsResponse,
                          types.SaveWeightsForSamplerResponse)):
        serialized_result = result.model_dump()
        print(f"RETRIEVE_FUTURE: Returning Pydantic model result: {serialized_result}")
        return serialized_result
    else:
        # For dict results (like OptimStepResponse)
        print(f"RETRIEVE_FUTURE: Returning dict result: {result}")
        return result

@app.post("/add_lora", response_model=types.UntypedAPIFuture)
async def add_lora(params: types.LoraAddParams):
    """Add a LoRA adapter to the model."""
    future_id = generate_future_id()

    # Generate new model_id with LoRA
    base_model = params["base_model"]
    lora_model_id = f"{base_model}_lora_{uuid.uuid4().hex[:8]}"

    # Store LoRA configuration
    if base_model not in lora_adapters:
        lora_adapters[base_model] = {}

    lora_adapters[base_model][lora_model_id] = {
        "rank": params.get("rank", 16),
        "alpha": params.get("alpha", 32),
        "created_at": datetime.now().isoformat()
    }

    # Create the result that will be retrieved later
    result = types.AddLoraResponse(model_id=lora_model_id)

    # Store the result for future retrieval
    futures_store[future_id] = {
        "result": result,
        "status": "completed",
        "created_at": datetime.now().isoformat()
    }

    return types.UntypedAPIFuture(
        request_id=future_id,
        model_id=lora_model_id
    )


@app.post("/remove_lora", response_model=types.UntypedAPIFuture)
async def remove_lora(params: types.LoraRemoveParams):
    """Remove a LoRA adapter from the model."""
    future_id = generate_future_id()

    model_id = params["model_id"]

    # Check if this is a LoRA model
    assert "_lora_" in model_id, f"Model {model_id} is not a LoRA model"

    # Remove from our tracking
    base_model_id = model_id.split("_lora_")[0]
    if base_model_id in lora_adapters and model_id in lora_adapters[base_model_id]:
        del lora_adapters[base_model_id][model_id]


    # Store the result for future retrieval
    futures_store[future_id] = {
        "result": result,
        "status": "completed",
        "created_at": datetime.now().isoformat()
    }

    return types.UntypedAPIFuture(
        request_id=future_id,
        model_id=params.get("model_id")
    )


@app.post("/load_weights", response_model=types.UntypedAPIFuture)
async def load_weights(params: types.LoadWeightsRequest):
    """Load model weights from a path."""
    future_id = generate_future_id()

    # Mock implementation - in reality this would load weights from storage
    result = LoadWeightsResponse(message=f"Weights loaded from {params.path}", success=True)

    # Store the result for future retrieval
    futures_store[future_id] = {
        "result": result,
        "status": "completed",
        "created_at": datetime.now().isoformat(),
    }

    return UntypedAPIFuture(request_id=future_id, model_id=params.model_id)


@app.post("/save_weights", response_model=types.UntypedAPIFuture)
async def save_weights(params: types.SaveWeightsRequest):
    """Save model weights to a path."""
    future_id = generate_future_id()

    # Mock implementation - in reality this would save weights to storage
    save_path = (
        f"{params.path or '/tmp'}/checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    result = SaveWeightsResponse(message=f"Weights saved to {save_path}", success=True)

    # Store the result for future retrieval
    futures_store[future_id] = {
        "result": result,
        "status": "completed",
        "created_at": datetime.now().isoformat(),
    }

    return UntypedAPIFuture(request_id=future_id, model_id=params.model_id)


@app.post("/save_weights_for_sampler", response_model=types.UntypedAPIFuture)
async def save_weights_for_sampler(params: types.SaveWeightsForSamplerRequest):
    """Save weights in a format suitable for the sampler."""
    future_id = generate_future_id()

    # Mock implementation
    save_path = (
        f"{params.path or '/tmp'}/sampler_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    result = types.SaveWeightsForSamplerResponse(
        message=f"Sampler weights saved to {save_path}", success=True
    )

    # Store the result for future retrieval
    futures_store[future_id] = {
        "result": result,
        "status": "completed",
        "created_at": datetime.now().isoformat(),
    }

    return types.UntypedAPIFuture(request_id=future_id, model_id=params.model_id)


@app.post("/get_info", response_model=types.GetInfoResponse)
async def get_info(params: types.ModelGetInfoParams):
    """Get information about a model."""
    model_id = params["model_id"]

    # Find the model in our supported models or check if it's a LoRA model
    model_info = None
    if "_lora_" in model_id:
        # Extract base model ID from LoRA model
        base_model_id = model_id.split("_lora_")[0]
        model_info = next((m for m in SUPPORTED_MODELS if m["model_id"] == base_model_id), None)
    else:
        model_info = next((m for m in SUPPORTED_MODELS if m["model_id"] == model_id), None)

    if not model_info:
        # Default model info for unknown models
        model_info = {"model_name": f"unknown/{model_id}", "arch": "unknown"}

    return types.GetInfoResponse(
        model_data={"model_name": model_info["model_name"], "arch": model_info["arch"]},
        model_id=model_id,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
