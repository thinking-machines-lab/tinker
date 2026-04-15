<h1 align="center">Tinker Python SDK</h1>
<div align="center">
  <img src="https://github.com/thinking-machines-lab/tinker/blob/829c151ba7c6740a84db02e41f03825084c37fed/docs/images/logo.png" width="60%" />

  Documentation:
  <a href="http://tinker-docs.thinkingmachines.ai/">tinker-docs.thinkingmachines.ai</a>
</div>

## Installation

```bash
# PyPI
pip install tinker

# uv (recommended)
uv add tinker
```

## Quickstart

### CLI

```bash
# Train a model
tinker train --name my-model --training-file training_data.jsonl

# List your models
tinker models list

# Get model details
tinker models get <model_id>
```

### Python SDK

```python
from tinker import Tinker

# Initialize client
client = Tinker(api_key="your-api-key")

# Create a training run
run = client.training.create(
    name="my-model",
    training_file="training_data.jsonl",
)
print(f"Training started: {run.id}")
```

### Using the API

```bash
# Set API key
export TINKER_API_KEY="your-api-key"

# Or use with Python
from tinker import Tinker
client = Tinker()  # Automatically reads TINKER_API_KEY
```

## Documentation

Full documentation available at [tinker-docs.thinkingmachines.ai](http://tinker-docs.thinkingmachines.ai/)

## License

[Apache-2.0](LICENSE)