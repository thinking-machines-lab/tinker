"""Fixtures for the live SDK suite: nothing here is mocked."""

from __future__ import annotations

import json
import os
import time
from functools import cache
from typing import NamedTuple

import httpx
import pytest

import tinker


class Env(NamedTuple):
    base_url: str
    key_var: str
    models: tuple[str, ...]
    behind_cloudflare: bool = False
    # The org rejects a static key, so the SDK insists on a credential command.
    credential_cmd: bool = False


# No staging: its org refuses API keys, and the JWT exchange too.
ENVS = {
    "prod": Env(
        base_url="https://tinker.thinkingmachines.dev/services/tinker-prod",
        key_var="TINKER_API_KEY_PROD",
        # Capabilities expose no model size, so the cheap default is a choice.
        models=("Qwen/Qwen3.5-4B",),
    ),
    "intern": Env(
        base_url="https://tinker.thinkingmachines.dev/services/tinker-intern",
        key_var="TINKER_API_KEY_INTERN",
        behind_cloudflare=True,
        # Intern's others cannot sample, or need a tokenizer the wheel lacks.
        models=("Qwen/Qwen3.5-9B",),
        credential_cmd=True,
    ),
}

DEFAULT_ENV = "prod"

# Tags what the suite leaves behind; the SDK cannot create a project.
RUN_METADATA = {"source": "tinker-sdk-e2e"}

LORA_RANK = 8


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: consumes real training or sampling capacity")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--models",
        default=os.environ.get("TINKER_E2E_MODELS", ""),
        help="'all' for every supported model, or a comma separated list. "
        "Defaults to the cheap model set.",
    )


def resolve_env() -> Env:
    name = os.environ.get("TINKER_E2E_ENV", DEFAULT_ENV).strip() or DEFAULT_ENV
    assert name in ENVS, f"unknown TINKER_E2E_ENV {name!r}, expected one of {[*ENVS]}"
    return ENVS[name]


@cache
def _service_client() -> tinker.ServiceClient:
    env = resolve_env()
    key = os.environ.get(env.key_var) or os.environ.get("TINKER_API_KEY", "")
    assert key, f"set {env.key_var} or TINKER_API_KEY"

    if env.credential_cmd:
        # Feed it our own key, or a developer's scontrol token stands in silently.
        os.environ["TINKER_E2E_CREDENTIAL"] = key
        os.environ["TINKER_CREDENTIAL_CMD"] = 'printf %s "$TINKER_E2E_CREDENTIAL"'

    return tinker.ServiceClient(
        api_key=key, base_url=env.base_url, default_headers=_cloudflare_headers(env)
    )


def _cloudflare_headers(env: Env) -> dict[str, str]:
    if not env.behind_cloudflare:
        return {}
    cf_id = os.environ.get("CF_ACCESS_CLIENT_ID", "")
    cf_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
    # Without these Cloudflare redirects, and the parse error hides why.
    assert cf_id and cf_secret, "set CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET"
    return {"CF-Access-Client-Id": cf_id, "CF-Access-Client-Secret": cf_secret}


@pytest.fixture(scope="session")
def service_client() -> tinker.ServiceClient:
    return _service_client()


@pytest.fixture(scope="session")
def rest_client(service_client: tinker.ServiceClient):
    return service_client.create_rest_client()


@pytest.fixture(scope="session")
def training_clients(service_client: tinker.ServiceClient):
    """One client per model: training runs cannot be deleted, so create few."""

    @cache
    def get(model: str):
        return service_client.create_lora_training_client(
            base_model=model, rank=LORA_RANK, user_metadata=RUN_METADATA
        )

    return get


@pytest.fixture(scope="session")
def forward_only_clients(service_client: tinker.ServiceClient):
    """For `forward` only: a zero init LoRA stands in for the base weights."""

    @cache
    def get(model: str):
        return service_client.create_lora_training_client(
            base_model=model, rank=LORA_RANK, user_metadata=RUN_METADATA
        )

    return get


@pytest.fixture
def new_training_client(service_client: tinker.ServiceClient):
    """A run of its own, for tests that move the weights a long way."""

    def make(model: str):
        return service_client.create_lora_training_client(
            base_model=model, rank=LORA_RANK, user_metadata=RUN_METADATA
        )

    return make


@pytest.fixture(scope="session")
def tokenizers(service_client: tinker.ServiceClient):
    """One tokenizer per model; each is a HuggingFace download."""

    @cache
    def get(model: str):
        return service_client.create_sampling_client(base_model=model).get_tokenizer()

    return get


@pytest.fixture(scope="session")
def oai_client():
    """The OpenAI compatible surface, which the SDK does not wrap."""
    env = resolve_env()
    key = os.environ.get(env.key_var) or os.environ.get("TINKER_API_KEY", "")
    assert key, f"set {env.key_var} or TINKER_API_KEY"
    headers = {"Authorization": f"Bearer {key}"} | _cloudflare_headers(env)
    with httpx.Client(base_url=f"{env.base_url}/oai/api/v1", headers=headers, timeout=120) as c:
        yield c


@cache
def _supported_models() -> tuple[str, ...]:
    """Read once: collection and the fixtures both need it."""
    capabilities = _service_client().get_server_capabilities()
    return tuple(m.model_name for m in capabilities.supported_models if m.model_name)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Any test taking a `model` argument runs once per selected model."""
    if "model" not in metafunc.fixturenames:
        return

    selection = metafunc.config.getoption("--models").strip()
    available = _supported_models()

    if selection == "all":
        models = available
    else:
        models = list(dict.fromkeys(m.strip() for m in selection.split(",") if m.strip()))
        models = models or list(resolve_env().models)
        missing = [m for m in models if m not in available]
        assert not missing, f"models not offered by this server: {missing}"

    metafunc.parametrize("model", models)


_SESSION_START = 0.0


def pytest_sessionstart(session: pytest.Session) -> None:
    global _SESSION_START
    _SESSION_START = time.time()


def pytest_terminal_summary(terminalreporter) -> None:
    """Counts for the Slack card, shaped like Playwright's."""
    path = os.environ.get("TINKER_E2E_RESULTS_JSON")
    if not path:
        return
    stats = terminalreporter.stats
    payload = {
        "stats": {
            "expected": len(stats.get("passed", [])),
            "unexpected": len(stats.get("failed", [])) + len(stats.get("error", [])),
            "flaky": 0,
            "skipped": len(stats.get("skipped", [])),
            "duration": (time.time() - _SESSION_START) * 1000,
        }
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle)
