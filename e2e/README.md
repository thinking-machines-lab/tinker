# Tinker SDK e2e

Live tests against a deployed Tinker. Nothing is mocked, so a failure means the service,
the credential or the built package is broken.

`TINKER_E2E_ENV` selects `prod` (default) or `intern`. Each carries its own base url,
credential and model set: the catalogs barely overlap. No staging, whose org refuses API
keys entirely.

## Run

```bash
uv venv /tmp/sdk-e2e && VIRTUAL_ENV=/tmp/sdk-e2e uv pip install ./public/tinker-python pytest
/tmp/sdk-e2e/bin/python -m pytest public/tinker-python/e2e -o addopts= -o filterwarnings=
```

Installed from a build rather than run against the checkout, because that is what a user
gets. The two `-o` flags clear the SDK's own pytest settings: `-n auto` would give each
xdist worker its own training run, and its `filterwarnings` make warnings fatal.

- `-m "not gpu"` runs only the free checks.
- `--models=all` runs every model the server offers.

## Credentials

`TINKER_API_KEY_PROD` and `TINKER_API_KEY_INTERN`, plus `CF_ACCESS_CLIENT_ID` and
`CF_ACCESS_CLIENT_SECRET` for intern, which sits behind Cloudflare Access. All live in
the `buildkite-tinker-console-e2e-creds` secret in `k8s-secrets-447022`.

## What it leaves behind

Around eight training runs per cycle, tagged `{"source": "tinker-sdk-e2e"}`. Most tests
share one client per model, but the checkpoint and correctness tests each need weights
nobody else has moved, so they take a run of their own. Training runs cannot be deleted
through the API, so they accumulate.

Checkpoints are deleted in teardown, every save site wrapped in `try/finally` and given a
one hour TTL as a backstop. `save_weights_and_get_sampling_client` is the exception and
needs none: its weights are ephemeral, never listed, and reclaimed by the server.
