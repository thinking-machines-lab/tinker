from typing_extensions import Literal, TypeAlias

__all__ = ["SamplingClientKind"]

SamplingClientKind: TypeAlias = Literal["snapshot", "inflight"]
"""Which weights a sampling client uses, sent in sampler-save requests to the API.

- `snapshot` (default) captures fixed weights. Use it when inference must stay on
  one saved set of weights.
- `inflight` publishes mutable weights shared by this training model's inflight
  sampling clients.

Inflight sampling contract:

- All inflight sampling clients for this training model refer to the same mutable
  weights target. Later inflight saves update that target for both existing and
  newly returned clients. Updates propagate asynchronously across serving replicas.
- Save completion means the checkpoint is saved and the shared target is updated.
  It does not wait for serving replicas to finish loading weights.
- Samples generated with the returned client are guaranteed to run on samplers
  using the weights saved when the client was created, or newer weights.
- Each returned client represents a sampling session with a fresh KV-cache
  namespace. Requests using that session, including serialized copies sent to
  worker processes, can reuse matching cached prefixes when available. Different
  inflight sessions do not share KV entries. A new save creates a new namespace
  even if no optimizer step occurred between saves.
- Publishing new weights does not clear an existing session's KV or change its
  namespace. Active generation may span updates and use KV computed under earlier
  weights. Outputs can depend on update timing even with the same prompt and seed.
  Responses do not identify which saved weights produced each token.
"""
