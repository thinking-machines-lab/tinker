"""Commands for listing the models available to the active credential.

This module implements the 'tinker model' commands, including:
- list: List the models this organization can train or sample
"""

from typing import TYPE_CHECKING, Any, Dict, List

import click

if TYPE_CHECKING:
    from tinker.types import SupportedModel

from ..client import create_service_client, handle_api_errors
from ..context import CLIContext
from ..exceptions import TinkerCliError
from ..output import OutputBase, format_bool, format_optional


class ModelListOutput(OutputBase):
    """Output for 'tinker model list' command."""

    def __init__(self, models: List["SupportedModel"]):
        self.models = models

    def to_dict(self) -> Dict[str, Any]:
        return {"models": [model.model_dump() for model in self.models]}

    def get_table_columns(self) -> List[str]:
        return ["Model", "Max Context", "Trainable", "Sampleable"]

    def get_table_rows(self) -> List[List[str]]:
        return [
            [
                format_optional(model.model_name),
                format_optional(model.max_context_length),
                format_optional(model.trainable, format_bool),
                format_optional(model.sampleable, format_bool),
            ]
            for model in self.models
        ]


@click.group()
def cli():
    """List available models."""
    pass


@cli.command(name="list")
@click.argument("substring", required=False)
@click.pass_obj
@handle_api_errors
def list_models(cli_context: CLIContext, substring: str | None) -> None:
    """List the models your organization can use.

    SUBSTRING keeps only the models whose name contains it, ignoring case.
    Pass the name in the Model column as base_model.
    """
    service_client = create_service_client()
    models = service_client.get_server_capabilities().supported_models
    if substring is not None:
        models = [
            model
            for model in models
            if model.model_name is not None and substring.lower() in model.model_name.lower()
        ]
        if not models:
            org_urn = service_client.create_rest_client().whoami().result().org_urn
            raise TinkerCliError(
                f"No model name contains '{substring}' for organization {org_urn}",
                "Run 'tinker model list' to see every model this organization can use.",
            )
    ModelListOutput(models).print(format=cli_context.format)
