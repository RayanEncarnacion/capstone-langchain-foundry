"""Persistence layer for the Foundry assistant: Azure Cosmos DB (tasks).

Phase 4 — the agent's task tools need a durable, per-user store. We reuse the
same Cosmos account/database/container the LangChain implementation uses so
both assistants operate on one source of truth:

    database  = COSMOS_DATABASE   (default: capstone-db)
    container = COSMOS_CONTAINER  (default: tasks, partition key /user_id)

Auth mirrors agent.py: prefer an explicit connection string, otherwise fall
back to DefaultAzureCredential (keyless — `az login` / managed identity).
"""

import os

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

# Partition key path for the tasks container, per the project's provisioning.
# Tasks are grouped per user, so the authenticated user id IS the partition.
COSMOS_PARTITION_KEY = "/user_id"


class CosmosSettings:
    """Cosmos DB connection settings read from the environment."""

    def __init__(self) -> None:
        self.connection_string = os.environ.get("COSMOS_CONNECTION_STRING")
        self.endpoint = os.environ.get("COSMOS_ENDPOINT")
        self.database = os.environ.get("COSMOS_DATABASE", "capstone-db")
        self.container = os.environ.get("COSMOS_CONTAINER", "tasks")

    def require(self) -> None:
        """Fail fast if no way to reach Cosmos DB is configured."""
        if not self.connection_string and not self.endpoint:
            raise RuntimeError(
                "Set COSMOS_CONNECTION_STRING or COSMOS_ENDPOINT."
            )


cosmos_settings = CosmosSettings()


def _cosmos_client():
    """Build a CosmosClient from a connection string or keyless endpoint."""
    from azure.cosmos import CosmosClient

    cosmos_settings.require()
    if cosmos_settings.connection_string:
        return CosmosClient.from_connection_string(cosmos_settings.connection_string)
    return CosmosClient(
        url=cosmos_settings.endpoint,
        credential=DefaultAzureCredential(),
    )


def get_tasks_container():
    """Return the tasks container client (assumes db/container provisioned)."""
    client = _cosmos_client()
    database = client.get_database_client(cosmos_settings.database)
    return database.get_container_client(cosmos_settings.container)
