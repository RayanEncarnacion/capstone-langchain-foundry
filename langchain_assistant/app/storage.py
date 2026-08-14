"""Persistence layer for RAG: Azure Blob Storage + Azure AI Search.

This phase adds retrieval, so we need three external pieces:

  1. Blob Storage  -> the raw note corpus lives here (the source of truth).
  2. Embeddings    -> turn chunk text into vectors (Foundry, OpenAI-compatible).
  3. Azure AI Search -> the index we query at request time (vectors + text).

Nothing is hard-coded: every endpoint / name / secret is read from the
environment. Auth mirrors agent.py: prefer an explicit key, otherwise fall
back to DefaultAzureCredential (keyless, `az login` / managed identity).
"""

import os

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceExistsError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Reuse the same connection settings / bearer logic as the chat model.
from .agent import _bearer_secret, settings

load_dotenv()


class RagSettings:
    """Blob + Search + embedding settings, all from the environment."""

    def __init__(self) -> None:
        # --- Azure Blob Storage (raw note corpus) -------------------------
        # Either a full connection string, OR an account URL used keyless.
        self.blob_connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        self.blob_account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
        self.blob_container = os.environ.get("AZURE_STORAGE_CONTAINER", "notes")

        # --- Azure AI Search (the searchable index) -----------------------
        self.search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
        self.search_api_key = os.environ.get("AZURE_SEARCH_API_KEY")
        self.search_index = os.environ.get("AZURE_SEARCH_INDEX", "notes-index")

        # --- Embeddings (Foundry v1, OpenAI-compatible) -------------------
        self.embedding_deployment = os.environ.get(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
        )

        # --- Azure Cosmos DB (task store) ---------------------------------
        # Prefer a full connection string, else an endpoint used keyless.
        self.cosmos_connection_string = os.environ.get("COSMOS_CONNECTION_STRING")
        self.cosmos_endpoint = os.environ.get("COSMOS_ENDPOINT")
        self.cosmos_database = os.environ.get("COSMOS_DATABASE", "capstone-db")
        self.cosmos_container = os.environ.get("COSMOS_CONTAINER", "tasks")

    def require_cosmos(self) -> None:
        """Fail fast if no way to reach Cosmos DB is configured."""
        if not self.cosmos_connection_string and not self.cosmos_endpoint:
            raise RuntimeError(
                "Set COSMOS_CONNECTION_STRING or COSMOS_ENDPOINT."
            )

    def require_search(self) -> None:
        """Fail fast if Search settings are missing."""
        if not self.search_endpoint:
            raise RuntimeError("Missing required environment variable: AZURE_SEARCH_ENDPOINT")

    def require_blob(self) -> None:
        """Fail fast if no way to reach Blob Storage is configured."""
        if not self.blob_connection_string and not self.blob_account_url:
            raise RuntimeError(
                "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL."
            )


rag_settings = RagSettings()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def build_embeddings() -> OpenAIEmbeddings:
    """Embeddings client pointed at the Foundry v1 endpoint.

    Same trick as the chat model: OpenAI-compatible route, so we pass a
    base_url and a bearer secret instead of Azure api-version wiring.
    """
    return OpenAIEmbeddings(
        base_url=settings.endpoint,
        model=rag_settings.embedding_deployment,
        api_key=_bearer_secret(),
        check_embedding_ctx_length=False,  # deployment name != tiktoken model name.
    )


# ---------------------------------------------------------------------------
# Azure Blob Storage
# ---------------------------------------------------------------------------
def _blob_service_client():
    """Build a BlobServiceClient from a connection string or keyless URL."""
    from azure.storage.blob import BlobServiceClient

    rag_settings.require_blob()
    if rag_settings.blob_connection_string:
        return BlobServiceClient.from_connection_string(rag_settings.blob_connection_string)
    return BlobServiceClient(
        account_url=rag_settings.blob_account_url,
        credential=DefaultAzureCredential(),
    )


def upload_seed_data(local_dir: str) -> int:
    """Upload every *.md file in `local_dir` into the blob container.

    Convenience so the corpus actually lives in Blob Storage (Phase 2 step 1).
    Returns the number of files uploaded.
    """
    client = _blob_service_client()
    container = client.get_container_client(rag_settings.blob_container)
    try:
        container.create_container()
    except ResourceExistsError:
        pass  # already exists.

    count = 0
    for name in os.listdir(local_dir):
        if not name.lower().endswith(".md"):
            continue
        with open(os.path.join(local_dir, name), "rb") as fh:
            container.upload_blob(name=name, data=fh, overwrite=True)
        count += 1
    return count


def iter_blob_notes():
    """Yield (blob_name, text) for every markdown blob in the container."""
    client = _blob_service_client()
    container = client.get_container_client(rag_settings.blob_container)
    for blob in container.list_blobs():
        if not blob.name.lower().endswith(".md"):
            continue
        data = container.download_blob(blob.name).readall()
        yield blob.name, data.decode("utf-8")


# ---------------------------------------------------------------------------
# Azure AI Search
# ---------------------------------------------------------------------------
def _search_credential():
    """Key credential if a key is set, else keyless DefaultAzureCredential."""
    if rag_settings.search_api_key:
        return AzureKeyCredential(rag_settings.search_api_key)
    return DefaultAzureCredential()


def get_search_client() -> SearchClient:
    """Client scoped to the notes index (used for upload + query)."""
    rag_settings.require_search()
    return SearchClient(
        endpoint=rag_settings.search_endpoint,
        index_name=rag_settings.search_index,
        credential=_search_credential(),
    )


def get_index_client() -> SearchIndexClient:
    """Client used to create / manage the index definition."""
    rag_settings.require_search()
    return SearchIndexClient(
        endpoint=rag_settings.search_endpoint,
        credential=_search_credential(),
    )


# One shared vector-search profile name referenced by the vector field.
_VECTOR_PROFILE = "notes-hnsw-profile"


def ensure_index(embedding_dim: int) -> None:
    """Create (or recreate) the notes index with a vector + text schema.

    Fields:
      id         - unique chunk key.
      content    - the chunk text (searchable for hybrid/keyword).
      embedding  - the chunk vector (searchable for vector queries).
      source     - originating blob / file name (for citations).
      title      - document title (for citations).
      chunk_id   - stable chunk identifier (for citations).
      doc_version- document version stamp.
    """
    index_client = get_index_client()

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dim,
            vector_search_profile_name=_VECTOR_PROFILE,
        ),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="title", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="doc_version", type=SearchFieldDataType.String, filterable=True),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="notes-hnsw")],
        profiles=[
            VectorSearchProfile(
                name=_VECTOR_PROFILE,
                algorithm_configuration_name="notes-hnsw",
            )
        ],
    )

    index = SearchIndex(
        name=rag_settings.search_index,
        fields=fields,
        vector_search=vector_search,
    )

    # Idempotent: replaces the definition if it already exists.
    # pylint: disable=no-member  # method added dynamically by the azure-search SDK patch layer.
    index_client.create_or_update_index(index)


# ---------------------------------------------------------------------------
# Azure Cosmos DB (task store)
# ---------------------------------------------------------------------------
# Partition key for the tasks container, per the project's provisioning.
COSMOS_PARTITION_KEY = "/user_id"


def _cosmos_client():
    """Build a CosmosClient from a connection string or keyless endpoint."""
    from azure.cosmos import CosmosClient

    rag_settings.require_cosmos()
    if rag_settings.cosmos_connection_string:
        return CosmosClient.from_connection_string(rag_settings.cosmos_connection_string)
    return CosmosClient(
        url=rag_settings.cosmos_endpoint,
        credential=DefaultAzureCredential(),
    )


def get_tasks_container():
    """Return the tasks container client (assumes db/container provisioned)."""
    client = _cosmos_client()
    database = client.get_database_client(rag_settings.cosmos_database)
    return database.get_container_client(rag_settings.cosmos_container)
