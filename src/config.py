import logging
from typing import Self
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KnowledgeGraphClientType(StrEnum):
    FUSEKI = "fuseki"
    """Apache Jena Fuseki SPARQL endpoint."""
    RDFLIB = "rdflib"
    """In-memory RDF store using rdflib (no external dependencies)."""


class KnowledgeGraphClientConfig(BaseModel):
    type: KnowledgeGraphClientType = Field(frozen=True, default=KnowledgeGraphClientType.RDFLIB)
    host: str = Field(frozen=True, default="http://localhost:3030")
    port: int = Field(frozen=True, default=3030, validate_default=True)
    username: str = Field(frozen=True, default="")
    password: str = Field(frozen=True, default="")
    dataset: str = Field(frozen=True, default="amazon")
    csv_path: str = Field(frozen=True, default="")


class VectorSearchType(StrEnum):
    FAISS = "faiss"
    """FAISS vector search index."""


class VectorSearchConfig(BaseModel):
    type: VectorSearchType = Field(frozen=True, default=VectorSearchType.FAISS)
    faiss_index_path: str = Field(frozen=True, default="data/indices/entity_embeddings.index")
    faiss_mapping_path: str = Field(frozen=True, default="data/indices/uri_mapping.json")
    embedding_model_name: str = Field(frozen=True, default="all-MiniLM-L6-v2")


class LargeLanguageModelProvider(StrEnum):
    OLLAMA = "ollama"
    """Locally running model served by Ollama."""


class OllamaConfig(BaseModel):
    model_name: str = Field(frozen=True, default="qwen2.5:7b")
    temperature: float = Field(frozen=True, default=0.0)
    context_window_size: int = Field(frozen=True, default=65536)


class LargeLanguageModelConfig(BaseModel):
    provider: LargeLanguageModelProvider = Field(
        frozen=True, default=LargeLanguageModelProvider.OLLAMA
    )
    ollama: OllamaConfig = Field(frozen=True, default_factory=OllamaConfig)
    enable_caching: bool = Field(frozen=True, default=True)
    count_input_tokens: bool = Field(frozen=True, default=False)


class AgentFlavor(StrEnum):
    AMAZON = "amazon"
    """Amazon Product Reviews knowledge graph."""

    def description(self) -> str:
        if self == AgentFlavor.AMAZON:
            return "Amazon Product Reviews"
        raise NotImplementedError(f"Unable to create description for agent flavor '{self}'")


class AgentConfig(BaseModel):
    max_steps: int = Field(frozen=True, default=5, validate_default=True)
    generate_reason: bool = Field(frozen=True, default=True, validate_default=True)
    tool_timeout: float | None = Field(frozen=True, default=None)
    max_tool_input: int | None = Field(frozen=True, default=None)
    max_tool_output_observed: int = Field(frozen=True, default=10, validate_default=True)
    max_relations: int = Field(frozen=True, default=50, validate_default=True)
    history_limit: int = Field(frozen=True, default=5, validate_default=True)


class CorsConfiguration(BaseModel):
    allow_origins: list[str] = Field(frozen=True, default=["*"], validate_default=True)
    allow_methods: list[str] = Field(frozen=True, default=["GET, POST, OPTIONS"], validate_default=True)
    allow_headers: list[str] = Field(frozen=True, default=["*"], validate_default=True)
    allow_credentials: bool = Field(frozen=True, default=False, validate_default=True)
    expose_headers: list[str] = Field(frozen=True, default=["Session-ID"], validate_default=True)


class OpenIdConnectConfig(BaseModel):
    server_url: str = Field(frozen=True, default="")
    issuer: str = Field(frozen=True, default="")
    client_id: str = Field(frozen=True, default="")


class ApiConfig(BaseModel):
    host: str = Field(frozen=True, default="0.0.0.0", validate_default=True)
    port: int = Field(frozen=True, default=8080, validate_default=True)
    cors: CorsConfiguration = Field(frozen=True, default_factory=CorsConfiguration)
    log_level: str = Field(frozen=True, default=logging.getLevelName(logging.DEBUG), validate_default=True)
    max_sessions: int = Field(frozen=True, default=100, validate_default=True)
    oidc: OpenIdConnectConfig = Field(frozen=True, default_factory=OpenIdConnectConfig)


class KnowledgeGraphReasonerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KGR_", env_nested_delimiter="__")

    knowledge_graph: KnowledgeGraphClientConfig = Field(default_factory=KnowledgeGraphClientConfig)
    vector_db: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    llm: LargeLanguageModelConfig = Field(default_factory=LargeLanguageModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    flavor: AgentFlavor = Field(default=AgentFlavor.AMAZON)
    api: ApiConfig = Field(default_factory=ApiConfig)


settings = KnowledgeGraphReasonerSettings()
