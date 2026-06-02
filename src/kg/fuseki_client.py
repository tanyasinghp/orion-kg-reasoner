from typing import Any
import httpx

from config import KnowledgeGraphClientConfig
from kg.client import KnowledgeGraphClient
from kg.sparql import SparqlJsonResponse


class FusekiKnowledgeGraphClient(KnowledgeGraphClient):
    def __init__(self, config: KnowledgeGraphClientConfig) -> None:
        self.sparql_endpoint = f"{config.host.rstrip('/')}/{config.dataset}/sparql"
        self.update_endpoint = f"{config.host.rstrip('/')}/{config.dataset}/update"
        self.auth = (config.username, config.password) if config.username else None

    async def execute_sparql_query(
        self,
        query: str,
        **kwargs: Any
    ) -> SparqlJsonResponse:
        async with httpx.AsyncClient(auth=self.auth) as client:
            response = await client.post(
                self.sparql_endpoint,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=kwargs.get("timeout", 60.0)
            )
            response.raise_for_status()
            return SparqlJsonResponse.model_validate(response.json())

    async def execute_sparql_update(self, update: str) -> None:
        async with httpx.AsyncClient(auth=self.auth) as client:
            response = await client.post(
                self.update_endpoint,
                data={"update": update},
                timeout=120.0
            )
            response.raise_for_status()

    def sync_execute_sparql_update(self, update: str) -> None:
        import httpx as sync_httpx
        with sync_httpx.Client(auth=self.auth) as client:
            response = client.post(
                self.update_endpoint,
                data={"update": update},
                timeout=120.0
            )
            response.raise_for_status()
