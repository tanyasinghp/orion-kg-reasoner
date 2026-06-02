from typing import Any
from abc import ABC, abstractmethod

from kg.sparql import SparqlJsonResponse


class KnowledgeGraphClient(ABC):
    """An abstract base class for a client interacting with a knowledge graph."""

    @abstractmethod
    async def execute_sparql_query(
        self,
        query: str,
        **kwargs: Any # noqa: ANN401
    ) -> SparqlJsonResponse:
        """Execute a SPARQL query against the knowledge graph.

        Args:
            query: The SPARQL query to execute.
            kwargs: Additional keyword arguments to be considered by the client implementation.

        Returns:
            The query results deserialized from a response in the SPARQL 1.1 query results JSON format
        """
        ...
