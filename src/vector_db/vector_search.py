from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class EntityEmbedding(BaseModel):
    """A simple wrapper class for the entity-related information attached to an entity embedding.

    Note that this class doesn't contain the actual embedding, i.e., the vector, but just the entity
    data the embedding relates to.

    Attributes:
        uri: URI of the entity.
        name: Name of the entity.
        types: Types of the entity.
        embedding_class: Embedding class used for the entity.
        similarity: Similarity score between this embedding and the search input.
    """
    uri: str = Field(frozen=True)
    name: str = Field(frozen=True)
    types: list[str] = Field(frozen=True, default_factory=list)
    embedding_class: str | None = Field(frozen=True, default=None)
    similarity: float | None = Field(frozen=True, default=None)


class VectorSearch(ABC):
    """An abstract base class for a vector search."""

    @abstractmethod
    async def search_entities(
        self,
        question: str,
        include_types: bool,
        limit: int,
        **kwargs: Any # noqa: ANN401
    ) -> list[EntityEmbedding]:
        """Search for entities similar to the given question.

        Args:
            question: Search question.
            include_types: Select entity types from the database table storing
                the embeddings and include them in the search result.
            limit: Maximum number of entity embeddings to return.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A list of entity embeddings most similar to the given question.
        """
        raise NotImplementedError("This method must be implemented.")
