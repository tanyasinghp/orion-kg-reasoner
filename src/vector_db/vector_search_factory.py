from config import VectorSearchType, settings
from vector_db.vector_search import VectorSearch
from vector_db.faiss_search import FaissVectorSearch


class VectorSearchFactory:
    @staticmethod
    def create_vector_search(
        vector_search_type: VectorSearchType,
        ontology_provider=None
    ) -> VectorSearch:
        if vector_search_type == VectorSearchType.FAISS:
            return FaissVectorSearch(config=settings.vector_db)
        else:
            raise Exception(f"Unsupported vector search type: {vector_search_type.value}")
