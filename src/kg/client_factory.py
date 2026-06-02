from config import KnowledgeGraphClientType, settings
from kg.client import KnowledgeGraphClient
from kg.fuseki_client import FusekiKnowledgeGraphClient
from kg.rdflib_client import RdfLibKnowledgeGraphClient


class KnowledgeGraphClientFactory:
    @staticmethod
    def create_knowledge_graph_client(client_type: KnowledgeGraphClientType, csv_path: str = "") -> KnowledgeGraphClient:
        if client_type == KnowledgeGraphClientType.FUSEKI:
            return FusekiKnowledgeGraphClient(config=settings.knowledge_graph)
        if client_type == KnowledgeGraphClientType.RDFLIB:
            config = settings.knowledge_graph.model_copy(update={"csv_path": csv_path})
            return RdfLibKnowledgeGraphClient(config=config)
        raise Exception(f"Unsupported knowledge graph client type: {client_type.value}")
