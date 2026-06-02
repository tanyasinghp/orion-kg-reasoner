import textwrap
from typing import override
from string import Template

from agent.agent_context import Entity, UnresolvedRelation
from agent.ontology import OntologyProvider
from agent.ranker import Ranker
from tools.entity_retriever.entity_retriever import EntityRetriever
from kg.client import KnowledgeGraphClient, SparqlJsonResponse
from kg.sparql import SparqlRdfTerm
from vector_db.vector_search import VectorSearch, EntityEmbedding
from trace import trace


class AmazonEntityRetriever(EntityRetriever):
    def __init__(self,
                 knowledge_graph_client: KnowledgeGraphClient,
                 vector_search: VectorSearch,
                 ontology_provider: OntologyProvider,
                 ranker: Ranker
                 ) -> None:
        super().__init__(knowledge_graph_client, ontology_provider, ranker)
        self.vector_search = vector_search

    @override
    async def search_entities_by_names(self, names: list[str], entity_type: str, limit: int) -> list[str]:
        conditions = " || ".join(
            f"CONTAINS(LCASE(?name), LCASE(\"{name}\"))" for name in names
        )
        query = Template(textwrap.dedent("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT DISTINCT ?uri ?name
        WHERE {
            ?uri rdf:type <http://amazon/kg/Product> .
            ?uri <http://amazon/kg/p_name> ?name .
            FILTER ( $conditions )
        }
        LIMIT $limit
        """)).substitute(conditions=conditions, limit=limit)

        trace("amazon_entity_retriever.py", f"SPARQL name search: names={names}, type={entity_type}")
        trace("amazon_entity_retriever.py", f"  Query: {query.replace(chr(10), ' ')}")
        response: SparqlJsonResponse = await self.knowledge_graph_client.execute_sparql_query(query)
        uris = [binding["uri"].value for binding in response.results.bindings]
        trace("amazon_entity_retriever.py", f"  Name search returned {len(uris)} URIs: {uris}")
        return uris

    @override
    async def search_entities_by_description(self, description: str, entity_type: str, limit: int) -> list[str]:
        trace("amazon_entity_retriever.py", f"Vector search: description='{description[:60]}', type={entity_type}")
        embeddings = await self.vector_search.search_entities(
            question=description,
            include_types=True,
            limit=limit
        )
        uris = [e.uri for e in embeddings]
        trace("amazon_entity_retriever.py", f"  Vector search returned {len(uris)} URIs")
        return uris

    @override
    async def get_relations_between_entities(self, uris: list[str], limit: int = 1000) -> list[UnresolvedRelation]:
        formatted_uris = " ".join(f"<{uri}>" for uri in uris)
        query = Template(textwrap.dedent("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT DISTINCT ?source ?relation ?target
        WHERE {
            VALUES ?source { $uris }
            VALUES ?target { $uris }
            ?source ?relation ?target .
            FILTER (?source != ?target)
        }
        LIMIT $limit
        """)).substitute(uris=formatted_uris, limit=limit)

        response: SparqlJsonResponse = await self.knowledge_graph_client.execute_sparql_query(query)
        unresolved: list[UnresolvedRelation] = []
        for binding in response.results.bindings:
            unresolved.append(UnresolvedRelation(
                source_uri=binding["source"].value,
                target_uri=binding["target"].value,
                relation=binding["relation"].value,
                properties={}
            ))
        return unresolved

    @override
    async def get_basic_entity_information(self, uris: list[str]) -> list[Entity]:
        if not uris:
            return []

        formatted_uris = " ".join(f"<{uri}>" for uri in uris)
        query = Template(textwrap.dedent("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?uri ?name ?description ?type
        WHERE {
            VALUES ?uri { $uris }
            OPTIONAL { ?uri <http://amazon/kg/p_name> ?name . }
            OPTIONAL { ?uri <http://amazon/kg/p_about_product> ?description . }
            OPTIONAL { ?uri rdf:type ?type . }
        }
        LIMIT 1000
        """)).substitute(uris=formatted_uris)

        response: SparqlJsonResponse = await self.knowledge_graph_client.execute_sparql_query(query)

        entity_map: dict[str, Entity] = {}
        for binding in response.results.bindings:
            uri = binding["uri"].value
            if uri not in entity_map:
                name = binding["name"].value if "name" in binding else ""
                description = binding["description"].value if "description" in binding else ""
                entity_map[uri] = Entity(
                    uri=uri,
                    id=self.entity_id_generator.generate_id(),
                    name=name,
                    description=description,
                    types=[],
                    properties={},
                    ranking_features={}
                )
            if "type" in binding:
                entity_map[uri].types.append(binding["type"].value)

        return list(entity_map.values())
