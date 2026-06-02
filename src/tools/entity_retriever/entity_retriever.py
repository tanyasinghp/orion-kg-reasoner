import textwrap
from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from string import Template

from agent.agent_context import Context, UnresolvedRelation, Entity, Relation
from agent.ontology import OntologyProvider
from agent.ranker import Ranker
from agent.util import IdGenerator
from kg.client import KnowledgeGraphClient, SparqlJsonResponse
from kg.sparql import SparqlRdfTerm


@dataclass
class ExtraProperties:
    """A simple wrapper class for the extra properties found for a given set of entities.

    Attributes:
        properties: A lookup table from entity URIs to a property dictionary for an entity containing
            the extra properties found for the entity.
        additional_entity_uris: URIs of additional entities found when resolving the found extra properties.
    """
    properties: dict[str, dict[str, Any]] # entity URI -> extra properties dictionary
    additional_entity_uris: list[str]


@dataclass
class EntitySearchResult:
    """A simple wrapper class for an entity search result.

    Attributes:
        entities: A list of the found entities meeting the search criteria.
        extra_entities: Additional entities found along the way, which are not related to the search. Such entities
            can be returned to make them available to the tool executor to be registered in the agent context.
    """
    entities: list[Entity]
    extra_entities: list[Entity]


class EntityRetriever(ABC):
    """A base class for the entity retriever.

    Attributes:
        knowledge_graph_client: A client to query the knowledge graph.
        ontology_provider: A provider for the knowledge graph ontology to be used by the entity retriever.
        ranker: The ranker to be used to calculate ranking features for each retrieved entity.
        entity_id_generator: The ID generator used to generate entity IDs.
    """

    def __init__(self,
                 knowledge_graph_client: KnowledgeGraphClient,
                 ontology_provider: OntologyProvider,
                 ranker: Ranker
                 ) -> None:
        self.knowledge_graph_client = knowledge_graph_client
        self.ontology_provider = ontology_provider
        self.ranker = ranker
        self.entity_id_generator = IdGenerator()

    @abstractmethod
    async def search_entities_by_names(self, names: list[str], entity_type: str, limit: int) -> list[str]:
        """Search entities by name in the knowledge graph.

        Args:
            names: The names of the entities to search for.
            entity_type: Preferred entity type.
            limit: The maximum number of entities to return.

        Returns:
            A list with the URIs of the entities whose name matches any of the given names.
        """
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    async def search_entities_by_description(self, description: str, entity_type: str, limit: int) -> list[str]:
        """Search entities by description in the vector database.

        Args:
            description: The description of the entities to search for.
            entity_type: Preferred entity type.
            limit: The maximum number of entities to return.

        Returns:
            A list with the URIs of the entities similar to the given description.
        """
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    async def get_relations_between_entities(self, uris: list[str], limit: int = 1000) -> list[UnresolvedRelation]:
        """Find all predefined relations between any two of the given entities.

        Args:
            uris: Entity URIs
            limit: The maximum number of relations to return.

        Returns:
            A list of predefined relations between any two of the given entities.
        """
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    async def get_basic_entity_information(self, uris: list[str]) -> list[Entity]:
        """Get the basic entity information (name, description, types) for the given entity URIs.

        Args:
            uris: Entity URIs.

        Returns:
            A list of entities with the basic entity information for the given entity URIs.
        """
        raise NotImplementedError("Not implemented.")

    async def search_matching_entities(
        self,
        context: Context,
        texts: list[str],
        entity_type: str,
        limit: int = 20
    ) -> EntitySearchResult:
        """Search entities whose name or description matches any of the given texts.

        While this method accepts an argument for a preferred entity type, this method doesn't only return entities
        with this type. Each entity will contain features for later ranking, and these ranking features will reflect
        whether the found entity has the given type or not.

        Args:
            context: Agent context.
            texts: Text snippets to search for.
            entity_type: Prefer entities with the given type.
            limit: Maximum number of entities to return.

        Returns:
            The entities whose name or description matches any of the given text snippets.
        """
        entity_uris: list[str] = await self.search_entities_by_names(names=texts, entity_type=entity_type, limit=limit)
        entities, extra_entities = await self.get_entities(entity_uris, context)

        # Update features for ranking in the found entities
        for entity in entities:
            entity.ranking_features["name_search"] = 1
            entity.ranking_features["type_match"] = self.ranker.calculate_type_match_score(entity.types, entity_type)
            trace("entity_retriever.py", f"  [name] Entity '{entity.name}' features={entity.ranking_features}")

        # If none of the found entities matches the given type, perform
        # a similarity search on the entity descriptions as a fallback
        if not any(entity_type in entity.types for entity in entities):
            description = "; ".join(texts)
            entity_uris = await self.search_entities_by_description(description, entity_type, limit)
            entities, extra_entities = await self.get_entities(entity_uris, context)

            # Update features for ranking in the found entities
            # TODO: The entities retrieved through the vector search are already sorted by the similarity score
            #  determined by the vector database. Before the HANA Cloud vector engine was used, this similarity score
            #  was internal to the used vector database and could not be directly returned. Therefore, an "artificial"
            #  similarity score is computed as defined below to be used for ranking later on. We should check if with
            #  the switch to the HANA Cloud vector engine we have access to the similarity score used by the database
            #  when doing a vector search, in this case we can omit this code block and directly use the score from
            #  the vector engine.
            type_match_count, type_mismatch_count = 0, 0
            for entity in entities:
                entity.ranking_features["type_match"] = self.ranker.calculate_type_match_score(entity.types, entity_type)

                if entity.ranking_features["type_match"]:
                    type_match_count += 1
                    entity.ranking_features["type_match_inverse_rank"] = 1 / type_match_count
                else:
                    type_mismatch_count += 1
                    entity.ranking_features["type_mismatch_inverse_rank"] = 1 / type_mismatch_count

        return EntitySearchResult(entities=entities, extra_entities=extra_entities)

    async def get_entities( # noqa: PLR0912
        self,
        entity_uris: list[str],
        context: Context
    ) -> tuple[list[Entity], list[Entity]]:
        """Get entity information for the given entity URIs.

        This method tries to look up the entity information in the agent context. If the entity is not known to the
        context, yet, the entity information will be retrieved from the knowledge graph. Additionally, the method
        resolves all extra properties for the entities defined in the knowledge graph ontology.

        The entity details for the given entity URIs are resolved using the following algorithm:

        Assuming the entities E-0, E-1, E-2 are already in the context, and given the URIs for the entities
        E-3, E-4, E-2 as input:

        1. All URIs for the entities which are not known in the agent context are determined (E-3, E-4).
        2. Then, search for any extra properties defined in the ontology, potentially introducing additional unknown
           entities (E-5).
        3. Query basic entity information (name, description, types) from the knowledge graph for all entities not
           known in the agent context (E-3, E-4, E-5).
        4. Add the extra properties found in step 2 to the retrieved entities (E-3, E-4, E-5).
        5. Collect requested entities from new entities retrieved from the knowledge graph and existing ones from the
           agent context (E-3, E-4, E-2).
        6. Collect any extra entities not relating to the given entity URIs which were found along the way but are not
           known in the agent context, yet (E-5).

        Args:
            entity_uris: URIs of entities to get information for.
            context: Agent context.

        Returns:
            A tuple of the entities and extra entities retrieved for the given entity URIs.
            All retrieved entities contain the basic entity information (name, description, types) as well as any
            extra properties defined for each entity in the knowledge graph ontology. Any extra entities returned do
            not relate to any of the given entity URIs but are entities found along the way when retrieving the entity
            information for the given URIs, e.g., when resolving extra properties for an entity
        """
        # Determine entities not available in the agent context, yet. For these entities,
        # all relevant information needs to be fetched from the knowledge graph.
        new_entity_uris: list[str] = [uri for uri in entity_uris if context.get_entity_by_uri(uri) is None]

        # First search for any extra properties defined in the ontology, potentially introducing newer entities
        extra_properties: ExtraProperties = await self._get_extra_properties(new_entity_uris)
        new_additional_entity_uris: list[str] = [
            uri for uri in extra_properties.additional_entity_uris
            if context.get_entity_by_uri(uri) is None
        ]
        new_entity_uris.extend(new_additional_entity_uris)

        # Get basic entity data (name, description, types) from the knowledge graph for all new entities
        new_entities: list[Entity] = await self.get_basic_entity_information(new_entity_uris)

        # Add found extra properties to the retrieved entities and build lookup table for new entities along the way
        new_entities_lookup_table: dict[str, Entity] = {}
        for entity in new_entities:
            if entity.uri in extra_properties.properties:
                entity.properties.update(extra_properties.properties[entity.uri])

            new_entities_lookup_table[entity.uri] = entity

        # Collect entities from new entities retrieved from the knowledge graph and existing ones from the context
        entities: list[Entity] = []
        for entity_uri in entity_uris:
            entity: Entity | None = context.get_entity_by_uri(entity_uri)
            if entity is not None:
                entities.append(entity)
            else: # noqa: PLR5501
                if entity_uri in new_entities_lookup_table:
                    entities.append(new_entities_lookup_table[entity_uri])
                else:
                    raise Exception(f"Could not find entity with URI '{entity_uri}' in knowledge graph or context")

        # Additionally, collect all entities relating to the extra properties which are not known in the agent context,
        # yet. This way these entities can be made available to the tool executor to register them in the agent context
        # later on.
        extra_entities: list[Entity] = []
        for entity_uri in new_additional_entity_uris:
            if context.get_entity_by_uri(entity_uri) is None:
                extra_entities.append(new_entities_lookup_table[entity_uri])

        # Replace the URI values in entity properties with the corresponding entity names
        for entity in entities:
            for property_name, property_value in entity.properties.items():
                if isinstance(property_value, list):
                    for i, property_value_element in enumerate(property_value):
                        # Check if the property value element is an entity already
                        # known in the context or in the entities retrieved previously
                        value_entity = context.get_entity_by_uri(
                            property_value_element, new_entities_lookup_table.get(property_value_element, None)
                        )
                        if value_entity is not None:
                            entity.properties[property_name][i] = value_entity.name

        return entities, extra_entities

    async def _get_extra_properties(self, entity_uris: list[str]) -> ExtraProperties:
        """Resolve any extra properties defined in the knowledge graph ontology for the given entities.

        The "extra" properties defined in the knowledge graph ontology are properties that are automatically
        looked up for every entity. Potential property values will contain "just" entity URIs, resolving these
        URIs to a human-readable entity attribute like entity name is not handled by this method and needs to
        be performed by the caller in a later step.

        Args:
            entity_uris: Entity URIs

        Returns:
            The extra properties of the given entities and the URIs of all entities introduced by the extra properties.
        """
        if not entity_uris:
            return ExtraProperties(properties={}, additional_entity_uris=[])

        param_to_path: dict[str, str] = {}
        value_clauses: list[str] = []

        for i, extra_property in enumerate(self.ontology_provider.extra_properties):
            param = f"value_{i}"
            param_to_path[param] = extra_property.alias or extra_property.path

            path_with_resolved_properties = self.ontology_provider.resolve_path_properties(extra_property.path)
            type_uri = self.ontology_provider.get_type_uri(extra_property.type, extra_property.type)
            value_clause = f"OPTIONAL {{\n\t?uri rdf:type <{type_uri}> .\n\t?uri {path_with_resolved_properties} ?{param} .\n}}"
            value_clauses.append(value_clause)

        select_params = " ".join([
            f'(GROUP_CONCAT(?{param} ; SEPARATOR=";;; ") as ?{param})'
            for param in param_to_path
        ])
        group_by_params = " ".join([f"?{param}" for param in param_to_path])

        # Format the given entity URIs as syntactically correct SPARQL resource identifiers
        # with '<' and '>' as delimiters, e.g., <http://example.org/book/book1>.
        formatted_entity_uris = " ".join(f"<{uri}>" for uri in entity_uris)

        formatted_value_clauses = "\n".join(value_clauses)

        query = Template(textwrap.dedent("""
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?uri $select_params
        WHERE {
            VALUES ?uri { $uris }
            $value_clauses
        }
        GROUP BY ?uri $group_by_params
        LIMIT $limit
        """)).substitute(
            uris=formatted_entity_uris,
            select_params=select_params,
            value_clauses=formatted_value_clauses,
            group_by_params=group_by_params,
            limit=1000
        )

        response: SparqlJsonResponse = await self.knowledge_graph_client.execute_sparql_query(query)

        extra_properties: dict[str, dict[str, list[Any]]] = {} # entity URI -> properties
        additional_entity_uris: set[str] = set()
        for binding in response.results.bindings:
            uri = binding["uri"].value
            properties: dict[str, list[Any]] = self.collect_entity_properties(binding, param_to_path)
            extra_properties[uri] = properties

            # Extra properties reference entities, i.e., all values of the extra properties are entity URIs
            for property_values in properties.values():
                additional_entity_uris.update(property_values)

        return ExtraProperties(properties=extra_properties, additional_entity_uris=list(additional_entity_uris))

    @staticmethod
    def collect_entity_properties(
            binding: dict[str, SparqlRdfTerm],
            param_to_path: dict[str, str]
    ) -> dict[str, list[Any]]:
        """Collect entity properties from the given SPARQL result bindings containing variables corresponding to
        filter conditions on properties used in the SPARQL query.

        Args:
            binding: SPARQL response result binding.
            param_to_path: Lookup table mapping the additional values which were selected in
                the SPARQL query to the properties defined in the conditions they correspond to.

        Returns:
            A dictionary with the entity properties present in the given result bindings.
        """
        # Update properties of the filtered entities with the properties resolved as part of the target
        # conditions. Given, for example, a target condition ['Name', 'contains', 'Foo'] the SPARQL query
        # returns the value(s) for the property, allowing us to add these property values to the collection
        # of known properties for the found target entities.
        properties: dict[str, list[Any]] = {}
        for param_name, path in param_to_path.items():
            if param_name in binding:
                param_value: str = binding[param_name].value

                # Parse parameter values which have been aggregated into a single string value using the
                # separator ";;;". This will be the case if the value of a property is a list and not just
                # a single value like a string, a number, a boolean value, etc.
                if isinstance(param_value, str):
                    # TODO: convert to the actual data type
                    param_values = param_value.split(";;; ")
                else:
                    param_values = [param_value]

                properties[path] = param_values
        return properties

    @staticmethod
    def resolve_relations(unresolved_relations: list[UnresolvedRelation], context: Context) -> list[Relation]:
        """Resolving the entity URIs in the given relations to the corresponding IDs of the entities stored in the
        agent context.

        Args:
            unresolved_relations: "Raw" relations obtained from the knowledge graph just containing the URIs of the
                source and the target of the relations.
            context: Agent context.

        Returns:
            A list of relations where the source and target entities have been resolved with entities present in the
            agent context.

        Raises:
            ValueError: If the source or target of a relation contains the URI of an entity not present in the agent
                context.
        """
        relations: list[Relation] = []

        for unresolved_relation in unresolved_relations:
            source_entity: Entity | None = context.get_entity_by_uri(unresolved_relation.source_uri)
            if source_entity is None:
                raise ValueError(f"Failed to resolve relation '{unresolved_relation.relation}'."
                                 f"Entity URI '{unresolved_relation.source_uri}' not found in agent context.")

            target_entity: Entity | None = context.get_entity_by_uri(unresolved_relation.target_uri)
            if target_entity is None:
                raise ValueError(f"Failed to resolve relation '{unresolved_relation.relation}'."
                                 f"Entity URI '{unresolved_relation.target_uri}' not found in agent context.")

            relation = Relation(target_uri=unresolved_relation.target_uri,
                                target_id=target_entity.id,
                                source_uri=unresolved_relation.source_uri,
                                source_id=source_entity.id,
                                relation=unresolved_relation.relation,
                                properties=unresolved_relation.properties)
            relations.append(relation)

        return relations
