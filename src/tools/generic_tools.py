import textwrap
from typing import TYPE_CHECKING
from string import Template
from dataclasses import dataclass

from agent.agent_context import Context, Entity, Relation, ToolLog
from agent.ontology import OntologyProvider
from tools.entity_retriever import EntityRetriever
from tools.tool_executor import ToolOutput
from tools.utils import format_value_for_sparql
from kg.client import KnowledgeGraphClient, SparqlJsonResponse
from trace import trace
if TYPE_CHECKING:
    from kg.sparql import SparqlRdfTerm


@dataclass
class QueryFiltersForTargetConditions:
    """A simple wrapper for the filter clauses in a SPARQL query corresponding to a set of conditions given to a tool.

    Attributes:
        param_to_path: Lookup table mapping the additional values which should be selected in the SPARQL
            query to the properties defined in the conditions they correspond to.
        where_clause: Expressions to be added to the WHERE clause in the SPARQL query.
        filter_conditions: BIND expressions defining conditions to be used in the FILTER expression.
        filter_clause: FILTER expressions to be used in the SPARQL query.
        select_params: Additional values that should be selected in the SPARQL query
        group_by_params: Identified parameters to group the query results by
    """
    param_to_path: dict[str, str]
    where_clause: str
    filter_conditions: str
    filter_clause: str
    select_params: str
    group_by_params: str


class GenericTools:
    """Generic agent tools relevant for all use cases.

    These tools must be made available to the agent for it to function properly. That is, these tools must be
    considered mandatory tools for each agent session.

    Attributes:
        knowledge_graph_client: Client used by the tools to query the knowledge graph.
        entity_retriever: Entity retriever used by the tools to search entities in the knowledge graph.
        ontology_provider: Provider for the ontology of the knowledge graph.
    """

    def __init__(
        self,
        knowledge_graph_client: KnowledgeGraphClient,
        entity_retriever: EntityRetriever,
        ontology_provider: OntologyProvider
    ) -> None:
        self.knowledge_graph_client = knowledge_graph_client
        self.entity_retriever = entity_retriever
        self.ontology_provider = ontology_provider

    # Note: All tools use keyword-only arguments
    async def tool_retrieve_entities(
        self,
        *,
        context: Context, # implicit tool argument, not documented in doc string
        texts: list[str],
        entity_type: str,
        require_type_match: bool = True,
        require_single_result: bool = False
    ) -> ToolOutput:
        """Find entities by name match or similarity search on their names and descriptions.

        Retrieved entities whose types match the given 'entity_type' have higher priority.

        CRITICAL: Only use this tool for a NEW entity you haven't searched before.
        If you already found an entity (it appears in ENTITIES or TOOL_LOG), use its ID
        (e.g. E-1) in other tools instead of calling this tool again for the same search.

        This tool should be invoked separately for each potential entity, which means the values in `texts`
        should possibly be variations (name, description) of a single entity.

        Args:
            texts: The name or description of the entities to search for.
            entity_type: The expected entity type.
            require_type_match: Flag indicating that the retrieved result must match the expected entity type.
                Set to true when you are expecting that the results of the function call match the entity type
                because it is either critical to the answer or the next step in the evaluation process.
            require_single_result: Flag indicating that the result must be one and only one entity.

        Returns:
            A tool output containing the found entities.
        """
        # Never require that the retrieved result must match the expected entity type if no entity type is given
        if not entity_type:
            require_type_match = False

        trace("generic_tools.py", f"tool_retrieve_entities(texts={texts}, entity_type={entity_type})")
        entity_search_result = await self.entity_retriever.search_matching_entities(context, texts, entity_type)
        trace("generic_tools.py", f"  Found {len(entity_search_result.entities)} entities, {len(entity_search_result.extra_entities)} extra")

        # Filter found entities based on their type (if a type match with the expected type is required) and ensure
        # a single result if requested. All filtered out entities will not be discarded but considered extra entities
        # as well. This way they end up in the context and don't need to be fetched from the knowledge graph in
        # future tool calls, but aren't considered part of the tool result.
        entities: list[Entity] = []
        extra_entities = entity_search_result.extra_entities
        for entity in entity_search_result.entities:
            collect_entity: bool = len(entities) < 1 if require_single_result else True
            if (not require_type_match or entity_type in entity.types) and collect_entity:
                entities.append(entity)
            else:
                extra_entities.append(entity)

        return ToolOutput(entities=entities, extra_entities=extra_entities)

    async def tool_get_relations_between_entities(
        self,
        *,
        context: Context, # implicit tool argument, not documented in doc string
        ids: list[str]
    ) -> ToolOutput:
        """Find all predefined relations between any two of the given entities.

        Args:
            ids: Entity IDs

        Returns:
            The predefined relations between any two of the given entities.
        """
        unresolved_relations = await self.entity_retriever.get_relations_between_entities(ids)
        relations = self.entity_retriever.resolve_relations(unresolved_relations, context)

        return ToolOutput(relations=relations)

    async def tool_select_entities(
        self,
        *,
        context: Context, # implicit tool argument, not documented in doc string
        tool_log_id: str,
        selected_entity_ids: list[str]
    ) -> None:
        """Select a subset of entities from a tool output, determine the relevant entities, and remove the rest.

        This tool is typically applied to the output of 'tool_retrieve_entities' to select the relevant candidates.
        This tool works in-place, i.e., the results are in the same tool_log, and the selection will be propagated
        to subsequent tool calls that use its output.

        Args:
            tool_log_id: The ID of the tool log to select output entities from.
            selected_entity_ids: IDs of the selected entities which should remain in the tool log output.
        """
        tool_log: ToolLog | None = context.get_tool_log_by_id(tool_log_id)
        if tool_log is not None:
            selected_entity_ids = set(selected_entity_ids)
            tool_log.selected_entities = [
                entity for entity in tool_log.selected_entities
                if entity.id in selected_entity_ids
            ]

    async def tool_filter_entities(
        self,
        *,
        context: Context, # implicit tool argument, not documented in doc string
        ids: list[str],
        target_conditions: list[list[str]] | None = None,
        match_mode: str = "all"
    ) -> ToolOutput:
        """Filter the given entities based on the given target conditions.

        Args:
            ids: Entity IDs to filter.
            target_conditions: Conditions to filter the given entities by depending on the given match mode.
            match_mode: Match mode. Supported values are 'all' or 'any'. If match mode 'all' is selected, an entity
                needs to match all the given target conditions. If match mode 'any' is selected, an entity only needs
                to match at least one of the given target conditions.

        Returns:
            A tool output containing the entities meeting the given conditions as well as all relations between the
            found entities.
        """
        # Format the given entity URIs as syntactically correct SPARQL resource identifiers
        # with '<' and '>' as delimiters, e.g., <http://example.org/book/book1>.
        # Note: Even though the parameter is called 'ids' and the doc string refers to entity IDs,
        # the tool executor will replace these entity IDs with entity URIs before calling the tool.
        formatted_uris = " ".join(f"<{uri}>" for uri in ids)

        parsed_target_conditions = self._parse_target_conditions(target_conditions, match_mode)

        query = Template(textwrap.dedent("""
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?uri ?source_uri ?source_relation $select_params
        WHERE {
            VALUES ?source_uri { $uris }
            BIND (?source_uri as ?uri)
            BIND ("filter" as ?source_relation)
            $where_clause
            $filter_conditions
            $filter_clause
        }
        GROUP BY ?uri ?source_uri ?source_relation $group_by_params
        LIMIT $limit

        """)).substitute(
            select_params=parsed_target_conditions.select_params,
            uris=formatted_uris,
            where_clause=parsed_target_conditions.where_clause,
            filter_conditions=parsed_target_conditions.filter_conditions,
            filter_clause=parsed_target_conditions.filter_clause,
            group_by_params=parsed_target_conditions.group_by_params,
            limit=1000
        )

        response: SparqlJsonResponse = await self.knowledge_graph_client.execute_sparql_query(query)

        # Make sure the resulting entities are sorted in the same order as the
        # given source entities as the SPARQL query is not guaranteeing this order
        self.order_sparql_response(response, ids, "source_uri")

        # Collect URIs of the filtered entities and build a lookup table mapping the
        # found entity URI to the result binding in the SPARQL response it stems from
        filtered_entity_uris: list[str] = []
        results_binding_by_uri_lookup_table: dict[str, dict[str, SparqlRdfTerm]] = {}
        for binding in response.results.bindings:
            uri = binding["uri"].value
            filtered_entity_uris.append(uri)
            results_binding_by_uri_lookup_table[uri] = binding

        # Get entity information for the filtered entities
        filtered_entities, extra_entities = await self.entity_retriever.get_entities(filtered_entity_uris, context)

        # Update properties of the filtered entities with the properties resolved as part of the target
        # conditions. Given, for example, a target condition ['Name', 'contains', 'Foo'] the SPARQL query
        # returns the value(s) for the property, allowing us to add these property values to the collection
        # of known properties for the filtered entities.
        for filtered_entity in filtered_entities:
            binding = results_binding_by_uri_lookup_table[filtered_entity.uri]
            properties = self.entity_retriever.collect_entity_properties(
                binding, parsed_target_conditions.param_to_path
            )
            filtered_entity.properties.update(properties)

        return ToolOutput(entities=filtered_entities, extra_entities=extra_entities)

    def _parse_target_conditions(
        self,
        target_conditions: list[list[str]] | None,
        match_mode: str
    ) -> QueryFiltersForTargetConditions:
        """Parse the given target conditions into corresponding filter clauses to be used in the SPARQL query.

        This method identifies the right WHERE, filter, and GROUP BY clauses needed for the given conditions and
        also defines additional values which should be selected in the SPARQL query based on the properties defined
        in the conditions.

        Given, for example, the target condition ['Name', 'contains', 'Foo'] this method will identify the
        following filter-related aspects to later be used in the SPARQL query:

        1. Additional values which should be selected in the SPARQL query based on the properties defined
           in the conditions: '(GROUP_CONCAT(?value_0 ; SEPARATOR=";;; ") as ?value_0)'
        2. An expression for a WHERE clause referencing the property defined in the condition:
           '?uri <http://sap.com/datasphere/deepsea/repository#o_name> ?value_0 .'
        3. A BIND expression defining a condition for the property to match the given value:
           'BIND(contains(?value_0, "Foo") AS ?cond_0)'
        4. A FILTER clause using the defined conditions: 'FILTER (?cond_0)'
        5. The selected values the SPARQL result should be grouped by: '?value_0'
        6. Lookup table mapping the additional values which should be selected in the SPARQL query to the
           properties defined in the conditions they correspond to: {'value_0': 'Name'}

        Args:
            target_conditions: Conditions to filter the target entities by depending on the given match mode.
            match_mode: Match mode. Supported values are 'all' or 'any'. If match mode 'all' is selected, an entity
                needs to match all the given target conditions. If match mode 'any' is selected, an entity only needs
                to match at least one of the given target conditions.

        Returns:
            The right WHERE, filter, and GROUP BY clauses needed for the given conditions as well as additional values
            which should be selected in the SPARQL query based on the properties defined in the conditions.
        """
        where_clauses: list[str] = []
        filter_conditions: list[str] = []
        filter_values: list[str] = []
        param_to_path: dict[str, str] = {}

        if target_conditions:
            for i, (path, comparator, value) in enumerate(target_conditions):
                path_with_resolved_properties = self.ontology_provider.resolve_path_properties(path)

                param = f"value_{i}"
                filter_value = f"?cond_{i}"
                param_to_path[param] = path
                if comparator in ["=", "!=", ">", "<", ">=", "<=", "contains", "startsWith", "endsWith"]:
                    formatted_value = format_value_for_sparql(value)
                    where_clauses.append(f"?uri {path_with_resolved_properties} ?{param} .")

                    if comparator == "contains":
                        filter_conditions.append(f"BIND(contains(?{param}, {formatted_value}) AS {filter_value})")
                    elif comparator == "startsWith":
                        filter_conditions.append(f"BIND(strstarts(?{param}, {formatted_value}) AS {filter_value})")
                    elif comparator == "endsWith":
                        filter_conditions.append(f"BIND(strends(?{param}, {formatted_value}) AS {filter_value})")
                    elif comparator == "in":
                        # TODO: This comparator was commented out in the original code, understand why
                        pass
                    else:  # =, !=, >, <, >=, <=
                        filter_conditions.append(f"BIND((?{param} {comparator} {formatted_value}) AS {filter_value})")
                    filter_values.append(filter_value)
                else:
                    raise ValueError(f"Unsupported comparison operator: {comparator}")

        formatted_where_clause = "\n".join(where_clauses) if where_clauses else ""
        formatted_filter_conditions = "\n".join(filter_conditions) if filter_conditions else ""
        operator = " && " if match_mode.lower() == "all" else " || "
        filter_clause = f"FILTER ({operator.join(filter_values)})" if filter_values else ""

        # Values for the parameters are aggregated into a single string value using the given separator (;;;)
        select_params = " ".join(
            [f'(GROUP_CONCAT(?{param} ; SEPARATOR=";;; ") as ?{param})' for param in param_to_path]
        )

        group_by_params = " ".join([f"?{param}" for param in param_to_path])

        return QueryFiltersForTargetConditions(param_to_path=param_to_path,
                                               where_clause=formatted_where_clause,
                                               filter_conditions=formatted_filter_conditions,
                                               filter_clause=filter_clause,
                                               select_params=select_params,
                                               group_by_params=group_by_params)

    @staticmethod
    def order_sparql_response(response: SparqlJsonResponse, uris: list[str], variable: str) -> None:
        """Order the result bindings in the given SPARQL JSON response by the order of the given entity URI collection.

        This method makes sure the result bindings in the SPARQL response are sorted in the same order as the
        given entities. Making this step explicit is required for SPARQL queries which are not guaranteeing input
        order in the response, e.g., in the tools accepting filter conditions.

        Changes to the given SPARQL JSON response are done in place.

        Args:
            response: SPARQL JSON response.
            uris: URIs to order the result bindings by.
            variable: Variable in the result bindings defining the order key.
        """
        source_uris_order: dict[str, int] = {uri: i for i, uri in enumerate(uris)}
        response.results.bindings.sort(key=lambda b: source_uris_order[b[variable].value])

    async def tool_navigate_path(
        self,
        *,
        context: Context, # implicit tool argument, not documented in doc string
        ids: list[str],
        path: str,
        target_types: list[str] | None = None,
        target_conditions: list[list[str]] | None = None,
        match_mode: str = "all",
    ) -> ToolOutput:
        """Navigate from the given source entities to the target entities by following the given path.

        Args:
            ids: IDs of the source entities
            path: Property path in the knowledge graph to navigate from the source entities
            target_types: Expected types of the target entities, i.e., only target entities are considered whose types
                match one of the given target types. This is an optional argument, if no target types are given all
                entities reachable from the given source entities via the given path will be considered.
            target_conditions: Conditions to filter the target entities by depending on the given match mode.
            match_mode: Match mode. Supported values are 'all' or 'any'. If match mode 'all' is selected, an entity
                needs to match all the given target conditions. If match mode 'any' is selected, an entity only needs
                to match at least one of the given target conditions.

        Returns:
            A tool output containing the entities which could be navigated to from the given source entities via the
            given path and which meet the given target conditions. Additionally, the tool output contains the relations
            between all identified entities.
        """
        # Method contract (not part of doc string as not relevant for the tool description presented to the LLM):
        # This method assumes that all source entities given through the 'ids' parameter are present in the context.
        # Ultimately, this is guaranteed by the tool executor which resolves the entity IDs the LLM provides as
        # arguments to the tool call with the URIs of these entities from the context. In case this method is not
        # called through the tool executor, the caller needs to make sure that all given source entities are present
        # in the given context as well.

        # Format the given entity URIs as syntactically correct SPARQL resource identifiers
        # with '<' and '>' as delimiters, e.g., <http://example.org/book/book1>.
        # Note: Even though the parameter is called 'ids' and the doc string refers to entity IDs,
        # the tool executor will replace these entity IDs with entity URIs before calling the tool.
        formatted_uris = " ".join(f"<{uri}>" for uri in ids)

        # Resolve all property names in the given path to their corresponding property URIs. For example,
        # given the ontology defines a property 'Is_In_Table', the path '^Is_In_Table' would be resolved
        # to the path '^<http://sap.com/datasphere/deepsea/repository#r_isInTable>'.
        path_with_resolved_properties = self.ontology_provider.resolve_path_properties(path)

        relation_name = f'"{path}"'

        types_clause = ""
        if target_types:
            formatted_types = " ".join([f"<{self.ontology_provider.get_type_uri(t, t)}>" for t in target_types])
            types_clause = f"\tVALUES ?type {{ {formatted_types} }}\n\t?uri a ?type ."

        parsed_target_conditions = self._parse_target_conditions(target_conditions, match_mode)

        query = Template(textwrap.dedent("""
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?uri ?source_uri ?source_relation $select_params
        WHERE {
            VALUES ?source_uri { $uris }
            ?source_uri $path ?uri .
            $types_clause
            BIND ($name as ?source_relation)
            $where_clause
            $filter_conditions
            $filter_clause
        }
        GROUP BY ?uri ?source_uri ?source_relation $group_by_params
        LIMIT $limit
        """)).substitute(
            select_params=parsed_target_conditions.select_params,
            uris=formatted_uris,
            path=path_with_resolved_properties,
            types_clause=types_clause,
            name=relation_name,
            where_clause=parsed_target_conditions.where_clause,
            filter_conditions=parsed_target_conditions.filter_conditions,
            filter_clause=parsed_target_conditions.filter_clause,
            group_by_params=parsed_target_conditions.group_by_params,
            limit=1000
        )

        response: SparqlJsonResponse = await self.knowledge_graph_client.execute_sparql_query(query)

        # Make sure the resulting target entities are sorted in the same order as the
        # given source entities as the SPARQL query is not guaranteeing this order
        # TODO: Why are we doing this? Is this needed?
        self.order_sparql_response(response, ids, "source_uri")

        # Collect URIs of the found target entities and build a lookup table mapping the
        # target entity URI to the result binding in the SPARQL response it stems from
        target_entity_uris: list[str] = []
        results_binding_by_uri_lookup_table: dict[str, dict[str, SparqlRdfTerm]] = {}
        for binding in response.results.bindings:
            uri = binding["uri"].value
            target_entity_uris.append(uri)
            results_binding_by_uri_lookup_table[uri] = binding

        # Get entity information for the found target entities
        target_entities, extra_entities = await self.entity_retriever.get_entities(target_entity_uris, context)

        # Collect the relations defined by the given path between source and target entities
        relations: list[Relation] = []
        for target_entity in target_entities:
            binding = results_binding_by_uri_lookup_table[target_entity.uri]

            source_relation = binding["source_relation"].value
            source_uri = binding["source_uri"].value
            source_id = context.get_entity_by_uri(source_uri).id
            relation = Relation(target_uri=target_entity.uri,
                                target_id=target_entity.id,
                                source_uri=source_uri,
                                source_id=source_id,
                                relation=source_relation)
            relations.append(relation)

            # Update properties of the found target entities with the properties resolved as part of the target
            # conditions. Given, for example, a target condition ['Name', 'contains', 'Foo'] the SPARQL query
            # returns the value(s) for the property, allowing us to add these property values to the collection
            # of known properties for the found target entities.
            properties = self.entity_retriever.collect_entity_properties(
                binding, parsed_target_conditions.param_to_path
            )

            target_entity.properties.update(properties)

        return ToolOutput(entities=target_entities, extra_entities=extra_entities, relations=relations)

    async def tool_get_entities_matching_conditions(
        self,
        *,
        context: Context, # implicit tool argument, not documented in doc string
        types: list[str],
        target_conditions: list[list[str]] | None = None,
        match_mode: str = "all",
    ) -> ToolOutput:
        """Find entities of a specific type.

        This tool yields all entities whose types match at least one of the given types.

        Args:
            types: Types to search entities for.
            target_conditions: Conditions to filter the found entities by depending on the given match mode.
            match_mode: Match mode. Supported values are 'all' or 'any'. If match mode 'all' is selected, an entity
                needs to match all the given target conditions. If match mode 'any' is selected, an entity only needs
                to match at least one of the given target conditions.

        Returns:
            A tool output containing the found entities which meet the given conditions.
        """
        parsed_target_conditions = self._parse_target_conditions(target_conditions, match_mode)

        formatted_types = " ".join([f"<{self.ontology_provider.get_type_uri(t, t)}>" for t in types])

        query = Template(textwrap.dedent("""
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?uri $select_params
        WHERE {
            VALUES ?type { $types }
            ?uri a ?type .
            $where_clause
            $filter_conditions
            $filter_clause
        }
        GROUP BY ?uri $group_by_params
        LIMIT $limit
        """)).substitute(
            select_params=parsed_target_conditions.select_params,
            types=formatted_types,
            where_clause=parsed_target_conditions.where_clause,
            filter_conditions=parsed_target_conditions.filter_conditions,
            filter_clause=parsed_target_conditions.filter_clause,
            group_by_params=parsed_target_conditions.group_by_params,
            limit=1000
        )

        response: SparqlJsonResponse = await self.knowledge_graph_client.execute_sparql_query(query)

        # Collect URIs of the found entities and build a lookup table mapping the
        # found entity URI to the result binding in the SPARQL response it stems from
        entity_uris: list[str] = []
        results_binding_by_uri_lookup_table: dict[str, dict[str, SparqlRdfTerm]] = {}
        for binding in response.results.bindings:
            uri = binding["uri"].value
            entity_uris.append(uri)
            results_binding_by_uri_lookup_table[uri] = binding

        # Get entity information for the found entities
        entities, extra_entities = await self.entity_retriever.get_entities(entity_uris, context)

        # Update properties of the found entities with the properties resolved as part of the target
        # conditions. Given, for example, a target condition ['Name', 'contains', 'Foo'] the SPARQL query
        # returns the value(s) for the property, allowing us to add these property values to the collection
        # of known properties for the found entities.
        for entity in entities:
            binding = results_binding_by_uri_lookup_table[entity.uri]
            properties = self.entity_retriever.collect_entity_properties(
                binding, parsed_target_conditions.param_to_path
            )
            entity.properties.update(properties)

        return ToolOutput(entities=entities, extra_entities=extra_entities)

    async def tool_get_properties(
        self,
        *,
        context: Context, # implicit tool argument, not documented in doc string
        ids: list[str],
        paths: list[str]
    ) -> ToolOutput:
        """Look up the properties defined by the given paths of the given entities.

        Each path can contain multiple relations that navigate from the entity to the property value.

        IMPORTANT: Do not call this tool with an empty list of entity IDs, it will always return no results!

        Args:
            ids: IDs of the source entities.
            paths: Possible property paths.

        Returns:
            A tool output containing the given entities whose properties have been updated with the properties
            found through the given paths.
        """
        if not ids:
            raise ValueError("List of entity IDs cannot be empty")

        # Format the given entity URIs as syntactically correct SPARQL resource identifiers
        # with '<' and '>' as delimiters, e.g., <http://example.org/book/book1>.
        # Note: Even though the parameter is called 'ids' and the doc string refers to entity IDs,
        # the tool executor will replace these entity IDs with entity URIs before calling the tool.
        formatted_uris = " ".join(f"<{uri}>" for uri in ids)

        # Create query filters based on the given property paths
        param_to_path: dict[str, str] = {}
        value_clauses: list[str] = []

        for i, path in enumerate(paths):
            param = f"value_{i}"
            param_to_path[param] = path
            path_with_resolved_properties = self.ontology_provider.resolve_path_properties(path)
            value_clause = f"OPTIONAL {{\n  ?uri {path_with_resolved_properties} ?{param} .\n}}"
            value_clauses.append(value_clause)

        formatted_value_clauses = "\n".join(value_clauses)

        select_params = " ".join(
            [f'(GROUP_CONCAT(?{param} ; SEPARATOR=";;; ") as ?{param})' for param in param_to_path]
        )
        group_by_params = " ".join([f"?{param}" for param in param_to_path])

        query = Template(textwrap.dedent("""
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?uri $select_params
        WHERE {
            VALUES ?uri { $uris }
            $value_clauses
        }
        GROUP BY ?uri $group_by_params
        LIMIT $limit
        """)).substitute(
            uris=formatted_uris,
            select_params=select_params,
            value_clauses=formatted_value_clauses,
            group_by_params=group_by_params,
            limit=1000
        )

        response: SparqlJsonResponse = await self.knowledge_graph_client.execute_sparql_query(query)

        # Make sure the resulting entities are sorted in the same order as the
        # given entities as the SPARQL query is not guaranteeing this order
        self.order_sparql_response(response, ids, "uri")

        # Collect URIs of the found entities and build a lookup table mapping the
        # found entity URI to the result binding in the SPARQL response it stems from
        entity_uris: list[str] = []
        results_binding_by_uri_lookup_table: dict[str, dict[str, SparqlRdfTerm]] = {}
        for binding in response.results.bindings:
            uri = binding["uri"].value
            entity_uris.append(uri)
            results_binding_by_uri_lookup_table[uri] = binding

        # Get entity information for the found entities
        entities, extra_entities = await self.entity_retriever.get_entities(entity_uris, context)

        # Update properties of the retrieved entities with the properties resolved through the given paths
        for entity in entities:
            binding = results_binding_by_uri_lookup_table[entity.uri]
            # TODO: For now we assume the property values are always literals
            properties = self.entity_retriever.collect_entity_properties(binding, param_to_path)
            entity.properties.update(properties)

        return ToolOutput(entities=entities, extra_entities=extra_entities)

