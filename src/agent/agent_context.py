import textwrap
from typing import Any, Self
from collections.abc import Iterable
from collections import defaultdict
from pprint import pformat
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field


class Intent(BaseModel):
    """Categories for user queries.

    Intents help the agent to categorize user queries and select the right tools to call
    to collect relevant information from the knowledge graph to answer a query.

    Attributes:
        id: ID for the intent data.
        intent: Intent descriptor.
        num_args: Number of arguments.
        description: Description of the intent.
    """
    id: str = Field(frozen=True)
    intent: str = Field(frozen=True)
    num_args: int = Field(frozen=True)
    description: str = Field(frozen=True)


class BackgroundInformation(BaseModel):
    """A class for background information provided to the agent.

    Background information are a mechanism to provide curated knowledge around a specified topic to the agent.

    Attributes:
        id: ID for the background information.
        topic: Topic of the background information.
        knowledge: Knowledge defined by the background information.
        mandatory: Flag indicating whether the background information should always be considered by the agent.
        add_tool: Name of the tool that should be added to the tool set of the agent when the background information
            is considered relevant for the agent session based on the user query.
    """
    id: str = Field(frozen=True)
    topic: str = Field(frozen=True)
    knowledge: str = Field(frozen=True)
    mandatory: bool = Field(frozen=True, default=False, validate_default=True)
    add_tool: str | None = Field(frozen=True, default=None, validate_default=True)


class Entity(BaseModel):
    """A class for entities in the knowledge graph.

    Attributes:
        uri: URI of the entity.
        name: Name of the entity.
        description: Description for the entity.
        types: Types of the entity.
        properties: Properties of the entity.
        index: Index of the entity, 0-based.
        ranking_features: Features for ranking.
        id_prefix: ID prefix used for generating the ID of the entity.
    """
    uri: str = Field(frozen=True)
    name: str
    description: str | None = Field(default=None)
    types: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    index: int = Field(frozen=True, default=0, validate_default=True)
    ranking_features: dict[str, float] = Field(default_factory=lambda: defaultdict(float))
    id_prefix: str = Field(frozen=True, default="E", validate_default=True)

    @property
    def id(self) -> str:
        """Get the entity ID.

        This method generates an entity ID from the 'id_prefix' and the 'index' fields in the format 'E-0', 'E-1', etc.

        Returns:
             The generated ID of the entity.
        """
        return f"{self.id_prefix}-{self.index}"

    def format_for_prompt(self) -> str:
        """Format this entity for usage in an LLM prompt.

        Returns:
             A string representation of this entity suitable for usage in an LLM prompt.
        """
        # TODO: Limit entity description length in prompt via configuration
        short_description = self.description

        # Comma-seperated properties in an assignment-like format, e.g., ", prop1=foo, prop2=bar, prop3=42"
        properties = (
            ", " + ", ".join(f"{name}={value}" for name, value in self.properties.items() if value != "")
            if self.properties else ""
        )

        types = "|".join(self.types) if self.types else "?"

        return f'({self.id}: name={self.name}, types={types}, description="{short_description}"{properties})'

    def __eq__(self, other: Self) -> bool:
        return self.uri == other.uri

    def __hash__(self) -> int:
        return hash(self.uri)


class UnresolvedRelation(BaseModel):
    """A class for relations between entities in the knowledge graph.

    This class represents the "raw" relation as obtained from the knowledge graph, i.e., with entity URIs
    representing the source and the target of the relation. It is typically used in the entity retriever
    when querying the relations in the knowledge graph. There the agent context isn't known, and resolving
    the entity URIs to the corresponding IDs of the entities stored in the agent context is happening at
    a later step in the tool execution layer.

    Attributes:
        target_uri: Target entity URI.
        source_uri: Source entity URI.
        relation: Name of the relation from source to target.
        properties: Additional properties of the relation.
    """
    target_uri: str = Field(frozen=True)
    source_uri: str = Field(frozen=True)
    relation: str = Field(frozen=True)
    properties: dict[str, Any] = Field(frozen=True, default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.source_uri, self.relation, self.target_uri))

class Relation(UnresolvedRelation):
    """A class for relations between entities in the knowledge graph.

    This class extends the UnresolvedRelation class with the IDs of the entities stored in the agent context
    corresponding to the entity URIs in the "raw" relation.

    Attributes:
        target_uri: Target entity URI.
        target_id: Target entity ID.
        source_uri: Source entity URI.
        source_id: Source entity ID.
        relation: Name of the relation from source to target.
        properties: Additional properties of the relation.
    """
    target_id: str = Field(frozen=True)
    source_id: str = Field(frozen=True)

    @property
    def signature(self) -> str:
        """Get the signature of this relation.

        The signature of a relation is a string representation made out of the source and target entity IDs,
        the relation name, and the properties, and can serve as a type of ID for the relation.

        Returns:
             The signature of the relation in the Cypher-like triple format '(source)-[relation]->(target)'.
        """
        properties = f", {self.properties}" if self.properties else ""
        return f"({self.source_id})-[{self.relation}{properties}]->({self.target_id})"

    def format_for_prompt(self) -> str:
        """Format this relation for usage in an LLM prompt.

        Returns:
            A string representation of this relation suitable for usage in an LLM prompt.
        """
        return self.signature

    def __hash__(self) -> int:
        return hash(self.signature)


class ToolLog(BaseModel):
    """A log record for a tool call.

    Attributes:
        id: Tool call log record ID.
        reason: Explanation why the tool was called.
        purposes: Intents that justified the tool call.
        tool_name: Name of the tool that was called.
        args: Arguments provided to the tool.
        info_only: Flag indicating whether the tool only provided information and no entities as output.
        info: Information provided by the tool.
        entities: All entities returned by the tool.
        selected_entities: Subset of the returned entities shown to the agent.
        relations: All relations returned by the tool.
        selected_relations: Subset of the returned relations shown to the agent.
        error: Runtime error of the tool
    """
    id: str = Field(frozen=True)
    reason: str | None = Field(frozen=True, default=None)
    purposes: list[str] = Field(frozen=True, default_factory=list)
    tool_name: str = Field(frozen=True)
    args: dict[str, Any] = Field(frozen=True, default_factory=dict)
    info_only: bool = Field(frozen=True, default=False)
    info: str | None = Field(frozen=True, default=None)
    entities: list[Entity] = Field(frozen=True, default_factory=list)
    selected_entities: list[Entity] = Field(frozen=True, default_factory=list)
    relations: list[Relation] = Field(frozen=True, default_factory=list)
    selected_relations: list[Relation] = Field(frozen=True, default_factory=list)
    error: str | None = Field(frozen=True, default=None)

    def format_for_prompt(self) -> str:
        """Format this tool call record for usage in an LLM prompt.

        Returns:
            A string representation of this tool call record suitable for usage in an LLM prompt.
        """
        if self.info_only:
            return f"ToolLog(id={self.id}, tool_name={self.tool_name}, info=```{self.info}```"
        else:
            error = f", error={self.error}" if self.error else ""
            reason = f'reason="{self.reason}", ' if self.reason else ""

            # Only show entity IDs to LLM as part of the tool log, not the full entity
            selected_entity_ids = [e.id for e in self.selected_entities]

            formatted_selected_relations = [r.format_for_prompt() for r in self.selected_relations]

            return (
                f"ToolLog(id={self.id}, {reason}tool_name={self.tool_name}, args={self.args}, "
                f"num_entities={len(self.entities)}, entities={selected_entity_ids}, "
                f"num_relations={len(self.relations)}, relations={formatted_selected_relations}"
                f"{error})"
            )


class ToolLogGroup(BaseModel):
    """A group of tool call logs semantically belonging together.

    A tool log group typically contains all the log records of all tool calls
    executed during the full agent loop when generating an answer for a query.

    Attributes:
        id: Tool log group ID unique within an agent session.
        tool_logs: Tool log records contained in the group.
    """
    id: str = Field(frozen=True)
    tool_logs: list[ToolLog] = Field(frozen=True, default_factory=list)


class History(BaseModel):
    """A class for the history of queries to the agent within an agent session.

    Attributes:
        question: Query to the agent.
        raw_answer: Raw, non-augmented answer generated by the agent.
        augmented_answer: Augmented answer of the agent with resolved entities.
        tool_log_group_id: ID of the tool log group containing the tool log records for
            all tools executed when generating an answer for the question.
        entities: IDs of the entities mentioned in the answer.
    """
    question: str = Field(frozen=True)
    raw_answer: str = Field(frozen=True)
    augmented_answer: str = Field(frozen=True)
    tool_log_group_id: str = Field(frozen=True)
    entities: list[str] = Field(frozen=True, default_factory=list)

    def format_for_prompt(self) -> str:
        """Format this query history record for usage in an LLM prompt.

        Returns:
             A string representation of this query history record suitable for usage in an LLM prompt.
        """
        return f'History(question="{self.question}", answer="{self.raw_answer}", entities (total {len(self.entities)}) = {self.entities[:10]})'


class UserContext(BaseModel):
    """User context.

    Attributes:
        locale: IETF BCP 47 language tag of the user's locale.
        time_zone: User time zone.
    """
    locale: str = Field(frozen=True, default="en")
    time_zone: ZoneInfo = Field(frozen=True, default=ZoneInfo("UTC"))


class Context(BaseModel):
    """A class for the agent context.

    Attributes:
        query: Original query sent to the agent
        user_context: Metadata about the user (e.g. locale, time zone) made available to the agent
        history: Conversation history
        ontology: String representation of the ontology
        relevant_types: Relevant entity types for the query
        target_types: Target entity types for the query
        query_specific_background_information_ids: IDs of relevant background information identified by the query parser
        background_information: Relevant background information identified by the query parser
        hint: Preset hint for the agent
        intents: IDs of the intents of the query identified by the query parser
        id_to_tool_log: Lookup table mapping entity IDs to tool log entries
        tool_log_groups: Lookup table mapping tool log group IDs to tool log groups
        entities: Entities exposed to the agent
        relations: Relations exposed to the agent
        uri_to_entity: Lookup table mapping entity URIs to entities
        id_to_entity: Lookup table mapping entity IDs to entities
        errors: All errors that occurred during the agent session
    """
    query: str | None = Field(default=None)
    user_context: UserContext = Field(default_factory=UserContext)
    history: list[History] = Field(default_factory=list)
    ontology: str | None = Field(default=None)
    relevant_types: list[str] = Field(default_factory=list)
    target_types: list[str] = Field(default_factory=list)

    query_specific_background_information_ids: set[str] = Field(default_factory=set)
    background_information: dict[str, BackgroundInformation] = Field(default_factory=dict)
    hint: dict[str, str] = Field(default_factory=dict)
    intents: list[str] = Field(default_factory=list)

    id_to_tool_log: dict[str, ToolLog] = Field(default_factory=dict)
    tool_log_groups: dict[str, ToolLogGroup] = Field(default_factory=dict)

    # dynamically updated, sent to the agent
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    # mappings for the entities
    uri_to_entity: dict[str, Entity] = Field(default_factory=dict)
    id_to_entity: dict[str, Entity] = Field(default_factory=dict)

    errors: list[str] = Field(default_factory=list)

    @property
    def tool_logs(self) -> Iterable[ToolLog]:
        """Tool call logs."""
        return self.id_to_tool_log.values()

    def register_entity(self, entity: Entity) -> None:
        """Register the given entity with the agent context.

        This makes the entity known to the context such that tools can look up the entity
        information without needing to query it from the knowledge graph again.

        Args:
             entity: The entity to register.
        """
        if entity.id not in self.id_to_entity:
            self.id_to_entity[entity.id] = entity
            self.uri_to_entity[entity.uri] = entity

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get the entity with the given ID.

        Args:
            entity_id: ID of the entity, e.g. 'E-0'.

        Returns:
            The entity with the given ID, or None if no such entity exists.
        """
        return self.id_to_entity.get(entity_id, None)

    def get_entities_by_ids(self, ids: list[str]) -> list[Entity]:
        """Get entities with the given IDs.

        This method provides a "fuzzy" lookup for entities in this context. That is, the given IDs are interpreted
        as entity IDs, entity URIs, as well as tool log IDs, and entities referenced by a tool log are returned as well.

        Args:
             ids: A list of IDs.

        Returns:
            All entities with the given IDs.

        Raises:
            ValueError: If a given ID is neither an entity ID, an entity URI, or a tool log ID.
        """
        entities: set[Entity] = set()
        for some_id in ids:
            # Given ID is an entity ID, e.g. 'E-5'
            if some_id in self.id_to_entity:
                entities.add(self.id_to_entity[some_id])
            # Given ID is a tool ID, e.g. 'T-1'
            elif some_id in self.id_to_tool_log:
                entities = entities.union(self.id_to_tool_log[some_id].entities)
            # Given ID is an entity URI in exceptional cases, e.g. 'http://schema.sap.com/ABAPTable/MARA'
            elif some_id in self.uri_to_entity:
                entities.add(self.uri_to_entity[some_id])
            else:
                raise ValueError(f"Unknown IDs '{some_id}'. The input IDs should be in a format like "
                                 f"['E-0', 'T-1'] and the IDs should already exist in the context.")

        return list(entities)

    def get_entity_by_uri(self, uri: str, default: Entity | None = None) -> Entity | None:
        """Get the entity with the given URI.

        Args:
            uri: Entity URI.
            default: Default entity to return if no entity with the given URI is found.

        Returns:
             The entity with the given URI, or the given default entity if no such entity exists.
        """
        return self.uri_to_entity.get(uri, default)

    def add_relevant_types(self, relevant_types: Iterable[str]) -> None:
        """Add relevant types to the agent context.

        This method only adds the given types to the agent context which are not known, yet.
        That is, this method prevents that the relevant types known to the agent contain any duplicates.

        Args:
             relevant_types: Names of relevant types to be added to the context.
        """
        for relevant_type in relevant_types:
            self.relevant_types.append(relevant_type) if relevant_type not in self.relevant_types else ...

    def add_tool_log_group(self, tool_log_group: ToolLogGroup) -> None:
        """Add the given tool log group to the agent context.

        Args:
            tool_log_group: Tool log group to be added to the agent context.
        """
        self.tool_log_groups[tool_log_group.id] = tool_log_group

    def add_tool_log(self, tool_log: ToolLog, tool_log_group: ToolLogGroup) -> None:
        """Add a tool execution log to the agent context.

        Args:
             tool_log: Tool execution log to be added to the agent context.
             tool_log_group: Tool log group to which the tool execution log should be added.
        """
        # Add tool log record
        self.id_to_tool_log[tool_log.id] = tool_log

        # Add tool log record to tool log group
        tool_log_group.tool_logs.append(tool_log)

    def get_tool_log_by_id(self, tool_log_id: str, default: ToolLog | None = None) -> ToolLog | None:
        """Get the tool log with the given ID from the agent context.

        Args:
            tool_log_id: ID of the tool log to get.
            default: Default tool log to return if no tool log with the given ID is found.

        Returns:
            The tool log with the given ID, or the given default if no such tool log exists.
        """
        return self.id_to_tool_log.get(tool_log_id, default)

    def revise_history(self, history_limit: int) -> None:
        history_to_keep = self.history[-history_limit:]  # last N items of the list
        history_to_prune = self.history[:-history_limit] # everything except the last N items of the list

        # Remove tool logs and tool log groups which are part of to be pruned history records from the context
        for history in history_to_prune:
            # Remove tool logs
            tool_log_group = self.tool_log_groups[history.tool_log_group_id]
            for tool_log in tool_log_group.tool_logs:
                del self.id_to_tool_log[tool_log.id]
            # Remove tool log group
            del self.tool_log_groups[history.tool_log_group_id]

        # Collect entities described in tools logs which should be kept
        entities_to_keep: set[Entity] = set()
        for history in history_to_keep:
            tool_log_group = self.tool_log_groups[history.tool_log_group_id]
            for tool_log in tool_log_group.tool_logs:
                entities_to_keep.update(tool_log.entities)

        # Remove all entities which are not part of the history which should be kept
        entities_to_remove: set[Entity] = {
            entity for entity in self.id_to_entity.values()
            if (entity not in entities_to_keep) and (entity not in self.entities)
        }
        for entity_to_remove in entities_to_remove:
            del self.id_to_entity[entity_to_remove.id]
            del self.uri_to_entity[entity_to_remove.uri]

        # Remove history which should be discarded
        self.history = history_to_keep

    def format_for_prompt(self) -> str:
        """Prepare a representation of the context only containing the elements which should be shown to the LLM.

        Returns:
            A string representation of the agent context suitable for usage in an LLM prompt.
        """
        formatted_history = pformat([h.format_for_prompt() for h in self.history], compact=True)
        hint = f"HINT = {self.hint}" if self.hint else ""
        background_information = sorted(self.background_information.values(), key=lambda b: b.id)
        formatted_entities = pformat([entity.format_for_prompt() for entity in self.entities], compact=True)
        formatted_relations = pformat([relation.format_for_prompt() for relation in self.relations], compact=True)
        formatted_tool_logs = pformat([tool_log.format_for_prompt() for tool_log in self.tool_logs], compact=True)

        # FIXME: textwrap.dedent isn't fully working here, maybe due to the use of pformat?
        context_representation_for_prompt = textwrap.dedent(f"""
        ====
        [HISTORY]:
        {formatted_history}
        [USER]:
        LOCALE = {self.user_context.locale}
        TIME_ZONE = {self.user_context.time_zone}
        [ONTOLOGY]:
        {self.ontology}
        [CONTEXT]:
        QUERY = {self.query}
        TARGET_TYPES = {self.target_types}
        {hint}
        BACKGROUNDS = {background_information}
        INTENTS = {self.intents}
        ENTITIES = {formatted_entities}
        RELATIONS = {formatted_relations}
        TOOL_LOG = {formatted_tool_logs}
        ====
        """)

        return context_representation_for_prompt
