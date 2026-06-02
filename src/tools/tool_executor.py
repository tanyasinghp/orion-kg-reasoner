import asyncio
import logging
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field
if TYPE_CHECKING:
    from langchain_core.tools.structured import StructuredTool

from agent.ontology import OntologyProvider
from config import AgentConfig
from agent.agent_context import Context, Entity, Relation, ToolLog, ToolLogGroup
from agent.ranker import Ranker
from agent.util import IdGenerator
from tools.tool_provider import ToolProvider
from trace import trace


logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    """A class for a tool call identified by the agent.

    Attributes:
        tool_name: Tool name
        reason: Explanation why the tool should be called
        intent_ids: The intents that justify the call
        args: Tool arguments
    """
    tool_name: str = Field(frozen=True)
    reason: str | None = Field(frozen=True, default=None)
    intent_ids: list[str] = Field(frozen=True, default_factory=list)
    args: dict[str, Any] = Field(frozen=True)


class ToolOutput(BaseModel):
    """The output of a tool call.

    Attributes:
        information: Information provided by the tool.
        entities: All entities returned by the tool.
        extra_entities: Additional entities found during tool execution which do not directly relate to the tool call.
            This field is typically populated when the agent is used with ontologies defining extra properties. Extra
            properties are centrally defined properties each entity has, e.g., a parent ABAP table or a parent CDS view
            in the S/4HANA data model, and are resolved in each tool call retrieving an entity from the knowledge graph.
            A tool can return these additional entities relating to these extra properties through this field. This
            allows the tool executor to also register these additional entities with the agent context, making them
            available in later tool calls without requiring them to be looked up in the knowledge graph again.
        relations: All relations returned by the tool.
    """
    information: str | None = Field(frozen=True, default=None)
    entities: list[Entity] = Field(frozen=True, default_factory=list)
    extra_entities: list[Entity] = Field(frozen=True, default_factory=list)
    relations: list[Relation] = Field(frozen=True, default_factory=list)


class ToolExecutorBlueprint:
    """A blueprint for a tool executor defining the static aspects of the executor.

    Here "static aspects" of the executor refer to the properties which can be defined when initializing the
    agent session, e.g., available tools, the used ontology, and the ranker. Other aspects of the executor, e.g.,
    the agent context used by the executor, only become available when the agent session is started.
    The agent session will then create a tool executor from this blueprint with the agent context determined by
    the query parser.

    Attributes:
        config: Agent configuration.
        tool_provider: Provider for the tools available to the agent.
        ontology_provider: Ontology provider.
        ranker: Ranker for tool outputs (entities and relations).
    """

    def __init__(self,
                 config: AgentConfig,
                 tool_provider: ToolProvider,
                 ontology_provider: OntologyProvider,
                 ranker: Ranker
                 ) -> None:
        self.config = config
        self.tool_provider = tool_provider
        self.ontology_provider = ontology_provider
        self.ranker = ranker


class ToolExecutor(ToolExecutorBlueprint):
    """An executor for the tools available to the agent.

    Attributes:
        config: Agent configuration.
        tool_provider: Provider for the tools available to the agent.
        ontology_provider: Ontology provider.
        ranker: Ranker for tool outputs (entities and relations).
        context: Agent context.
    """

    def __init__(self,
                 config: AgentConfig,
                 tool_provider: ToolProvider,
                 ontology_provider: OntologyProvider,
                 ranker: Ranker,
                 tool_log_id_generator: IdGenerator,
                 context: Context
                 ) -> None:
        super().__init__(config, tool_provider, ontology_provider, ranker)
        self.tool_log_id_generator = tool_log_id_generator
        self.context = context

    @staticmethod
    def from_blueprint(
        blueprint: ToolExecutorBlueprint,
        tool_log_id_generator: IdGenerator,
        context: Context
    ) -> "ToolExecutor":
        """Create a tool executor from a blueprint.

        Args:
            blueprint: Tool executor blueprint.
            tool_log_id_generator: Generator for fresh tool log IDs
            context: Agent context.

        Returns:
            A tool executor from the given blueprint using the given context.
        """
        return ToolExecutor(config=blueprint.config,
                            tool_provider=blueprint.tool_provider,
                            ontology_provider=blueprint.ontology_provider,
                            ranker=blueprint.ranker,
                            tool_log_id_generator=tool_log_id_generator,
                            context=context)

    async def execute_tool(self, tool_call: ToolCall, tool_log_group: ToolLogGroup) -> ToolOutput | None:
        """Execute a tool call identified by the agent.

        1. Collect all relevant arguments for the given tool call from the agent context. In this step the values of
           all arguments referencing entity IDs are translated to the actual entity URIs stored in the agent context.
        2. Execute the tool call.
        3. Add a tool execution log to agent context (except for in-place tools).
        4. Register all entities in the tool output (independent of whether they should be shown to the LLM or not)
           in the agent context to have them available in later tool calls.
        5. Update the entities and relations in the agent context which should be visible to the agent. This is
           achieved by ranking the entities and relations in all tool execution logs and the conversation history
           based on the ranking features set during tool execution.

        Args:
            tool_call: Tool call identified by the agent.
            tool_log_group: Tool log group in which the record of the executed tool call should be collected.

        Returns:
            The output of the tool call if the tool execution succeeded.
        """
        tool_output: ToolOutput | None = None
        error: str | None = None

        # Lookup registered tool to be called in the tool provider.
        # Before executing the tool, deal with a frequently hallucinated prefix of the tool name.
        # TODO: Should we just silently sanitize this or give the LLM a high prio message
        #  instructing it to not add anything to the tool names?
        tool_name: str = tool_call.tool_name.replace("functions.", "")
        tool: StructuredTool = self.tool_provider.registered_tools[tool_name]

        try:
            # Collect tool arguments
            tool_args: dict[str, Any] = self._collect_tool_arguments(tool_call)
            trace("tool_executor.py", f"Collected args: { {k: (str(v)[:80] + '...') if isinstance(v, (list, str)) and len(str(v)) > 80 else v for k, v in tool_args.items() if k != 'context'} }")

            # Call tool function
            trace("tool_executor.py", f"Calling tool '{tool_name}'...")
            tool_output = await asyncio.wait_for(tool.arun(tool_args), timeout=self.config.tool_timeout)
            trace("tool_executor.py", f"Tool '{tool_name}' returned: {len(tool_output.entities)} entities, {len(tool_output.relations)} relations")
        except TimeoutError:
            error = f"Tool '{tool_name}' timed out after {self.config.tool_timeout} seconds"
        except Exception as e:
            error = f"Failed to execute tool '{tool_name}': {e}"
            logger.exception(error)

        # Add tool log to context (except for in-place tools)
        if tool_output is not None:
            tool_log = self._create_tool_log(tool_call, tool_output, error)
            self.context.add_tool_log(tool_log, tool_log_group)

        # Register all entities in the tool output in the agent context to have them available in later tool calls
        if tool_output is not None:
            for entity in tool_output.entities:
                self.context.register_entity(entity)
            for extra_entity in tool_output.extra_entities:
                self.context.register_entity(extra_entity)

        # In case an error occurred, also add it to the error log in the agent context
        if error is not None:
            self.context.errors.append(error)

        # Update the entities and relations in the agent context which should be visible to the agent
        await self._update_agent_context()

        return tool_output

    def _collect_tool_arguments(self, tool_call: ToolCall) -> dict[str, Any]:
        """Collect all relevant arguments for the given tool call from the agent context.

        This method translates the values of all arguments referencing entity IDs to the actual entity URIs stored
        in the agent context.

        Note: This implementation assumes that all tool arguments referencing multiple entity IDs are of type 'List'.
        That is, the translated values of these tool arguments will be a list, and tool implementations need to adhere
        to this contract.

        Args:
            tool_call: Tool call

        Returns:
             A dictionary with the tool arguments where all entity IDs have been replaced with entity URIs.
        """
        tool_args: dict[str, Any] = {"context": self.context}
        for arg_name, arg_value in tool_call.args.items():
            # Ignore the argument for the context from the args given in the tool call if present
            if arg_name == "context":
                pass
            # Translate all tool inputs with the suffix "_ids" to entity URIs
            elif arg_name == "ids" or arg_name.endswith("_ids"):
                # Translate IDs to entities and limit the number of input entities provided to the tool call
                entities = self.context.get_entities_by_ids(arg_value)
                selected_entities = self.ranker.rank_and_select_entities(entities, self.config.max_tool_input)

                # Replace entity IDs with actual entity URIs in the tool input
                tool_args[arg_name] = [e.uri for e in selected_entities]
            else:
                tool_args[arg_name] = arg_value
        return tool_args

    async def _update_agent_context(self) -> None:
        """Update the entities and relations in the agent context which should be visible to the agent.

        Select the entities and relations which should be visible to the agent from the tool logs and the
        conversation history and make sure relevant entity types exposed to the agent contain the types of
        all these selected entities.

        This method should be called after each tool execution to make sure the most relevant entities and
        relations identified by the tool are visible to the agent.
        """
        # Select the entities from the tool logs and the conversation history as well
        # as the relations from the tool logs which should be visible to the agent
        selected_entities: set[Entity] = set()
        selected_relations_from_tool_logs: set[Relation] = set()

        for history in self.context.history:
            entities_from_history = {
                self.context.id_to_entity[entity_id] for entity_id in history.entities
            }
            selected_entities.update(entities_from_history)

        for tool_log in self.context.tool_logs:
            selected_entities.update(tool_log.selected_entities)

            for relation in tool_log.selected_relations:
                selected_entities.add(self.context.id_to_entity[relation.source_id])
                selected_entities.add(self.context.id_to_entity[relation.target_id])

                selected_relations_from_tool_logs.add(relation)

        # Set entities which should be visible to the agent
        self.context.entities = sorted(selected_entities, key=lambda e: e.index)

        # Set relations which should be visible to the agent
        self.context.relations = self.ranker.rank_and_select_relations(
            self.context, selected_relations_from_tool_logs, self.config.max_relations
        )

        # Add the types of the selected entities to the relevant types to make sure the LLM knows their details
        types_of_selected_entities: set[str] = set()
        for entity in selected_entities:
            types_of_selected_entities.update(entity.types)
        self.context.add_relevant_types(types_of_selected_entities)

        # Reduce the ontology contained in the agent context to only contain information related
        # to the relevant types known to the agent
        self.context.ontology = self.ontology_provider.format_subset(self.context.relevant_types)

    def _create_tool_log(self,
                         tool_call: ToolCall,
                         tool_output: ToolOutput | None,
                         error: str | None
                         ) -> ToolLog:
        """Create a tool execution log for the given tool call and its output.

        In case the tool output contains entities or relations, this method ranks the entities
        and relations and selects a subset from the full tool output which will become visible
        to the agent. The size of this subset is defined through the agent configuration.

        Args:
             tool_call: Tool call.
             tool_output: Tool output in case the tool call succeeded.
             error: Error message in case the tool call didn't succeed.

        Returns:
            A log record containing all relevant information about the tool execution.
        """
        # Generate a fresh tool log ID
        tool_log_id = f"T-{self.tool_log_id_generator.fresh_id()}"

        if error is None and tool_output is not None:
            # Tool only provided information and no entities as output
            if tool_output.information is not None:
                return ToolLog(id=tool_log_id,
                               reason=tool_call.reason,
                               purposes=tool_call.intent_ids,
                               tool_name=tool_call.tool_name,
                               args=tool_call.args,
                               info_only=True,
                               info=tool_output.information)
            # Tool provided entities or relations as output
            else:
                # Rank and select tool output (entities, relations) which should be made available to the agent
                selected_entities = self.ranker.rank_and_select_entities(
                    tool_output.entities, self.config.max_tool_output_observed
                )

                selected_relations = self.ranker.rank_and_select_relations(
                    self.context, tool_output.relations, self.config.max_tool_output_observed
                )

                return ToolLog(id=tool_log_id,
                               reason=tool_call.reason,
                               purposes=tool_call.intent_ids,
                               tool_name=tool_call.tool_name,
                               args=tool_call.args,
                               entities=tool_output.entities,
                               selected_entities=selected_entities,
                               relations=tool_output.relations,
                               selected_relations=selected_relations)
        else:
            # Error log
            return ToolLog(id=tool_log_id,
                           reason=tool_call.reason,
                           purposes=tool_call.intent_ids,
                           tool_name=tool_call.tool_name,
                           args=tool_call.args,
                           error=error)
