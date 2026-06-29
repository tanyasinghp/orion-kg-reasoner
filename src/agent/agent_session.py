import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field

from config import AgentConfig
from agent.agent import Agent, AgentResponse, AfterReasoningStepCallback, AfterToolCallback
from agent.agent_context import Entity, ToolLog, History, Context, Relation, UserContext, ToolLogGroup
from agent.ontology import OntologyProvider
from agent.query_parser import QueryParser
from agent.util import InputTokenCounter, IdGenerator
from tools.tool_executor import ToolExecutor, ToolExecutorBlueprint
from trace import trace, set_collector
from trace_event import TraceCollector


logger = logging.getLogger(__name__)


class DebugInformation(BaseModel):
    """Additional debug information available as part of the agent response.

    Attributes:
        tool_execution_log: Logs of all tools calls and their responses.
        history: Conversation history.
        errors: All errors that occurred during the agent session.
    """
    tool_execution_log: list[ToolLog] = Field(frozen=True, default_factory=list)
    history: list[History] = Field(frozen=True, default_factory=list)
    errors: list[str] = Field(frozen=True, default_factory=list)


class AugmentedAnswer(BaseModel):
    """A class for the augmented agent response.

    In the augmented answer, all placeholders for entity groups or single entities
    have been resolved and replaced with the names of the corresponding entities.

    Attributes:
        description: Augmented agent response where all placeholders for entities have
            been resolved and replaced with the names of the corresponding entities.
            This augmented answer might not contain the names of all relevant entities,
            as some of them might have been omitted for brevity's sake.
        links:
            Entities related to the augmented answer. Each entity name in the augmented
            answer will have a corresponding entity in this collection.
    """
    description: str = Field(frozen=True)
    links: list[Entity] = Field(frozen=True, default_factory=list)


class Answer(BaseModel):
    """An answer to a query posted to the agent.

    Attributes:
        raw_answer: The raw response from the agent, including placeholders, e.g. <(E-5)>, for entities.
        entities: Lookup table for all entities relevant for the answer.
        relations: All relations relevant for the answer.
        augmented_answer: Augmented agent response where all placeholders for entities have been resolved.
            The number of entities contained in the augmented answer is typically limited for brevity's sake.
        answers: Entities forming the answer to the query posted to the agent. This collection contains
            all entities identified by the agent and not only the ones selected for the augmented answer.
        duration: Total number of seconds the agent took for generating an answer.
        debug_info: Additional debug information
    """
    raw_answer: str = Field(frozen=True)
    entities: dict[str, Entity] = Field(default_factory=dict)
    relations: list[Relation] = Field(default_factory=list)
    augmented_answer: AugmentedAnswer | None = Field(default=None)
    answers: list[Entity] = Field(default_factory=list)
    duration: float = Field(default=0.0)
    debug_info: DebugInformation | None = Field(default=None)


@dataclass
class ResolvedAnswerPlaceholder:
    """A placeholder in an agent response which has been resolved to the entities and relations it represents.

    Attributes:
        text: Full placeholder as used in the agent response, e.g., <[FULL:T-1, T-2]> or <(E-5)>.
        mode: Placeholder mode to differentiate between group IDs ('FULL', 'SAMPLE') and single IDs.
        entity_ids: Entity IDs resolved for the placeholder.
        relations: Relations resolved for the placeholder.
    """
    text: str
    mode: str # TODO: Use enum
    entity_ids: list[str]
    relations: list[Relation]


@dataclass
class RelevantEntitiesAndRelationsInAnswer:
    """All relevant entities and relations identified in an agent response.

    Attributes:
        mentioned_ids: All placeholder IDs mentioned in the agent response.
        relevant_entity_ids: IDs of all entities relevant for the answer.
        resolved_answer_placeholders: Placeholders in the agent response which
            have been resolved to the entities they represent.
    """
    mentioned_ids: set[str]
    relevant_entity_ids: set[str]
    resolved_answer_placeholders: list[ResolvedAnswerPlaceholder]


type AfterQueryParserCallback = Callable[[Context, float, int], None]
"""Callback to be invoked after the query parser has analyzed the query.

Args:
    context: Initial context resulting from the query parser.
    duration: Total number of seconds the query parser took to analyze the query.
    num_tokens: Number of input tokens consumed by the LLM invoked during query analysis.
"""


class AgentSession:
    """A class for an agent session.

    Attributes:
        agent_config: Agent configuration.
        llm: Large language model (LLM) used by the agent.
        query_parser: Query analyzer for selecting query-specific grounding information for the agent.
        ontology_provider: Provider for the ontology of the knowledge graph the agent should reason over.
        tool_executor_blueprint: Blueprint for the executor of the tools available to the agent.
        after_query_parser_callback: Callback to be invoked after the query parser has analyzed the query.
        after_reasoning_step_callback: Callback to be invoked after the agent performed a reasoning step.
        after_tool_callback: Callback to be invoked after the agent executed a tool.
        input_token_counter: Counter for the input tokens consumed by the LLM throughout the agent session.
        agent: Agent initialized for the session.
        context: Context of the agent session.
        active_conversation: Flag indicating that the session holds an active conversation, i.e., an answer
            to a query was previously generated in this session and is recorded in the history of the session
            context. This flag is used to determine whether a query is part of a continuous conversation and
            (parts of) the context of previous queries should be considered for the respecitve query as well.
    """

    def __init__(self,
                 agent_config: AgentConfig,
                 llm: BaseLanguageModel,
                 query_parser: QueryParser,
                 ontology_provider: OntologyProvider,
                 tool_executor_blueprint: ToolExecutorBlueprint,
                 after_query_parser_callback: AfterQueryParserCallback | None = None,
                 after_reasoning_step_callback: AfterReasoningStepCallback | None = None,
                 after_tool_callback: AfterToolCallback | None = None,
                 input_token_counter: InputTokenCounter | None = None,
                 trace_collector: TraceCollector | None = None
                 ) -> None:
        """Create an agent session.

        Args:
            agent_config: Agent configuration.
            llm: Large language model (LLM) used by the agent.
            query_parser: Query analyzer for selecting query-specific grounding information for the agent.
            ontology_provider: Provider for the ontology of the knowledge graph the agent should reason over.
            tool_executor_blueprint: Blueprint for the executor of the tools available to the agent.
            after_query_parser_callback: Callback to be invoked after the query parser has analyzed the query.
            after_reasoning_step_callback: Callback to be invoked after the agent performed a reasoning step.
            after_tool_callback: Callback to be invoked after the agent executed a tool.
        """
        self.agent_config = agent_config
        self.llm = llm
        self.query_parser = query_parser
        self.ontology_provider = ontology_provider
        self.tool_executor_blueprint = tool_executor_blueprint
        self.after_query_parser_callback = after_query_parser_callback
        self.after_reasoning_step_callback = after_reasoning_step_callback
        self.after_tool_callback = after_tool_callback
        self.trace_collector = trace_collector
        self.agent: Agent = ...
        self.context: Context = ...
        self.active_conversation: bool = False

        # Create a fresh input token counter counting the LLM tokens consumed
        # throughout the session if none is provided through the method argument
        self.input_token_counter = input_token_counter or InputTokenCounter(self.llm)

        # Generated IDs for tool logs and tools log groups are unique within the agent session only
        self._tool_log_id_generator = IdGenerator()
        self._tool_log_group_id_generator = IdGenerator()

    async def generate_answer(
        self,
        query: str,
        top_n: int | None = None,
        debug: bool = False,
        user_context: UserContext | None = None,
    ) -> Answer:
        """Generate an answer for the given query.

        When generating an answer, the agent session runs the full reasoning process:

        1. (LLM) query parser: analyze the query and select query-specific grounding information
           (intents, background information) for the agent.
        2. (LLM) agent:
           - Reason what (knowledge graph) tools to execute in a loop.
           - Generate a final answer with placeholders (entity IDs).
        3. (non-LLM) augment answer: translate the placeholder IDs in the answer into entities
           stored in the agent context.

        Args:
            query: Query the agent should generate an answer for.
            top_n: Maximum number of entities to consider when replacing the placeholder IDs in the agent response
                with entities. The full set of entities identified by the agent response will be contained in the
                answer as well; this parameter only controls how many of the entities will be stated in the augmented
                textual answer.
            debug: Add additional debug information to the answer.
            user_context: Customer user context (locale, time zone) to be considered by the agent.

        Returns:
            An answer containing the augmented agent response and all resolved entities referenced by placeholders
            in the agent response.
        """
        start_time = datetime.now()

        # Set trace collector for UI event capture
        if self.trace_collector is not None:
            set_collector(self.trace_collector)

        # Analyze the query to identify intents and background information relevant to the query.
        # Possibly exit early if the user asked an inappropriate question which is rejected by the LLM.
        try:
            trace("agent_session.py", "Phase 1: Query parsing...", event_type="phase", data={"phase": "query_parser", "query": query})
            query_parser_start_time = datetime.now()
            context = await self.query_parser.parse_query(query, self.input_token_counter)
            query_parser_end_time = datetime.now()
            query_parser_duration = (query_parser_end_time - query_parser_start_time).total_seconds()
            trace("agent_session.py", f"Phase 1 done: intents={context.intents}, target_types={context.target_types}", data={
                "intents": context.intents,
                "target_types": context.target_types,
                "relevant_types": context.relevant_types,
            })

            if self.after_query_parser_callback is not None:
                self.after_query_parser_callback(context, query_parser_duration, self.input_token_counter.last_count)
        except Exception as exc:
            trace("agent_session.py", f"Phase 1 error: {exc}", event_type="error", data={"error": str(exc)})
            answer = (
                "We cannot answer this question, the possible reason could be that the question is inappropriate "
                "or the LLM has encountered a temporary issue. Please retry or ask a more relevant question."
            )
            logger.exception("Unable to answer question")

            # Early exit
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return Answer(raw_answer=answer, duration=duration)

        # If the session holds an active conversation, update the existing agent context held by this session with
        # the query-specific information determined by the query parser, i.e., query-specific intents and background
        # information, target types, etc. If this is the first query to the agent session, initialize the agent
        # context of this session with the initial context created by the query parser.
        if not self.active_conversation:
            self.context = context
        else:
            self._update_existing_session_context(context)
            logger.debug(
                "Continuous conversation detected. Updated existing agent context with query analyzer results."
            )

        # Set custom user context if provided
        if user_context is not None:
            self.context.user_context = user_context

        # Create a tool log group for collecting the records of all tool calls executed when
        # generating an answer for the given query and register the group in the agent context
        tool_log_group_id = f"TG-{self._tool_log_group_id_generator.fresh_id()}"
        tool_log_group = ToolLogGroup(id=tool_log_group_id)
        self.context.add_tool_log_group(tool_log_group)

        # Run the agent loop with the defined initial context as the initial agent state
        trace("agent_session.py", "Phase 2: Starting agent reasoning loop...", event_type="phase", data={"phase": "agent_loop"})
        tool_executor = ToolExecutor.from_blueprint(
            self.tool_executor_blueprint, self._tool_log_id_generator, self.context
        )
        # Wrap callbacks to emit trace events for the UI
        trace_collector = self.trace_collector

        def _step_callback(response, duration, num_tokens):
            if self.after_reasoning_step_callback:
                self.after_reasoning_step_callback(response, duration, num_tokens)
            if response.type == "answer":
                trace("agent.py", f"Step: LLM returned ANSWER ({len(response.answer)} chars)", event_type="step",
                      data={"step_response_type": "ANSWER", "duration": duration})
            else:
                tc = response.tool_call
                trace("agent.py", f"Step: LLM decided TOOL_CALL '{tc.tool_name}'", event_type="tool_call",
                      data={
                          "tool_name": tc.tool_name,
                          "reason": tc.reason,
                          "intent_ids": tc.intent_ids,
                          "args": tc.args,
                          "duration": duration,
                      })

        def _tool_callback(tool_call, tool_output, duration):
            if self.after_tool_callback:
                self.after_tool_callback(tool_call, tool_output, duration)
            n_entities = len(tool_output.entities) if tool_output else 0
            n_relations = len(tool_output.relations) if tool_output else 0
            trace("tool_executor.py", f"Tool '{tool_call.tool_name}' returned: {n_entities} entities, {n_relations} relations",
                  event_type="tool_result",
                  data={
                      "tool_name": tool_call.tool_name,
                      "entities_count": n_entities,
                      "relations_count": n_relations,
                  })

        self.agent = Agent(config=self.agent_config,
                           llm=self.llm,
                           ontology_provider=self.ontology_provider,
                           tool_executor=tool_executor,
                           context=self.context,
                           after_reasoning_step_callback=_step_callback,
                           after_tool_callback=_tool_callback,
                           input_token_counter=self.input_token_counter)
        agent_response = await self.agent.run_agent_loop(tool_log_group)

        # Augment answer in agent response with data from agent context
        trace("agent_session.py", "Phase 3: Augmenting answer with entity data...", event_type="phase", data={"phase": "augmentation"})
        answer = self._augment_answer(agent_response, self.agent.context, top_n)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        answer.duration = duration
        logger.debug("Agent returned response of type %s", agent_response.type)

        # Add debug information if requested
        if debug:
            debug_info = DebugInformation(tool_execution_log=tool_log_group.tool_logs,
                                          history=self.context.history,
                                          errors=self.context.errors)
            answer.debug_info = debug_info

        # Record query and the found answer in the history of the agent context
        history = History(question=query,
                          raw_answer=answer.raw_answer,
                          augmented_answer=answer.augmented_answer.description,
                          tool_log_group_id=tool_log_group.id,
                          entities=[entity.id for entity in answer.augmented_answer.links])
        self.context.history.append(history)

        # Adapt context to only store the history for a configured maximum number of the last questions and
        # discard history, tool logs, and found entities and relations for any older questions to prevent the
        # context from growing infinitely big
        if len(self.context.history) > self.agent_config.history_limit:
            self.context.revise_history(self.agent_config.history_limit)
            logger.debug("Agent context history limit (%s) reached, pruned oldest history records",
                         self.agent_config.history_limit)

        # Mark the session as an active conversation after an answer was generated for a query at least once
        self.active_conversation = True

        return answer

    def _update_existing_session_context(self, context: Context) -> None:
        """Update the existing agent context held by this session with context information from the query parser.

        If this session holds an active conversation, i.e., an agent context with a history containing the generated
        answer for at least one previous query, this method updates the query-specific portions of the agent context
        held by this session, i.e., query-specific intents and background information, target types, etc., with the
        data provided by the given context.

        Args:
            context: Initial agent context created by the query parser for a specific query.
        """
        self.context.intents = context.intents

        self.context.query_specific_background_information_ids.update(context.query_specific_background_information_ids)
        for background_info_id, background_info in context.background_information.items():
            if background_info_id not in self.context.background_information:
                self.context.background_information[background_info_id] = background_info

        self.context.target_types = context.target_types
        self.context.relevant_types = context.relevant_types

        self.context.query = context.query
        self.context.hint = context.hint
        self.context.ontology = context.ontology

    def _augment_answer(self,
                        agent_response: AgentResponse,
                        context: Context,
                        top_n: int | None = None
                        ) -> Answer:
        """Augment the agent response by replacing all placeholder IDs in the agent response with the resolved
        entities and relations they represent.

        Args:
            agent_response: Agent response.
            context: Agent context.
            top_n: Maximum number of entities to consider when replacing the placeholder IDs in the agent response
                with entities. The full set of entities identified by the agent response will be contained in the
                answer as well; this parameter only controls how many of the entities will be stated in the augmented
                textual answer.

        Returns:
            An answer containing the augmented agent response and all resolved entities referenced through placeholders
            in the agent response.
        """
        raw_answer: str = agent_response.answer

        # Map each ID in the raw answer, e.g., group IDs like <[FULL:T-1, T-2]> or <[SAMPLE: M-0]>,
        # or single entity IDs like <(E-5)>, to the entity IDs it represents. In case a group ID refers
        # to a tool output, collect the relations contained in the tool output as well.
        resolved_entity_groups: dict[str, list[str]] = self._resolve_entity_group_ids(context)
        relations_in_tool_outputs: dict[str, list[Relation]] = self._collect_relations_in_tool_outputs(context)
        relevant_entities_and_relations_in_answer = self._find_relevant_entities_and_relations_in_answer(
            raw_answer, resolved_entity_groups, relations_in_tool_outputs
        )

        # Build lookup table for all relevant entities identified in the answer
        relevant_entities: dict[str, Entity] = {
            entity_id: context.id_to_entity[entity_id]
            for entity_id in relevant_entities_and_relations_in_answer.relevant_entity_ids
        }

        # Augment answer by replacing all group IDs and single entity IDs with the names of the resolved entities
        answers: list[Entity] = []
        augmented_answer: str = raw_answer
        links: list[Entity] = []
        relevant_relations: set[Relation] = set()
        recorded_entities: set[str] = set() # entities which have been selected to be part of the answer already
        for resolved_answer_placeholder in relevant_entities_and_relations_in_answer.resolved_answer_placeholders:
            replacement = ""

            # Entity group IDs with mode 'FULL', e.g., <[FULL:T-1, T-2]>
            if resolved_answer_placeholder.mode == "FULL":
                # Add all relevant entities to the answers
                answers.extend([relevant_entities[entity_id] for entity_id in resolved_answer_placeholder.entity_ids])

                # Collect relations the placeholder refers in relations relevant for the answer
                relevant_relations.update(resolved_answer_placeholder.relations)

                # Limit the number of entities considered for the augmented answer (including links)
                selected_entity_ids = (
                    resolved_answer_placeholder.entity_ids[:top_n]
                    if (top_n is not None) and (top_n > 0) else resolved_answer_placeholder.entity_ids
                )

                # Add entities resolved for the placeholder to the entity links provided as part of the answer
                for selected_entity_id in selected_entity_ids:
                    if selected_entity_id not in recorded_entities:
                        selected_entity = relevant_entities[selected_entity_id]
                        links.append(selected_entity)

                        # Mark the selected entity as seen and make sure it's not added to the links
                        # again if it is encountered as part of another placeholder ID again
                        recorded_entities.add(selected_entity_id)

            # Entity group IDs with mode 'SAMPLE', e.g., <[SAMPLE: M-0]>
            elif resolved_answer_placeholder.mode == "SAMPLE":
                for selected_entity_id in resolved_answer_placeholder.entity_ids:
                    selected_entity = relevant_entities[selected_entity_id]
                    if selected_entity_id not in recorded_entities:
                        links.append(selected_entity)
                        recorded_entities.add(selected_entity_id)

                    # TODO: Format entity name for augmented answer
                    if selected_entity.name is not None:
                        entity_name: str = selected_entity.name
                        replacement = ", ".join([replacement, entity_name]) if replacement else entity_name

            # Single entity IDs, e.g. <(E-5)>
            elif resolved_answer_placeholder.mode == "SINGLE":
                selected_entity_id = resolved_answer_placeholder.entity_ids[0]
                selected_entity = relevant_entities[selected_entity_id]
                if selected_entity_id not in recorded_entities:
                    links.append(selected_entity)
                    recorded_entities.add(selected_entity_id)

                # Remove the placeholder marker (entity name is already in the answer text)
                replacement = ""

            # Replace ID placeholders in answer with names of resolved entities
            augmented_answer = augmented_answer.replace(resolved_answer_placeholder.text, replacement)

        return Answer(raw_answer=raw_answer,
                      answers=answers,
                      entities=relevant_entities,
                      relations=list(relevant_relations),
                      augmented_answer=AugmentedAnswer(description=augmented_answer, links=links))

    @staticmethod
    def _resolve_entity_group_ids(context: Context) -> dict[str, list[str]]:
        """Resolve all entity group IDs in the tool logs of the agent context.

        This method identifies all tool logs in the agent context with arguments which reference entity IDs
        and maps the IDs of these tools to the IDs of the entities in the output of these tools.

        Args
            context: Agent context.

        Returns:
            A dictionary mapping tool IDs to the IDs of the entities in the output of these tools. Additionally,
            the dictionary maps the IDs of all entities in the given agent context to themselves.
        """
        # Map each entity ID to itself
        resolved_ids: dict[str, list[str]] = {entity_id: [entity_id] for entity_id in context.id_to_entity.keys()}

        # Map tool logs IDs to the entities referenced in the tool log
        for tool_log in context.tool_logs:
            # Determine entities referenced in a tool argument
            tool_log_args_ids: set[str] = set()
            for tool_arg in tool_log.args.get("ids", []):
                entity_ids_for_tool_arg = resolved_ids.get(tool_arg, [])
                tool_log_args_ids.update(entity_ids_for_tool_arg)

            # Record all entity IDs referenced in the tool output
            entity_ids = {entity.id for entity in tool_log.entities}

            for relation in tool_log.relations:
                # TODO: Why are we only recording entities from relations
                #  where the source of the relation is part of a tool input?
                if relation.source_id in tool_log_args_ids:
                    entity_ids.add(relation.target_id)

            resolved_ids[tool_log.id] = list(entity_ids)

        return resolved_ids

    @staticmethod
    def _collect_relations_in_tool_outputs(context: Context) -> dict[str, list[Relation]]:
        """Collect all relations in the tool outputs from the tool logs stored in the agent context.

        Args
            context: Agent context.

        Returns:
            A dictionary mapping tool IDs to the relations in the output of these tools.
        """
        relations_in_tool_outputs: dict[str, list[Relation]] = {
            tool_log.id: tool_log.relations for tool_log in context.tool_logs
        }
        return relations_in_tool_outputs

    @staticmethod
    def _find_relevant_entities_and_relations_in_answer(
        answer: str,
        resolved_entity_groups: dict[str, list[str]],
        relations_in_tool_outputs: dict[str, list[Relation]]
    ) -> RelevantEntitiesAndRelationsInAnswer:
        """Find all relevant entities in the given answer.

        This method parses all placeholders for entity groups and single entities in the given answer
        and maps them to the resolved entities they represent.

        Args:
            answer: The answer from the agent response containing placeholders for entities and entity groups.
            resolved_entity_groups: A lookup table mapping entity groups IDs, e.g., tool log IDs, to the IDs of
                the entities they represent.
            relations_in_tool_outputs: A dictionary mapping tool IDs to the relations in the output of these tools.

        Returns:
            All relevant entities and relations identified in the given answer.
        """
        mentioned_ids: set[str] = set()
        relevant_entity_ids: set[str] = set()
        mapped_answer_ids: list[ResolvedAnswerPlaceholder] = []

        # Search for group IDs with FULL and SAMPLE mode, e.g., <[FULL:T-1, T-2]> or <[SAMPLE: M-0]>,
        # and identify all entities these group IDs are resolved to
        # (Also handle LLM typos like `]}` instead of `]>` — rdflib bug workaround)
        for match in re.finditer(r"<\[(FULL|SAMPLE):\s*([TE]-\d+(?:,\s*[TE]-\d+)*)\][>}]", answer):
            text: str = match.group(0)
            mode: str = match.group(1)
            group_ids: list[str] = re.findall(r"[TE]-\d+", match.group(2))
            mentioned_ids.update(group_ids)

            selected_entity_ids: set[str] = set()
            selected_relations: set[Relation] = set()
            for group_id in group_ids:
                selected_entity_ids.update(resolved_entity_groups.get(group_id, []))
                selected_relations.update(relations_in_tool_outputs.get(group_id, []))

            sorted_selected_entity_ids: list[str] = sorted(selected_entity_ids, key=lambda x: int(x.split("-")[1]))
            sorted_selected_relations: list[Relation] = sorted(selected_relations, key=lambda r: r.source_id)

            # Limit the number of samples
            # TODO: Why are we doing this? And why here, instead of controlling it with something like the 'top_n'
            #  parameter in the '_augment_answer' method?
            if mode == "SAMPLE":
                sorted_selected_entity_ids = sorted_selected_entity_ids[:3]
                sorted_selected_relations = sorted_selected_relations[:5]

            mapped_answer_id = ResolvedAnswerPlaceholder(
                text=text, mode=mode, entity_ids=sorted_selected_entity_ids, relations=sorted_selected_relations
            )
            mapped_answer_ids.append(mapped_answer_id)

            relevant_entity_ids.update(selected_entity_ids)

        # Search for single entity IDs like <(E-5)> or <(E-5: Name)> and record them
        raw_entity_id_placeholders = re.findall(r"\<\(E\-\d+(?::[^)]*)?\)\>", answer)
        for raw_entity_id_placeholder in raw_entity_id_placeholders:
            entity_id = raw_entity_id_placeholder.strip("<()>").split(":")[0].strip()

            mentioned_ids.add(entity_id)
            relevant_entity_ids.add(entity_id)

            mapped_answer_id = ResolvedAnswerPlaceholder(
                text=raw_entity_id_placeholder, mode="SINGLE", entity_ids=[entity_id], relations=[]
            )
            mapped_answer_ids.append(mapped_answer_id)

        return RelevantEntitiesAndRelationsInAnswer(mentioned_ids=mentioned_ids,
                                                    relevant_entity_ids=relevant_entity_ids,
                                                    resolved_answer_placeholders=mapped_answer_ids)
