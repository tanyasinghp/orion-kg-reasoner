import re
import logging
from typing import Any, Self, TYPE_CHECKING
from collections.abc import Callable
from enum import StrEnum
from datetime import datetime

from pydantic import BaseModel, Field, model_validator
from langchain_core.runnables.base import Runnable
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.exceptions import OutputParserException
if TYPE_CHECKING:
    from langchain_core.messages.base import BaseMessage

from config import AgentConfig
from agent.ontology import OntologyProvider
from agent.agent_context import Context, ToolLogGroup
from agent.prompts import format_max_prio_message, get_agent_prompt
from agent.util import InputTokenCounter
from tools.tool_executor import ToolExecutor, ToolOutput, ToolCall
from trace import trace


logger = logging.getLogger(__name__)


class AgentResponseType(StrEnum):
    """An enumeration class for the agent response type."""
    ANSWER = "answer"
    """The agent found an answer."""
    TOOL_CALL = "tool_call"
    """The agent identified a tool to be called."""


class AgentResponse(BaseModel):
    """A class for the response from the agent.

    Attributes:
        answer: The final answer to the query given to the agent.
        tool_call: A tool call identified by the agent which should be executed next.
    """
    answer: str | None = Field(frozen=True, default=None)
    tool_call: ToolCall | None = Field(frozen=True, default=None)

    @property
    def type(self) -> AgentResponseType:
        return AgentResponseType.TOOL_CALL if self.tool_call is not None else AgentResponseType.ANSWER

    @model_validator(mode="after")
    def check_response_type(self) -> Self:
        # Neither answer nore tool call is given
        if (self.answer is None) and (self.tool_call is None):
            raise ValueError("Either answer or tool call must be provided")
        # Both answer and tool call are given
        if (self.answer is not None) and (self.tool_call is not None):
            raise ValueError("Either just an answer or just a tool call can be provided, not both.")

        return self


type AfterReasoningStepCallback = Callable[[AgentResponse, float, int], None]
"""Callback to be invoked after the agent completed a reasoning step.

Args:
    agent_response: Response returned by the internal LangChain agent after it has been invoked.
    duration: Total number of seconds the agent took to perform the reasoning step.
    num_tokens: Number of input tokens consumed by the LLM invoked during agent execution.
"""

type AfterToolCallback = Callable[[ToolCall, ToolOutput, float], None]
"""Callback to be invoked after the agent executed a tool call.

Args:
    tool_call: Tool call the agent executed.
    tool_output: Output produced by the tool that was executed.
    duration: Total number of seconds the agent took to execute the tool call.
"""


class Agent:
    """An agent instance.

    Attributes:
        config: Agent configuration.
        llm: Large Language Model (LLM) used by the agent.
        tool_executor: Tool executor used by the agent to execute tool calls.
        context: Agent context.
        after_reasoning_step_callback: Callback to be invoked after the agent performed a reasoning step.
        after_tool_callback: Callback to be invoked after the agent executed a tool.
        llm_input_token_counter: Counter for the input tokens consumed by the LLM of the agent.
        llm_agent: LangChain-based agent used by the agent instance.
        max_prio_messages: Maximum priority messages exposed to the LLM.
    """

    def __init__(self,
                 config: AgentConfig,
                 llm: BaseLanguageModel,
                 ontology_provider: OntologyProvider,
                 tool_executor: ToolExecutor,
                 context: Context,
                 after_reasoning_step_callback: AfterReasoningStepCallback | None = None,
                 after_tool_callback: AfterToolCallback | None = None,
                 input_token_counter: InputTokenCounter | None = None
                 ) -> None:
        self.config = config
        self.llm = llm
        self.tool_executor = tool_executor
        self.context = context
        self.after_reasoning_step_callback = after_reasoning_step_callback
        self.after_tool_callback = after_tool_callback
        self.llm_input_token_counter: InputTokenCounter = input_token_counter or InputTokenCounter(self.llm)
        self.llm_agent = self._setup_llm_agent(ontology_provider)
        self.max_prio_messages: list[BaseMessage] = []

    def _setup_llm_agent(self, ontology_provider: OntologyProvider) -> Runnable[dict[str, Any], AgentResponse]:
        """Set up the LLM-based agent to be used by the agent instance.

        Args:
            ontology_provider: Provider for the ontology of the knowledge graph the agent should reason over.

        Returns:
            A LangChain runnable created from chaining the relevant prompt template to the given input token counter,
            the LLM, and an output parser transforming the LLM output into the Pydantic model for the agent response.
        """
        # Create the LLM prompt template for the agent with input variables
        # for tools and the ontology extension already filled in.
        agent_prompt = get_agent_prompt(tool_info=self.tool_executor.tool_provider.available_tools(),
                                        ontology_extension=ontology_provider.ontology_extension,
                                        generate_reason=self.config.generate_reason)

        # Create the LLM-based agent by chaining the prompt template to an input token counter, the LLM,
        # and an output parser transforming the LLM output into the Pydantic model for the agent response.
        # The input for the LangChain runnable resulting from the chain is the input for the prompt template,
        # i.e., a dictionary providing values for the message placeholders in the template. The output of the
        # chain is the result of the output parser, i.e., the agent response.
        def _strip_md(text: str) -> str:
            for token in ("```json", "```", "**"):
                text = text.replace(token, "")
            text = text.strip()
            text = re.sub(r'\bTrue\b', 'true', text)
            text = re.sub(r'\bFalse\b', 'false', text)
            text = re.sub(r'\bNone\b', 'null', text)
            return text

        llm_agent: Runnable[dict[str, Any], AgentResponse] = (
                agent_prompt
                | self.llm_input_token_counter.count
                | self.llm
                | StrOutputParser()
                | RunnableLambda(_strip_md)
                | PydanticOutputParser(pydantic_object=AgentResponse)
        )

        return llm_agent

    async def run_agent_loop(self, tool_log_group: ToolLogGroup) -> AgentResponse:
        """Run the agent loop.

        In each step, the LLM-based agent is invoked to determine whether an answer is available based on the
        information collected in the agent context, or if a tool should be called to gather more information.

        Args:
            tool_log_group: Tool log group in which the records of all tool calls executed when running the
                agent loop should be collected.

        Returns:
            A response to the query contained in the agent context the agent was initialized with.
        """
        step = 0
        consecutive_parse_errors = 0
        max_parse_errors = 2
        consecutive_tool_errors = 0

        while step <= self.config.max_steps:
            if step == self.config.max_steps - 1:
                self.max_prio_messages.append(
                    format_max_prio_message("YOU REACHED THE MAXIMUM NUMBER OF STEPS, NOW SUMMARIZE THE FINAL ANSWER!")
                )

            step += 1

            try:
                trace("agent.py", f"Step {step}: Invoking LLM...", event_type="step", data={"step": step, "status": "invoking"})
                invoke_agent_start_time = datetime.now()
                response: AgentResponse = await self._invoke_agent()
                invoke_agent_end_time = datetime.now()
                invoke_agent_duration = (invoke_agent_end_time - invoke_agent_start_time).total_seconds()
                consecutive_parse_errors = 0

                if response.type == AgentResponseType.ANSWER:
                    trace("agent.py", f"Step {step}: LLM returned ANSWER (no more tools needed)", event_type="step",
                          data={"step": step, "status": "answer", "duration": invoke_agent_duration})
                else:
                    tc = response.tool_call
                    trace("agent.py", f"Step {step}: LLM decided TOOL_CALL '{tc.tool_name}' args={tc.args}",
                          event_type="step",
                          data={"step": step, "status": "tool_call", "tool_name": tc.tool_name, "args": tc.args, "duration": invoke_agent_duration})

                if self.after_reasoning_step_callback is not None:
                    self.after_reasoning_step_callback(
                        response, invoke_agent_duration, self.llm_input_token_counter.last_count
                    )
            except OutputParserException as e:
                consecutive_parse_errors += 1
                self.context.errors.append(f"Failed to parse agent response at step {step}: {e}")

                if consecutive_parse_errors >= max_parse_errors:
                    logger.error("Agent failed to produce valid JSON %d times. Returning early.", max_parse_errors)
                    return AgentResponse(answer="I encountered a technical issue generating a response. Please try rephrasing your question.")

                self.max_prio_messages.append(
                    format_max_prio_message(
                        "Failed to parse the previous response. Now do it again and make sure the output is pure JSON!"
                    )
                )
                logger.exception("Failed to parse agent response")
                continue

            if response.type == AgentResponseType.ANSWER:
                # ---- Fix 5: Hallucination guard ----
                if consecutive_tool_errors > 0:
                    trace("agent.py",
                          f"Step {step}: LLM attempted answer but last {consecutive_tool_errors} tool(s) "
                          f"returned errors. Blocking hallucination.",
                          event_type="step",
                          data={"step": step, "status": "hallucination_guard"})
                    self.max_prio_messages.append(
                        format_max_prio_message(
                            "The previous tool call(s) failed with an error and returned no data. "
                            "Do NOT answer from memory or training knowledge. "
                            "Either retry with a corrected tool call or state that the information "
                            "could not be retrieved from the knowledge graph."
                        )
                    )
                    continue
                return response
            elif response.type == AgentResponseType.TOOL_CALL:
                tool_call_start_time = datetime.now()
                tool_output = await self.tool_executor.execute_tool(response.tool_call, tool_log_group)
                tool_call_end_time = datetime.now()
                tool_call_duration = (tool_call_end_time - tool_call_start_time).total_seconds()
                logger.debug("Executed tool '%s'", response.tool_call.tool_name)

                if self.after_tool_callback is not None:
                    self.after_tool_callback(response.tool_call, tool_output, tool_call_duration)
                    logger.debug("Executed after tool callback")

                # ---- Fix 5: Track consecutive tool errors for hallucination guard ----
                if tool_output is not None and tool_output.error is not None:
                    consecutive_tool_errors += 1
                else:
                    consecutive_tool_errors = 0

                # Guardrail: after tool_retrieve_entities finds entities, tell LLM not to re-search
                if response.tool_call.tool_name == "tool_retrieve_entities" and tool_output is not None:
                    if tool_output.entities:
                        found_names = [e.name or e.id for e in tool_output.entities[:3]]
                        found_ids = [e.id for e in tool_output.entities[:3]]
                        self.max_prio_messages.append(
                            format_max_prio_message(
                                f"You ALREADY found {len(tool_output.entities)} entity(ies): {', '.join(found_names)} (IDs: {', '.join(found_ids)}). "
                                f"DO NOT call tool_retrieve_entities again. "
                                f"Use these entity IDs in tool_navigate_path to find related entities (reviews, categories, etc.)."
                            )
                        )

        return AgentResponse(answer="Agent was terminated because it reached the maximum number of steps!")

    async def _invoke_agent(self) -> AgentResponse:
        """Invoke the LLM-based agent.

        Returns:
            A response containing either the answer to the query contained in the agent context
            or the tool call which should be executed next to gather more information.
        """
        # Prepare a representation of the context only containing the elements which should be shown to the LLM
        context_representation_for_prompt = self.context.format_for_prompt()

        response = await self.llm_agent.ainvoke(
            {
                "context": context_representation_for_prompt,
                "max_prio_messages": self.max_prio_messages
            }
        )

        return response
