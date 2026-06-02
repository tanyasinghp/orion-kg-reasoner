from typing import Self, Any, TYPE_CHECKING
from enum import Enum

import re
import yaml
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables.base import Runnable
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field, model_validator
from difflib import SequenceMatcher
if TYPE_CHECKING:
    from langchain_core.messages.base import BaseMessage

from agent.agent_context import Intent, BackgroundInformation, Context
from agent.ontology import OntologyProvider
from agent.prompts import get_query_parser_prompt
from agent.util import InputTokenCounter
from tools.tool_provider import ToolProvider


class IntentProvider:
    """A simple wrapper for intent data.

    This class provides convenient lookup functionality for the intent data.

    Attributes:
        intents: List of intent data.
        intents_lookup_table: Lookup table mapping intent IDs to intents.
    """

    def __init__(self, intents: list[Intent], intents_lookup_table: dict[str, Intent]) -> None:
        self.intents = intents
        self.intents_lookup_table = intents_lookup_table

    @staticmethod
    def from_yaml(path: str) -> "IntentProvider":
        """Load intent data from a YAML file.

        This method generates IDs in the format 'I-1', 'I-2', etc. for each intent object contained in the data.

        Args:
            path: Path to the YAML file containing the intent data.

        Returns:
            A fully initialized IntentProvider providing the intent data given in the YAML file.
        """
        intents: list[Intent] = []
        intents_lookup_table: dict[str, Intent] = {}

        with open(path) as f:
            for i, intent_data in enumerate(yaml.safe_load(f)["intents"]):
                intent_id = f"I-{i}"
                intent = Intent(id=intent_id, **intent_data)

                intents.append(intent)
                intents_lookup_table[intent_id] = intent

        return IntentProvider(intents, intents_lookup_table)

    def get(self, intent_id: str) -> Intent | None:
        """Get intent data by ID.

        Args:
             intent_id: ID of the intent data.

        Returns:
            The intent data with the given ID if and only if it exists, or None otherwise.
        """
        return self.intents_lookup_table.get(intent_id)


class BackgroundInformationProvider:
    """A simple wrapper for background information.

    This class provides convenient lookup functionality for the background information data.

    Attributes:
        background_information_lookup_table: Lookup table mapping background information IDs to background information.
    """

    def __init__(self, background_information_lookup_table: dict[str, BackgroundInformation]) -> None:
        self.background_information_lookup_table = background_information_lookup_table

    @staticmethod
    def from_yaml(path: str) -> "BackgroundInformationProvider":
        """Load background information data from a YAML file.

        This method generates IDs in the format 'B-1', 'B-2', etc. for each
        background information object contained in the data.

        Args:
            path: Path to the YAML file containing the background information data.

        Returns:
            A fully initialized BackgroundInformationProvider providing the
            background information data given in the YAML file.
        """
        background_information_lookup_table: dict[str, BackgroundInformation] = {}

        with open(path) as f:
            for i, background_info_data in enumerate(yaml.safe_load(f)):
                background_info_id = f"B-{i}"
                background_info = BackgroundInformation(id=background_info_id, **background_info_data)
                background_information_lookup_table[background_info_id] = background_info

        return BackgroundInformationProvider(background_information_lookup_table)

    def mandatory_background_information_ids(self) -> set[str]:
        """Get the IDs of all background information marked as mandatory.

        Returns:
             A set containing the IDs of all background information marked as mandatory.
        """
        return {background_info_id for background_info_id, background_info
                in self.background_information_lookup_table.items()
                if background_info.mandatory}

    def get(self, background_information_id: str) -> BackgroundInformation:
        """Get background information by ID.

        Args:
             background_information_id: ID of the background information.

        Returns:
            The background information with the given ID if and only if it exists, or None otherwise.
        """
        return self.background_information_lookup_table.get(background_information_id)

    def get_all_topics(self) -> dict[str, str]:
        """Get all topics in the background information.

        Returns:
             A dictionary mapping background information IDs to the topic of the background information.
        """
        return {background_info_id: background_info.topic for background_info_id, background_info
                in self.background_information_lookup_table.items()}


class PresetAnswerType(Enum):
    """An enumeration class for the type of preset question and answer pair."""
    ANSWER = "answer"
    """An answer to the question."""
    COMMAND = "command"
    """Instead of an answer to the question, a command (to the LLM) is provided."""


class PresetAnswer(BaseModel):
    """A class for a preset question and answer pair.

    Such question and answer pairs form examples which can be provided as hints to the LLM.

    Attributes:
        question: Question.
        answer: Answer to the question.
        command: Instead of an answer to the question, a command (to the LLM) can be provided.
    """
    question: str = Field(frozen=True)
    answer: str | None = Field(frozen=True, default=None)
    command: str | None = Field(frozen=True, default=None)

    @model_validator(mode="after")
    def validate_either_answer_or_command_is_provided(self) -> Self:
        # Neither answer nore command is given
        if (self.answer is None) and (self.command is None):
            raise ValueError("Either answer or command must be provided.")
        # Both answer and command are given
        if (self.answer is not None) and (self.command is not None):
            raise ValueError("Either just an answer or just a command can be provided, not both.")

        return self

    @property
    def type(self) -> PresetAnswerType:
        """Type of the preset question and answer pair."""
        if self.answer is not None:
            return PresetAnswerType.ANSWER
        elif self.command is not None:
            return PresetAnswerType.COMMAND
        else:
            raise Exception("Either answer or command must be specified.")

    def as_llm_hint(self) -> dict[str, str]:
        """Dump the model into a dictionary only containing the present fields.

        Returns:
            A dictionary containing just the properties of this model which are present, i.e., not 'None'.
        """
        hint: dict[str, str] = {
            "question": self.question
        }

        if self.answer is not None:
            hint["answer"] = self.answer
        if self.command is not None:
            hint["command"] = self.command

        return hint



class PresetAnswerProvider:
    """A simple provider for preset question and answer pairs.

    This class provides convenient lookup functionality for the preset question and answer data.

    Attributes:
        preset_answers: Lookup table from questions to full preset question and answer pairs.
    """

    def __init__(self, preset_answers: dict[str, PresetAnswer]) -> None:
        self.preset_answers = preset_answers

    @staticmethod
    def empty() -> "PresetAnswerProvider":
        """Create an empty provider with no preset question and answer pairs."""
        return PresetAnswerProvider({})

    @staticmethod
    def from_yaml(path: str) -> "PresetAnswerProvider":
        """Load preset question and answer data from a YAML file.

        Args:
            path: Path to the YAML file containing the preset question and answer data.

        Returns:
            A fully initialized PresetAnswerProvider providing the question and answer data given in the YAML file.
        """
        with open(path) as f:
            # YAML file format:
            #
            # - questions:
            #   - "Question 1"
            #   - "Question 2"
            #   answer: "Answer"
            # - questions:
            #   - "Question 1"
            #   - "Question 2"
            #   command: "Command"
            items = yaml.safe_load(f)

            preset_answers: dict[str, PresetAnswer] = {} # question -> preset answer

            for item in items:
                for question in item["questions"]:
                    normalized_question = PresetAnswerProvider.normalize_question(question)
                    answer: str | None = item.get("answer")
                    command: str | None = item.get("command")

                    preset_answers[normalized_question] = PresetAnswer(question=normalized_question,
                                                                       answer=answer, command=command)

        return PresetAnswerProvider(preset_answers)

    @staticmethod
    def normalize_question(question: str) -> str:
        return question.lower()

    def search_answer(self, question: str, min_similarity: float = 0.8) -> PresetAnswer | None:
        """Search a matching preset question and answer pair for a given query.

        Args:
             question: The question to search for.
             min_similarity: The minimum similarity score between the given question and one from the preset data for
                the two questions to be considered similar

        Returns:
            A preset question and answer pair matching the given query w.r.t the given similarity score, None otherwise.
        """
        if len(self.preset_answers) > 0:
            normalized_question = PresetAnswerProvider.normalize_question(question)

            # Try if the given question exactly matches one of the preset questions
            answer = self.preset_answers.get(normalized_question)

            # If not, search for similar questions
            if answer is None:
                matched_questions = [(SequenceMatcher(None, question, preset_question).ratio(), preset_question)
                                     for preset_question in self.preset_answers.keys()]
                similarity, question = max(matched_questions)
                if similarity >= min_similarity:
                    return self.preset_answers[question]

            return answer
        else:
            return None


class QueryParser:
    """An LLM-based agent query analyzer for selecting query-specific grounding information for the agent.

    Attributes:
        llm: Large Language Model (LLM) used by the query parser.
        intent_provider: A provider for all intents defined for the agent.
        background_information_provider: A provider for all background information defined for the agent.
        ontology_provider: A provider for the ontology of the knowledge graph used by agent.
        preset_answer_provider: A provider for all preset question and answer pairs available as potential hints
            for the LLM prompt done by the query parser.
    """

    def __init__(self,
                 llm: BaseLanguageModel,
                 intent_provider: IntentProvider,
                 background_information_provider: BackgroundInformationProvider,
                 ontology_provider: OntologyProvider,
                 preset_answer_provider: PresetAnswerProvider,
                 tool_provider: ToolProvider
                 ) -> None:
        self.llm = llm
        self.intent_provider = intent_provider
        self.background_information_provider = background_information_provider
        self.ontology_provider = ontology_provider
        self.preset_answer_provider = preset_answer_provider
        self.tool_provider = tool_provider

    async def parse_query(self, query: str,
                          input_token_counter: InputTokenCounter | None = None) -> Context:
        # Create the LLM prompt template for the query parser with input variables for intents,
        # background information, and the knowledge graph ontology already filled in.
        query_parser_prompt = get_query_parser_prompt(
            intents=self.intent_provider.intents,
            background_info_topics=self.background_information_provider.get_all_topics(),
            ontology=self.ontology_provider.format()
        )

        # Create a fresh input token counter if none is provided through the method argument
        llm_input_token_counter = input_token_counter or InputTokenCounter(self.llm)

        # Create the LLM-based query parser by chaining the prompt template to an input token counter, the LLM,
        # and an output parser transforming the LLM output into the Pydantic model for the agent context.
        # The input for the LangChain runnable resulting from the chain is the input for the prompt template,
        # i.e., a dictionary providing values for the input variables in the template. The output of the chain
        # is the result of the output parser, i.e., the initial agent context.
        def _strip_md(text: str) -> str:
            return re.sub(r'\*\*|```json|```', '', text).strip()

        llm_query_parser: Runnable[dict[str, Any], Context] = (
                query_parser_prompt
                | llm_input_token_counter.count
                | self.llm
                | StrOutputParser()
                | RunnableLambda(_strip_md)
                | PydanticOutputParser(pydantic_object=Context)
        )

        # Construct a hint for the LLM from relevant preset questions and answers
        # TODO: Do we need to unpack this into a dictionary before adding it to the prompt,
        #  or could we simply add the (string representation) of the Pydantic model?
        preset_answer: PresetAnswer | None = self.preset_answer_provider.search_answer(question=query)
        hint: dict[str, str] = preset_answer.as_llm_hint() if preset_answer is not None else {}

        # Parse the given query with the help of an LLM and build the initial context model
        context: Context = await llm_query_parser.ainvoke({"query": query, "hint": hint})
        context.query = query
        context.hint = hint

        # Add the ontology which has been reduced to only contain information related
        # to the relevant types identified by the query parser to the initial context
        context.ontology = self.ontology_provider.format_subset(context.relevant_types)

        # TODO: Set user context. The user context is provided to the LLM as part of the agent prompt,
        #  beyond that it is currently not used anywhere.

        # The query parser identified query-specific background information but only populated the
        # 'query_specific_background_information_ids' property of the context model. First, we add
        # the IDs of all mandatory background information to the set of query-specific ones, before
        # we populate the background information lookup table in the context model with all identified
        # background information.
        background_info_ids: set[str] = context.query_specific_background_information_ids.union(
            self.background_information_provider.mandatory_background_information_ids()
        )

        for background_info_id in background_info_ids:
            background_info = self.background_information_provider.get(background_info_id)
            context.background_information[background_info_id] = background_info

            # Add the (optional) tool to the set of tools available to the agent
            # if the background information is associated with a tool.
            if background_info.add_tool is not None:
                self.tool_provider.consider_tool(tool_name=background_info.add_tool)

        return context
