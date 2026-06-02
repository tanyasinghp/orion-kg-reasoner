import os
from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from config import settings, AgentFlavor
from kg.client_factory import KnowledgeGraphClientFactory
from vector_db.vector_search_factory import VectorSearchFactory
from agent.ontology import OntologyProvider
from agent.ranker import Ranker
from agent.query_parser import QueryParser, IntentProvider, BackgroundInformationProvider, PresetAnswerProvider
from agent.llm import get_llm
from agent.agent_session import AgentSession, AfterQueryParserCallback
from agent.agent import AfterReasoningStepCallback, AfterToolCallback
from tools.entity_retriever.amazon_entity_retriever import AmazonEntityRetriever
from tools.generic_tools import GenericTools
from tools.amazon_tools import AmazonTools
from tools.tool_provider import ToolProvider
from tools.tool_executor import ToolExecutorBlueprint


def absolute_file_path(path: str) -> str:
    return os.path.join(Path(__file__).parent.absolute(), path)


def project_root_path(path: str) -> str:
    return os.path.join(Path(__file__).parent.parent.absolute(), path)


def create_amazon_agent_session(
    after_query_parser_callback: AfterQueryParserCallback | None = None,
    after_reasoning_step_callback: AfterReasoningStepCallback | None = None,
    after_tool_callback: AfterToolCallback | None = None
) -> AgentSession:
    """Create an agent session for the Amazon Product Reviews knowledge graph."""
    kg_client = KnowledgeGraphClientFactory.create_knowledge_graph_client(
        client_type=settings.knowledge_graph.type,
        csv_path=project_root_path("data/raw/amazon.csv")
    )
    vector_search = VectorSearchFactory.create_vector_search(vector_search_type=settings.vector_db.type)

    ontology_provider = OntologyProvider.from_yaml(absolute_file_path("data/amazon/ontology.yaml"))
    intent_provider = IntentProvider.from_yaml(absolute_file_path("data/amazon/intents.yaml"))
    preset_answer_provider = PresetAnswerProvider.empty()
    background_info_provider = BackgroundInformationProvider.from_yaml(
        absolute_file_path("data/amazon/background_information.yaml")
    )

    llm: BaseChatModel = get_llm()

    ranker = Ranker.from_yaml(absolute_file_path("data/amazon/ranking_weights.yaml"))
    entity_retriever = AmazonEntityRetriever(kg_client, vector_search, ontology_provider, ranker)

    generic_tools = GenericTools(
        knowledge_graph_client=kg_client,
        entity_retriever=entity_retriever,
        ontology_provider=ontology_provider
    )
    amazon_tools = AmazonTools(generic_tools=generic_tools)

    tool_provider = (
        ToolProvider()
        .register_async_tool(generic_tools.tool_retrieve_entities)
        .register_async_tool(generic_tools.tool_get_relations_between_entities)
        .register_async_tool(generic_tools.tool_select_entities)
        .register_async_tool(generic_tools.tool_filter_entities)
        .register_async_tool(generic_tools.tool_navigate_path)
        .register_async_tool(generic_tools.tool_get_entities_matching_conditions)
        .register_async_tool(generic_tools.tool_get_properties)
        .register_async_tool(amazon_tools.optional_tool_get_top_products, optional=True)
        .register_async_tool(amazon_tools.optional_tool_get_category_summary, optional=True)
    )

    tool_executor_blueprint = ToolExecutorBlueprint(
        config=settings.agent,
        tool_provider=tool_provider,
        ontology_provider=ontology_provider,
        ranker=ranker
    )

    query_parser = QueryParser(
        llm=llm,
        intent_provider=intent_provider,
        background_information_provider=background_info_provider,
        ontology_provider=ontology_provider,
        preset_answer_provider=preset_answer_provider,
        tool_provider=tool_provider
    )

    agent_session = AgentSession(
        agent_config=settings.agent,
        llm=llm,
        query_parser=query_parser,
        ontology_provider=ontology_provider,
        tool_executor_blueprint=tool_executor_blueprint,
        after_query_parser_callback=after_query_parser_callback,
        after_reasoning_step_callback=after_reasoning_step_callback,
        after_tool_callback=after_tool_callback
    )

    return agent_session


def create_agent_session(
    after_query_parser_callback: AfterQueryParserCallback | None = None,
    after_reasoning_step_callback: AfterReasoningStepCallback | None = None,
    after_tool_callback: AfterToolCallback | None = None
) -> AgentSession:
    if settings.flavor == AgentFlavor.AMAZON:
        return create_amazon_agent_session(
            after_query_parser_callback=after_query_parser_callback,
            after_reasoning_step_callback=after_reasoning_step_callback,
            after_tool_callback=after_tool_callback
        )
    else:
        raise ValueError(f"Unsupported agent flavor '{settings.flavor}'")
