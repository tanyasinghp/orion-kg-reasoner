from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from http import HTTPStatus
from zoneinfo import ZoneInfo

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from agent.agent_context import Context, BackgroundInformation, Intent, UserContext
from agent.agent_session import AgentSession, Answer
from flavors import create_agent_session
from api.session import SessionStore, SessionDependency
from api.auth import UserDependency
from api.logs import log_config


class QueryAgentInput(BaseModel):
    """Input to query the agent."""
    query: str = Field(
        frozen=True,
        min_length=1,
        max_length=1000,
        description="Query the agent should generate an answer for."
    )
    top_n: int | None = Field(
        frozen=True,
        default=None,
        gt=0,
        description="Maximum number of entities to consider when replacing the placeholder IDs in the agent response "
                    "with entities. The full set of entities identified by the agent response will be contained in "
                    "the answer as well; this parameter only controls how many of the entities will be stated in the "
                    "augmented textual answer.")
    debug: bool = Field(
        frozen=True,
        default=False,
        description="Add additional debug information to the answer."
    )
    locale: str = Field(
        frozen=True,
        default="en",
        max_length=10,
        description="IETF BCP 47 language tag to be considered by the agent as the user's locale."
    )
    time_zone: ZoneInfo = Field(
        frozen=True,
        default=ZoneInfo("UTC"),
        description="IANA time zone to be considered by the agent as the user's time zone."
    )


class AnalyzeQueryInput(BaseModel):
    """Input to analyze a query."""
    query: str = Field(
        frozen=True,
        min_length=1,
        max_length=1000,
        description="Query the agent should analyze."
    )


class AnalyzeQueryResponse(BaseModel):
    """Response to a request to analyze a query."""
    query: str | None = Field(
        frozen=True,
        default=None,
        description="Query the agent should analyze.")
    relevant_types: list[str] = Field(
        frozen=True,
        default_factory=list,
        description="Relevant entity types for the query.")
    target_types: list[str] = Field(
        frozen=True,
        default_factory=list,
        description="Target entity types for the query.")
    query_specific_background_information: list[BackgroundInformation] = Field(
        frozen=True,
        default_factory=list,
        description="Relevant background information identified by the query parser."
    )
    intents: list[Intent] = Field(
        frozen=True,
        default_factory=list,
        description="Intents of the query identified by the query parser."
    )


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
    fastapi_app.state.session_store = SessionStore({}, settings.api.max_sessions)
    yield


app = FastAPI(title="Knowledge Graph Reasoner API",
              description="A web service for an agent that answers questions around the "
                          "meta data of a system's data model stored in a knowledge graph.",
              version="1.0.0",
              root_path="/v1",
              lifespan=lifespan)

# Configure and add CORS (Cross-Origin Resource Sharing) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors.allow_origins,
    allow_methods=settings.api.cors.allow_methods,
    allow_headers=settings.api.cors.allow_headers,
    allow_credentials=settings.api.cors.allow_credentials,
    expose_headers=settings.api.cors.expose_headers
)


@app.post(
    "/agent/ask",
    tags=["Agent"],
    name="askAgent",
    description="Query the agent to generate an answer.",
    status_code=HTTPStatus.OK,
    responses={
        HTTPStatus.OK: {
            "headers": {"Session-ID": {"description": "Unique ID of the session.", "type": "string", "format": "uuid"}},
        }
    }
)
async def ask(
    query_input: QueryAgentInput,
    user: UserDependency,
    session: SessionDependency,
    response: Response
) -> Answer:
    user_context = UserContext(locale=query_input.locale, time_zone=query_input.time_zone)

    answer = await session.agent_session.generate_answer(
        query=query_input.query, top_n=query_input.top_n, debug=query_input.debug, user_context=user_context
    )

    # Add session ID to response headers to support continuous conversations with the same agent session
    response.headers["Session-ID"] = str(session.id)

    return answer


@app.post(
    "/agent/analyzeQuery",
    tags=["Agent"],
    name="analyzeQuery",
    description="Analyze the query.",
    status_code=HTTPStatus.OK
)
async def analyze_query(analyze_query_input: AnalyzeQueryInput, user: UserDependency) -> AnalyzeQueryResponse:
    agent_session: AgentSession = create_agent_session()
    initial_context: Context = await agent_session.query_parser.parse_query(query=analyze_query_input.query)

    response = AnalyzeQueryResponse(
        query=analyze_query_input.query,
        relevant_types=initial_context.relevant_types,
        target_types=initial_context.target_types,
        query_specific_background_information=[
            initial_context.background_information[background_info_id]
            for background_info_id in initial_context.query_specific_background_information_ids
        ],
        intents=[
            agent_session.query_parser.intent_provider.get(intent_id)
            for intent_id in initial_context.intents
        ]
    )

    return response


@app.get("/health", tags=["Health"], status_code=HTTPStatus.OK)
def health_check(): # noqa: ANN201
    return {}


@app.get("/health/live", tags=["Health"], status_code=HTTPStatus.OK)
def liveness_check(): # noqa: ANN201
    return {}


@app.get("/health/ready", tags=["Health"], status_code=HTTPStatus.OK)
def readiness_check(): # noqa: ANN201
    return {}


if __name__ == "__main__":
    uvicorn.run(app, host=settings.api.host, port=settings.api.port, log_config=log_config())
