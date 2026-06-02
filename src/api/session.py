import uuid
import logging
from typing import Annotated
from dataclasses import dataclass
from datetime import datetime, UTC
from uuid import UUID

from fastapi import Request, Depends, Header

from agent.agent_session import AgentSession
from flavors import create_agent_session


logger = logging.getLogger(__name__)


@dataclass
class Session:
    """A session of the API service.

    Attributes:
        id: Unique identifier of the session.
        created_at: Timestamp when the session was created.
        agent_session: Agent session associated with the API session.
    """
    id: UUID
    created_at: datetime
    agent_session: AgentSession

    @staticmethod
    def for_agent_session(agent_session: AgentSession) -> "Session":
        """Creates a new Session from the given agent session.

        Args:
            agent_session: Agent session to be associated with the API session.

        Returns:
            A new API session associated with the given agent session.
        """
        return Session(id=uuid.uuid4(), created_at=datetime.now(UTC), agent_session=agent_session)


@dataclass
class SessionStore:
    """A basic session store.

    This is a very simple sessions store aiming to provide some means to support continuous conversations
    with the agent through the API. This is not a production-grade implementation, important aspects like
    session persistency, thread safety, etc. are not handled in this class.

    Attributes:
        sessions: Lookup table mapping sessions IDs to sessions.
        max_sessions: Maximum number of sessions the service is allowed to hold at the same time.
    """
    sessions: dict[UUID, Session]
    max_sessions: int

    def create_session(self) -> Session:
        """Create a new session.

        In case the maximum number of sessions the service is allowed to hold at the same time
        is already reached, the method evicts the oldest session to be able to create a new one.

        Returns:
             The created session.
        """
        # Evict the oldest session from the session store if the session limit is reached
        logger.debug("Session capacity: %s/%s", len(self.sessions), self.max_sessions)
        if len(self.sessions) >= self.max_sessions:
            self._evict_oldest_session()

        # Create a new session and add it to the session store
        agent_session: AgentSession = create_agent_session()
        api_session = Session.for_agent_session(agent_session)
        self.sessions[api_session.id] = api_session
        logger.info("Created session '%s'", api_session.id)

        return api_session

    def get_session(self, session_id: UUID, default: Session | None = None) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Unique identifier of the session.
            default: Default session to return if a session with the given ID does not exist in this session store.

        Returns:
            The session with the given ID if present in this session store, the given default session otherwise.
        """
        return self.sessions.get(session_id, default)

    def _evict_oldest_session(self) -> None:
        """Remove the oldest session from this session store."""
        oldest_session = min(self.sessions.values(), key=lambda session: session.created_at)
        del self.sessions[oldest_session.id]
        logger.info("Evicted oldest session '%s' (created at '%s')", oldest_session.id, oldest_session.created_at)


SessionIdHeader = Annotated[
    UUID,
    Header(description="Use for continuous conversations by asking multiple questions within the same session.")
]


async def resolve_session(request: Request, session_id: SessionIdHeader = None) -> Session:
    """Resolve the session for a request from the session ID header.

    If no session ID header is provided, a new session will be created.

    Args:
        request: Request object.
        session_id: Session identifier provided through a corresponding request header.

    Returns:
        The session with the ID defined in the request header if present in the session
        store, otherwise a newly created session.
    """
    session_store: SessionStore = request.app.state.session_store

    existing_session: Session | None = session_store.get_session(session_id) if session_id is not None else None
    if existing_session is not None:
        return existing_session
    else:
        return session_store.create_session()


SessionDependency = Annotated[Session, Depends(resolve_session)]
