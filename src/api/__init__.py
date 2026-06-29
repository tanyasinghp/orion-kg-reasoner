import json
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from http import HTTPStatus
from zoneinfo import ZoneInfo

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from config import settings
from agent.agent_context import Context, BackgroundInformation, Intent, UserContext
from agent.agent_session import AgentSession, Answer
from flavors import create_agent_session
from api.session import SessionStore, SessionDependency
from api.auth import UserDependency
from api.logs import log_config
from trace_event import TraceCollector
from trace import set_collector


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


@app.post(
    "/agent/ask/trace",
    tags=["Agent"],
    name="askAgentWithTrace",
    description="Query the agent and return structured trace events for UI visualization.",
    status_code=HTTPStatus.OK
)
async def ask_with_trace(
    query_input: QueryAgentInput,
    user: UserDependency,
    session: SessionDependency,
    response: Response
) -> dict:
    user_context = UserContext(locale=query_input.locale, time_zone=query_input.time_zone)
    collector = TraceCollector()
    session.agent_session.trace_collector = collector
    set_collector(collector)

    try:
        answer = await session.agent_session.generate_answer(
            query=query_input.query, top_n=query_input.top_n, debug=query_input.debug, user_context=user_context
        )
    finally:
        response.headers["Session-ID"] = str(session.id)
        set_collector(None)

    return {
        "answer": answer.model_dump(mode="json") if hasattr(answer, "model_dump") else str(answer),
        "events": collector.get_events(),
    }


@app.get("/ui", tags=["UI"], response_class=HTMLResponse)
async def ui():
    return HTMLResponse(content=UI_HTML)


UI_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KG Reasoner - Agent Visualizer</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/vis-network@9.1.6/styles/vis-network.min.css">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f14; color: #e0e0e0; height: 100vh; overflow: hidden; }
.app { display: flex; flex-direction: column; height: 100vh; }

/* Header */
.header { display: flex; align-items: center; gap: 12px; padding: 12px 20px; background: #1a1a24; border-bottom: 1px solid #2a2a3a; }
.header h1 { font-size: 16px; font-weight: 600; color: #b0b0c0; }
.header input { flex: 1; padding: 8px 14px; border-radius: 8px; border: 1px solid #2a2a3a; background: #0f0f14; color: #e0e0e0; font-size: 14px; outline: none; }
.header input:focus { border-color: #4a6cf7; }
.header button { padding: 8px 20px; border-radius: 8px; border: none; background: #4a6cf7; color: white; font-size: 14px; cursor: pointer; font-weight: 500; }
.header button:disabled { opacity: 0.5; cursor: not-allowed; }
.header .status { font-size: 12px; color: #6a6a8a; margin-left: auto; }

/* Main layout */
.main { display: flex; flex: 1; overflow: hidden; }

/* Left: Timeline */
.timeline-panel { width: 380px; min-width: 380px; overflow-y: auto; border-right: 1px solid #2a2a3a; padding: 12px; background: #13131c; }
.timeline-panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #6a6a8a; margin-bottom: 12px; }

.step-card { background: #1a1a24; border-radius: 8px; margin-bottom: 8px; padding: 12px; border-left: 3px solid #3a3a4a; cursor: pointer; transition: all .15s; }
.step-card:hover { background: #1e1e2a; }
.step-card.active { border-left-color: #4a6cf7; background: #1e1e2e; }
.step-card .header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.step-card .step-label { font-size: 11px; font-weight: 600; color: #8a8aaa; text-transform: uppercase; }
.step-card .step-badge { font-size: 10px; padding: 2px 8px; border-radius: 10px; background: #2a2a3a; color: #a0a0c0; }
.step-card .step-badge.tool-call { background: #2a4a3a; color: #6ad4a0; }
.step-card .step-badge.answer { background: #3a2a4a; color: #b080e0; }
.step-card .step-badge.phase { background: #2a3a5a; color: #80b0e0; }
.step-card .step-badge.sparql { background: #4a3a2a; color: #e0b080; }
.step-card .step-badge.error { background: #4a2a2a; color: #e08080; }
.step-card .step-msg { font-size: 13px; color: #c0c0d0; word-break: break-word; }
.step-card .step-detail { margin-top: 8px; padding: 8px; background: #0f0f14; border-radius: 6px; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 11px; color: #8a8aaa; max-height: 200px; overflow-y: auto; white-space: pre-wrap; display: none; }
.step-card .step-detail.open { display: block; }
.step-card .step-detail .key { color: #6a9fcf; }
.step-card .step-detail .string { color: #8fc98f; }
.step-card .step-detail .number { color: #cf8f6a; }

/* Center: KG Graph */
.graph-panel { flex: 1; position: relative; }
#graph-container { width: 100%; height: 100%; }
#graph-container .no-data { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #4a4a5a; text-align: center; }
#graph-container .no-data h3 { font-size: 18px; margin-bottom: 8px; }
#graph-container .no-data p { font-size: 13px; }

/* Bottom controls */
.controls { display: flex; align-items: center; gap: 8px; padding: 8px 16px; background: #1a1a24; border-top: 1px solid #2a2a3a; }
.controls button { background: none; border: 1px solid #2a2a3a; color: #a0a0c0; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.controls button:hover { background: #2a2a3a; }
.controls button:disabled { opacity: 0.3; cursor: not-allowed; }
.controls .step-info { font-size: 12px; color: #6a6a8a; margin-left: auto; }
.controls input[type="range"] { width: 80px; accent-color: #4a6cf7; }

/* Entity detail popup */
.entity-popup { display: none; position: absolute; top: 12px; right: 12px; width: 300px; background: #1a1a24; border-radius: 8px; border: 1px solid #2a2a3a; padding: 16px; z-index: 100; }
.entity-popup.open { display: block; }
.entity-popup h3 { font-size: 14px; margin-bottom: 8px; color: #e0e0e0; }
.entity-popup .prop { display: flex; justify-content: space-between; padding: 4px 0; font-size: 12px; border-bottom: 1px solid #2a2a3a; }
.entity-popup .prop .key { color: #8a8aaa; }
.entity-popup .prop .val { color: #c0c0d0; max-width: 180px; overflow: hidden; text-overflow: ellipsis; }
.entity-popup .close { float: right; background: none; border: none; color: #6a6a8a; cursor: pointer; font-size: 18px; }
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <h1>KG Reasoner</h1>
    <input id="query-input" type="text" placeholder="e.g. headphones" value="headphones" onkeydown="if(event.key==='Enter')runQuery()">
    <button id="run-btn" onclick="runQuery()">Run</button>
    <span class="status" id="status">ready</span>
  </div>
  <div class="main">
    <div class="timeline-panel" id="timeline"></div>
    <div class="graph-panel">
      <div id="graph-container">
        <div class="no-data" id="no-data">
          <h3>Run a query</h3>
          <p>Enter a query above to see the agent's reasoning step by step.</p>
        </div>
      </div>
      <div class="entity-popup" id="entity-popup">
        <button class="close" onclick="closePopup()">&times;</button>
        <h3 id="popup-title">Entity</h3>
        <div id="popup-props"></div>
      </div>
    </div>
  </div>
  <div class="controls">
    <button id="play-btn" onclick="togglePlay()">▶ Play</button>
    <button id="prev-btn" onclick="stepTo(Math.max(0, currentStep-1))" disabled>◀</button>
    <button id="next-btn" onclick="stepTo(currentStep+1)" disabled>▶</button>
    <span class="step-info" id="step-info">Step 0 / 0</span>
    <span style="font-size:12px;color:#6a6a8a;margin-left:12px;">Speed:</span>
    <input type="range" id="speed-slider" min="1" max="5" value="2" step="1" oninput="updateSpeed()">
  </div>
</div>

<script>
let events = [];
let currentStep = 0;
let isPlaying = false;
let playInterval = null;
let speedMs = 2000;
let nodes = new vis.DataSet();
let edges = new vis.DataSet();
let network = null;
let currentDisplayStep = -1;

function updateSpeed() {
  const val = document.getElementById('speed-slider').value;
  speedMs = [0, 3000, 2000, 1000, 500, 200][val] || 2000;
}

function togglePlay() {
  isPlaying = !isPlaying;
  const btn = document.getElementById('play-btn');
  btn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
  if (isPlaying) {
    if (currentStep >= events.length - 1) currentStep = 0;
    playInterval = setInterval(() => {
      if (currentStep < events.length - 1) {
        stepTo(currentStep + 1);
      } else {
        togglePlay();
      }
    }, speedMs);
  } else {
    clearInterval(playInterval);
  }
}

async function runQuery() {
  const query = document.getElementById('query-input').value.trim();
  if (!query) return;
  
  const runBtn = document.getElementById('run-btn');
  const status = document.getElementById('status');
  runBtn.disabled = true;
    status.textContent = '🔄 running (this may take 1-2 min)...';
  events = [];
  currentStep = 0;
  currentDisplayStep = -1;
  if (network) { network.destroy(); network = null; }
  nodes.clear();
  edges.clear();
  document.getElementById('no-data').style.display = 'none';
  document.getElementById('timeline').innerHTML = '<h2>Execution Trace</h2><p style="color:#6a6a8a;font-size:13px;padding:12px;">Waiting for agent response...</p>';
  
  try {
    const resp = await fetch('/v1/agent/ask/trace', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, debug: true })
    });
    if (!resp.ok) {
      status.textContent = 'HTTP ' + resp.status;
      return;
    }
    const data = await resp.json();
    events = data.events || [];
    status.textContent = events.length + ' events';
    
    renderTimeline();
    
    document.getElementById('prev-btn').disabled = false;
    document.getElementById('next-btn').disabled = false;
    
    togglePlay();
  } catch (err) {
    status.textContent = 'error: ' + err.message;
    document.getElementById('timeline').innerHTML = '<h2>Execution Trace</h2><p style="color:#e08080;font-size:13px;padding:12px;">Error: ' + escapeHtml(err.message) + '</p>';
  } finally {
    runBtn.disabled = false;
  }
}

function renderTimeline() {
  const container = document.getElementById('timeline');
  container.innerHTML = '<h2>Execution Trace</h2>';
  
  events.forEach((evt, i) => {
    const card = document.createElement('div');
    card.className = 'step-card';
    card.dataset.index = i;
    card.onclick = () => stepTo(i);
    
    const badgeClass = getBadgeClass(evt.event_type);
    const time = new Date(evt.timestamp).toLocaleTimeString();
    
    card.innerHTML = `
      <div class="header-row">
        <span class="step-label">${evt.event_type.toUpperCase()}</span>
        <span class="step-badge ${badgeClass}">${time}</span>
      </div>
      <div class="step-msg">${escapeHtml(evt.message)}</div>
      <div class="step-detail" id="detail-${i}"></div>
    `;
    container.appendChild(card);
  });
}

function getBadgeClass(type) {
  if (type === 'tool_call' || type === 'tool_execution' || type === 'tool_result') return 'tool-call';
  if (type === 'answer' || type === 'phase') return 'phase';
  if (type === 'sparql' || type === 'sparql_result') return 'sparql';
  if (type === 'error' || type === 'sparql_error') return 'error';
  return '';
}

function stepTo(index) {
  if (index < 0 || index >= events.length) return;
  currentStep = index;
  
  // Update timeline
  document.querySelectorAll('.step-card').forEach((c, i) => {
    c.classList.toggle('active', i === index);
    
    // Show detail for active card
    const detail = c.querySelector('.step-detail');
    if (i === index) {
      detail.classList.add('open');
      detail.textContent = formatData(events[i].data || {});
    } else {
      detail.classList.remove('open');
    }
  });
  
  // Scroll to active card
  const active = document.querySelector('.step-card.active');
  if (active) active.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  
  // Update graph to reflect state up to this step
  updateGraphForStep(index);
  
  // Update controls
  document.getElementById('step-info').textContent = `Step ${index + 1} / ${events.length}`;
  
  // Scroll timeline to bottom when playing
  const timeline = document.getElementById('timeline');
  timeline.scrollTop = timeline.scrollHeight;
}

function updateGraphForStep(index) {
  const seenEntities = new Map();
  const seenEdges = new Set();
  
  // Collect real entity and relation data from trace events
  for (let i = 0; i <= index; i++) {
    const evt = events[i];
    const data = evt.data || {};
    
    // Entities from tool results (now enriched with entity names)
    if (evt.event_type === 'tool_result' && data.entities) {
      data.entities.forEach(e => {
        if (!seenEntities.has(e.id)) {
          seenEntities.set(e.id, { id: e.id, label: e.name || e.id, type: e.type || 'entity' });
        }
      });
    }
    
    // Relations from tool results
    if (evt.event_type === 'tool_result' && data.relations) {
      data.relations.forEach(r => {
        const edgeId = r.source + '->' + r.target;
        if (!seenEdges.has(edgeId)) {
          seenEdges.add(edgeId);
        }
      });
    }
    
    // Link tool calls to their inputs/outputs
    if (evt.event_type === 'tool_execution') {
      const tcId = 'tool-' + i;
      seenEntities.set(tcId, { id: tcId, label: data.tool_name || 'tool', type: 'tool_call' });
      
      // Link entities used as args
      const args = data.args || {};
      if (args.ids) {
        args.ids.forEach(id => {
          const edgeId = id + '->' + tcId;
          if (!seenEdges.has(edgeId)) seenEdges.add(edgeId);
        });
      }
    }
    
    // Phase 3 augmentation adds the final answer entities
    if (evt.event_type === 'phase' && data.phase === 'augmentation') {
      // These are included from tool_results above
    }
  }
  
  // Add nodes for each distinct entity
  const nodeArr = Array.from(seenEntities.values()).map(e => ({
    id: e.id,
    label: (e.label || e.id).length > 25 ? (e.label || e.id).slice(0, 25) + '...' : (e.label || e.id),
    color: getNodeColor(e.type),
    shape: e.type === 'tool_call' ? 'box' : 'ellipse',
    size: e.type === 'tool_call' ? 25 : 30,
    title: e.id + ': ' + (e.label || ''),
  }));
  
  const edgeArr = Array.from(seenEdges).map(id => {
    const [from, to] = id.split('->');
    return { from, to, arrows: 'to', color: '#3a3a4a', width: 1 };
  });
  
  nodes.clear();
  edges.clear();
  if (nodeArr.length > 0) nodes.add(nodeArr);
  if (edgeArr.length > 0) edges.add(edgeArr);
  
  if (!network) {
    const container = document.getElementById('graph-container');
    network = new vis.Network(container, { nodes, edges }, {
      physics: { solver: 'forceAtlas2Based', forceAtlas2Based: { gravitationalConstant: -60, springConstant: 0.02, springLength: 120 } },
      interaction: { hover: true, tooltipDelay: 200, zoomView: true },
      nodes: { font: { color: '#c0c0d0', size: 13, face: 'sans-serif' } },
      edges: { font: { color: '#6a6a8a', size: 10 } },
    });
    network.on('click', (params) => {
      if (params.nodes.length > 0) showEntityPopup(params.nodes[0], seenEntities);
      else closePopup();
    });
  }
}

function getNodeColor(type) {
  switch((type || '').toLowerCase()) {
    case 'tool_call': return '#4a6cf7';
    case 'product': return '#6ad4a0';
    case 'review': return '#f0c060';
    case 'category': return '#b080e0';
    case 'user': return '#e08080';
    default: return '#6a9fcf';
  }
}

function showEntityPopup(nodeId, entityMap) {
  const popup = document.getElementById('entity-popup');
  const title = document.getElementById('popup-title');
  const props = document.getElementById('popup-props');
  
  const entity = entityMap.get(nodeId);
  if (!entity) return;
  
  title.textContent = entity.label;
  props.innerHTML = Object.entries(entity).map(([k, v]) =>
    `<div class="prop"><span class="key">${k}</span><span class="val">${escapeHtml(String(v))}</span></div>`
  ).join('');
  
  popup.classList.add('open');
}

function closePopup() {
  document.getElementById('entity-popup').classList.remove('open');
}

function formatData(data) {
  if (!data || Object.keys(data).length === 0) return '';
  try {
    return JSON.stringify(data, null, 2);
  } catch (e) {
    return String(data);
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
</script>
</body>
</html>
"""


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
