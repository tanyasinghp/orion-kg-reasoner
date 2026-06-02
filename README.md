# Orion KG Reasoner

Ontology-guided KG reasoning agent for Amazon Product Reviews(here) metadata. An open-source replica of the SAP BKG Knowledge Graph Reasoner architecture, adapted for the Amazon product domain.
The base architecture of the agent is domain agnostic.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Agent Session                      │
│  ┌──────────────┐    ┌──────────────────────────┐    │
│  │ Query Parser  │───>│    Agent Loop (ReAct)    │    │
│  │  (LLM)       │    │  ┌────────────────────┐  │    │
│  └──────────────┘    │  │   LLM (Ollama)      │  │    │
│                      │  │  prompt → response  │  │    │
│                      │  └─────────┬──────────┘  │    │
│                      │            │             │    │
│                      │  ┌─────────▼──────────┐  │    │
│                      │  │   Tool Executor     │  │    │
│                      │  │  ┌───────────────┐  │  │    │
│                      │  │  │ GenericTools   │  │  │    │
│                      │  │  │ ─ retrieve     │  │  │    │
│                      │  │  │ ─ navigate     │  │  │    │
│                      │  │  │ ─ filter       │  │  │    │
│                      │  │  │ ─ select       │  │  │    │
│                      │  │  └───────┬───────┘  │  │    │
│                      │  └─────────┬──────────┘  │    │
│                      └────────────┼─────────────┘    │
│                                   │                   │
│  ┌────────────────────────────────▼───────────────┐  │
│  │         Answer Augmentation                     │  │
│  │  (resolve entity placeholders → names/props)   │  │
│  └────────────────────────────────┬───────────────┘  │
└───────────────────────────────────┼──────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   Grounded Answer    │
                         │  (entities + text)   │
                         └─────────────────────┘
```

## Knowledge Graph Backends

Two backends are supported:

| Backend | Type | Setup |
|---|---|---|
| **RDFLib** (default) | In-memory RDF store | Zero setup — loads CSV directly on startup |
| **Apache Jena Fuseki** | SPARQL-over-HTTP | Requires `docker compose up -d fuseki` + CSV load script |

## Agent Flow (Detailed)

### Phase 1: Query Parsing
1. User submits a query (e.g., "what is the most preferred product?")
2. LLM classifies the query into **intents** (e.g., `I-4` = "compare/rank products")
3. LLM selects relevant **background information** topics
4. LLM identifies **relevant types** and **target types** (e.g., `Product`)
5. Output: a `Context` object with intents, background IDs, types

### Phase 2: Agent Loop (ReAct, max N steps)
Each step:
1. Build prompt with: system instructions + ontology + tools + current context
2. LLM decides either:
   - **ANSWER**: Final answer with entity placeholders (e.g., `<(E-3)>`, `<[FULL: T-1, T-2]>`)
   - **TOOL_CALL**: Which tool to call with which arguments and a reasoning string
3. If tool call: `ToolExecutor` runs the tool, results go back into context, loop continues
4. If answer or max steps reached: exit loop

### Phase 3: Answer Augmentation
- Replace entity placeholders (`<(E-3)>`) with actual entity names
- Replace group placeholders (`<[FULL: T-1]>`) with enumerated entity lists
- Build final `Answer` with raw text, augmented text, and linked entities

## Key Components

### `src/agent/`
| File | Role |
|---|---|
| `agent.py` | Agent class: `run_agent_loop()`, `_invoke_agent()`, `_setup_llm_agent()` |
| `agent_session.py` | Orchestrator: `generate_answer()` runs all 3 phases |
| `agent_context.py` | `Context`, `Entity`, `Relation`, `ToolLog`, `History` data model |
| `prompts.py` | Prompt templates for query parser and agent loop |
| `query_parser.py` | LLM-based query analysis → `Context` |
| `ontology.py` | Ontology provider: types, relations, hierarchy, extra properties |
| `ranker.py` | Scores/ranks entities and relations by configured weights |
| `llm.py` | LLM factory (Ollama) |
| `util.py` | `InputTokenCounter`, `IdGenerator` |

### `src/tools/`
| Tool | Description |
|---|---|
| `tool_retrieve_entities` | Search entities by name/description (SPARQL CONTAINS + vector fallback) |
| `tool_get_relations_between_entities` | Find relations connecting two entities |
| `tool_navigate_path` | Traverse relation paths (e.g., `has_review / ^written_by`) |
| `tool_get_entities_matching_conditions` | Find entities by type + property conditions |
| `tool_filter_entities` | Filter existing entity set by conditions |
| `tool_select_entities` | Manually select subset of entities |
| `tool_get_properties` | Retrieve property values for entities |

### `src/kg/`
| Component | Description |
|---|---|
| `rdflib_client.py` | In-memory RDF store using rdflib 7.x, loads CSV on init |
| `fuseki_client.py` | SPARQL-over-HTTP client for Apache Jena Fuseki |
| `client_factory.py` | Factory to create KG client by configured type |
| `sparql.py` | `SparqlJsonResponse` Pydantic model |

## Ontology (4 Entity Types)

```
Product ──belongs_to──> Category
Product ──has_review──> Review
Review  ──written_by──> User
```

**Properties:** rating, price, name, discount, etc. for Product; category_name for Category; review_title, review_content, review_rating for Review; user_name for User.

**Extra property:** `Category.Name` is auto-resolved from `Product.belongs_to.Category.p_category_name`.

## Dataset

- Source: [yasserh/amazon-product-reviews-dataset](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset)
- 54 unique products, 56 category tags, 2 brands, ~1,600 reviews
- Downloaded via `kagglehub` to `data/raw/amazon.csv`
- Loaded into RDF triples on startup (rdflib backend) or via `scripts/load_kg.py` (Fuseki)

## Setup

```bash
poetry install

# Option A: RDFLib (default, no external services)
python src/cli.py

# Option B: Fuseki + Vector search
docker compose up -d fuseki
pip install sentence-transformers faiss-cpu
python scripts/download_data.py
python scripts/load_kg.py
python scripts/build_vector_index.py
python src/cli.py
```

### Environment Variables

All settings are configurable via env vars with prefix `KGR_`:

| Variable | Default | Description |
|---|---|---|
| `KGR_KNOWLEDGE_GRAPH__TYPE` | `rdflib` | KG backend: `rdflib` or `fuseki` |
| `KGR_KNOWLEDGE_GRAPH__CSV_PATH` | `data/raw/amazon.csv` | Path to CSV for rdflib loader |
| `KGR_LLM__OLLAMA__MODEL_NAME` | `llama3.1` | Ollama model |
| `KGR_LLM__OLLAMA__TEMPERATURE` | `0.0` | LLM temperature |
| `KGR_FLAVOR` | `amazon` | Agent flavor |
| `KGR_AGENT__MAX_STEPS` | `5` | Max agent loop iterations |
| `KGR_AGENT__GENERATE_REASON` | `true` | Include reasoning in tool calls |

## CLI Commands

```
>>> what products are available?     # Run agent on query
>>> tools                            # List available tools
>>> parse_query <query>              # Only run query parser
>>> show_context                     # Show current agent context
>>> exit                             # Exit CLI
```

## API

```bash
python src/api.py
curl -X POST http://localhost:8080/query -H "Content-Type: application/json" -d '{"query": "what is the most preferred product?"}'
```

## Tracing

Execution traces are printed to stderr with the `⚡` prefix, showing:
- Phase transitions (query parsing, agent loop, augmentation)
- SPARQL queries and result counts
- LLM decisions (tool calls with args)
- Tool execution results (entity/relation counts)
- Errors and parse failures

Set `KGR_LLM__OLLAMA__MODEL_NAME` to switch models (tested with llama3.1, qwen2, mistral).

## Project Structure

```
src/
├── agent/                 # Core agent logic
│   ├── agent.py           # Agent loop (ReAct)
│   ├── agent_session.py   # Full session orchestrator
│   ├── agent_context.py   # Data models
│   ├── prompts.py         # LLM prompt templates
│   ├── query_parser.py    # Query analysis
│   ├── ontology.py        # KG schema
│   ├── ranker.py          # Entity/relation ranking
│   ├── llm.py             # LLM (Ollama) factory
│   └── util.py            # Token counter, ID generator
├── tools/
│   ├── generic_tools.py   # 7 core knowledge graph tools
│   ├── amazon_tools.py    # Domain-specific tools
│   ├── tool_executor.py   # Tool execution engine
│   ├── tool_provider.py   # Tool registry
│   └── entity_retriever/  # Entity search (SPARQL + vector)
├── kg/
│   ├── rdflib_client.py   # In-memory RDF store (default)
│   ├── fuseki_client.py   # SPARQL-over-HTTP client
│   ├── client.py          # Abstract KG client
│   └── sparql.py          # SPARQL response model
├── vector_db/             # Vector search (FAISS)
├── data/amazon/           # Ontology, intents, config
├── config.py              # Pydantic settings
├── flavors.py             # Agent flavor factory
├── trace.py               # Execution tracer
├── api.py                 # FastAPI entry point
└── cli.py                 # CLI entry point
```

## Known Issues

- **LLM reasoning**: llama3.1 often fails to switch strategies when a tool returns empty results (tends to repeat same tool call)
- **Vector search**: Requires `sentence-transformers` and `faiss-cpu` (not in pyproject.toml); returns 0 results when not installed
- **Query parser**: Complex queries may cause the LLM to return only intents instead of the full Context JSON
- **RDFLib VALUES**: rdflib 7.6.0 crashes on `VALUES ?v { }` (empty set); gracefully handled by returning empty results
