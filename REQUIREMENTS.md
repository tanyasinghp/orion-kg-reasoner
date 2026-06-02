# Requirements

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.12 | 3.12+ |
| RAM | 8 GB | 16 GB |
| Disk | 2 GB | 5 GB |
| Ollama | 0.1.x | Latest |

## Runtime Dependencies

### Core (pyproject.toml)

| Package | Version | Purpose |
|---|---|---|
| `langchain` | ^0.3 | Agent framework, runnables, tool abstractions |
| `langchain-core` | ^0.3 | Core LangChain abstractions |
| `langchain-ollama` | ^0.2 | Ollama LLM integration |
| `pydantic` | ^2.0 | Data models, settings validation |
| `pydantic-settings` | ^2.0 | Environment-based configuration |
| `rdflib` | >=6.0 | In-memory RDF store, SPARQL evaluation |
| `httpx` | ^0.28 | HTTP client (Fuseki, Ollama) |
| `fastapi` | ^0.115 | REST API |
| `uvicorn` | ^0.34 | ASGI server |
| `PyJWT` | ^2.0 | JWT decoding (OIDC auth) |
| `SPARQLWrapper` | ^2.0 | SPARQL-over-HTTP (Fuseki backend) |
| `pyyaml` | ^6.0 | YAML config files |
| `networkx` | >=3.2 | Graph algorithms (ontology inference) |
| `numpy` | ^2.0 | Numerical operations |

### Optional (not in pyproject.toml)

| Package | Version | Purpose | Installation |
|---|---|---|---|
| `sentence-transformers` | latest | Embedding generation for vector search | `pip install sentence-transformers` |
| `faiss-cpu` | latest | Vector similarity search index | `pip install faiss-cpu` |

### External Services

| Service | Required for | Notes |
|---|---|---|
| [Ollama](https://ollama.ai) | LLM inference | Must be running: `ollama serve`. Default model: `llama3.1` (`ollama pull llama3.1`) |
| Apache Jena Fuseki | Fuseki KG backend | Optional. Start via `docker compose up -d fuseki` or [standalone](https://jena.apache.org/documentation/fuseki2/) |

## Configuration

All settings are managed via environment variables with the `KGR_` prefix, using `__` as nested delimiter. These map to the Pydantic model in `src/config.py`.

### Knowledge Graph

| Variable | Default | Options | Description |
|---|---|---|---|
| `KGR_KNOWLEDGE_GRAPH__TYPE` | `rdflib` | `rdflib`, `fuseki` | KG backend |
| `KGR_KNOWLEDGE_GRAPH__HOST` | `http://localhost:3030` | | Fuseki host URL |
| `KGR_KNOWLEDGE_GRAPH__PORT` | `3030` | | Fuseki port |
| `KGR_KNOWLEDGE_GRAPH__USERNAME` | `""` | | Fuseki basic auth username |
| `KGR_KNOWLEDGE_GRAPH__PASSWORD` | `""` | | Fuseki basic auth password |
| `KGR_KNOWLEDGE_GRAPH__DATASET` | `amazon` | | Fuseki dataset name |
| `KGR_KNOWLEDGE_GRAPH__CSV_PATH` | `""` | | Path to CSV (rdflib only; defaults to `data/raw/amazon.csv` in code) |

### LLM

| Variable | Default | Description |
|---|---|---|
| `KGR_LLM__PROVIDER` | `ollama` | LLM provider (only Ollama supported) |
| `KGR_LLM__OLLAMA__MODEL_NAME` | `llama3.1` | Ollama model |
| `KGR_LLM__OLLAMA__TEMPERATURE` | `0.0` | Generation temperature |
| `KGR_LLM__OLLAMA__CONTEXT_WINDOW_SIZE` | `65536` | Max context window |
| `KGR_LLM__ENABLE_CACHING` | `true` | Cache LLM responses |
| `KGR_LLM__COUNT_INPUT_TOKENS` | `false` | Count input tokens |

### Agent

| Variable | Default | Description |
|---|---|---|
| `KGR_AGENT__MAX_STEPS` | `5` | Max iterations in agent loop |
| `KGR_AGENT__GENERATE_REASON` | `true` | Show reasoning in tool calls |
| `KGR_AGENT__TOOL_TIMEOUT` | `null` | Tool execution timeout (seconds) |
| `KGR_AGENT__MAX_TOOL_INPUT` | `null` | Max items in tool input lists |
| `KGR_AGENT__MAX_TOOL_OUTPUT_OBSERVED` | `10` | Max entities shown to LLM per tool result |
| `KGR_AGENT__MAX_RELATIONS` | `50` | Max relations shown to LLM |
| `KGR_AGENT__HISTORY_LIMIT` | `5` | Max conversation history turns |

### Vector Database

| Variable | Default | Description |
|---|---|---|
| `KGR_VECTOR_DB__TYPE` | `faiss` | Vector search type |
| `KGR_VECTOR_DB__FAISS_INDEX_PATH` | `data/indices/entity_embeddings.index` | Path to FAISS index file |
| `KGR_VECTOR_DB__FAISS_MAPPING_PATH` | `data/indices/uri_mapping.json` | Path to URI ↔ index mapping |
| `KGR_VECTOR_DB__EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence transformer model |

### API

| Variable | Default | Description |
|---|---|---|
| `KGR_API__HOST` | `0.0.0.0` | Bind address |
| `KGR_API__PORT` | `8080` | HTTP port |
| `KGR_API__LOG_LEVEL` | `DEBUG` | Logging level |
| `KGR_API__MAX_SESSIONS` | `100` | Max concurrent sessions |
| `KGR_API__OIDC__SERVER_URL` | `""` | OIDC issuer URL |
| `KGR_API__OIDC__ISSUER` | `""` | JWT issuer |
| `KGR_API__OIDC__CLIENT_ID` | `""` | JWT audience |

### Example

```bash
# Use Fuseki backend
export KGR_KNOWLEDGE_GRAPH__TYPE=fuseki
export KGR_KNOWLEDGE_GRAPH__HOST=http://localhost:3030

# Switch to a different model
export KGR_LLM__OLLAMA__MODEL_NAME=qwen2

# 10-step agent loop
export KGR_AGENT__MAX_STEPS=10

# Disable reasoning
export KGR_AGENT__GENERATE_REASON=false
```

## Data Files

| File | Format | Description |
|---|---|---|
| `data/raw/amazon.csv` | CSV | ~1600 rows, 54 products, Kaggle dataset |
| `src/data/amazon/ontology.yaml` | YAML | KG schema: types, relations, properties |
| `src/data/amazon/intents.yaml` | YAML | Query intent definitions |
| `src/data/amazon/background_information.yaml` | YAML | Background knowledge for query analysis |
| `src/data/amazon/ranking_weights.yaml` | YAML | Entity/relation scoring weights |

## Current Limitations

1. **LLM output reliability**: The agent prompt was designed for the SAP/BDC domain. Queries on the Amazon ontology may produce unexpected LLM behavior (repeating failed tool calls, invalid JSON).
2. **JSON parsing**: LLMs occasionally output Python-style booleans (`True`/`False`) or unescaped strings. The `_strip_md` function in `agent.py` normalizes these before JSON parsing.
3. **LangChain-Ollama**: `ChatOllama` returns empty string when invoked with only a `SystemMessage`. The agent prompt includes a `HumanMessage` to work around this.
4. **RDFLib 7.x**: Lazy SPARQL evaluation means errors surface during iteration, not during `query()`. All SPARQL execution is wrapped in try/except. Empty `VALUES` clauses (`VALUES ?v { }`) crash the evaluator.
5. **Vector search**: Requires separate `pip install` of `sentence-transformers` and `faiss-cpu`. These are not declared in `pyproject.toml`.
