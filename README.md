# FedeLangGraph

A hands-on exploration of [LangGraph](https://langchain-ai.github.io/langgraph/) — building stateful, multi-step AI agent workflows with LangChain and Ollama.

## Project Structure

```
quickstart/          # Basic ReAct agent with tools, structured output, and memory
chatbot/             # Conversational chatbot with tool routing
  ├── chat_graph.py          # Core graph: state, tool routing, conditional edges
  ├── main.py                # Interactive chat loop with streaming
  ├── web_search.py          # SerpAPI web search tool
  ├── ToolHandler.py         # Custom BasicToolNode implementation
  ├── human_tools.py         # Human-in-the-loop interrupt tools
  └── complex_state_main.py  # Extended state with human review & corrections
web search/          # Multi-node graph with web search routing and Postgres checkpointing
examples/            # Prompt generation via information-gathering conversation
```

## What's Covered

- **ReAct agents** — prebuilt agent with tools, custom prompts, and structured output (`quickstart/`)
- **Stateful graphs** — custom `StateGraph` with `add_messages` reducer and conditional tool routing (`chatbot/`)
- **Tool integration** — SerpAPI web search, custom tool nodes, `ToolNode` from LangGraph prebuilt
- **Human-in-the-loop** — `interrupt` / `Command(resume=...)` for human review and state correction
- **Checkpointing** — `InMemorySaver` and `PostgresSaver` for conversation persistence
- **Multi-node routing** — conditional edges to route between chatbot, web search, and tool nodes
- **Prompt engineering agent** — conversational info-gathering that generates prompt templates (`examples/`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `chatbot/.env` with your API keys:

```
SERPAPI_API_KEY=<your-serpapi-key>
```

The project uses **Ollama** locally (`qwen3:0.6b`). Make sure Ollama is running:

```bash
ollama pull qwen3:0.6b
```

For the web search example with Postgres checkpointing, run a Postgres instance:

```bash
docker run -d --name langgraph-postgres -e POSTGRES_USER=langgraph -e POSTGRES_PASSWORD=langgraph -p 5433:5432 postgres
```

## Usage

```bash
# Quickstart — ReAct agent
python quickstart/main.py

# Interactive chatbot with tools
python chatbot/main.py

# Human-in-the-loop with state correction
python chatbot/complex_state_main.py

# Web search with Postgres persistence
python "web search/web_search.py"

# Prompt generation agent
python examples/information-gather-prompting.py
```
