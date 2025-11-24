from typing import Annotated

from langchain.chat_models import init_chat_model
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from chatbot.web_search import serpapi


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


AVAILABLE_TOOLS = [serpapi]


def create_chat_model():
    chat_model = init_chat_model(model="ollama:qwen3:0.6b", reasoning=True, num_predict=1000, temperature=0.0)

    return chat_model.bind_tools(AVAILABLE_TOOLS)


def __route_tools__(state: State):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
    it is fine directly responding. This conditional routing defines the main agent loop.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


def handle_graph_state(state: State):
    chat_model_with_tools = create_chat_model()
    return {"messages": [chat_model_with_tools.invoke(state["messages"])]}


def build_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(State)
    # add nodes
    graph_builder.add_node("chatbot", handle_graph_state)
    tool_node = ToolNode(tools=AVAILABLE_TOOLS)
    graph_builder.add_node("tools", tool_node)

    # add edges
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    # add conditional edges
    graph_builder.add_conditional_edges(
        "chatbot",
        __route_tools__
    )
    return graph_builder.compile()
