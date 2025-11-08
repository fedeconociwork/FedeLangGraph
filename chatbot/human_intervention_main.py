from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from psycopg import Connection

from chatbot.chat_graph import State, create_chat_model, AVAILABLE_TOOLS, __route_tools__


def chatbot(state: State):
    llm_with_tools = create_chat_model()
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


def compile_graph(checkpointer):
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=AVAILABLE_TOOLS)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges(
        "chatbot",
        __route_tools__,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    return graph_builder.compile(checkpointer=checkpointer)


if __name__ == '__main__':
    DB_URI = "postgresql://langgraph:langgraph@localhost:5433/postgres?sslmode=disable"
    conn = Connection.connect(DB_URI, **{
        "autocommit": True,
        "prepare_threshold": 0,
    })
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()

    graph = compile_graph(checkpointer)

    config = {"configurable": {"thread_id": "3"}}
    events = graph.stream(
        {"messages": [{"role": "user",
                       "content": "I need some expert guidance for building an AI agent. Could you request assistance "
                                  "for me"}]},
        stream_mode="values",
        config=config
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    human_command = Command(
        resume={"data": "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
                        " It's much more reliable and extensible than simple autonomous agents."})

    events = graph.stream(
        human_command,
        config=config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    print(graph.get_state(config))
