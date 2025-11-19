from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from chatbot.chat_graph import __route_tools__
from chatbot.web_search import serpapi


class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


@tool
# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
def human_assistance(
        name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)


TOOLS = [human_assistance, serpapi]


def chatbot(state: State):
    chat_model = init_chat_model(model="ollama:qwen3:0.6b", reasoning=True, num_predict=1000, temperature=0.0)
    chat_model.bind_tools(TOOLS)
    message = chat_model.invoke(state["messages"])
    assert (len(message.tool_calls) <= 1)
    return {"messages": [message]}


def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=TOOLS)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges(
        "chatbot",
        __route_tools__,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    memory = InMemorySaver()
    return graph_builder.compile(checkpointer=memory)


if __name__ == '__main__':
    graph = build_graph()
    config = {"configurable": {"thread_id": "1"}}
    events = graph.stream(
        {"messages": [{
                          "system": "You must do a web search if you are not sure about the answer. If you dont know the answer after the web search reply with 'I don't know'",
                          "role": "user", "content": "Can you look up when LangGraph was released? "
                                                     "When you have the answer, use the human_assistance tool for review."}]},
        config,
        stream_mode="values",
    )
    for event in events:
        print(event)

    human_command = Command(
        resume={
            "name": "LangGraph",
            "birthday": "Jan 17, 2024",
        },
    )

    events = graph.stream(human_command, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
