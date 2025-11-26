import os
from random import random
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from psycopg import Connection
from pydantic import BaseModel
from serpapi import GoogleSearch
from typing_extensions import TypedDict

from chatbot.web_search import serpapi


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    search_web_query: str
    # prompts: list[ChatPromptTemplate]
    # current_node: str


# we use the article object for parsing serpapi results later
class Article(BaseModel):
    title: str
    source: str
    link: str
    snippet: str

    @classmethod
    def from_serpapi_result(cls, result: dict) -> "Article":
        return cls(
            title=result["title"],
            source=result["source"],
            link=result["link"],
            snippet=result["snippet"],
        )


@tool
def serpapi(query: str) -> list[Article]:
    """Use this tool to search the web."""
    load_dotenv()
    params = {
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "engine": "google",
        "q": query,
        "google_domain": "google.com",
        "gl": "it",
        "hl": "en"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return [Article.from_serpapi_result(organic_result) for organic_result in results["organic_results"]]


@tool
class WebSearchQuery(BaseModel):
    """ hold the query for the web search """
    query: str


chat_model = init_chat_model(model="ollama:qwen3:0.6b", reasoning=True, num_predict=1000, temperature=0.0)
chat_model_with_tools = chat_model.bind_tools([WebSearchQuery, serpapi])


def chatbot(state: State):
    # Only for this node add a system msg that instructs the LLM to ask permission to use tools
    llm_input = state["messages"] + [
        SystemMessage(
            content="If the user wants to search the web then you must be sure that you can construct a valid query."
                    "If the query is valid then use the provided tool")]
    message = chat_model_with_tools.invoke(llm_input)
    if message.tool_calls:
        return {"messages": [message], "search_web_query": message.tool_calls[0]["args"]["query"]}
    return {"messages": [message]}


def search_web(state: State):
    query = state["search_web_query"]
    llm_input = [SystemMessage(
        content=f"Use the web search with the following query: '{query}'.")]
    response = chat_model_with_tools.invoke(llm_input)
    return {"messages": [response]}


def should_continue(state: State):
    if state["search_web_query"]:
        return "search_web"
    return END


def compile_graph(checkpointer):
    graph_builder = StateGraph(State)

    # Nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("search_web", search_web)
    graph_builder.add_node("tool", ToolNode(tools=[serpapi]))
    # Edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("search_web", "tool")
    graph_builder.add_edge("tool", END)

    # Conditional edges
    graph_builder.add_conditional_edges(
        "chatbot",
        should_continue,
        ["search_web", END]
    )
    compiled = graph_builder.compile(checkpointer=checkpointer)
    print(compiled.get_graph().draw_ascii())
    return compiled


if __name__ == '__main__':
    DB_URI = "postgresql://langgraph:langgraph@localhost:5433/postgres?sslmode=disable"
    conn = Connection.connect(DB_URI, **{
        "autocommit": True,
        "prepare_threshold": 0,
    })
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()

    graph = compile_graph(checkpointer)

    config: RunnableConfig = {"configurable": {"thread_id": random()}}

    while True:
        user = input("User (q/Q to quit): ")
        if user in {"q", "Q"}:
            print("AI: Byebye")
            exit()
        human_msg = HumanMessage(content=user)
        for event in graph.stream({"messages": [human_msg]}, config=config, stream_mode="updates"):
            last_msg = next(iter(event.values()))["messages"][-1]
            print(last_msg)
