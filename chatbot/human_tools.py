from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.config import get_store
from langgraph.types import interrupt


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a Human"""
    human_response = interrupt({"query": query})
    return human_response["data"]


def get_user_info(config: RunnableConfig) -> str:
    """Look up user info."""
    store = get_store()
    user_id = config["configurable"].get("user_id")
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"
