# This is a sample Python script.
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    user_name = config["configurable"].get("user_name")
    system_msg = f"You are a helpful assistant. Address the user as {user_name}."
    return [{"role": "system", "content": system_msg}] + state["messages"]


chatModel = init_chat_model(model="ollama:qwen3:0.6b", reasoning=True, num_predict=1000, temperature=0.0)
class WeatherResponse(BaseModel):
    conditions: str

agent = create_react_agent(
    model=chatModel,
    tools=[get_weather],
    prompt=prompt,
    checkpointer=InMemorySaver(),
    response_format=WeatherResponse
)


if __name__ == '__main__':
    config = {"configurable": {"thread_id": "1"}}
    print(agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]},
                       config=config))
