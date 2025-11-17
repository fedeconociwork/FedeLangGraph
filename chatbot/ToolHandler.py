import json

from langchain_core.messages import ToolMessage


class BasicToolNode:

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            last_message = messages[-1]
        else:
            raise ValueError("no last_message in input")
        outputs = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_result = self.tools_by_name[tool_name].invoke(tool_args)
            tool_message = ToolMessage(
                content=tool_result[0].model_dump(),
                name=tool_name,
                tool_call_id=tool_call["id"]
            )
            outputs.append(tool_message)
        return {"messages": outputs}
