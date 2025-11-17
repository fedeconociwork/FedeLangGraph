from langgraph.graph.state import CompiledStateGraph

from chatbot.chat_graph import build_graph


def stream_graph_updates(user_input: str, graph: CompiledStateGraph):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


if __name__ == '__main__':
    graph = build_graph()
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit"]:
                print("Goodbye")
                break
            stream_graph_updates(user_input, graph)
        except Exception as e:
            print("Exception ", e)
            break
