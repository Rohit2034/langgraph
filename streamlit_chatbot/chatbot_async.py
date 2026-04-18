from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import os
import asyncio
import aiosqlite

load_dotenv()

# -------------------
# 1. LLM
# -------------------
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT_EUS2')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_APIKEY_EUS2')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2025-04-01-preview')
LLM_DEPLOYMENT_NAME = os.getenv('LLM_DEPLOYMENT_NAME', os.getenv('MODEL_NAME'))

model = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=LLM_DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.2,
)

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchResults(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool("joke_generator")
def joke_generator(user_input: str) -> dict:
    """Generate a joke based on the user's input."""
    prompt = f"Explain the joke {user_input} in detail"
    response = model.invoke(prompt).content
    return {'explanation': response}

tools = [search_tool, joke_generator, calculator]
llm_with_tools = model.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 5. Graph builder
# -------------------
def build_graph(checkpointer):
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    return graph.compile(checkpointer=checkpointer)

# -------------------
# 6. Helper — retrieve all thread IDs
# -------------------
async def retrieve_all_threads(checkpointer):
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

# -------------------
# 7. Main entrypoint — everything runs inside one async context
# -------------------
async def main():
    async with  aiosqlite.connect("chatbot.db") as conn:
        checkpointer = AsyncSqliteSaver(conn=conn)
        chatbot = build_graph(checkpointer)

        result = await chatbot.ainvoke(
            {"messages": [HumanMessage(content="what is the recipe to make pasta")]},
            config={"configurable": {"thread_id": "test-thread"}}
        )
        print(result['messages'][-1].content)

if __name__ == "__main__":
    asyncio.run(main())