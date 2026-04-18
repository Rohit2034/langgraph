from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import os

load_dotenv()

# What it does tools_conditionChecks if LLM's reply has tool_calls → routes to tools or END
# ToolNodeReads those tool_calls, runs the actual functions, returns results to LLM, then routes back to chat node for next LLM response

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

class Arithmetic(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    ans: float

@tool
def multiply(first: float, secondnum: float) -> dict:
    """Multiply two numbers."""
    return {"ans": first * secondnum}

@tool
def add(first: float, secondnum: float) -> dict:
    """Add two numbers."""
    return {"ans": first + secondnum}

tools = [multiply, add]
model_with_tools = model.bind_tools(tools)

def chat_node(state: Arithmetic):
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}   

tool_node = ToolNode(tools)  

graph = StateGraph(Arithmetic)
graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)      
graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)  
graph.add_edge("tools", "chat")         
checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer, interrupt_before=["tools"])

config1 = {"configurable": {"thread_id": "1"}}

# First invoke — pauses before tools
response = chatbot.invoke({"messages": [HumanMessage("what is 10 multiplied by 5, then add 30?")]}, config=config1)

state = chatbot.get_state(config1)
print(state.next)  # → ('tools',)  ✅ now there's something to resume

# Update state with a different question
chatbot.update_state(config1, {"messages": [HumanMessage("what is 20 multiplied by 5, then add 50?")]})

# Now stream(None) works — resumes from the interrupt point
for event in chatbot.stream(None, config=config1, stream_mode="values"):
    event['messages'][-1].pretty_print()