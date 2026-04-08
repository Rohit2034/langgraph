from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.sqlite import SqliteServer
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
import sqlite3
load_dotenv()



conn =sqlite3.connect(database='chatbot.db',check_same_thread=False)

AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT_EUS2')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_APIKEY_EUS2')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2025-04-01-preview')
LLM_DEPLOYMENT_NAME = os.getenv('LLM_DEPLOYMENT_NAME', os.getenv('MODEL_NAME'))


llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=LLM_DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.2,
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = SqliteServer(conn = conn )

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

config1 = {"configurable": {"thread_id": "1"}}
chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

 
# stream =chatbot.stream({
#     "messages":[HumanMessage(content= "what is the recipes to make pasta")]
# },config = config1,stream_mode = 'messages')


# for message_chunk , metadata in chatbot.stream(
#     {'messages':[HumanMessage(content="what is the recipe to make pasta")]},
#     config = config1,
#     stream_mode='messages'
# ):
#     if message_chunk.content:
#         print(message_chunk.content,end=" ",flush=True)









# response = chatbot.invoke({"messages":"How are you?"},config = config1)
# print(response['messages'][-1])