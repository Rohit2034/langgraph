
import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
try:
    import httpx
    _original_client = httpx.Client
    _original_async_client = httpx.AsyncClient
    class PatchedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs['verify'] = False
            super().__init__(*args, **kwargs)
    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs['verify'] = False
            super().__init__(*args, **kwargs)
    httpx.Client = PatchedClient
    httpx.AsyncClient = PatchedAsyncClient
except Exception:
    pass
try:
    import requests
    original_request = requests.Session.request
    def patched_request(self, *args, **kwargs):
        kwargs['verify'] = False
        return original_request(self, *args, **kwargs)
    requests.Session.request = patched_request
except Exception:
    pass

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()



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
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

config1 = {"configurable": {"thread_id": "1"}}
chatbot = graph.compile(checkpointer=checkpointer)

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