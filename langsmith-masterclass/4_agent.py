import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import httpx
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


# ─── Imports ────────────────────────────────────────────────────────────────
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import requests
import os

load_dotenv()

# ─── Tools ──────────────────────────────────────────────────────────────────
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city.
    Use this when the user asks about current temperature or weather conditions.
    """
    url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'
    response = requests.get(url)
    data = response.json()

    # Parse and return a clean string instead of raw JSON
    try:
        location = data['location']['name']
        country = data['location']['country']
        temp_c = data['current']['temperature']
        feels_like = data['current']['feelslike']
        description = data['current']['weather_descriptions'][0]
        humidity = data['current']['humidity']
        wind_speed = data['current']['wind_speed']

        return (
            f"Weather in {location}, {country}:\n"
            f"  Temperature : {temp_c}°C (Feels like {feels_like}°C)\n"
            f"  Condition   : {description}\n"
            f"  Humidity    : {humidity}%\n"
            f"  Wind Speed  : {wind_speed} km/h"
        )
    except (KeyError, IndexError):
        return f"Could not parse weather data: {data}"


# ─── LLM ────────────────────────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT    = os.getenv('AZURE_OPENAI_ENDPOINT_EUS2')
AZURE_OPENAI_API_KEY     = os.getenv('AZURE_OPENAI_APIKEY_EUS2')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2025-04-01-preview')
LLM_DEPLOYMENT_NAME      = os.getenv('LLM_DEPLOYMENT_NAME', os.getenv('MODEL_NAME'))

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=LLM_DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.2,
)

# ─── Prompt ─────────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the tools available to answer the user's questions accurately."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ─── Agent ───────────────────────────────────────────────────────────────────
tools = [search_tool, get_weather_data]

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# ─── Run ─────────────────────────────────────────────────────────────────────
queries = [
    "What is the current temp of Gurgaon?",
    "What is the release date of Dhadak 2?",
    "Identify the birthplace city of Kalpana Chawla and give its current temperature.",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    response = agent_executor.invoke({"input": query})
    print(f"\nFinal Answer: {response['output']}")