from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain import hub
from pydantic import BaseModel, Field
from openai import OpenAI
import chromadb
import os

load_dotenv()

# -------------------
# 1. Environment
# -------------------
AZURE_OPENAI_ENDPOINT     = os.getenv("AZURE_OPENAI_ENDPOINT_EUS2")
AZURE_OPENAI_API_KEY      = os.getenv("AZURE_OPENAI_APIKEY_EUS2")
AZURE_OPENAI_API_VERSION  = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
LLM_DEPLOYMENT_NAME       = os.getenv("LLM_DEPLOYMENT_NAME", os.getenv("MODEL_NAME"))
EMBEDDING_API_KEY         = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME", "BT-Embedding")
EMBEDDING_ENDPOINT        = os.getenv("EMBEDDING_ENDPOINT", "")
DB_PATH                   = "./chroma_db"
COLLECTION_NAME_ATTENTION     = "attention_is_all_you_need"
COLLECTION_NAME_OPENAI_REPORT = "openai_technical_report"

# -------------------
# 2. Clients — single model instance reused everywhere
# -------------------
model = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=LLM_DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.2,
)

embedding_client = OpenAI(
    base_url=EMBEDDING_ENDPOINT,
    api_key=EMBEDDING_API_KEY,
)

chroma_client = chromadb.PersistentClient(path=DB_PATH)

# -------------------
# 3. Embedding Function
# -------------------
def get_embedding(text: str) -> list:
    response = embedding_client.embeddings.create(
        input=text,
        model=EMBEDDING_DEPLOYMENT_NAME,
    )
    return response.data[0].embedding

# -------------------
# 4. Ingest Helper
# -------------------
def ingest_pdfs(
    pdf_files: list,
    collection_name: str,
    id_prefix: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    batch_size: int = 50,
):
    docs_raw = []
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(pdf)
            docs_raw.extend(loader.load())
            print(f"  [load]  Loaded: {pdf}")
        except Exception as e:
            print(f"  [load]  Failed: {pdf} -> {e}")

    if not docs_raw:
        print(f"  [warn]  No documents loaded for '{collection_name}'. Skipping.")
        return

    print(f"  [load]  Total pages: {len(docs_raw)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs_raw)
    print(f"  [split] Total chunks: {len(splits)}")

    collection = chroma_client.get_or_create_collection(name=collection_name)

    for i in range(0, len(splits), batch_size):
        batch      = splits[i : i + batch_size]
        ids        = [f"{id_prefix}_chunk_{i + j}" for j in range(len(batch))]
        texts      = [doc.page_content for doc in batch]
        metadatas  = [doc.metadata for doc in batch]
        embeddings = [get_embedding(text) for text in texts]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"  [store] Batch {i // batch_size + 1} — chunks {i} to {min(i + batch_size, len(splits))} stored.")

    print(f"  [done]  {len(splits)} chunks in '{collection_name}'\n")

# -------------------
# 5. Ingest Both Papers
# -------------------
print("=== Ingesting: Attention Is All You Need ===")
ingest_pdfs(
    pdf_files=["paper-1.pdf"],
    collection_name=COLLECTION_NAME_ATTENTION,
    id_prefix="attention",
)

print("=== Ingesting: OpenAI Technical Report ===")
ingest_pdfs(
    pdf_files=["paper-3.pdf"],
    collection_name=COLLECTION_NAME_OPENAI_REPORT,
    id_prefix="openai_report",
)

# -------------------
# 6. Custom Embedding Wrapper
# -------------------
class AzureCustomEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return get_embedding(text)

embedding_function = AzureCustomEmbeddings()

# -------------------
# 7. Vectorstores + Retrievers
# -------------------
vectorstore_attention = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME_ATTENTION,
    embedding_function=embedding_function,
)
vectorstore_openai = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME_OPENAI_REPORT,
    embedding_function=embedding_function,
)

retriever_attention = vectorstore_attention.as_retriever(search_kwargs={"k": 3})
retriever_openai    = vectorstore_openai.as_retriever(search_kwargs={"k": 3})

# -------------------
# 8. Retriever Tools
# -------------------
tool_attention = create_retriever_tool(
    retriever_attention,
    "retriever_attention_is_all_you_need",
    "Search and retrieve information about the Transformer architecture, "
    "attention mechanism, multi-head attention, positional encoding, "
    "encoder-decoder structure from the 'Attention Is All You Need' paper.",
)
tool_openai = create_retriever_tool(
    retriever_openai,
    "retriever_openai_technical_report",
    "Search and retrieve information about GPT-4, RLHF, reinforcement learning "
    "from human feedback, alignment, safety, model capabilities and evaluations "
    "from the OpenAI technical report.",
)

tools = [tool_attention, tool_openai]

# -------------------
# 9. Agent State
# -------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# -------------------
# 10. Nodes
# -------------------
def agent(state):
    print("---CALL AGENT---")
    messages = state["messages"]
    model_with_tools = model.bind_tools(tools)   # ✅ different name — no scoping conflict
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def grade_documents(state) -> Literal["generate", "rewrite"]:
    print("---CHECK RELEVANCE---")

    class grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    llm_with_tool = model.with_structured_output(grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    chain = prompt | llm_with_tool

    messages  = state["messages"]
    question  = messages[0].content
    docs      = messages[-1].content

    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewrite"


def generate(state):
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    docs     = messages[-1].content

    prompt    = hub.pull("rlm/rag-prompt")
    rag_chain = prompt | model | StrOutputParser()
    response  = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


def rewrite(state):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f"""\n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]
    response = model.invoke(msg)   # ✅ directly invoke global model — no shadowing
    return {"messages": [response]}

# -------------------
# 11. Build Graph
# -------------------
workflow = StateGraph(AgentState)

workflow.add_node("agent",    agent)
workflow.add_node("retrieve", ToolNode([tool_attention, tool_openai]))
workflow.add_node("rewrite",  rewrite)
workflow.add_node("generate", generate)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)

workflow.add_edge("generate", END)
workflow.add_edge("rewrite",  "agent")

graph = workflow.compile()

# -------------------
# 12. Run
# -------------------
result = graph.invoke({"messages": [HumanMessage(content="What is attention learning?")]})
print(result["messages"][-1].content)