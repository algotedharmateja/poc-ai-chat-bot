import asyncio
from typing import Optional

from starlette import status
from sse_starlette import EventSourceResponse, ServerSentEvent

from fastapi import FastAPI, Request, Depends, status, BackgroundTasks
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

import logging
import os
import shutil
import json

from dotenv import load_dotenv

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from llama_index.core import Settings

import qdrant_client

from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader

from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import FunctionAgent

from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentStream, ToolCallResult

from pydantic import BaseModel
from fastapi.responses import JSONResponse

UPLOAD_DIR = "uploads"

# Logger setup
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Stream:
    def __init__(self) -> None:
        self._queue: Optional[asyncio.Queue[ServerSentEvent]] = None

    @property
    def queue(self) -> asyncio.Queue[ServerSentEvent]:
        if self._queue is None:
            self._queue = asyncio.Queue[ServerSentEvent]()
        return self._queue

    def __aiter__(self) -> "Stream":
        return self

    async def __anext__(self) -> ServerSentEvent:
        return await self.queue.get()

    async def asend(self, value: ServerSentEvent) -> None:
        await self.queue.put(value)


# Initialize FastAPI app
app = FastAPI()
_stream = Stream()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

azure_endpoint=os.getenv("AZURE_ENDPOINT")
api_key=os.getenv("API_KEY")
api_version="2024-05-01-preview"

def get_llm():
	llm = AzureOpenAI(
	    deployment_name="gpt-4o",
	    api_key=api_key,
	    azure_endpoint=azure_endpoint,
	    api_version=api_version,
	)

	return llm

def get_embedder():
	embed_model = AzureOpenAIEmbedding(
	    deployment_name="text-embedding-ada-002",
	    api_key=api_key,
	    azure_endpoint=azure_endpoint,
	    api_version=api_version,
	)

	return embed_model


# Azure LLM Initialization
def init_azure_llm():
    return AzureChatOpenAI(
        azure_endpoint="https://evaln-openai.openai.azure.com/",
        azure_deployment="gpt-4o",
        api_version="2024-05-01-preview",
        api_key=api_key,
        temperature=0,
        streaming=True,
    )

class CollectionRequest(BaseModel):
    collection_name: str

llama_idx_llm = get_llm()
embed_model = get_embedder()

Settings.llm = llama_idx_llm
Settings.embed_model = embed_model

qdaclient = qdrant_client.AsyncQdrantClient(
    host="localhost",
    port=6333
)

qdclient = qdrant_client.QdrantClient(
    host="localhost",
    port=6333
)

ctx_map = {}

logger.info(f"Qdrant async client instantiated:{qdaclient}")
logger.info(f"Qdrant sync client instantiated:{qdclient}")

# Global conversation history
conv_history = []

# Initialize LLM
try:
    llm = init_azure_llm()
    logger.info("✅ Azure chat init successful")
except Exception as e:
    logger.error(f"❌ Azure chat init failed: {e}")
    raise e

def get_collection_agent_response(collection_name : str = "default", query : str = ""):
    vector_store = QdrantVectorStore(aclient=qdaclient, collection_name=collection_name)
    loaded_index = VectorStoreIndex.from_vector_store(
        vector_store,
        # Embedding model should match the original embedding model
        # embed_model=Settings.embed_model
    )
    qq_engine = loaded_index.as_query_engine(similarity_top_k=3)

    retrieval_tool = QueryEngineTool.from_defaults(
        query_engine=qq_engine,
        name="retrieval_engine",
        description=(
            "Use this retrieval tool to get information from indexed documents"
            "Use a detailed plain text question as input to the tool."
        ),
    )

    # Create an agent workflow with our retrieval tool
    retrieval_agent = FunctionAgent(
        name="RetrievalAgent",
        tools=[retrieval_tool],
        llm=llama_idx_llm,
        system_prompt="You are a helpful assistant that retrieves information using the retrieval tool",
    )

    if collection_name in ctx_map:
        agent_ctx = ctx_map.get(collection_name, None)
    else:
        agent_ctx = Context(retrieval_agent)
        ctx_map[collection_name] = agent_ctx

    handler = retrieval_agent.run(query,ctx=agent_ctx)

    return handler


# Stream LLM response
def stream_response(llm, query):
    user_msg = HumanMessage(content=query)
    conv_history.append(user_msg)
    ai_msg = AIMessage(content="")
    conv_history.append(ai_msg)

    for chunk in llm.stream(conv_history):
        conv_history[-1].content += chunk.content
        yield chunk.content

# Stream summary response
def stream_summary(llm):
    summary_prompt = "Now summarize the whole history of conversation."
    history_for_summary = conv_history + [HumanMessage(content=summary_prompt)]

    for chunk in llm.stream(history_for_summary):
        yield chunk.content

@app.get("/sse")
async def sse(stream: Stream = Depends(lambda: _stream)) -> EventSourceResponse:
    return EventSourceResponse(stream)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Preflight OPTIONS for CORS
@app.options("/chat/")
async def options_chat():
    return {}

@app.options("/summary/")
async def options_summary():
    return {}

# Chat endpoint
@app.post("/chat/")
async def chat(request: Request):
    try:
        data = await request.json()
        human_message_content = data.get("human_message_content", "")
        logger.info(f"📥 User: {human_message_content}")
        return StreamingResponse(stream_response(llm, human_message_content), media_type="text/plain")
    except Exception as e:
        logger.error(f"❌ Error in /chat/: {e}")
        return {"error": "Failed to process chat request."}

# Summary endpoint
@app.get("/summary/")
async def summary():
    try:
        return StreamingResponse(stream_summary(llm), media_type="text/plain")
    except Exception as e:
        logger.error(f"❌ Error in /summary/: {e}")
        return {"error": "Failed to process summary request."}


async def start_indexing(collection_name: str, file_path: str, stream: Stream = Depends(lambda: _stream)):
    # [TODO] Add Docling indexing code here
    # 
    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
    node_parser = DoclingNodeParser()

    vector_store = QdrantVectorStore(client=qdclient, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = reader.load_data(file_path)
    VectorStoreIndex.from_documents(
        documents,
        transformations=[node_parser],
        storage_context=storage_context,
    )
    await stream.asend(ServerSentEvent(data=f"Indexing complete for {file_path} in collection {collection_name}"))


@app.post("/upload-start-indexing/", status_code=status.HTTP_202_ACCEPTED)
async def upload_and_start_indexing(
    background_tasks: BackgroundTasks,
    collection_name: str = None, file: UploadFile = File(...),
    stream: Stream = Depends(lambda: _stream)
) -> dict:
    try:
        # Ensure the uploads directory exists
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        # Ensure the collection-specific directory exists
        collection_dir = os.path.join(UPLOAD_DIR, collection_name)
        if not os.path.exists(collection_dir):
            os.makedirs(collection_dir)

        # Save the uploaded file to the server
        file_path = os.path.join(collection_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        background_tasks.add_task(start_indexing, collection_name, file_path, stream)

        return {"message": f"File '{file.filename}' uploaded successfully and indexing started"}
    except Exception as e:
        logger.exception("File upload and indexing failed")
        return JSONResponse(status_code=500, content={"error": f"{str(e)}"})


# [TODO] Support multiple file uploads with indexing in the background
# How does the user know that the indexing is done? Events? Notifications?
@app.post("/upload-file/")
async def upload_and_index(collection_name: str = None, file: UploadFile = File(...)):
    try:
        # Ensure the uploads directory exists
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        # Ensure the collection-specific directory exists
        collection_dir = os.path.join(UPLOAD_DIR, collection_name)
        if not os.path.exists(collection_dir):
            os.makedirs(collection_dir)

        # Save the uploaded file to the server
        file_path = os.path.join(collection_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        collection_name = collection_name

        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        vector_store = QdrantVectorStore(client=qdclient, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )

        return {"message": f"File '{file.filename}' uploaded successfully with indexing done"}
    except Exception as e:
        logger.exception("File upload and indexing failed")
        return JSONResponse(status_code=500, content={"error": f"{str(e)}"})

@app.post("/ask-collection-agent/")
async def ask_collection_agent(collection_name: str = None, query: str=None):
    try:
        handler = get_collection_agent_response(collection_name, query)
        response = await handler
        return {"response": f"{str(response)}"}
    except Exception as e:
        logger.exception("Collection agent response failed")
        return JSONResponse(status_code=500, content={"error": f"{str(e)}"})

# [TODO] Make it a streaming response with citations
@app.post("/stream-collection-agent/")
async def ask_collection_agent(collection_name: str = None, query: str=None):
    try:
        handler = get_collection_agent_response(collection_name, query)
        logger.info(f"collection agent ({collection_name}) response for:{query}")
        async def stream_response(handler):
            async for event in handler.stream_events():
                if isinstance(event, AgentStream):
                        logger.info(event)
                        yield json.dumps({
                            "type": "output_chunk",
                            "chunk": event.delta
                        }) + "\n"
                elif isinstance(event, ToolCallResult):
                        logger.info(event)
                        raw_output = event.tool_output.raw_output
                        for source_node in raw_output.source_nodes:
                                yield json.dumps({
                                    "type": "citation",
                                    "citation": {
                                        "content": source_node.get_content(),
                                        "score": source_node.get_score(),
                                        "metadata": source_node.metadata
                                    }
                                }) + "\n"

        return StreamingResponse(stream_response(handler), media_type="text/event-stream")
    except Exception as e:
        logger.exception("Collection agent response failed")
        return JSONResponse(status_code=500, content={"error": f"{str(e)}"})

