from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import logging

# Logger setup
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure LLM Initialization
def init_azure_llm():
    return AzureChatOpenAI(
        azure_endpoint="https://evaln-openai.openai.azure.com/",
        azure_deployment="gpt-4o",
        api_version="2024-05-01-preview",
        api_key="",  # ⚠️ Replace with env var in production
        temperature=0,
        streaming=True,
    )

# Global conversation history
conv_history = []

# Initialize LLM
try:
    llm = init_azure_llm()
    logger.info("✅ Azure chat init successful")
except Exception as e:
    logger.error(f"❌ Azure chat init failed: {e}")
    raise e

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
