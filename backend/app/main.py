# --------------------------------------------------------
# 1. Patch nltk.download before anything else
# --------------------------------------------------------

# now import the rest
import asyncio
import logging
from fastapi import FastAPI

# --------------------------------------------------------
# 2. Now import the rest of your application
# --------------------------------------------------------
import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Your routers and services
from app.routes import chat
from app.services.bot import load_pdf_async, pdf_path  


# --------------------------------------------------------
# 3. App setup remains unchanged
# --------------------------------------------------------
logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="LLaMA Chatbot API",
    version="1.0.0",
    description="A FastAPI backend for LLaMA-powered chatbot integrated with React frontend."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://172.19.112.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api", tags=["Chat"])

@app.get("/")
async def root():
    return {"message": "Backend is running 🚀"}

@app.on_event("startup")
async def on_startup():
    try:
        await load_pdf_async(pdf_path)
    except asyncio.CancelledError:
        logger.warning("Startup vector store loading was cancelled.")
    except Exception as e:
        logger.error(f"Error loading vector store on startup: {e}")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Application shutdown: cleaning up resources if any.")


import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")