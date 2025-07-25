from fastapi import APIRouter, HTTPException
from app.models.schema import ChatRequest
from app.services.bot import generate_response
from pydantic import BaseModel
import logging

router = APIRouter()
logger = logging.getLogger("uvicorn.error")

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        reply = await generate_response(request.message)
        return {"response": reply}
    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        raise HTTPException(status_code=500, detail="Chatbot error")
