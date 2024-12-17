from fastapi import  APIRouter
from pydantic import BaseModel
from typing import Optional
from main_agent import MainAgent  

chat_router = APIRouter()
agent = MainAgent()

class MessageRequest(BaseModel):
    message: str
    config: Optional[dict] = {}  

# Endpoint to interact with the chatbot
@chat_router.post("/chat", summary="Chat with the E-commerce bot")
async def chat_endpoint(request: MessageRequest):
    """
    Receives user input, runs the chatbot graph, and returns a response.
    """
    user_message = request.message
    config = {'configurable': {'thread_id' : 1}}
    response = agent.run(config, user_message)
    return {"response": response}

