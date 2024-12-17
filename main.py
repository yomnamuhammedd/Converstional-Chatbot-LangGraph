from fastapi import FastAPI
from dotenv import load_dotenv
from APIS import chatbot_api

app = FastAPI(title="E-commerce Chatbot API")
app.include_router(chatbot_api.chat_router, prefix="/api", tags=["chatbot"])

@app.get("/")
async def root():
    return {"message": "Welcome to Slash Chatbot"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)