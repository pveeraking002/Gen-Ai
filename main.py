import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key="")

app = FastAPI(title="AI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.get("/")
def root():
    return {"message": "AI Chatbot API is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful and friendly AI assistant."
        }
    ]

    messages.extend(request.history)
    messages.append({
        "role": "user",
        "content": request.message
    })

    response = client.responses.create(
        model="gpt-5.6",
        input=messages
    )

    return {
        "reply": response.output_text
    }
