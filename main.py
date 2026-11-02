from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOllama(
    model="gemma4:cloud",
    temperature=0,
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"
        }
    },
)

@app.get("/health")
async def health():
    return {"status": "ok"}


class NameRequest(BaseModel):
    name: str


@app.post("/transliterate")
async def transliterate(request: NameRequest):
    prompt = f"""
Transliterate this Bangla person's name into natural English spelling.

Rules:
- Output ONLY the English name.
- Do not explain anything.
- Do not translate the meaning.
- Preserve the person's name as closely as possible.

Bangla name: {request.name}
"""

    response = await llm.ainvoke(prompt)

    return {
        "bangla_name": request.name,
        "english_name": response.content.strip(),
    }
