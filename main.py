from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import unicodedata
from pydantic import BaseModel, Field, field_validator


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
    name: str = Field(
        ...,
        strict=True,
        min_length=1,
        max_length=100,
        description="Bangla name containing only Bangla letters and spaces",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        # Allow only Bangla letters, combining marks, and regular spaces.
        for char in value:
            if char == " ":
                continue

            is_bangla = "\u0980" <= char <= "\u09FF"
            is_letter_or_mark = unicodedata.category(char).startswith(("L", "M"))

            if not (is_bangla and is_letter_or_mark):
                raise ValueError(
                    "Name must contain only Bangla letters and spaces"
                )

        value = value.strip()

        if not value or not any(
            unicodedata.category(char).startswith("L") for char in value
        ):
            raise ValueError("Name must contain at least one Bangla letter")

        return value


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
