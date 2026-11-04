from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import unicodedata
from pydantic import BaseModel, Field, field_validator
import json


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

class NameResponse(BaseModel):
    """The three names transliterated into English."""

    name: str = Field(description="Person's name in English")
    fathers_name: str = Field(description="Father's name in English")
    mothers_name: str = Field(description="Mother's name in English")


structured_llm = llm.with_structured_output(
    NameResponse,
    method="function_calling",
)

@app.get("/health")
async def health():
    return {"status": "ok"}


class NameRequest(BaseModel):
    name: str = Field(..., strict=True, min_length=1, max_length=100)
    fathers_name: str = Field(..., strict=True, min_length=1, max_length=100)
    mothers_name: str = Field(..., strict=True, min_length=1, max_length=100)

    @field_validator("name", "fathers_name", "mothers_name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        # Allow only Bangla letters, combining marks, and regular spaces.
        for char in value:
            if char in " :.":
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
Transliterate these three Bangla names into natural English spelling.

Rules:
- Do not translate the meanings.
- Preserve each person's name as closely as possible.
- Return all three English names using the provided output schema.

Input:
{json.dumps(request.model_dump(), ensure_ascii=False)}
"""

    result = await structured_llm.ainvoke(prompt)
    return result