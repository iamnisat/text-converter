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

    name: str | None = Field(default=None, description="Person's name in English")
    fathers_name: str | None = Field(default=None, description="Father's name in English")
    mothers_name: str | None = Field(default=None, description="Mother's name in English")


structured_llm = llm.with_structured_output(
    NameResponse,
    method="function_calling",
)

@app.get("/health")
async def health():
    return {"status": "ok"}


class NameRequest(BaseModel):
    name: str | None = Field(default=None, strict=True, min_length=1, max_length=100)
    fathers_name: str | None = Field(default=None, strict=True, min_length=1, max_length=100)
    mothers_name: str | None = Field(default=None, strict=True, min_length=1, max_length=100)

    @field_validator("name", "fathers_name", "mothers_name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        # Allow only Bangla letters, combining marks, and regular spaces.
        if value is None:
            return None

        for char in value:
            if char in " :.":
                continue
            is_english = "A" <= char <= "Z" or "a" <= char <= "z"
            is_bangla = (
                "\u0980" <= char <= "\u09FF"
                and unicodedata.category(char).startswith(("L", "M"))
            )
            if not (is_english or is_bangla):
                raise ValueError(
                    "Name must contain only Bangla or English letters, spaces, :, and period"
                )

        value = value.strip()

        if not value or not any(
            unicodedata.category(char).startswith("L") for char in value
        ):
            raise ValueError("Name must contain at least one Bangla or English letter")

        return value


@app.post("/transliterate")
async def transliterate(request: NameRequest):
    prompt = f"""
Transliterate these Bangla names into natural English spelling.

Rules:
- Do not translate the meanings.
- Preserve each person's name as closely as possible.
- Return all English names using the provided output schema.
- For any input field that is null or missing, return null for that field.

Input:
{json.dumps(request.model_dump(), ensure_ascii=False)}
"""

    result = await structured_llm.ainvoke(prompt)
    return result