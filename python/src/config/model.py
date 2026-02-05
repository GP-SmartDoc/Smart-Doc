import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()

MODEL_BACKEND = os.environ["MODEL_BACKEND"].lower()

if MODEL_BACKEND == "groq":
    model = ChatGroq(
        model=os.getenv(
            "GROQ_MODEL",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0)),
    )

elif MODEL_BACKEND == "ollama":
    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0)),
    )
