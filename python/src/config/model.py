import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

__all__ = ["model"]

load_dotenv()

MODEL_BACKEND = os.environ["MODEL_BACKEND"].lower()

if MODEL_BACKEND == "groq":
    model_name = os.getenv("GROQ_MODEL")
    if not model_name:
        raise Exception("GROQ_MODEL not provided in .env")
    model = ChatGroq(
        model=model_name,
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0)),
    )

elif MODEL_BACKEND == "ollama":
    model_name = os.getenv("OLLAMA_MODEL")
    if not model_name:
        raise Exception("OLLAMA_MODEL not provided in .env")
    model = ChatOllama(
        model=model_name,
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0)),
    )
