# config/model.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()  # load .env file

MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "groq").lower()

if MODEL_BACKEND == "groq":
    model = ChatGroq(
        model=os.getenv(
            "GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
        ),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0)),
    )

elif MODEL_BACKEND == "openai":
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "moonshotai/Kimi-K2.5"),
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.inference.wandb.ai/v1"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0)),
    )

elif MODEL_BACKEND == "ollama":
    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0)),
    )
else:
    raise ValueError(f"Unsupported MODEL_BACKEND: {MODEL_BACKEND}")