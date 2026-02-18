# config/model.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()  # load .env file

# =============================== 
# DEFAULT MODEL
# ===============================

MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "groq").lower()

if MODEL_BACKEND == "groq":
    model = ChatGroq(
        model=os.getenv(
            "GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
        ),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0)),
    )

elif MODEL_BACKEND == "ollama":
    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0)),
    )
else:
    raise ValueError(f"Unsupported MODEL_BACKEND: {MODEL_BACKEND}")

# =============================== 
# OTHER MODELS
# ===============================

visualization_model = ChatGroq(
        model=os.getenv(
            "VISUALIZATION_MODEL", "qwen/qwen3-32b"
        ),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0)),
    )