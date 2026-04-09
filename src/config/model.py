# config/model.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "groq").lower()
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", 0))

# ===============================
# TEXT MODEL
# ===============================

if MODEL_BACKEND == "groq":

    text_model = ChatGroq(
        model=os.getenv(
            "TEXT_MODEL",
            "meta-llama/llama-4-scout-17b-16e-instruct"
        ),
        temperature=TEMPERATURE,
        api_key=os.getenv("GROQ_API_KEY")
    )

    image_model = ChatGroq(
        model=os.getenv(
            "IMAGE_MODEL",
            "meta-llama/llama-4-scout-17b-16e-instruct"
        ),
        temperature=TEMPERATURE,
        api_key=os.getenv("GROQ_API_KEY")
    )

elif MODEL_BACKEND == "openai":

    text_model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "moonshotai/Kimi-K2.5"),
        openai_api_base=os.getenv(
            "OPENAI_API_BASE",
            "https://api.inference.wandb.ai/v1"
        ),
        temperature=TEMPERATURE
    )

    image_model = text_model


elif MODEL_BACKEND == "ollama":

    text_model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        temperature=TEMPERATURE
    )

    image_model = text_model

else:
    raise ValueError(f"Unsupported MODEL_BACKEND: {MODEL_BACKEND}")


# ===============================
# VISUALIZATION MODEL
# ===============================

visualization_model = ChatGroq(
    model=os.getenv(
        "VISUALIZATION_MODEL",
        "qwen/qwen3-32b"
    ),
    temperature=TEMPERATURE
)