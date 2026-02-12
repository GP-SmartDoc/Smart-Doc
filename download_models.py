import os
from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import Path


# =========================================================
# Install This Once to Download All Models Locally
# to download run python download_models.py in terminal (cmd)
# =========================================================



BASE_DIR = "./models"
os.makedirs(BASE_DIR, exist_ok=True)

# =========================================================
# Sentence Transformers (Text Embeddings)
# =========================================================
print("Downloading all-MiniLM-L6-v2...")
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir=f"{BASE_DIR}/all-MiniLM-L6-v2",
    local_dir_use_symlinks=False
)

# =========================================================
# BLIP Image Captioning
# =========================================================
print("Downloading BLIP image captioning model...")
snapshot_download(
    repo_id="Salesforce/blip-image-captioning-base",
    local_dir=f"{BASE_DIR}/blip-image-captioning-base",
    local_dir_use_symlinks=False
)

# =========================================================
# YOLO Document Layout Model
# =========================================================
MODELS = [
    "yolo11n_doc_layout.pt",
    "yolo11s_doc_layout.pt",
    "yolo11m_doc_layout.pt",
]

REPO_ID = "Armaggheddon/yolo11-document-layout"
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)

for model in MODELS:
    print(f"Downloading {model} ...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=model,
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False
    )

print("✅ All YOLOv11 document layout models downloaded locally.")

# run these commands in terminal (cmd)
#setx HF_HUB_OFFLINE 1
#setx TRANSFORMERS_OFFLINE 1
