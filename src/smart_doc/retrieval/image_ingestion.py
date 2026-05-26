import os
import uuid

import torch
from PIL import Image


def caption_image(pil_image, caption_processor, caption_model):
    inputs = caption_processor(
        images=pil_image,
        return_tensors="pt"
    )
    device = next(caption_model.parameters()).device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        out = caption_model.generate(
            **inputs,
            max_new_tokens=50
        )

    return caption_processor.decode(
        out[0],
        skip_special_tokens=True
    )


def add_image_file(
    file_path,
    image_collection,
    get_collection,
    detect_language,
    caption_processor,
    caption_model,
    file_hash
):
    abs_path = os.path.abspath(file_path)
    image = Image.open(abs_path)
    caption = caption_image(image, caption_processor, caption_model)
    caption_language = detect_language(caption)
    image_id = str(uuid.uuid4())

    image_collection.add(
        ids=[image_id],
        uris=[abs_path],
        metadatas=[{
            "source": abs_path,
            "caption": caption,
            "file_hash": file_hash,
            "content_type": "image",
            "source_type": "standalone_image",
            "document": os.path.basename(file_path)
        }]
    )

    get_collection(caption).add(
        documents=[caption],
        ids=[f"{image_id}_caption"],
        metadatas=[{
            "type": "image_caption",
            "image_id": image_id,
            "source": abs_path,
            "file_hash": file_hash,
            "content_type": "text",
            "source_type": "image_caption",
            "document": os.path.basename(file_path),
            "language": caption_language
        }]
    )
