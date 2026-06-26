import io
import os
import shutil
import uuid

import fitz
from langchain_community.document_loaders import PyMuPDFLoader
from PIL import Image


def add_pdf_file(
    file_path,
    documents_path,
    blob_storage_path,
    file_hash,
    parent_splitter,
    child_splitter,
    yolo,
    device,
    ignored_layout_classes,
    get_collection_by_language,
    detect_language,
    image_collection
):
    filename = os.path.basename(file_path)

    stored_path = os.path.join(documents_path, filename)
    if not os.path.exists(stored_path):
        shutil.copy(file_path, stored_path)

    file_path = stored_path

    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    pdf = fitz.open(file_path)

    for page_index, doc in enumerate(docs):
        _index_page_text(
            doc.page_content,
            filename,
            page_index,
            file_hash,
            parent_splitter,
            child_splitter,
            get_collection_by_language,
            detect_language,
            os.path.abspath(file_path)
        )

        _index_page_images(
            pdf[page_index],
            filename,
            page_index,
            file_hash,
            blob_storage_path,
            yolo,
            device,
            ignored_layout_classes,
            image_collection
        )

    print(f"PDF indexed correctly: {filename}")


def _index_page_text(
    page_content,
    filename,
    page_index,
    file_hash,
    parent_splitter,
    child_splitter,
    get_collection_by_language,
    detect_language,
    source
):
    parent_chunks = parent_splitter.split_text(page_content)
    batches = {}

    for p_id, parent in enumerate(parent_chunks):
        child_chunks = child_splitter.split_text(parent)

        for c_id, child in enumerate(child_chunks):
            language = detect_language(child)
            target_col = get_collection_by_language(language)
            batch = batches.setdefault(
                language,
                {"collection": target_col, "documents": [], "ids": [], "metadatas": []}
            )
            batch["documents"].append(child)
            batch["ids"].append(f"{filename}_p{page_index}_P{p_id}_C{c_id}")
            batch["metadatas"].append({
                "page": page_index,
                "document": filename,
                "source": source,
                "file_hash": file_hash,
                "content_type": "text",
                "source_type": "pdf",
                "language": language,
                "parent_chunk_index": p_id,
                "child_chunk_index": c_id
            })

    # One add per language collection is much cheaper than one add per chunk.
    for batch in batches.values():
        batch["collection"].add(
            documents=batch["documents"],
            ids=batch["ids"],
            metadatas=batch["metadatas"]
        )


def _index_page_images(
    page,
    filename,
    page_index,
    file_hash,
    blob_storage_path,
    yolo,
    device,
    ignored_layout_classes,
    image_collection
):
    pix = page.get_pixmap(dpi=200)
    page_img = Image.open(
        io.BytesIO(pix.tobytes("png"))
    )

    results = yolo(page_img, conf=0.5, device=device)
    if not results:
        return

    result = results[0]

    for det_id, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        class_name = result.names[class_id]

        if class_name in ignored_layout_classes:
            continue

        x0, y0, x1, y1 = map(
            int,
            box.xyxy[0].tolist()
        )

        crop = page_img.crop(
            (x0, y0, x1, y1)
        )

        img_path = os.path.join(
            blob_storage_path,
            f"{filename}_p{page_index}_fig{det_id}.png"
        )

        crop.save(img_path)
        image_id = str(uuid.uuid4())

        image_collection.add(
            ids=[image_id],
            uris=[os.path.abspath(img_path)],
            metadatas=[{
                "source": img_path,
                "page": page_index,
                "document": filename,
                "file_hash": file_hash,
                "content_type": "image",
                "source_type": "pdf_crop",
                "layout_class": class_name,
                "detection_index": det_id
            }]
        )
