import os


def add_text_file(
    file_path,
    child_splitter,
    get_collection_by_language,
    detect_language,
    file_hash
):
    filename = os.path.basename(file_path)
    source = os.path.abspath(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = child_splitter.split_text(text)
    batches = {}

    for i, chunk in enumerate(chunks):
        language = detect_language(chunk)
        target_col = get_collection_by_language(language)
        batch = batches.setdefault(
            language,
            {"collection": target_col, "documents": [], "ids": [], "metadatas": []}
        )
        batch["documents"].append(chunk)
        batch["ids"].append(f"{filename}_chunk_{i}")
        batch["metadatas"].append({
            "document": filename,
            "source": source,
            "file_hash": file_hash,
            "content_type": "text",
            "source_type": "txt",
            "language": language,
            "chunk_index": i
        })

    # Batch inserts avoid many small Chroma writes for larger text files.
    for batch in batches.values():
        batch["collection"].add(
            documents=batch["documents"],
            ids=batch["ids"],
            metadatas=batch["metadatas"]
        )
