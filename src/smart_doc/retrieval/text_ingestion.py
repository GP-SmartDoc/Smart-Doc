import os


def add_text_file(
    file_path,
    child_splitter,
    get_collection,
    detect_language,
    file_hash
):
    filename = os.path.basename(file_path)
    source = os.path.abspath(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = child_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        language = detect_language(chunk)
        target_col = get_collection(chunk)
        target_col.add(
            documents=[chunk],
            ids=[f"{filename}_chunk_{i}"],
            metadatas=[{
                "document": filename,
                "source": source,
                "file_hash": file_hash,
                "content_type": "text",
                "source_type": "txt",
                "language": language,
                "chunk_index": i
            }]
        )
