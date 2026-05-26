import os


def add_text_file(file_path, child_splitter, get_collection):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = child_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        target_col = get_collection(chunk)
        target_col.add(
            documents=[chunk],
            ids=[f"{os.path.basename(file_path)}_chunk_{i}"]
        )
