from smart_doc.utils.image import encode_image_from_path


def query_collections(
    prompt,
    get_collection,
    image_collection,
    k_text=6,
    k_image=4,
    document=None,
    include_encoded_images=True
):
    where_filter = None

    if document and document != "all":
        where_filter = {
            "document": document
        }

    target_col = get_collection(prompt)
    text_res = target_col.query(
        query_texts=[prompt],
        n_results=k_text,
        where=where_filter,
        include=["documents", "metadatas"]
    )

    img_res = image_collection.query(
        query_texts=[prompt],
        n_results=k_image,
        where=where_filter,
        include=["uris", "metadatas"]
    )

    encoded_images = []
    paths = []
    image_metadata = img_res.get("metadatas", [[]])[0]

    # Encoding images is useful for multimodal QA, but callers that only need
    # file paths can skip it to avoid extra disk I/O and base64 work.
    for uri, meta in zip(
        img_res.get("uris", [[]])[0],
        image_metadata
    ):
        if include_encoded_images:
            encoded_images.append(
                encode_image_from_path(uri)
            )
        paths.append(uri)

    text_metadata = text_res.get("metadatas", [[]])[0]
    citations = _build_citations(text_metadata, image_metadata)

    return {
        "text": text_res.get(
            "documents",
            [[]]
        )[0],
        "text_metadata": text_metadata,
        "images": encoded_images,
        "image_metadata": image_metadata,
        "citations": citations,
        "paths": paths
    }


def _build_citations(text_metadata, image_metadata):
    citations = []
    seen = set()

    for metadata in [*text_metadata, *image_metadata]:
        citation = _citation_from_metadata(metadata)
        if not citation:
            continue

        key = (
            citation.get("document"),
            citation.get("page"),
            citation.get("source_type"),
            citation.get("source"),
        )
        if key in seen:
            continue

        seen.add(key)
        citations.append(citation)

    return citations


def _citation_from_metadata(metadata):
    if not metadata:
        return None

    citation = {
        "document": metadata.get("document"),
        "page": metadata.get("page"),
        "source": metadata.get("source"),
        "source_type": metadata.get("source_type"),
        "content_type": metadata.get("content_type"),
    }

    return {
        key: value
        for key, value in citation.items()
        if value is not None
    }
