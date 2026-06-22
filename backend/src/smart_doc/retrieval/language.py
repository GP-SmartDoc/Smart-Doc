from langdetect import detect


def detect_text_language(text) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"


def get_text_collection(text, arabic_collection, english_collection):
    """Route text to the Arabic collection when detected, otherwise English."""
    return get_text_collection_by_language(
        detect_text_language(text),
        arabic_collection,
        english_collection
    )


def get_text_collection_by_language(language, arabic_collection, english_collection):
    return arabic_collection if language == "ar" else english_collection
