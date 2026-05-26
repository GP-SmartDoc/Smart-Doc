from langdetect import detect


def detect_text_language(text) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"


def get_text_collection(text, arabic_collection, english_collection):
    """Route text to the Arabic collection when detected, otherwise English."""
    return arabic_collection if detect_text_language(text) == "ar" else english_collection
