from langdetect import detect


def get_text_collection(text, arabic_collection, english_collection):
    """Route text to the Arabic collection when detected, otherwise English."""
    try:
        return arabic_collection if detect(text) == "ar" else english_collection
    except Exception:
        return english_collection
