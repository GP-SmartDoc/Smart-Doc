import json
import re

def safe_json_parse(text, fallback):
    if isinstance(text, dict):
        return text

    cleaned = re.sub(r"```json|```", "", str(text)).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return fallback
