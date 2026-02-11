import json
import re

def safe_json_parse(text, fallback):
    try:
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)
    except Exception:
        return fallback
