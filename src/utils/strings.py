import re 

def clean_json_string(s):
    s = re.sub(r'```json\s*', '', s)
    s = re.sub(r'```', '', s)
    return s.strip()

def remove_thinking_from_content(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()
