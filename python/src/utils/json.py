import re 

def clean_json_string(s):
    s = re.sub(r'```json\s*', '', s)
    s = re.sub(r'```', '', s)
    return s.strip()