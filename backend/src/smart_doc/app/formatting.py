import re


def format_qa_output(raw_text: str) -> str:
    text = raw_text.strip()
    text = text.replace("\\n", "\n")
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'\s*(\d+)\.\s+', r'\n\1. ', text)
    text = re.sub(r'\s*\((\d+)\)\s*', r'\n(\1) ', text)
    text = re.sub(r'\s-\s+(?=[A-Z])', r'\n- ', text)
    text = re.sub(r'\s*-\s+', r'\n- ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    lines = [line.strip() for line in text.split('\n')]
    text = "\n".join(line for line in lines if line)
    return text.strip()


def format_summarize_output(raw_text: str) -> str:
    answer = raw_text.strip()
    answer = answer.replace("\\n", " ")
    answer = re.sub(r'(\b\d)\.\s*\n\s*(\d)', r'\1.\2', answer)
    answer = re.sub(r'\s+', ' ', answer)
    sentences = re.split(r'(?<=[.!?]) +', answer)
    paragraphs = [" ".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
    return "\n\n".join(paragraphs).strip()
