import tiktoken

class SummarizationMemory:
    """Handling the memory of the current summarization session."""

    def __init__(self, max_entries=10, max_tokens=2000):
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self.history = []
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text):
        return len(self.encoding.encode(text))

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        self.trim()

    def trim(self):
        while len(self.history) > self.max_entries:
            self.history.pop(0)

        while True:
            tokens = sum(self.count_tokens(item["content"]) for item in self.history)
            if tokens <= self.max_tokens or len(self.history) <= 1:
                break
            self.history.pop(0)

    def build_context(self, user_question):
        history_text = "\n".join(
            f'{entry["role"].capitalize()}: {entry["content"]}'
            for entry in self.history
        )

        return f"""
Summarization History:
{history_text}

Current Summarization Request:
{user_question}
""".strip()
