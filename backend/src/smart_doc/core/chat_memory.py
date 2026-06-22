import tiktoken

class ChatMemory:

    def __init__(self, max_messages=10, max_tokens=2000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.history = []
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text):
        return len(self.encoding.encode(text))

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        self.trim()

    def trim(self):

        while len(self.history) > self.max_messages:
            self.history.pop(0)

        while True:
            tokens = sum(self.count_tokens(m["content"]) for m in self.history)

            if tokens <= self.max_tokens or len(self.history) <= 1:
                break

            self.history.pop(0)

    def build_context(self, user_msg):

        history_text = "\n".join(
            f'{m["role"].capitalize()}: {m["content"]}'
            for m in self.history
        )

        return f"""
Conversation History:
{history_text}

Current User Question:
{user_msg}
""".strip()