class FakeEmbedder:
    """Stands in for the real model in tests: texts with shared words get similar vectors."""

    size = 64

    def vector(self, text: str) -> list:
        numbers = [0.0] * self.size
        for word in text.lower().split():
            position = sum(ord(letter) for letter in word) % self.size
            numbers[position] += 1.0
        length = sum(x * x for x in numbers) ** 0.5 or 1.0
        return [x / length for x in numbers]

    def embed_documents(self, texts: list[str]) -> list:
        return [self.vector(text) for text in texts]

    def embed_query(self, text: str) -> list:
        return self.vector(text)
