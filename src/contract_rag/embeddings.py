from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Wraps the model, so the rest of the code only sees two simple methods."""

    def __init__(self, model_name: str = MODEL_NAME, query_prefix: str = QUERY_PREFIX):
        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list:
        """Embed passages (contract chunks). Returns one vector per text."""
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vectors.tolist()

    def embed_query(self, text: str) -> list:
        """Embed one question."""
        vector = self.model.encode(self.query_prefix + text, normalize_embeddings=True)
        return vector.tolist()
