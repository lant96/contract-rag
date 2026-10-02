"""Store chunk embeddings in Chroma and search them inside one contract."""

from pathlib import Path
from typing import cast

import chromadb
from chromadb.config import Settings

from contract_rag.schemas import Chunk, Hit


class ChunkIndex:
    """A Chroma collection of chunks.

    `embedder` can be any object with embed_documents(texts) and embed_query(text).
    Tests use a fake one, so they do not need to download the real model.
    """

    def __init__(self, folder: Path, name: str, embedder):
        client = chromadb.PersistentClient(
            path=str(folder), settings=Settings(anonymized_telemetry=False)
        )
        # "cosine" means similar texts get a small distance
        self.collection = client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = embedder

    def count(self) -> int:
        return self.collection.count()

    def chunk_count(self, contract_id: str) -> int:
        """How many chunks of this contract are stored."""
        found = self.collection.get(where={"contract_id": contract_id}, include=[])
        return len(found["ids"])

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Embed the chunks and store them. Adding the same chunk twice just replaces it."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = self.embedder.embed_documents([chunk.text for chunk in batch])

            metadatas = []
            for chunk in batch:
                metadatas.append(
                    {"contract_id": chunk.contract_id, "start": chunk.start, "end": chunk.end}
                )

            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                embeddings=vectors,
                documents=[chunk.text for chunk in batch],
                metadatas=metadatas,
            )

    def search(self, query: str, contract_id: str, k: int = 5) -> list[Hit]:
        """Find the k chunks most similar to the query, looking only inside one contract."""
        vector = self.embedder.embed_query(query)
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=k,
            where={"contract_id": contract_id},
        )

        # Chroma returns one list per query; we sent one query, so we take the first list.
        ids = result["ids"][0]
        documents = result["documents"]
        metadatas = result["metadatas"]
        distances = result["distances"]
        # Chroma's types say these can be None, but we asked for them, so they are filled.
        assert documents is not None
        assert metadatas is not None
        assert distances is not None

        hits = []
        for i in range(len(ids)):
            metadata = metadatas[0][i]
            chunk = Chunk(
                id=ids[i],
                contract_id=cast(str, metadata["contract_id"]),
                start=cast(int, metadata["start"]),
                end=cast(int, metadata["end"]),
                text=documents[0][i],
            )
            hits.append(Hit(chunk=chunk, score=1 - distances[0][i]))
        return hits
