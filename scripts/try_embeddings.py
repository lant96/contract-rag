"""Try the embedding model: similarity scores, chunk lengths in tokens, and speed."""

import statistics
import time

import numpy as np

from contract_rag.chunking import CHUNKER_CONFIGS, chunk_contract
from contract_rag.data.download import download_cuad
from contract_rag.data.loader import load_contracts
from contract_rag.embeddings import Embedder


def main() -> None:
    embedder = Embedder()
    print(f"max input length of the model: {embedder.model.max_seq_length} tokens")

    # 1. Does a question land closest to the passage that answers it?
    question = "Which law governs this agreement?"
    passages = {
        "governing law": "This Agreement shall be governed by the laws of the State of New York.",
        "payment": "The Customer shall pay all undisputed invoices within thirty days.",
        "termination": "Either party may terminate this Agreement with sixty days written notice.",
    }
    question_vector = embedder.embed_query(question)
    passage_vectors = embedder.embed_documents(list(passages.values()))
    print(f"\nvector size: {len(question_vector)}")
    print(f"question: {question}")
    names = list(passages)
    for i in range(len(names)):
        # The vectors have length 1, so the dot product is the cosine similarity
        score = np.dot(question_vector, passage_vectors[i])
        print(f"  similarity to '{names[i]}' passage: {score:.3f}")

    # 2. Do the chunks fit into the model's input limit?
    contracts = load_contracts(download_cuad())[:20]
    limit = embedder.model.max_seq_length
    print(f"\nchunk length in tokens (20 contracts, limit = {limit}):")
    for config_name in CHUNKER_CONFIGS:
        token_counts = []
        for contract in contracts:
            for chunk in chunk_contract(contract, config_name):
                token_counts.append(len(embedder.model.tokenizer(chunk.text)["input_ids"]))
        too_long = sum(1 for n in token_counts if n > limit)
        print(
            f"  {config_name:16s} median {int(statistics.median(token_counts)):4d}  "
            f"max {max(token_counts):5d}  too long: {too_long / len(token_counts):.1%}"
        )

    # 3. How fast is it? / How long building an index will take
    sample = [c.text for c in chunk_contract(contracts[0], "fixed_1000_200")[:64]]
    start_time = time.time()
    embedder.embed_documents(sample)
    seconds = time.time() - start_time
    print(f"\nspeed: {len(sample) / seconds:.1f} chunks per second ({len(sample)} chunks)")


if __name__ == "__main__":
    main()
