"""
RAG pipeline: ties together document loading, chunking, embedding,
retrieval, and generation into a single class.
"""
from typing import List, Dict, Optional

from .loader import load_documents, load_folder
from .chunker import chunk_documents
from .vector_store import VectorStore
from .generator import generate_answer


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        embedding_backend: str = "tfidf",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 4,
        llm_model: str = "claude-sonnet-5",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.llm_model = llm_model
        self.store = VectorStore(backend=embedding_backend, model_name=embedding_model)

    # ---- Ingestion ----
    def ingest_paths(self, paths: List[str]) -> int:
        docs = load_documents(paths)
        return self._ingest(docs)

    def ingest_folder(self, folder: str) -> int:
        docs = load_folder(folder)
        return self._ingest(docs)

    def _ingest(self, docs: List[Dict]) -> int:
        chunks = chunk_documents(docs, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return 0
        if self.store.embeddings is None:
            self.store.build(chunks)
        else:
            self.store.add(chunks)
        return len(chunks)

    # ---- Persistence ----
    def save(self, path: str) -> None:
        self.store.save(path)

    def load(self, path: str) -> None:
        self.store = VectorStore.load(path)

    # ---- Query ----
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Dict]:
        return self.store.search(question, top_k=top_k or self.top_k)

    def ask(self, question: str, top_k: Optional[int] = None) -> Dict:
        chunks = self.retrieve(question, top_k=top_k)
        answer = generate_answer(question, chunks, model=self.llm_model)
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"source": c["source"], "chunk_index": c["chunk_index"], "score": c["score"]}
                for c in chunks
            ],
        }
