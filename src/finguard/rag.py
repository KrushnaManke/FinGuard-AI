"""Local, cited retrieval over approved compliance documents."""

from __future__ import annotations

from pathlib import Path
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ComplianceRetriever:
    MIN_RELEVANCE_SCORE = 0.22

    def __init__(self, documents_dir: str | Path = "data/compliance", backend: str = "tfidf"):
        self.documents_dir = Path(documents_dir)
        self.documents = []
        paths = sorted(self.documents_dir.glob("*.md")) + sorted(self.documents_dir.glob("*.txt"))
        for path in paths:
            if path.name.lower() == "readme.md":
                continue
            text = path.read_text(encoding="utf-8").strip()
            if text:
                self.documents.append(
                    {"source": path.name, "text": text, "chunks": self._chunks(path.name, text)}
                )
        self.chunks = [chunk for document in self.documents for chunk in document["chunks"]]
        self.vectorizer = None
        self.matrix = None
        self.collection = None
        if self.chunks:
            if backend == "tfidf":
                self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                self.matrix = self.vectorizer.fit_transform([chunk["text"] for chunk in self.chunks])
            elif backend == "chroma":
                try:
                    import chromadb
                    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
                except ImportError as error:
                    raise RuntimeError("Install vector/RAG dependencies with `pip install -e '.[llm]'`.") from error
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY is required for Chroma's OpenAI embedding function.")
                client = chromadb.PersistentClient(path=str(self.documents_dir / ".chroma"))
                self.collection = client.get_or_create_collection(
                    name="compliance_documents",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=OpenAIEmbeddingFunction(api_key=api_key),
                )
                identifiers = [f"{chunk['source']}:{index}" for index, chunk in enumerate(self.chunks)]
                existing_ids = self.collection.get()["ids"]
                obsolete = sorted(set(existing_ids) - set(identifiers))
                if obsolete:
                    self.collection.delete(ids=obsolete)
                self.collection.upsert(
                    ids=identifiers,
                    documents=[chunk["text"] for chunk in self.chunks],
                    metadatas=[{"source": chunk["source"]} for chunk in self.chunks],
                )
            else:
                raise ValueError("RAG backend must be 'tfidf' or 'chroma'.")
        elif backend not in {"tfidf", "chroma"}:
            raise ValueError("RAG backend must be 'tfidf' or 'chroma'.")

    @staticmethod
    def _chunks(source, text, size=1200):
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        chunks, current = [], ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) > size:
                chunks.append({"source": source, "text": current})
                current = ""
            current = f"{current}\n\n{paragraph}".strip()
        if current:
            chunks.append({"source": source, "text": current})
        return chunks

    def search(self, query: str, top_k: int = 4, minimum_score: float = MIN_RELEVANCE_SCORE) -> list[dict]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least one.")
        if not 0 <= minimum_score <= 1:
            raise ValueError("minimum_score must be between 0 and 1.")
        if not self.chunks:
            return []
        if self.collection is not None:
            result = self.collection.query(query_texts=[query], n_results=min(top_k, len(self.chunks)))
            matches = []
            for index, document in enumerate(result["documents"][0]):
                matches.append(
                    {
                        "source": result["metadatas"][0][index]["source"],
                        "text": document,
                        "score": 1.0 - float(result["distances"][0][index]),
                    }
                )
            return [match for match in matches if match["score"] >= minimum_score]
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        best = scores.argsort()[::-1][:top_k]
        return [
            {**self.chunks[index], "score": float(scores[index])}
            for index in best
            if scores[index] >= minimum_score
        ]
