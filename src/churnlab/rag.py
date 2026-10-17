"""RAG = Retrieval-Augmented Generation: before the LLM answers a policy question, fetch the most relevant
paragraphs from our own documents (knowledge/*.md) and make it answer FROM them, with citations.
Backends: 'tfidf' (default, no download, used in tests) or 'chroma' (ChromaDB + BGE embeddings; pip install chromadb sentence-transformers)."""
from __future__ import annotations
import os, re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from . import config


@dataclass
class Chunk:
    source: str
    heading: str
    text: str


def load_chunks(folder: Path = config.KNOWLEDGE, max_words: int = 160) -> list[Chunk]:
    """Split every markdown file at '## ' headings; long sections are split again into ~160-word pieces."""
    chunks = []
    for f in sorted(p for p in Path(folder).glob("*.md") if p.name.lower() != "readme.md"):
        title, heading = f.stem, f.stem
        for block in re.split(r"\n(?=## )", f.read_text(encoding="utf-8")):
            lines = block.strip().splitlines()
            if not lines:
                continue
            if lines[0].startswith("# "):
                title = lines[0][2:].strip(); lines = lines[1:]
            if lines and lines[0].startswith("## "):
                heading = lines[0][3:].strip(); lines = lines[1:]
            words = " ".join(lines).split()
            for i in range(0, len(words), max_words):
                piece = " ".join(words[i:i + max_words])
                if piece:
                    chunks.append(Chunk(f.name, f"{title} > {heading}", piece))
    return chunks


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = chunks
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.M = self.vec.fit_transform([f"{c.heading} {c.heading} {c.text}" for c in chunks])

    def search(self, query: str, k: int = 3):
        s = (self.M @ self.vec.transform([query]).T).toarray().ravel()
        idx = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in idx if s[i] > 0]


class ChromaRetriever:
    def __init__(self, chunks: list[Chunk], model: str = "BAAI/bge-small-en-v1.5"):
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        self.chunks = chunks
        client = chromadb.PersistentClient(path=str(config.ARTIFACTS / "chroma"))
        self.col = client.get_or_create_collection("playbook", embedding_function=SentenceTransformerEmbeddingFunction(model_name=model),
                                                   metadata={"hnsw:space": "cosine"})
        if self.col.count() != len(chunks):
            client.delete_collection("playbook")
            self.col = client.get_or_create_collection("playbook", embedding_function=SentenceTransformerEmbeddingFunction(model_name=model),
                                                       metadata={"hnsw:space": "cosine"})
            self.col.add(ids=[str(i) for i in range(len(chunks))], documents=[f"{c.heading}\n{c.text}" for c in chunks])

    def search(self, query: str, k: int = 3):
        r = self.col.query(query_texts=[query], n_results=k)
        return [(self.chunks[int(i)], 1 - float(d)) for i, d in zip(r["ids"][0], r["distances"][0])]


_RET = None


def retriever():
    global _RET
    if _RET is None:
        chunks = load_chunks()
        if os.getenv("RAG_BACKEND", "tfidf") == "chroma":
            try:
                _RET = ChromaRetriever(chunks)
            except ImportError:
                _RET = TfidfRetriever(chunks)
        else:
            _RET = TfidfRetriever(chunks)
    return _RET


NUM_WORDS = {"1": "one single", "2": "two", "3": "three", "4": "four"}
SYNONYMS = {r"\b(women|woman|female|men|man|male|gender|german|french|spanish|nationality|country|race|religion)\b": " gender nationality protected attributes fair treatment",
            r"\b(discount|incentive|voucher|cashback)\b": " offer", r"\b(leave|leaving|quit|close)\b": " churn",
            r"\b(dormant|inactive|not active)\b": " inactive members re-engagement", r"\b(retrain|retraining|drift)\b": " retrained population stability"}


def expand_query(q: str) -> str:
    """Query expansion: bridge the words people type and the words the documents use (e.g. '3' -> 'three', 'women' -> 'gender')."""
    q = re.sub(r"[^,.;?!]*\[(EMAIL|PHONE|CARD|IBAN)\][^,.;?!]*[,.;]?", " ", q)     # drop clauses that held personal data
    q = re.sub(r"\b([1-4])\b", lambda m: NUM_WORDS[m.group(1)], q)
    low = q.lower()
    return q + "".join(extra for pat, extra in SYNONYMS.items() if re.search(pat, low))


def search(query: str, k: int = 3) -> list[dict]:
    return [{"source": c.source, "section": c.heading, "text": c.text, "score": round(s, 3)} for c, s in retriever().search(expand_query(query), k)]
