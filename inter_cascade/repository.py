"""The strategy repository: a persistent, similarity-searchable store of
(question, strategy, answer) triples produced by the strong model.

Questions are embedded with SentenceTransformers and indexed with FAISS
(inner product over L2-normalized vectors = cosine similarity). Everything is
persisted to ``dir`` (documents.json / index.faiss / meta.json) with atomic
writes, so a repository keeps growing across runs and survives interruption.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StrategyDoc:
    question: str
    strategy: str
    answer: str
    score: Optional[float] = None  # cosine similarity to the query, when retrieved


class StrategyRepository:
    def __init__(
        self,
        dir: str = "strategy_repo",
        embedding_model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        # Heavy imports stay local so that CLI paths that never touch the
        # repository (init, stats, --help) do not pay for them.
        from sentence_transformers import SentenceTransformer

        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)
        self.embedding_model = embedding_model
        self.embedder = SentenceTransformer(embedding_model, device=device)

        self.docs_file = os.path.join(self.dir, "documents.json")
        self.index_file = os.path.join(self.dir, "index.faiss")
        self.meta_file = os.path.join(self.dir, "meta.json")

        self.documents: List[Dict[str, Any]] = []
        self.index = None
        self._load()

    def __len__(self) -> int:
        return len(self.documents)

    # ------------------------------------------------------------------ I/O

    def _load(self) -> None:
        import faiss

        if os.path.exists(self.docs_file):
            with open(self.docs_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            stored = meta.get("embedding_model")
            if stored and stored != self.embedding_model:
                raise ValueError(
                    f"Repository at '{self.dir}' was built with embedding model "
                    f"'{stored}', not '{self.embedding_model}'. Use a fresh dir "
                    "or the original model."
                )

    def _save(self) -> None:
        import faiss

        self._atomic_write(self.docs_file, json.dumps(self.documents, ensure_ascii=False, indent=2))
        if self.index is not None:
            tmp = self.index_file + ".tmp"
            faiss.write_index(self.index, tmp)
            os.replace(tmp, self.index_file)
        meta = {
            "embedding_model": self.embedding_model,
            "size": len(self.documents),
        }
        self._atomic_write(self.meta_file, json.dumps(meta, indent=2))

    @staticmethod
    def _atomic_write(path: str, text: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    # ------------------------------------------------------------ embedding

    def _encode(self, texts: List[str]):
        import numpy as np

        embs = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return embs.astype(np.float32)

    # -------------------------------------------------------------- updates

    def add(self, question: str, strategy: str, answer: str) -> bool:
        """Add one solved query. Returns False when the question is already stored."""
        return self.add_many([StrategyDoc(question=question, strategy=strategy, answer=answer)]) > 0

    def add_many(self, docs: List[StrategyDoc]) -> int:
        """Add a batch, deduplicating by normalized question text. Returns #added."""
        import faiss

        existing = {d["question"].strip().lower() for d in self.documents}
        to_add = []
        for doc in docs:
            key = doc.question.strip().lower()
            if key in existing:
                continue
            existing.add(key)
            to_add.append({"question": doc.question, "strategy": doc.strategy, "answer": doc.answer})
        if not to_add:
            return 0

        embs = self._encode([d["question"] for d in to_add])
        if self.index is None:
            self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        self.documents.extend(to_add)
        self._save()
        return len(to_add)

    # ------------------------------------------------------------ retrieval

    def retrieve(self, query: str, k: int) -> List[StrategyDoc]:
        """Top-k most similar stored questions, most similar first."""
        if self.index is None or self.index.ntotal == 0:
            return []
        k = max(1, min(k, self.index.ntotal))
        sims, idxs = self.index.search(self._encode([query]), k)
        return [
            StrategyDoc(score=float(s), **self.documents[int(i)])
            for s, i in zip(sims[0], idxs[0])
        ]

    def retrieve_random(self, k: int, rng: Optional[random.Random] = None) -> List[StrategyDoc]:
        """k uniformly random entries (the paper's random-strategy ablation)."""
        if not self.documents:
            return []
        rng = rng or random
        k = max(1, min(k, len(self.documents)))
        picks = rng.sample(range(len(self.documents)), k)
        return [StrategyDoc(**self.documents[i]) for i in picks]

    def to_dicts(self) -> List[dict]:
        return [dict(d) for d in self.documents]
