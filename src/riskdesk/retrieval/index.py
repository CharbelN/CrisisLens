"""Embeddings and the vector index of past events ("historical twins").

- Embedding model: BAAI/bge-small-en-v1.5 via fastembed (runs on CPU, no PyTorch).
  Events are embedded as passages ("name. description"); headlines as queries,
  which for bge adds a short search instruction in front.
- Index: a NumPy file with one normalised vector per event. Similarity is the
  cosine similarity = dot product of normalised vectors. ~120 events need nothing fancier.
- The index stores the model name and a hash of events.csv. If either no longer
  matches, the index is stale and must be rebuilt (same model in preparation and live).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from riskdesk.config import PROJECT_ROOT
from riskdesk.data.library import DAY1_PREFIX, EVENTS_PATH, file_hash, load_library
from riskdesk.engine.drivers import DRIVER_IDS

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_PATH = PROJECT_ROOT / "data" / "event_index.npz"
MODEL_CACHE_DIR = PROJECT_ROOT / ".cache" / "fastembed"  # git-ignored; not in Temp, which Windows may clean
TOP_K = 4


class Embedder(Protocol):
    name: str

    def passages(self, texts: list[str]) -> np.ndarray: ...
    def query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    def __init__(self, model: str = EMBED_MODEL) -> None:
        self.name = model
        self._model = None

    def _get(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.name, cache_dir=str(MODEL_CACHE_DIR))
        return self._model

    def passages(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._get().passage_embed(texts)))

    def query(self, text: str) -> np.ndarray:
        return np.array(list(self._get().query_embed([text]))[0])


def _normalise(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norms == 0, 1, norms)


def event_text(row: pd.Series) -> str:
    return f"{row['name']}. {row['description']}"


def build_index(library: pd.DataFrame, embedder: Embedder, events_hash: str, path: Path = INDEX_PATH) -> None:
    vectors = _normalise(embedder.passages([event_text(r) for _, r in library.iterrows()]))
    np.savez(path, vectors=vectors.astype(np.float32), ids=np.array(library["id"].tolist()),
             model=np.array(embedder.name), events_hash=np.array(events_hash))


class StaleIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class Twin:
    id: str
    name: str
    date: str
    description: str
    risk_type: str
    similarity: float
    moves: dict[str, float | None]  # real 20-day moves; None = no data
    day1: dict[str, float | None] | None = None  # real first-day reaction; None = not in the library


class Retriever:
    def __init__(self, library: pd.DataFrame, vectors: np.ndarray, ids: list[str], embedder: Embedder) -> None:
        if list(library["id"]) != ids:
            raise StaleIndexError("Index ids do not match the crisis library. Rebuild: uv run python data/build_library.py")
        self.library = library.reset_index(drop=True)
        self.vectors = vectors
        self.embedder = embedder

    @classmethod
    def load(
        cls,
        embedder: Embedder | None = None,
        index_path: Path = INDEX_PATH,
        events_path: Path = EVENTS_PATH,
        library: pd.DataFrame | None = None,
    ) -> "Retriever":
        embedder = embedder or FastEmbedder()
        data = np.load(index_path)
        if str(data["model"]) != embedder.name:
            raise StaleIndexError(f"Index was built with {data['model']}, not {embedder.name}. Rebuild the index.")
        if str(data["events_hash"]) != file_hash(events_path):
            raise StaleIndexError("events.csv changed since the index was built. Rebuild: uv run python data/build_library.py")
        lib = library if library is not None else load_library()
        return cls(lib, data["vectors"], [str(i) for i in data["ids"]], embedder)

    def search(self, text: str, k: int = TOP_K) -> list[Twin]:
        q = _normalise(self.embedder.query(text))
        scores = self.vectors @ q
        order = np.argsort(-scores)[:k]
        out = []
        for i in order:
            row = self.library.iloc[int(i)]
            moves = {d: _num(row[d]) for d in DRIVER_IDS}
            day1 = {d: _num(row[DAY1_PREFIX + d]) for d in DRIVER_IDS} if DAY1_PREFIX + DRIVER_IDS[0] in row else None
            out.append(Twin(str(row["id"]), row["name"], str(row["date"]), row["description"],
                            row["risk_type"], round(float(scores[i]), 3), moves, day1))
        return out


def _num(v) -> float | None:
    return None if pd.isna(v) else float(v)


def twin_average(twins: list[Twin]) -> dict[str, float]:
    """Similarity-weighted average of the twins' real moves; n/a values are skipped."""
    out: dict[str, float] = {}
    for d in DRIVER_IDS:
        pairs = [(max(t.similarity, 0.0), t.moves[d]) for t in twins if t.moves[d] is not None]
        wsum = sum(w for w, _ in pairs)
        if not pairs:
            out[d] = 0.0
        elif wsum == 0:
            out[d] = sum(v for _, v in pairs) / len(pairs)
        else:
            out[d] = sum(w * v for w, v in pairs) / wsum
    return out
