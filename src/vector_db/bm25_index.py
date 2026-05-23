"""
Memory-efficient BM25Plus index using scipy sparse matrices.

Replaces rank_bm25.BM25Plus:
  - TF matrix  : scipy CSR sparse (n_docs × vocab_size)  — replaces list[dict] doc_freqs
  - IDF        : numpy float32 array (vocab_size,)        — replaces Python dict
  - doc_len    : numpy float32 array (n_docs,)            — replaces list[int]
  - vocab      : plain dict[str, int]                     — word → column index

Serialization uses numpy/scipy native binary formats instead of pickle:
  tf_matrix.npz  — scipy sparse save_npz
  arrays.npz     — numpy savez_compressed (idf, doc_len, scalar params)
  vocab.json     — compact JSON (no whitespace)
"""
import json
import logging
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

logger = logging.getLogger(__name__)

_K1: float = 1.5
_B: float = 0.75
_DELTA: float = 1.0

_TF_FILE = "tf_matrix.npz"
_ARRAYS_FILE = "arrays.npz"
_VOCAB_FILE = "vocab.json"


class BM25PlusIndex:
    """
    Vectorized BM25Plus backed by a scipy CSR sparse TF matrix.

    Scoring formula (per query term q, document d):
        score(q, d) = IDF(q) * (tf(q,d)*(k1+1) / (tf(q,d) + k1*(1-b+b*|d|/avgdl)) + delta)
    where IDF(q) = log((N+1) / df(q)).
    """

    def __init__(self, k1: float = _K1, b: float = _B, delta: float = _DELTA):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.tf_matrix: csr_matrix | None = None
        self.doc_len: np.ndarray | None = None
        self.avgdl: float = 0.0
        self.corpus_size: int = 0

    def build(self, tokenized_corpus: list[list[str]]) -> None:
        """Build index from a pre-tokenized corpus."""
        n_docs = len(tokenized_corpus)
        self.corpus_size = n_docs

        # --- vocabulary ---
        vocab: dict[str, int] = {}
        for tokens in tokenized_corpus:
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab
        vocab_size = len(vocab)

        # --- document lengths ---
        doc_len = np.array([len(tokens) for tokens in tokenized_corpus], dtype=np.float32)
        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean()) if n_docs > 0 else 1.0

        # --- build sparse TF matrix (COO → CSR) ---
        rows, cols, vals = [], [], []
        nd = np.zeros(vocab_size, dtype=np.int32)  # document frequency per term

        for doc_id, tokens in enumerate(tokenized_corpus):
            freq: dict[int, int] = {}
            for t in tokens:
                wi = vocab[t]
                freq[wi] = freq.get(wi, 0) + 1
            for wi, tf in freq.items():
                rows.append(doc_id)
                cols.append(wi)
                vals.append(float(tf))
                nd[wi] += 1

        self.tf_matrix = csr_matrix(
            (vals, (rows, cols)), shape=(n_docs, vocab_size), dtype=np.float32
        )

        # --- IDF: log((N+1) / df) ---
        self.idf = np.log((n_docs + 1.0) / nd.astype(np.float32)).astype(np.float32)

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Return BM25Plus scores (float32) for all documents."""
        if self.tf_matrix is None or not query_tokens:
            return np.zeros(self.corpus_size, dtype=np.float32)

        q_idx = [self.vocab[t] for t in query_tokens if t in self.vocab]
        if not q_idx:
            return np.zeros(self.corpus_size, dtype=np.float32)

        # Dense slice for query terms only: (n_docs, n_q)
        tf_sub = self.tf_matrix[:, q_idx].toarray()
        dl_ratio = self.doc_len[:, np.newaxis] / self.avgdl  # (n_docs, 1)

        numer = tf_sub * (self.k1 + 1.0)
        denom = tf_sub + self.k1 * (1.0 - self.b + self.b * dl_ratio)
        q_idf = self.idf[q_idx]  # (n_q,)

        return (q_idf * (numer / denom + self.delta)).sum(axis=1)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, cache_dir: Path) -> None:
        """Serialize to cache_dir using scipy/numpy native formats."""
        cache_dir.mkdir(parents=True, exist_ok=True)

        save_npz(str(cache_dir / _TF_FILE), self.tf_matrix)

        np.savez_compressed(
            str(cache_dir / _ARRAYS_FILE),
            idf=self.idf,
            doc_len=self.doc_len,
            params=np.array(
                [self.k1, self.b, self.delta, self.avgdl, float(self.corpus_size)],
                dtype=np.float64,
            ),
        )

        with open(cache_dir / _VOCAB_FILE, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def load(cls, cache_dir: Path) -> "BM25PlusIndex":
        """Deserialize from cache_dir."""
        obj = cls()
        obj.tf_matrix = load_npz(str(cache_dir / _TF_FILE))

        arr = np.load(str(cache_dir / _ARRAYS_FILE))
        obj.idf = arr["idf"]
        obj.doc_len = arr["doc_len"]
        p = arr["params"]
        obj.k1, obj.b, obj.delta, obj.avgdl = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        obj.corpus_size = int(p[4])

        with open(cache_dir / _VOCAB_FILE, encoding="utf-8") as f:
            obj.vocab = json.load(f)

        return obj
