"""
Memory-efficient BM25Plus index using scipy sparse matrices.

Replaces rank_bm25.BM25Plus:
  - TF matrix  : scipy CSC sparse (n_docs x vocab_size)  — replaces list[dict] doc_freqs
  - IDF        : numpy float32 array (vocab_size,)        — replaces Python dict
  - doc_len    : numpy float32 array (n_docs,)            — replaces list[int]
  - vocab      : plain dict[str, int]                     — word → column index

CSC (Compressed Sparse Column) is chosen over CSR because get_scores accesses
the matrix column-by-column (one column per query token). CSC stores each column
contiguously in indptr/indices/data, so a column slice is a zero-copy O(1) view.

Serialization uses numpy/scipy native binary formats instead of pickle:
  tf_matrix.npz  — scipy sparse save_npz (CSC preserved)
  arrays.npz     — numpy savez_compressed (idf, doc_len, scalar params)
  vocab.json     — compact JSON (no whitespace)
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, load_npz, save_npz

logger = logging.getLogger(__name__)

_K1: float = 1.5
_B: float = 0.75
_DELTA: float = 1.0

_TF_FILE = "tf_matrix.npz"
_ARRAYS_FILE = "arrays.npz"
_VOCAB_FILE = "vocab.json"


class BM25PlusIndex:
    """
    Vectorized BM25Plus backed by a scipy CSC sparse TF matrix.

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
        self.tf_matrix: csc_matrix | None = None
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

        # --- build sparse TF matrix (COO → CSC) ---
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

        self.tf_matrix = csc_matrix((vals, (rows, cols)), shape=(n_docs, vocab_size), dtype=np.float32)

        # --- IDF: log((N+1) / df) ---
        self.idf = np.log((n_docs + 1.0) / nd.astype(np.float32)).astype(np.float32)

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Return BM25Plus scores (float32) for all documents.

        Uses CSC column slicing to process only non-zero (doc, term) pairs,
        avoiding a full toarray() dense expansion that would spike to
        O(n_docs x n_query_terms) memory regardless of corpus sparsity.
        """
        if self.tf_matrix is None or not query_tokens:
            return np.zeros(self.corpus_size, dtype=np.float32)

        q_idx = [self.vocab[t] for t in query_tokens if t in self.vocab]
        if not q_idx:
            return np.zeros(self.corpus_size, dtype=np.float32)

        scores = np.zeros(self.corpus_size, dtype=np.float32)
        # CSC indptr lets us slice column qi in O(1) with zero allocation:
        #   indices[indptr[qi]:indptr[qi+1]] → doc ids that contain the term
        #   data[indptr[qi]:indptr[qi+1]]    → their raw TF values
        for qi in q_idx:
            start, end = self.tf_matrix.indptr[qi], self.tf_matrix.indptr[qi + 1]
            if start == end:
                continue
            doc_ids = self.tf_matrix.indices[start:end]
            tf_vals = self.tf_matrix.data[start:end]
            dl = self.doc_len[doc_ids]
            numer = tf_vals * (self.k1 + 1.0)
            denom = tf_vals + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            scores[doc_ids] += self.idf[qi] * (numer / denom + self.delta)

        return scores

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

        with np.load(str(cache_dir / _ARRAYS_FILE)) as arr:
            obj.idf = arr["idf"]
            obj.doc_len = arr["doc_len"]
            p = arr["params"]
            obj.k1, obj.b, obj.delta, obj.avgdl = float(p[0]), float(p[1]), float(p[2]), float(p[3])
            obj.corpus_size = int(p[4])

        with open(cache_dir / _VOCAB_FILE, encoding="utf-8") as f:
            obj.vocab = json.load(f)

        return obj
