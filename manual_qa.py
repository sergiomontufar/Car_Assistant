"""
manual_qa.py — Page-cited PDF manual Q&A module

Key features:
- Builds a page-level search index from a PDF (TF-IDF).
- Retrieves top-K relevant pages for a question.
- Calls OpenAI Responses API with those pages as context.
- Returns:
  - answer_text (with inline citations like "(p. 123)")
  - citations: structured list with page numbers + snippet previews

Usage (as a library):
    from manual_qa import ManualQA

    qa = ManualQA(
        pdf_path="GR86_Manual.pdf",
        cache_dir=".manual_cache",
        model="gpt-4.1-mini"
    )

    result = qa.ask("What is the wheel lug torque?")
    print(result["answer"])
    print(result["citations"])

CLI:
    python manual_qa.py --pdf GR86_Manual.pdf --q "What is the oil drain plug torque?"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise ImportError("PyMuPDF not installed. Run: pip install pymupdf") from e

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# OpenAI Python SDK (v1+)
# Docs: Responses API reference :contentReference[oaicite:1]{index=1}
try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError("openai SDK not installed. Run: pip install openai") from e


@dataclass
class PageChunk:
    page_number_1based: int
    text: str


def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _default_cache_key(pdf_path: str) -> str:
    # Cache key based on filename + mtime + size
    p = Path(pdf_path)
    stat = p.stat()
    return f"{p.name}__mtime{int(stat.st_mtime)}__size{stat.st_size}"


class PDFManualIndex:
    """
    Builds and queries a page-level TF-IDF index of a PDF manual.
    """

    def __init__(self, pdf_path: str, cache_dir: str = ".manual_cache"):
        self.pdf_path = str(pdf_path)
        self.cache_dir = Path(cache_dir)
        _safe_mkdir(self.cache_dir)

        self.cache_key = _default_cache_key(self.pdf_path)
        self.index_dir = self.cache_dir / self.cache_key
        _safe_mkdir(self.index_dir)

        self.pages: List[PageChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.page_matrix: Optional[np.ndarray] = None  # shape: (n_pages, n_terms)

    def build_or_load(self, force_rebuild: bool = False) -> None:
        meta_path = self.index_dir / "meta.json"
        pages_path = self.index_dir / "pages.jsonl"
        vec_path = self.index_dir / "tfidf_vocab.json"
        mat_path = self.index_dir / "tfidf_matrix.npy"

        if not force_rebuild and meta_path.exists() and pages_path.exists() and vec_path.exists() and mat_path.exists():
            self._load(meta_path, pages_path, vec_path, mat_path)
            return

        self._build()
        self._save(meta_path, pages_path, vec_path, mat_path)

    def _extract_pages(self) -> List[PageChunk]:
        doc = fitz.open(self.pdf_path)
        pages: List[PageChunk] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            text = _clean_text(text)
            pages.append(PageChunk(page_number_1based=i + 1, text=text))
        doc.close()
        return pages

    def _build(self) -> None:
        self.pages = self._extract_pages()

        # Some PDFs have many mostly-empty pages; keep them but they won't rank highly.
        corpus = [p.text if p.text else "" for p in self.pages]

        # Page-level TF-IDF. Good default for manuals without requiring embeddings.
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_df=0.95,
            min_df=1,
            ngram_range=(1, 2),
        )
        X = self.vectorizer.fit_transform(corpus)  # sparse matrix
        X = normalize(X, norm="l2", axis=1)
        self.page_matrix = X

    def _save(self, meta_path: Path, pages_path: Path, vec_path: Path, mat_path: Path) -> None:
        assert self.vectorizer is not None
        assert self.page_matrix is not None

        meta = {
            "pdf_path": self.pdf_path,
            "built_at": time.time(),
            "n_pages": len(self.pages),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        with pages_path.open("w", encoding="utf-8") as f:
            for p in self.pages:
                f.write(json.dumps({"page": p.page_number_1based, "text": p.text}, ensure_ascii=False) + "\n")

        # Save vocabulary & idf to reconstruct vectorizer without pickles.
        vocab = self.vectorizer.vocabulary_
        idf = self.vectorizer.idf_.tolist()
        vec_payload = {
            "vocabulary": vocab,
            "idf": idf,
            "params": {
                "lowercase": True,
                "stop_words": "english",
                "max_df": 0.95,
                "min_df": 1,
                "ngram_range": (1, 2),
            },
        }
        vec_path.write_text(json.dumps(vec_payload), encoding="utf-8")

        # Store sparse matrix efficiently as npz-like via scipy? We avoid extra deps:
        # Convert to CSR components and store as .npz using numpy.
        X = self.page_matrix.tocsr()
        np.savez_compressed(
            mat_path.with_suffix(".npz"),
            data=X.data,
            indices=X.indices,
            indptr=X.indptr,
            shape=X.shape,
        )

    def _load(self, meta_path: Path, pages_path: Path, vec_path: Path, mat_path: Path) -> None:
        # Load pages
        pages: List[PageChunk] = []
        with pages_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                pages.append(PageChunk(page_number_1based=int(obj["page"]), text=str(obj["text"])))
        self.pages = pages

        # Rebuild vectorizer
        vec_payload = json.loads(vec_path.read_text(encoding="utf-8"))
        params = vec_payload.get("params", {})
        vocab = vec_payload["vocabulary"]
        idf = np.array(vec_payload["idf"], dtype=float)

        self.vectorizer = TfidfVectorizer(**params)
        self.vectorizer.vocabulary_ = {k: int(v) for k, v in vocab.items()}
        self.vectorizer.idf_ = idf
        self.vectorizer._tfidf._idf_diag = None  # will be lazily rebuilt

        # Load matrix
        npz_path = mat_path.with_suffix(".npz")
        z = np.load(npz_path, allow_pickle=False)
        data = z["data"]
        indices = z["indices"]
        indptr = z["indptr"]
        shape = tuple(z["shape"])
        # Construct CSR without importing scipy by using sklearn's sparse (it uses scipy).
        from scipy.sparse import csr_matrix  # scikit-learn depends on scipy

        X = csr_matrix((data, indices, indptr), shape=shape)
        self.page_matrix = X

    def search(self, query: str, top_k: int = 6) -> List[Tuple[PageChunk, float]]:
        """
        Returns top_k (page, score) pairs.
        """
        if not query.strip():
            return []

        assert self.vectorizer is not None and self.page_matrix is not None

        q = self.vectorizer.transform([query])
        q = normalize(q, norm="l2", axis=1)

        # cosine similarity for L2-normalized TF-IDF is dot product
        scores = (self.page_matrix @ q.T).toarray().ravel()
        if scores.size == 0:
            return []

        idx = np.argsort(-scores)[:top_k]
        results: List[Tuple[PageChunk, float]] = [(self.pages[i], float(scores[i])) for i in idx if scores[i] > 0]
        return results


class ManualQA:
    """
    High-level Q&A interface: retrieve relevant pages, ask model, enforce page citations.
    """

    def __init__(
        self,
        pdf_path: str,
        cache_dir: str = ".manual_cache",
        model: str = "gpt-4.1-mini",
        api_key_env: str = "OPENAI_API_KEY",
        max_context_chars_per_page: int = 4500,
    ):
        self.index = PDFManualIndex(pdf_path=pdf_path, cache_dir=cache_dir)
        self.index.build_or_load()
        self.model = model
        self.max_context_chars_per_page = int(max_context_chars_per_page)

        # OpenAI client uses OPENAI_API_KEY by default, but we keep this explicit.
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key env var {api_key_env}. "
                f"Set it (e.g., export {api_key_env}=...) then retry."
            )
        self.client = OpenAI(api_key=api_key)

    def ask(
        self,
        question: str,
        top_k_pages: int = 6,
        min_score: float = 0.0,
    ) -> Dict[str, object]:
        """
        Returns dict:
          {
            "answer": "... (p. X) ...",
            "citations": [{"page": X, "score": 0.12, "snippet": "..."}, ...],
            "pages_used": [X, Y, ...]
          }
        """
        retrieved = self.index.search(question, top_k=top_k_pages)
        retrieved = [(p, s) for (p, s) in retrieved if s >= min_score]

        if not retrieved:
            return {
                "answer": "I couldn't find relevant pages in the manual for that question.",
                "citations": [],
                "pages_used": [],
            }

        # Build context with explicit page labels for exact citations
        context_blocks: List[str] = []
        citations: List[Dict[str, object]] = []
        pages_used: List[int] = []

        for page, score in retrieved:
            pages_used.append(page.page_number_1based)
            text = page.text[: self.max_context_chars_per_page]
            context_blocks.append(f"=== PAGE {page.page_number_1based} ===\n{text}\n")
            citations.append(
                {
                    "page": page.page_number_1based,
                    "score": score,
                    "snippet": (text[:350].replace("\n", " ") + ("…" if len(text) > 350 else "")),
                }
            )

        # Strong instruction: must cite pages inline.
        system_instructions = (
            "You answer questions using ONLY the provided manual pages.\n"
            "You MUST include exact page citations inline in the form (p. N) for every factual claim.\n"
            "If the manual pages do not contain the answer, say you cannot find it in the provided pages.\n"
            "Do NOT invent values.\n"
        )

        user_prompt = (
            f"{system_instructions}\n"
            f"QUESTION: {question}\n\n"
            f"MANUAL EXCERPTS:\n\n{''.join(context_blocks)}\n"
            "Write a clear answer. Every sentence that contains a fact must end with a citation like (p. 123)."
        )

        # Responses API call :contentReference[oaicite:2]{index=2}
        resp = self.client.responses.create(
            model=self.model,
            input=user_prompt,
        )
        answer = getattr(resp, "output_text", None) or ""

        # Safety net: if model forgets citations, append a citations footer
        if not re.search(r"\(p\.\s*\d+\)", answer):
            used_pages_str = ", ".join(str(p) for p in pages_used)
            answer = answer.strip() + f"\n\nCitations: (p. {used_pages_str})"

        return {
            "answer": answer.strip(),
            "citations": citations,
            "pages_used": pages_used,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to the manual PDF")
    ap.add_argument("--q", required=True, help="Question to ask")
    ap.add_argument("--model", default="gpt-4.1-mini", help="Model name")
    ap.add_argument("--cache", default=".manual_cache", help="Cache directory")
    ap.add_argument("--topk", type=int, default=6, help="Top-K pages to retrieve")
    args = ap.parse_args()

    qa = ManualQA(pdf_path=args.pdf, cache_dir=args.cache, model=args.model)
    result = qa.ask(args.q, top_k_pages=args.topk)

    print("\n=== ANSWER ===\n")
    print(result["answer"])
    print("\n=== PAGES USED ===\n")
    print(result["pages_used"])
    print("\n=== CITATIONS (debug) ===\n")
    for c in result["citations"]:
        print(f"- p. {c['page']}  score={c['score']:.4f}  snippet={c['snippet']}")


if __name__ == "__main__":
    main()