"""
manual_qa_vectors.py — Page-cited PDF manual Q&A with vector retrieval.

This module keeps the same high-level API as manual_qa.py:
    qa = ManualQAVectors(pdf_path="GR86 user manual.pdf", cache_dir=".manual_cache")
    result = qa.ask("What is the tire pressure?", top_k_pages=6)

It uses OpenAI embeddings for retrieval and Chat Completions for final answers.
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

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError("openai SDK not installed. Run: pip install openai") from e

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


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
    p = Path(pdf_path)
    stat = p.stat()
    return f"{p.name}__mtime{int(stat.st_mtime)}__size{stat.st_size}"


class PDFManualVectorIndex:
    """Builds/loads a page-level vector index using OpenAI embeddings."""

    def __init__(
        self,
        pdf_path: str,
        cache_dir: str = ".manual_cache",
        embedding_model: str = "text-embedding-3-small",
        api_key_env: str = "OPENAI_API_KEY",
    ):
        self.pdf_path = str(pdf_path)
        self.cache_dir = Path(cache_dir)
        _safe_mkdir(self.cache_dir)

        self.cache_key = _default_cache_key(self.pdf_path)
        self.index_dir = self.cache_dir / f"{self.cache_key}__vectors"
        _safe_mkdir(self.index_dir)

        self.embedding_model = embedding_model
        self.pages: List[PageChunk] = []
        self.page_matrix: Optional[np.ndarray] = None  # shape: (n_pages, emb_dim), L2-normalized

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key env var {api_key_env}. "
                f"Set it (e.g., export {api_key_env}=...) then retry."
            )
        self.client = OpenAI(api_key=api_key)

    def build_or_load(self, force_rebuild: bool = False) -> None:
        meta_path = self.index_dir / "meta.json"
        pages_path = self.index_dir / "pages.jsonl"
        emb_path = self.index_dir / "embeddings.npy"

        if not force_rebuild and meta_path.exists() and pages_path.exists() and emb_path.exists():
            self._load(meta_path, pages_path, emb_path)
            return

        self._build()
        self._save(meta_path, pages_path, emb_path)

    def _extract_pages(self) -> List[PageChunk]:
        doc = fitz.open(self.pdf_path)
        pages: List[PageChunk] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = _clean_text(page.get_text("text") or "")
            pages.append(PageChunk(page_number_1based=i + 1, text=text))
        doc.close()
        return pages

    @staticmethod
    def _l2_normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def _embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            safe_chunk = [t if t.strip() else "[empty page]" for t in chunk]
            resp = self.client.embeddings.create(model=self.embedding_model, input=safe_chunk)
            # Preserve order by sorting by index
            data = sorted(resp.data, key=lambda d: d.index)
            for item in data:
                vectors.append(np.array(item.embedding, dtype=np.float32))
        mat = np.vstack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)
        return self._l2_normalize(mat)

    def _build(self) -> None:
        self.pages = self._extract_pages()
        corpus = [p.text for p in self.pages]
        self.page_matrix = self._embed_texts(corpus)

    def _save(self, meta_path: Path, pages_path: Path, emb_path: Path) -> None:
        assert self.page_matrix is not None
        meta = {
            "pdf_path": self.pdf_path,
            "built_at": time.time(),
            "n_pages": len(self.pages),
            "embedding_model": self.embedding_model,
            "embedding_dim": int(self.page_matrix.shape[1]) if self.page_matrix.ndim == 2 else 0,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        with pages_path.open("w", encoding="utf-8") as f:
            for p in self.pages:
                f.write(json.dumps({"page": p.page_number_1based, "text": p.text}, ensure_ascii=False) + "\n")

        np.save(emb_path, self.page_matrix)

    def _load(self, meta_path: Path, pages_path: Path, emb_path: Path) -> None:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("embedding_model") != self.embedding_model:
            # Different embedding model; force rebuild with current model.
            self._build()
            self._save(meta_path, pages_path, emb_path)
            return

        pages: List[PageChunk] = []
        with pages_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                pages.append(PageChunk(page_number_1based=int(obj["page"]), text=str(obj["text"])))
        self.pages = pages

        self.page_matrix = np.load(emb_path, allow_pickle=False)
        self.page_matrix = self._l2_normalize(self.page_matrix.astype(np.float32))

    def search(self, query: str, top_k: int = 6) -> List[Tuple[PageChunk, float]]:
        if not query.strip():
            return []
        assert self.page_matrix is not None
        if self.page_matrix.shape[0] == 0:
            return []

        q_vec = self._embed_texts([query])[0]  # already normalized
        scores = self.page_matrix @ q_vec
        idx = np.argsort(-scores)[:top_k]
        return [(self.pages[i], float(scores[i])) for i in idx if scores[i] > 0]


class ManualQAVectors:
    """High-level Q&A interface with vector retrieval + page citations."""

    def __init__(
        self,
        pdf_path: str,
        cache_dir: str = ".manual_cache",
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        api_key_env: str = "OPENAI_API_KEY",
        max_context_chars_per_page: int = 4500,
    ):
        self.index = PDFManualVectorIndex(
            pdf_path=pdf_path,
            cache_dir=cache_dir,
            embedding_model=embedding_model,
            api_key_env=api_key_env,
        )
        self.index.build_or_load()
        self.model = model
        self.max_context_chars_per_page = int(max_context_chars_per_page)

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
        retrieved = self.index.search(question, top_k=top_k_pages)
        retrieved = [(p, s) for (p, s) in retrieved if s >= min_score]

        if not retrieved:
            return {
                "answer": "I couldn't find relevant pages in the manual for that question.",
                "citations": [],
                "pages_used": [],
            }

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
                    "snippet": (text[:350].replace("\n", " ") + ("..." if len(text) > 350 else "")),
                }
            )

        system_instructions = (
            "You answer questions using ONLY the provided manual pages.\n"
            "You MUST include exact page citations inline in the form (p. N) for every factual claim.\n"
            "If the manual pages do not contain the answer, say you cannot find it in the provided pages.\n"
            "Do NOT invent values.\n"
        )
        user_prompt = (
            f"QUESTION: {question}\n\n"
            f"MANUAL EXCERPTS:\n\n{''.join(context_blocks)}\n\n"
            "Write a clear answer. Every sentence that contains a fact must end with a citation like (p. 123)."
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip()

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
    ap.add_argument("--model", default="gpt-4o-mini", help="Answer model name")
    ap.add_argument("--embedding_model", default="text-embedding-3-small", help="Embedding model name")
    ap.add_argument("--cache", default=".manual_cache", help="Cache directory")
    ap.add_argument("--topk", type=int, default=6, help="Top-K pages to retrieve")
    ap.add_argument("--min_score", type=float, default=0.0, help="Min cosine score to accept a page")
    args = ap.parse_args()

    qa = ManualQAVectors(
        pdf_path=args.pdf,
        cache_dir=args.cache,
        model=args.model,
        embedding_model=args.embedding_model,
    )
    result = qa.ask(args.q, top_k_pages=args.topk, min_score=args.min_score)

    print("\n=== ANSWER ===\n")
    print(result["answer"])
    print("\n=== PAGES USED ===\n")
    print(result["pages_used"])
    print("\n=== CITATIONS (debug) ===\n")
    for c in result["citations"]:
        print(f"- p. {c['page']}  score={c['score']:.4f}  snippet={c['snippet']}")


if __name__ == "__main__":
    main()
