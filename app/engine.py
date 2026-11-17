from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import fitz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResearchEngine:
    """Local, evidence-retrieval research assistant. Not a clinical decision system."""
    def __init__(self, data_dir: str = "data/documents", use_ollama: bool = False,
                 ollama_model: str = "qwen2.5:3b", ollama_url: str = "http://localhost:11434"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url.rstrip("/")
        self.chunks: list[dict[str, Any]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.rebuild_index()

    @staticmethod
    def extract_pdf(path: Path) -> str:
        doc = fitz.open(path)
        pages = []
        for page_no, page in enumerate(doc, start=1):
            txt = page.get_text("text").strip()
            if txt:
                pages.append(f"[Page {page_no}] {txt}")
        return "\n".join(pages)

    @staticmethod
    def split_text(text: str, chunk_words: int = 180, overlap: int = 35) -> list[str]:
        words = re.findall(r"\S+", text)
        if not words:
            return []
        chunks = []
        step = max(1, chunk_words - overlap)
        for start in range(0, len(words), step):
            part = " ".join(words[start:start + chunk_words])
            if part:
                chunks.append(part)
            if start + chunk_words >= len(words):
                break
        return chunks

    def add_pdf(self, source_path: str | Path) -> dict:
        source = Path(source_path)
        if source.suffix.lower() != ".pdf":
            raise ValueError("PDF files only.")
        text = self.extract_pdf(source)
        if len(text.strip()) < 40:
            raise ValueError("PDF-ல் போதுமான selectable text இல்லை. OCR செய்யப்பட்ட PDF பயன்படுத்தவும்.")
        target = self.data_dir / source.name
        if source.resolve() != target.resolve():
            target.write_bytes(source.read_bytes())
        self.rebuild_index()
        return {"filename": target.name, "characters": len(text), "chunks_added": len(self.split_text(text))}

    def rebuild_index(self) -> None:
        self.chunks = []
        for pdf in sorted(self.data_dir.glob("*.pdf")):
            try:
                text = self.extract_pdf(pdf)
                for i, chunk in enumerate(self.split_text(text)):
                    self.chunks.append({"source": pdf.name, "chunk": i + 1, "text": chunk})
            except Exception:
                continue
        if self.chunks:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
            self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])
        else:
            self.vectorizer, self.matrix = None, None

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        if not question.strip():
            raise ValueError("Research question காலியாக இருக்கக்கூடாது.")
        if self.vectorizer is None or self.matrix is None:
            return []
        qv = self.vectorizer.transform([question])
        scores = cosine_similarity(qv, self.matrix).ravel()
        indexes = np.argsort(scores)[::-1][:max(1, min(top_k, 10))]
        results = []
        for idx in indexes:
            score = float(scores[idx])
            if score > 0:
                item = dict(self.chunks[int(idx)])
                item["score"] = round(score, 4)
                results.append(item)
        return results

    def _ask_ollama(self, question: str, evidence: list[dict]) -> str | None:
        try:
            import requests
            context = "\n\n".join(
                f"[{i+1}] Source: {e['source']}, chunk {e['chunk']}\n{e['text']}"
                for i, e in enumerate(evidence)
            )
            prompt = (
                "You are a cautious biomedical literature research assistant. "
                "Use only the supplied excerpts. Do not diagnose, prescribe, or recommend treatment. "
                "Clearly distinguish evidence from hypothesis, state when evidence is insufficient, "
                "and cite excerpt numbers like [1]. Answer in Tamil.\n\n"
                f"Research question: {question}\n\nEvidence:\n{context}\n\n"
                "Provide: சுருக்கம், ஆதாரங்கள், வரம்புகள், அடுத்த ஆராய்ச்சி கேள்விகள்."
            )
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip() or None
        except Exception:
            return None

    def answer(self, question: str, top_k: int = 5) -> dict:
        evidence = self.retrieve(question, top_k)
        if not evidence:
            return {
                "answer": "பதிவேற்றப்பட்ட ஆவணங்களில் இந்தக் கேள்விக்குப் பொருத்தமான ஆதாரம் கிடைக்கவில்லை. தொடர்புடைய ஆராய்ச்சி PDF ஆவணங்களைச் சேர்த்து மீண்டும் முயற்சிக்கவும்.",
                "evidence": [],
                "mode": "no_evidence",
                "limitations": ["இந்த அமைப்பு பதிவேற்றப்பட்ட ஆவணங்களை மட்டுமே தேடுகிறது."]
            }
        generated = self._ask_ollama(question, evidence) if self.use_ollama else None
        if generated:
            answer, mode = generated, "local_ollama"
        else:
            excerpts = "\n\n".join(
                f"[{i+1}] {e['source']} (பகுதி {e['chunk']}, relevance {e['score']}):\n{e['text']}"
                for i, e in enumerate(evidence)
            )
            answer = (
                "கிடைத்த ஆவணங்களின் அடிப்படையிலான ஆதாரத் தொகுப்பு:\n\n" + excerpts +
                "\n\nஆராய்ச்சி இடைவெளி: இந்தச் சிறிய ஆவணத் தொகுப்பில் பதிலளிக்கப்படாத கேள்விகள், "
                "முரண்பட்ட முடிவுகள் மற்றும் மாதிரி அளவு போன்ற வரம்புகளை ஆய்வாளர் தனியாகச் சரிபார்க்க வேண்டும்."
            )
            mode = "tfidf_evidence_fallback"
        return {
            "answer": answer,
            "evidence": [{"source": e["source"], "chunk": e["chunk"], "score": e["score"], "excerpt": e["text"][:700]} for e in evidence],
            "mode": mode,
            "limitations": [
                "TF-IDF பொருத்தம் என்பது அறிவியல் உண்மை அல்லது மருத்துவ ஆதாரத்தின் தரத்தை உறுதிப்படுத்தாது.",
                "ஆவணங்களின் வெளியீட்டு தேதி, peer-review நிலை, ஆய்வு வடிவமைப்பு மற்றும் bias ஆகியவற்றை மனிதர் சரிபார்க்க வேண்டும்.",
                "இது நோயறிதல் அல்லது சிகிச்சை முடிவுகளுக்கான கருவி அல்ல."
            ]
        }

    def list_documents(self) -> list[str]:
        return [p.name for p in sorted(self.data_dir.glob("*.pdf"))]
