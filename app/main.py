from pathlib import Path
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from app.engine import ResearchEngine

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "documents"
engine = ResearchEngine(str(DATA))

app = FastAPI(
    title="AI Cancer Research Assistant",
    description="Local literature retrieval prototype. Not for diagnosis or treatment.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1500)
    top_k: int = Field(default=5, ge=1, le=10)

@app.get("/health")
def health():
    return {"status": "ok", "documents": engine.list_documents(), "chunks": len(engine.chunks)}

@app.get("/documents")
def documents():
    return {"documents": engine.list_documents(), "count": len(engine.list_documents())}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF file மட்டுமே அனுமதிக்கப்படுகிறது.")
    DATA.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    destination = DATA / safe_name
    try:
        content = await file.read()
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="அதிகபட்ச கோப்பு அளவு 25 MB.")
        destination.write_bytes(content)
        result = engine.add_pdf(destination)
        return {"message": "PDF சேர்க்கப்பட்டது", **result}
    except HTTPException:
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/query")
def query(payload: QueryRequest):
    try:
        return engine.answer(payload.question, payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
