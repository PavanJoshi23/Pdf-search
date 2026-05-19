import os
import io
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import PyPDF2
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

load_dotenv()

app = FastAPI(title="PDF Chat RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

azure_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

CHAT_DEPLOYMENT = os.environ["AZURE_OPENAI_API_DEPLOYMENT_NAME"]
VECTOR_SIZE = 384   # all-MiniLM-L6-v2 output dimension
COLLECTION = "pdf_chunks"

# Local embedding model — no Azure quota needed
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

qdrant = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)


def init_collection() -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


init_collection()


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_text(pdf_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def embed(text: str) -> list[float]:
    return _embedder.encode(text).tolist()


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


class ChatRequest(BaseModel):
    pdf_id: str
    question: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    raw = await file.read()
    text = extract_text(raw)
    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from this PDF.")

    chunks = chunk_text(text)
    pdf_id = str(uuid.uuid4())

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embed(chunk),
            payload={
                "pdf_id": pdf_id,
                "text": chunk,
                "filename": file.filename,
                "chunk_index": idx,
            },
        )
        for idx, chunk in enumerate(chunks)
    ]

    qdrant.upsert(collection_name=COLLECTION, points=points)

    return {
        "pdf_id": pdf_id,
        "filename": file.filename,
        "chunks": len(chunks),
        "pages": len(PyPDF2.PdfReader(io.BytesIO(raw)).pages),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=embed(req.question),
        limit=5,
        query_filter=Filter(
            must=[FieldCondition(key="pdf_id", match=MatchValue(value=req.pdf_id))]
        ),
    )

    if not hits:
        return {"answer": "No relevant content found in the document.", "sources": []}

    # Build context from top-k retrieved chunks
    context = "\n\n---\n\n".join(h.payload["text"] for h in hits)

    completion = azure_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions strictly from "
                    "the provided document context. Be concise and accurate. "
                    "If the answer is not present in the context, reply with: "
                    "'I could not find that information in the document.'"
                ),
            },
            {
                "role": "user",
                "content": f"Document context:\n{context}\n\nQuestion: {req.question}",
            },
        ],
    )

    return {
        "answer": completion.choices[0].message.content,
        "sources": [
            {"text": h.payload["text"][:400], "score": round(h.score, 3)}
            for h in hits
        ],
    }
