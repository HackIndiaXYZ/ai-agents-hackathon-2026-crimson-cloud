"""
api.py
======
FastAPI wrapper for the Indian Legal Localisation engine.

Endpoints:
  GET  /                    – Serves the index.html UI (or Health check)
  POST /api/v1/analyze      – Upload a document → full summary, OR
                               upload a document + question → grounded Q&A answer
  POST /api/v1/ask          – Standalone text Q&A (no file required)

Run locally:
  python api.py
  # or directly:
  uvicorn api:app --reload --port 8000

Docs auto-generated at:
  http://localhost:8000/docs      (Swagger UI)
  http://localhost:8000/redoc     (ReDoc)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Union

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from llm_chain import (
    AnalysisError,
    LegalAnalysis,
    QueryAnswer,
    answer_legal_query,
    orchestrate_analysis,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

SUPPORTED_LANGUAGES = [
    "hindi", "tamil", "telugu", "marathi", "bengali",
    "kannada", "gujarati", "punjabi", "odia", "english",
]

API_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Lifespan – startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on startup and once on shutdown."""
    log.info("=" * 55)
    log.info("  Indian Legal Localisation API  v%s", API_VERSION)
    log.info("  Web UI       : http://localhost:8000/")
    log.info("  Swagger docs : http://localhost:8000/docs")
    log.info("=" * 55)
    yield
    log.info("API shutting down. Goodbye.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Indian Legal Localisation API",
    description=(
        "Upload a legal document (PDF or image) and receive a plain-language "
        "analysis grounded in Indian bare acts, delivered in your language."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS – allow all origins
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    log.info("→  %s %s  (client: %s)", request.method, request.url.path,
             request.client.host if request.client else "unknown")
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    log.info("←  %s %s  %d  %.1f ms",
             request.method, request.url.path, response.status_code, elapsed)
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Serve UI or Health Check", tags=["Status"])
async def root():
    """
    Serves the frontend index.html if it exists in the root directory.
    Otherwise, returns a JSON health status.
    """
    if os.path.exists("index.html"):
        return FileResponse("index.html")
        
    return JSONResponse(content={
        "status": "ok",
        "service": "Indian Legal Localisation API",
        "version": API_VERSION,
        "message": "Backend is active. Drop index.html in the root folder to see the UI.",
        "supported_languages": SUPPORTED_LANGUAGES,
    })


@app.post(
    "/api/v1/analyze",
    summary="Analyse a legal document, or ask a question about it",
    tags=["Analysis"],
)
async def analyze(
    file: UploadFile = File(
        ...,
        description="Legal document to analyse. Accepted: .pdf, .png, .jpg, .jpeg",
    ),
    target_language: str = Form(
        "hindi",
        description="Language for the output.",
    ),
    question: Optional[str] = Form(
        None,
        description="Optional question about the document.",
    ),
):
    # ── 1. File extension validation ─────────────────────────────────────────
    filename  = file.filename or ""
    _, ext    = os.path.splitext(filename.lower())

    log.info("Upload received | file: '%s' | lang: %s", filename, target_language)

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")

    # ── 2. Read file bytes asynchronously ────────────────────────────────────
    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read the uploaded file: {exc}")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── 3. Run the orchestration pipeline ────────────────────────────────────
    user_question = question.strip() if question and question.strip() else None

    try:
        result = orchestrate_analysis(
            file_bytes=file_bytes,
            file_ext=ext,
            target_language=target_language,
            user_question=user_question,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline failed unexpectedly: {exc}")

    if isinstance(result, AnalysisError):
        raise HTTPException(
            status_code=500,
            detail={"error": result.error, "stage": result.stage, "detail": result.detail},
        )

    return result


@app.post(
    "/api/v1/ask",
    summary="Ask a standalone legal question",
    tags=["Analysis"],
)
async def ask(
    question: str = Form(...),
    target_language: str = Form("hindi"),
    file: Optional[UploadFile] = File(None),
):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    file_bytes: Optional[bytes] = None
    file_ext:   Optional[str]   = None

    if file is not None and file.filename:
        _, ext = os.path.splitext(file.filename.lower())
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")
        try:
            file_bytes = await file.read()
            file_ext   = ext
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not read attached file: {exc}")

    try:
        result = answer_legal_query(
            question=question.strip(),
            target_language=target_language,
            file_bytes=file_bytes,
            file_ext=file_ext,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Q&A pipeline failed unexpectedly: {exc}")

    if isinstance(result, AnalysisError):
        raise HTTPException(
            status_code=500,
            detail={"error": result.error, "stage": result.stage, "detail": result.detail},
        )

    return result


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
