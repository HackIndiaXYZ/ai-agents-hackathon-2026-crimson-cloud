"""
llm_chain.py
============
Core AI orchestration layer for the Indian Legal Localisation app.
Uses Groq (free tier) with llama-3.3-70b-versatile.

Pipeline (orchestrate_analysis):
  1. Text Extraction     – extract text from PDF/image bytes
  2. Keyword Extraction  – Groq extracts 2-3 precise legal search terms
  3. Vector Search       – rag_engine.retrieve_legal_context() fetches bare-act clauses
  4. Contextual Synthesis – combines original doc + RAG chunks into a rich prompt
  5. Generation + Translation – Groq produces a structured JSON response
  6. Pydantic Validation – guarantees schema safety before returning to FastAPI

Environment variables (in .env):
  GROQ_API_KEY   – Groq free-tier key (starts with gsk_)
  GROQ_MODEL     – optional override (default: llama-3.3-70b-versatile)
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from groq import Groq
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("llm_chain")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    log.warning(
        "GROQ_API_KEY is not set!\n"
        "  Add GROQ_API_KEY=gsk_... to your .env file\n"
        "  Get a free key: https://console.groq.com/keys"
    )
else:
    log.info("Groq configured → model: %s", GROQ_MODEL)


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------

class LegalAnalysis(BaseModel):
    """
    Structured output returned by orchestrate_analysis().
    All text fields are in the requested target_language.
    FastAPI serialises this model directly.
    """
    legal_issue: str = Field(
        description="Plain-language summary. No jargon."
    )
    timeline_urgency: str = Field(
        description="Deadlines or response windows. 'No specific deadline' if none."
    )
    action_plan: list[str] = Field(
        description="Step-by-step list of concrete actions the recipient must take."
    )
    is_hallucinated_warning: bool = Field(
        description="True if any advice goes beyond the retrieved legal context."
    )
    source_acts: list[str] = Field(
        default_factory=list,
        description="Indian bare acts that informed this analysis."
    )
    search_terms_used: list[str] = Field(
        default_factory=list,
        description="Legal search terms extracted in Step 2."
    )

    @field_validator("action_plan", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [line.strip("•- ").strip() for line in v.splitlines() if line.strip()]
        return v


class QueryAnswer(BaseModel):
    """
    Structured output for the Q&A mode — returned when the user asks a
    specific question (with or without an accompanying document).
    """
    answer: str = Field(
        description="Direct, plain-language answer to the user's question."
    )
    relevant_acts: list[str] = Field(
        default_factory=list,
        description="Indian bare acts that informed this answer."
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Specific sections/chapters referenced, e.g. 'BNS Section 138'."
    )
    is_hallucinated_warning: bool = Field(
        description="True if any part of the answer goes beyond the retrieved legal context."
    )
    search_terms_used: list[str] = Field(
        default_factory=list,
        description="Legal search terms extracted from the question."
    )


class AnalysisError(BaseModel):
    """Returned instead of LegalAnalysis/QueryAnswer when the pipeline fails."""
    error: str
    stage: str
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def _get_client() -> Groq:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )
    return Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_groq(client: Groq, system_prompt: str, user_prompt: str) -> str:
    """Calls Groq chat completions and returns the response text."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def _extract_json_block(text: str) -> str:
    """Strips markdown fences from model output, returning raw JSON."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _build_rag_context_block(chunks: list[dict]) -> str:
    """Formats RAG chunks into a labelled context block for the prompt."""
    if not chunks:
        return "No relevant bare-act clauses were found in the local database."
    lines = ["=== RETRIEVED LEGAL CONTEXT (Indian Bare Acts) ===\n"]
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] Act: {chunk['act_name']} | "
            f"Chapter: {chunk.get('chapter', 'N/A')} | "
            f"Section: {chunk.get('section', 'N/A')}\n"
            f"{chunk['content']}\n"
            f"{'-' * 60}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 0 – Text extraction from file bytes
# ---------------------------------------------------------------------------

def _ocr_image(image) -> str:
    """Runs pytesseract OCR on a PIL Image and returns extracted text."""
    try:
        import pytesseract
        # lang="eng" is safe default; add "+hin" etc. if tesseract lang packs installed
        return pytesseract.image_to_string(image, lang="eng")
    except ImportError:
        raise RuntimeError(
            "pytesseract not installed. Run: pip install pytesseract\n"
            "Also install the engine: sudo apt install tesseract-ocr"
        )


def _extract_text_from_file(file_bytes: bytes, file_ext: str) -> str:
    """
    Converts raw file bytes into plain text based on file extension.

    PDF strategy (two-pass):
      Pass 1 – PyMuPDF digital text extraction (fast, perfect for typed PDFs)
      Pass 2 – If extracted text < 100 chars (scanned/image PDF), fall back to
               page-by-page OCR via pytesseract on rasterised page images.

    Image strategy (.png / .jpg / .jpeg):
      Direct pytesseract OCR on the image.

    Fallback:
      UTF-8 decode for any other file type.
    """
    import io
    from PIL import Image

    ext = file_ext.lower().strip(".")

    # ── PDF ──────────────────────────────────────────────────────────────────
    if ext == "pdf":
        try:
            import fitz                             # PyMuPDF
        except ImportError:
            raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")

        doc = fitz.open(stream=file_bytes, filetype="pdf")

        # Pass 1: digital text extraction
        pages_text = [page.get_text() for page in doc]
        full_text  = "\n".join(pages_text).strip()

        if len(full_text) >= 100:
            doc.close()
            log.info("PDF (digital) | %d characters extracted", len(full_text))
            return full_text

        # Pass 2: scanned PDF — rasterise each page and OCR it
        log.info(
            "PDF appears scanned (only %d chars from digital pass) — "
            "switching to OCR …", len(full_text)
        )

        ocr_pages = []
        for page_num, page in enumerate(doc, start=1):
            # Render page to a pixel map at 2x zoom for better OCR accuracy
            mat      = fitz.Matrix(2.0, 2.0)
            pix      = page.get_pixmap(matrix=mat, alpha=False)
            img_data = pix.tobytes("png")
            image    = Image.open(io.BytesIO(img_data))
            page_text = _ocr_image(image)
            ocr_pages.append(page_text)
            log.info("  OCR page %d | %d chars", page_num, len(page_text))

        doc.close()
        result = "\n".join(ocr_pages).strip()
        log.info("PDF (OCR) | %d total characters extracted", len(result))
        return result

    # ── Images ───────────────────────────────────────────────────────────────
    elif ext in {"png", "jpg", "jpeg"}:
        image = Image.open(io.BytesIO(file_bytes))
        text  = _ocr_image(image)
        log.info("Image OCR | %d characters extracted", len(text))
        return text

    # ── Fallback ─────────────────────────────────────────────────────────────
    else:
        return file_bytes.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Step 1 – Keyword extraction
# ---------------------------------------------------------------------------

def _step1_extract_keywords(client: Groq, document_text: str) -> list[str]:
    log.info("Step 1: Extracting legal keywords …")

    system = (
        "You are a precise Indian legal analyst. "
        "Output ONLY a valid JSON array of 2–3 search terms. "
        "No explanation, no preamble, no markdown fences."
    )
    user = textwrap.dedent(f"""
        Read this legal document and output a JSON array of 2–3 search terms
        to retrieve relevant sections from Indian bare acts (BNS, CrPC, BSA,
        Transfer of Property Act, Real Estate Act, Delhi Rent Control Act, etc.)

        Rules:
        - Be specific: e.g. "Section 138 cheque dishonour", "unlawful eviction tenant notice"
        - Avoid generic words like "law" or "India"
        - Output ONLY a JSON array: ["term one", "term two"]

        DOCUMENT:
        {document_text[:4000]}
    """).strip()

    raw  = _call_groq(client, system, user)
    raw_json = _extract_json_block(raw)

    try:
        terms = json.loads(raw_json)
        if not isinstance(terms, list):
            raise ValueError("Expected JSON array")
        terms = [str(t).strip() for t in terms if t][:3]
        log.info("  Keywords: %s", terms)
        return terms
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("  Keyword parse failed (%s) — using fallback", exc)
        return [document_text.split(".")[0].strip()[:120]]


# ---------------------------------------------------------------------------
# Step 2 – RAG vector search
# ---------------------------------------------------------------------------

def _step2_vector_search(search_terms: list[str], k: int = 4) -> list[dict]:
    log.info("Step 2: Querying RAG database …")

    try:
        from rag_engine import retrieve_legal_context
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import rag_engine. Ensure rag_engine.py is present and "
            "the DB is built: python rag_engine.py --build"
        ) from exc

    chunks = retrieve_legal_context(" | ".join(search_terms), k=k)
    log.info("  Retrieved %d chunks.", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Step 3 – Synthesis prompt
# ---------------------------------------------------------------------------

def _step3_build_synthesis_prompt(
    document_text: str,
    rag_chunks: list[dict],
    target_language: str,
    search_terms: list[str],
) -> tuple[str, str]:
    log.info("Step 3: Building synthesis prompt …")

    rag_block   = _build_rag_context_block(rag_chunks)
    source_acts = list({c["act_name"] for c in rag_chunks})

    system = textwrap.dedent(f"""
        You are an empathetic Indian legal assistant helping ordinary citizens
        understand legal documents in their own language.

        Write ALL human-readable values in: {target_language.upper()}
        (JSON keys stay in English.)

        Output ONLY a single raw JSON object — no preamble, no markdown fences.

        Rules:
        1. legal_issue, timeline_urgency, and every action_plan item must be in {target_language}.
        2. Set is_hallucinated_warning=true if you add ANY fact not in the document or clauses.
        3. action_plan must have at least 3 items.
        4. Be compassionate — the reader may be frightened or confused.
    """).strip()

    user = textwrap.dedent(f"""
        ── LEGAL DOCUMENT ───────────────────────────────────────────────────
        {document_text[:5000]}

        ── RETRIEVED BARE ACT CLAUSES ───────────────────────────────────────
        {rag_block}

        ── REQUIRED JSON OUTPUT ─────────────────────────────────────────────
        {{
          "legal_issue": "<plain-language summary in {target_language}>",
          "timeline_urgency": "<deadlines or 'No specific deadline' in {target_language}>",
          "action_plan": [
            "<step 1 in {target_language}>",
            "<step 2 in {target_language}>",
            "<step 3 in {target_language}>"
          ],
          "is_hallucinated_warning": <true or false>,
          "source_acts": {json.dumps(source_acts)},
          "search_terms_used": {json.dumps(search_terms)}
        }}
    """).strip()

    return system, user


# ---------------------------------------------------------------------------
# Step 4 – Generate, translate, validate
# ---------------------------------------------------------------------------

def _step4_generate_and_validate(
    client: Groq,
    system_prompt: str,
    user_prompt: str,
    search_terms: list[str],
    rag_chunks: list[dict],
) -> LegalAnalysis:
    log.info("Step 4: Generating final analysis …")

    raw_text = _call_groq(client, system_prompt, user_prompt)
    raw_json = _extract_json_block(raw_text)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        log.warning("  JSON parse failed — asking model to self-correct …")
        fixed    = _call_groq(
            client,
            "You are a JSON repair tool. Return ONLY valid JSON, nothing else.",
            f"Fix the JSON in this text:\n\n{raw_text[:3000]}",
        )
        raw_json = _extract_json_block(fixed)
        data     = json.loads(raw_json)

    data.setdefault("source_acts",       list({c["act_name"] for c in rag_chunks}))
    data.setdefault("search_terms_used", search_terms)

    result = LegalAnalysis(**data)
    log.info("  Analysis validated successfully.")
    return result


# ---------------------------------------------------------------------------
# Q&A mode – Steps 3b / 4b
# ---------------------------------------------------------------------------

def _step1_extract_keywords_from_question(client: Groq, question: str) -> list[str]:
    """
    Like _step1_extract_keywords, but tuned for short user questions
    rather than long documents.
    """
    log.info("Step 1: Extracting legal keywords from question …")

    system = (
        "You are a precise Indian legal analyst. "
        "Output ONLY a valid JSON array of 2–3 search terms. "
        "No explanation, no preamble, no markdown fences."
    )
    user = textwrap.dedent(f"""
        Read this user question and output a JSON array of 2–3 search terms
        to retrieve relevant sections from Indian bare acts (BNS, CrPC, BSA,
        Transfer of Property Act, Real Estate Act, Delhi Rent Control Act, etc.)

        Rules:
        - Be specific: e.g. "Section 138 cheque dishonour", "tenant eviction notice period"
        - Avoid generic words like "law" or "India"
        - Output ONLY a JSON array: ["term one", "term two"]

        QUESTION:
        {question[:1000]}
    """).strip()

    raw      = _call_groq(client, system, user)
    raw_json = _extract_json_block(raw)

    try:
        terms = json.loads(raw_json)
        if not isinstance(terms, list):
            raise ValueError("Expected JSON array")
        terms = [str(t).strip() for t in terms if t][:3]
        log.info("  Keywords: %s", terms)
        return terms
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("  Keyword parse failed (%s) — using fallback", exc)
        return [question[:120]]


def _build_qa_prompt(
    question: str,
    document_text: Optional[str],
    rag_chunks: list[dict],
    target_language: str,
    search_terms: list[str],
) -> tuple[str, str]:
    """
    Builds the system/user prompt pair for Q&A mode.
    If document_text is provided, the answer is grounded in BOTH the
    document and the retrieved bare-act clauses. Otherwise it relies
    purely on the retrieved clauses.
    """
    log.info("Building Q&A synthesis prompt …")

    rag_block   = _build_rag_context_block(rag_chunks)
    source_acts = list({c["act_name"] for c in rag_chunks})

    system = textwrap.dedent(f"""
        You are an empathetic Indian legal assistant. A user has asked you
        a direct legal question. Answer it clearly and accurately.

        Write ALL human-readable values in: {target_language.upper()}
        (JSON keys stay in English.)

        Output ONLY a single raw JSON object — no preamble, no markdown fences.

        Rules:
        1. "answer" must be in {target_language}, plain language, no jargon.
        2. Base your answer ONLY on the retrieved bare-act clauses (and the
           attached document, if provided).
        3. "citations" should list specific sections referenced, e.g.
           "BNS Section 138", "Delhi Rent Control Act Chapter VIII".
        4. Set is_hallucinated_warning=true if you add ANY fact not present
           in the retrieved clauses or document.
        5. If the clauses don't fully answer the question, say so honestly
           in the answer rather than guessing.
    """).strip()

    doc_block = ""
    if document_text:
        doc_block = textwrap.dedent(f"""
            ── ATTACHED DOCUMENT (for context) ──────────────────────────────────
            {document_text[:4000]}
        """).strip() + "\n\n"

    user = textwrap.dedent(f"""
        {doc_block}── USER QUESTION ────────────────────────────────────────────────────
        {question[:1500]}

        ── RETRIEVED BARE ACT CLAUSES ───────────────────────────────────────
        {rag_block}

        ── REQUIRED JSON OUTPUT ─────────────────────────────────────────────
        {{
          "answer": "<direct answer in {target_language}>",
          "relevant_acts": {json.dumps(source_acts)},
          "citations": ["<e.g. BNS Section 138>"],
          "is_hallucinated_warning": <true or false>,
          "search_terms_used": {json.dumps(search_terms)}
        }}
    """).strip()

    return system, user


def _generate_qa_answer(
    client: Groq,
    system_prompt: str,
    user_prompt: str,
    search_terms: list[str],
    rag_chunks: list[dict],
) -> QueryAnswer:
    """Calls Groq for Q&A mode and validates into a QueryAnswer model."""
    log.info("Generating Q&A answer …")

    raw_text = _call_groq(client, system_prompt, user_prompt)
    raw_json = _extract_json_block(raw_text)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        log.warning("  JSON parse failed — asking model to self-correct …")
        fixed    = _call_groq(
            client,
            "You are a JSON repair tool. Return ONLY valid JSON, nothing else.",
            f"Fix the JSON in this text:\n\n{raw_text[:3000]}",
        )
        raw_json = _extract_json_block(fixed)
        data     = json.loads(raw_json)

    data.setdefault("relevant_acts",      list({c["act_name"] for c in rag_chunks}))
    data.setdefault("search_terms_used",  search_terms)
    data.setdefault("citations", [])

    result = QueryAnswer(**data)
    log.info("  Q&A answer validated successfully.")
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_legal_query(
    question: str,
    target_language: str = "hindi",
    rag_k: int = 4,
    file_bytes: Optional[bytes] = None,
    file_ext: Optional[str] = None,
) -> QueryAnswer | AnalysisError:
    """
    Standalone Q&A pipeline. Answers a user's legal question, optionally
    grounded in an attached document.

    Parameters
    ----------
    question        : The user's legal question (required).
    target_language : Output language for the answer.
    rag_k           : Number of RAG chunks to retrieve.
    file_bytes      : Optional — bytes of an accompanying document.
    file_ext        : Optional — extension of that document, e.g. ".pdf".

    Returns
    -------
    QueryAnswer on success, AnalysisError on failure.
    """

    if not question or not question.strip():
        return AnalysisError(error="Empty question.", stage="validation")

    # ── Groq client ──────────────────────────────────────────────────────────
    try:
        client = _get_client()
    except RuntimeError as exc:
        return AnalysisError(error=str(exc), stage="groq_init")

    # ── Optional: extract text from attached document ───────────────────────
    document_text: Optional[str] = None
    if file_bytes and file_ext:
        try:
            log.info("Step 0: Extracting text from attached %s file …", file_ext)
            document_text = _extract_text_from_file(file_bytes, file_ext)
            if not document_text.strip():
                document_text = None
        except Exception as exc:
            log.exception("Step 0 failed (continuing without document)")
            document_text = None

    # ── Step 1: Keyword extraction from the question ────────────────────────
    try:
        search_terms = _step1_extract_keywords_from_question(client, question)
    except Exception as exc:
        log.exception("Step 1 failed")
        return AnalysisError(error="Keyword extraction failed.", stage="step1", detail=str(exc))

    # ── Step 2: RAG vector search ────────────────────────────────────────────
    try:
        rag_chunks = _step2_vector_search(search_terms, k=rag_k)
    except Exception as exc:
        log.exception("Step 2 failed")
        return AnalysisError(error="RAG retrieval failed.", stage="step2", detail=str(exc))

    # ── Step 3: Build Q&A prompt ──────────────────────────────────────────────
    try:
        system_prompt, user_prompt = _build_qa_prompt(
            question, document_text, rag_chunks, target_language, search_terms
        )
    except Exception as exc:
        log.exception("Step 3 failed")
        return AnalysisError(error="Prompt synthesis failed.", stage="step3", detail=str(exc))

    # ── Step 4: Generate + validate ──────────────────────────────────────────
    try:
        result = _generate_qa_answer(
            client, system_prompt, user_prompt, search_terms, rag_chunks
        )
    except json.JSONDecodeError as exc:
        return AnalysisError(
            error="Model returned malformed JSON even after retry.",
            stage="step4", detail=str(exc),
        )
    except Exception as exc:
        log.exception("Step 4 failed")
        return AnalysisError(error="Answer generation failed.", stage="step4", detail=str(exc))

    return result


def orchestrate_analysis(
    file_bytes: bytes,
    file_ext: str,
    target_language: str = "hindi",
    rag_k: int = 4,
    user_question: Optional[str] = None,
) -> LegalAnalysis | QueryAnswer | AnalysisError:
    """
    Full RAG + LLM pipeline. Called by api.py.

    Parameters
    ----------
    file_bytes      : Raw bytes of the uploaded legal document.
    file_ext        : File extension including dot, e.g. ".pdf", ".jpg".
    target_language : Output language for human-readable fields.
    rag_k           : Number of RAG chunks to retrieve.
    user_question   : Optional. If provided, the pipeline answers this
                       specific question (grounded in the document + RAG)
                       instead of producing a full structured summary.

    Returns
    -------
    - If user_question is None  → LegalAnalysis on success
    - If user_question is given → QueryAnswer on success
    - AnalysisError on failure (either mode)
    """

    if not file_bytes:
        return AnalysisError(error="Empty file.", stage="validation")

    # ── Groq client ──────────────────────────────────────────────────────────
    try:
        client = _get_client()
    except RuntimeError as exc:
        return AnalysisError(error=str(exc), stage="groq_init")

    # ── Step 0: Extract text ─────────────────────────────────────────────────
    try:
        log.info("Step 0: Extracting text from %s file …", file_ext)
        document_text = _extract_text_from_file(file_bytes, file_ext)
        if not document_text.strip():
            return AnalysisError(
                error="No text could be extracted from the file.",
                stage="text_extraction",
                detail="The document may be blank or image-based without OCR support.",
            )
    except Exception as exc:
        log.exception("Step 0 failed")
        return AnalysisError(error="Text extraction failed.", stage="step0", detail=str(exc))

    # ── Q&A mode: a specific question was asked about this document ─────────
    if user_question and user_question.strip():
        log.info("User question provided — routing to Q&A mode.")

        try:
            search_terms = _step1_extract_keywords_from_question(client, user_question)
        except Exception as exc:
            log.exception("Step 1 failed")
            return AnalysisError(error="Keyword extraction failed.", stage="step1", detail=str(exc))

        try:
            rag_chunks = _step2_vector_search(search_terms, k=rag_k)
        except Exception as exc:
            log.exception("Step 2 failed")
            return AnalysisError(error="RAG retrieval failed.", stage="step2", detail=str(exc))

        try:
            system_prompt, user_prompt = _build_qa_prompt(
                user_question, document_text, rag_chunks, target_language, search_terms
            )
        except Exception as exc:
            log.exception("Step 3 failed")
            return AnalysisError(error="Prompt synthesis failed.", stage="step3", detail=str(exc))

        try:
            return _generate_qa_answer(
                client, system_prompt, user_prompt, search_terms, rag_chunks
            )
        except json.JSONDecodeError as exc:
            return AnalysisError(
                error="Model returned malformed JSON even after retry.",
                stage="step4", detail=str(exc),
            )
        except Exception as exc:
            log.exception("Step 4 failed")
            return AnalysisError(error="Answer generation failed.", stage="step4", detail=str(exc))

    # ── Summary mode (default): full structured analysis ────────────────────
    # ── Step 1: Keyword extraction ───────────────────────────────────────────
    try:
        search_terms = _step1_extract_keywords(client, document_text)
    except Exception as exc:
        log.exception("Step 1 failed")
        return AnalysisError(error="Keyword extraction failed.", stage="step1", detail=str(exc))

    # ── Step 2: RAG vector search ────────────────────────────────────────────
    try:
        rag_chunks = _step2_vector_search(search_terms, k=rag_k)
    except Exception as exc:
        log.exception("Step 2 failed")
        return AnalysisError(error="RAG retrieval failed.", stage="step2", detail=str(exc))

    # ── Step 3: Build synthesis prompt ───────────────────────────────────────
    try:
        system_prompt, user_prompt = _step3_build_synthesis_prompt(
            document_text, rag_chunks, target_language, search_terms
        )
    except Exception as exc:
        log.exception("Step 3 failed")
        return AnalysisError(error="Prompt synthesis failed.", stage="step3", detail=str(exc))

    # ── Step 4: Generate + validate ──────────────────────────────────────────
    try:
        result = _step4_generate_and_validate(
            client, system_prompt, user_prompt, search_terms, rag_chunks
        )
    except json.JSONDecodeError as exc:
        return AnalysisError(
            error="Model returned malformed JSON even after retry.",
            stage="step4", detail=str(exc),
        )
    except Exception as exc:
        log.exception("Step 4 failed")
        return AnalysisError(error="Analysis generation failed.", stage="step4", detail=str(exc))

    return result


# ---------------------------------------------------------------------------
# CLI runner (for quick text-based testing without the API server)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the legal pipeline from CLI")
    parser.add_argument("--file",     type=str, help="Path to legal document (.pdf/.jpg/.png)")
    parser.add_argument("--question", type=str, help="A specific legal question to ask")
    parser.add_argument("--lang",     type=str, default="hindi", help="Target language")
    parser.add_argument("--k",        type=int, default=4,       help="RAG chunks to retrieve")
    args = parser.parse_args()

    if not args.file and not args.question:
        parser.error("Provide --file, --question, or both.")

    print(f"\n{'='*60}\n  LEGAL PIPELINE  |  lang={args.lang}  |  k={args.k}\n{'='*60}\n")

    if args.file:
        import os as _os
        _, ext = _os.path.splitext(args.file)
        with open(args.file, "rb") as f:
            raw = f.read()

        if args.question:
            # File + question → Q&A mode (document used as context)
            output = orchestrate_analysis(
                raw, ext, target_language=args.lang, rag_k=args.k,
                user_question=args.question,
            )
        else:
            # File only → full summary mode
            output = orchestrate_analysis(raw, ext, target_language=args.lang, rag_k=args.k)
    else:
        # Question only, no file → standalone Q&A
        output = answer_legal_query(args.question, target_language=args.lang, rag_k=args.k)

    if isinstance(output, AnalysisError):
        print(f"❌  Failed at stage: {output.stage}\n   {output.error}\n   {output.detail}")

    elif isinstance(output, QueryAnswer):
        print(f"✅  Answer ready!\n")
        print(f"🔍  Terms     : {output.search_terms_used}")
        print(f"📚  Acts      : {output.relevant_acts}")
        print(f"📖  Citations : {output.citations}")
        print(f"\n💬  ANSWER\n{output.answer}")
        print(f"\n⚠️   Hallucinated: {output.is_hallucinated_warning}")

    else:  # LegalAnalysis
        print(f"✅  Done!\n")
        print(f"🔍  Terms  : {output.search_terms_used}")
        print(f"📚  Acts   : {output.source_acts}")
        print(f"\n📋  ISSUE\n{output.legal_issue}")
        print(f"\n⏰  URGENCY\n{output.timeline_urgency}")
        print(f"\n📝  ACTION PLAN")
        for i, s in enumerate(output.action_plan, 1):
            print(f"   {i}. {s}")
        print(f"\n⚠️   Hallucinated: {output.is_hallucinated_warning}")
