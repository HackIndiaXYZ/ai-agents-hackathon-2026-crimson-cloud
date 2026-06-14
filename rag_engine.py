"""
rag_engine.py
=============
Local, $0-budget RAG pipeline for Indian Bare Acts.
Uses LangChain + HuggingFace (all-MiniLM-L6-v2) + Qdrant (local persistence).

Sources:
  - Markdown : Delhi Rent Control Act, Real Estate Act 2016, Transfer of Property Act
  - CSV      : BNS (358 sections), CrPC (534 sections), BSA (170 sections)

Usage
-----
# Step 1 – build the vector DB (run once):
    python rag_engine.py --build

# Step 2 – query:
    python rag_engine.py --query "What are the bail provisions under BNSS?"
    python rag_engine.py --query "Can a landlord evict a tenant in Delhi?" --k 5
"""

import argparse
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR   = Path(os.path.expanduser("~/Code/AiAgentHack_LawSummary/Dataset"))
QDRANT_PATH   = "./qdrant_db"
COLLECTION    = "indian_bare_acts"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# Markdown files  →  (filename, act_name)
MD_FILES = [
    ("Delhi Rent Control Act.md",       "Delhi Rent Control Act"),
    ("realestate act2016.md",           "Real Estate Act 2016"),
    ("The Transfer of Property Act.md", "Transfer of Property Act"),
]

# CSV files  →  (filename, act_name, encoding)
CSV_FILES = [
    ("bns_sections.csv",  "BNS",  "utf-8"),
    ("crpc_sections.csv", "CrPC", "latin-1"),
    ("bsa_sections.csv",  "BSA",  "latin-1"),
]

# Markdown header levels to split on
HEADERS_TO_SPLIT = [
    ("#",   "chapter"),
    ("##",  "section"),
    ("###", "subsection"),
]


# ===========================================================================
# _chunk_csv_row()  –  helper that guarantees every sub-chunk has legal text
# ===========================================================================

def _chunk_csv_row(row, act_name: str, filename: str, char_splitter) -> list:
    """
    Splits one CSV row into 1-N LangChain Documents.

    Key design: the header stamp (Chapter / Section line) is prepended to
    EVERY sub-chunk so that no chunk ever loses its legal identity after
    RecursiveCharacterTextSplitter breaks a long Description into pieces.
    """
    from langchain_core.documents import Document

    chapter_str = str(row["Chapter_name"]).strip()
    section_str = str(row["Section _name"]).strip()

    # Header stamp that will be injected into every sub-chunk
    header_stamp = (
        f"Act: {act_name}\n"
        f"Chapter {row['Chapter']} – {chapter_str}\n"
        f"Section {row['Section']} – {section_str}\n\n"
    )

    description = str(row["Description"]).strip()
    # Normalise Windows-style line endings from the CSV
    description = description.replace("\r\n", "\n").replace("\r", "\n")

    full_content = header_stamp + description

    base_metadata = {
        "act_name":    act_name,
        "source_file": filename,
        "chapter":     chapter_str,
        "section":     section_str,
        "subsection":  "",
    }

    # If the whole thing fits in one chunk, return as-is
    if len(full_content) <= CHUNK_SIZE:
        return [Document(page_content=full_content, metadata=base_metadata)]

    # Otherwise split only the Description, then re-attach the header stamp
    # to every piece — this is the critical fix vs the naive approach
    desc_doc = Document(page_content=description, metadata=base_metadata)
    desc_chunks = char_splitter.split_documents([desc_doc])

    result = []
    for chunk in desc_chunks:
        # Re-prepend the header so the chunk is always self-contained
        chunk.page_content = header_stamp + chunk.page_content
        # Ensure metadata survived the split
        chunk.metadata.update(base_metadata)
        result.append(chunk)

    return result


# ===========================================================================
# initialize_database()
# ===========================================================================

def initialize_database() -> None:
    """
    Ingests all Markdown and CSV files, chunks them, embeds with
    all-MiniLM-L6-v2, and persists everything to a local Qdrant collection.
    Run once; re-running wipes and rebuilds the collection cleanly.
    """

    import pandas as pd
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    print("=" * 60)
    print("  RAG PIPELINE – DATABASE INITIALISATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Splitters
    # ------------------------------------------------------------------
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []

    # ------------------------------------------------------------------
    # PASS A – Markdown files
    # ------------------------------------------------------------------
    print("\n  ── Markdown files ──────────────────────────────────────")

    for filename, act_name in MD_FILES:
        filepath = DATASET_DIR / filename

        if not filepath.exists():
            print(f"  [WARN]  Not found, skipping: {filepath}")
            continue

        print(f"\n  📄  {filename}  ({act_name})")
        raw_text = filepath.read_text(encoding="utf-8")
        print(f"        Raw length               : {len(raw_text):,} chars")

        header_chunks = md_splitter.split_text(raw_text)
        print(f"        After MarkdownHeaderSplit : {len(header_chunks)} chunks")

        fine_chunks = char_splitter.split_documents(header_chunks)
        print(f"        After RecursiveCharSplit  : {len(fine_chunks)} chunks")

        for chunk in fine_chunks:
            chunk.metadata["act_name"]    = act_name
            chunk.metadata["source_file"] = filename

        all_chunks.extend(fine_chunks)

    # ------------------------------------------------------------------
    # PASS B – CSV files (BNS / CrPC / BSA)
    # ------------------------------------------------------------------
    print("\n  ── CSV files ───────────────────────────────────────────")

    for filename, act_name, encoding in CSV_FILES:
        filepath = DATASET_DIR / filename

        if not filepath.exists():
            print(f"  [WARN]  Not found, skipping: {filepath}")
            continue

        print(f"\n  📄  {filename}  ({act_name})")
        df = pd.read_csv(filepath, encoding=encoding)
        print(f"        Rows loaded              : {len(df)}")

        csv_chunks = []
        for _, row in df.iterrows():
            csv_chunks.extend(
                _chunk_csv_row(row, act_name, filename, char_splitter)
            )

        print(f"        After chunking           : {len(csv_chunks)} chunks")

        # Sanity check: flag any chunk whose content is suspiciously short
        thin = [c for c in csv_chunks if len(c.page_content) < 80]
        if thin:
            print(f"  [WARN]  {len(thin)} very short chunks (<80 chars) – check source data")

        all_chunks.extend(csv_chunks)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n  ✅  Total chunks across all acts : {len(all_chunks)}")

    # Verify no chunk is missing its description (quick sanity check)
    empty = [c for c in all_chunks if len(c.page_content.strip()) < 50]
    if empty:
        print(f"  [WARN]  {len(empty)} chunks are nearly empty – investigate source files")
    else:
        print("  ✅  All chunks contain substantive text.")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    print(f"\n  🔄  Loading embedding model  →  {EMBED_MODEL}")
    print("       (first run downloads ~90 MB — subsequent runs use cache)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("  ✅  Embedding model ready.")

    # ------------------------------------------------------------------
    # Qdrant – create / recreate collection
    # ------------------------------------------------------------------
    print(f"\n  🗄️   Setting up Qdrant  →  {QDRANT_PATH}")
    client = QdrantClient(path=QDRANT_PATH)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"       Collection '{COLLECTION}' exists – recreating …")
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=384,               # all-MiniLM-L6-v2 → 384 dims
            distance=Distance.COSINE,
        ),
    )
    print(f"       Collection '{COLLECTION}' created.")

    # ------------------------------------------------------------------
    # Embed & upsert in batches
    # ------------------------------------------------------------------
    print(f"\n  ⚙️   Embedding & storing {len(all_chunks)} chunks …")
    print("       (may take a few minutes on CPU — grab a chai ☕)")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )

    BATCH = 64
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i : i + BATCH]
        vector_store.add_documents(batch)
        done = min(i + BATCH, len(all_chunks))
        print(f"       Stored {done:>5} / {len(all_chunks)} chunks …", end="\r")

    print(f"\n  ✅  All chunks persisted to '{QDRANT_PATH}'.")
    print("=" * 60)
    print("  DATABASE READY.  Run with --query to search.")
    print("=" * 60)


# ===========================================================================
# retrieve_legal_context()
# ===========================================================================

def retrieve_legal_context(query: str, k: int = 3) -> list[dict]:
    """
    Returns the top-k most relevant chunks for *query*.

    Each result dict:
        {
            "rank":        int,
            "act_name":    str,    # e.g. "BNS", "Delhi Rent Control Act"
            "source_file": str,    # e.g. "bns_sections.csv"
            "chapter":     str,
            "section":     str,
            "subsection":  str,
            "content":     str,    # full chunk text (always has legal text)
            "score":       float,  # cosine similarity
        }
    """

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    if not Path(QDRANT_PATH).exists():
        raise FileNotFoundError(
            f"Qdrant DB not found at '{QDRANT_PATH}'. "
            "Run  python rag_engine.py --build  first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    client = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )

    results_with_scores = vector_store.similarity_search_with_score(query, k=k)

    output = []
    for rank, (doc, score) in enumerate(results_with_scores, start=1):
        meta = doc.metadata
        output.append(
            {
                "rank":        rank,
                "act_name":    meta.get("act_name",    "Unknown"),
                "source_file": meta.get("source_file", "Unknown"),
                "chapter":     meta.get("chapter",     ""),
                "section":     meta.get("section",     ""),
                "subsection":  meta.get("subsection",  ""),
                "content":     doc.page_content,
                "score":       round(float(score), 4),
            }
        )

    return output


# ===========================================================================
# CLI helpers
# ===========================================================================

def _print_results(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print(f"  TOP {len(results)} RELEVANT CHUNKS")
    print("=" * 60)
    for r in results:
        print(f"\n  Rank #{r['rank']}  |  Act: {r['act_name']}  |  Score: {r['score']}")
        if r["chapter"]:
            print(f"  Chapter    : {r['chapter']}")
        if r["section"]:
            print(f"  Section    : {r['section']}")
        if r["subsection"]:
            print(f"  Subsection : {r['subsection']}")
        print(f"  File       : {r['source_file']}")
        print("-" * 60)
        # Show first 600 chars of the actual legal text
        snippet = r["content"][:600].replace("\n", " ")
        print(f"  {snippet} …" if len(r["content"]) > 600 else f"  {snippet}")
    print("=" * 60)


# ===========================================================================
# Entry-point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local RAG engine for Indian Bare Acts"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--build",
        action="store_true",
        help="Ingest all files and build the Qdrant vector DB.",
    )
    group.add_argument(
        "--query",
        type=str,
        metavar="QUERY",
        help="Natural-language legal query.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3).",
    )

    args = parser.parse_args()

    if args.build:
        initialize_database()
    else:
        print(f"\n  🔍  Query : {args.query}")
        results = retrieve_legal_context(args.query, k=args.k)
        _print_results(results)
