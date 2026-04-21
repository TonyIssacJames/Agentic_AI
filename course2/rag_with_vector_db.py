# ============================================================
# LOCAL RAG PIPELINE — Ollama + ChromaDB
# ============================================================
# Install dependencies:
#   pip install pypdf chromadb ollama langchain
#
# Setup Ollama models (run once in terminal):
#   ollama pull nomic-embed-text
#   ollama pull llama3
#
# Run:
#   python rag_chromadb.py
# ============================================================

import sys
import ollama
import chromadb
from pypdf import PdfReader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
PDF_PATH    = "./data/khaja.pdf"   # ← Change this to your PDF path
EMBED_MODEL = "nomic-embed-text"    # Ollama embedding model
CHAT_MODEL  = "phi3"              # Ollama chat model
DB_PATH     = "./chroma_db"         # Folder where ChromaDB saves files
COLLECTION  = "pdf_collection"      # Name for this document's collection


# ─────────────────────────────────────────────
# STEP 1 — Extract text from PDF
# ─────────────────────────────────────────────
def load_pdf(path):
    try:
        reader = PdfReader(path)
    except FileNotFoundError:
        print(f"ERROR: File not found → '{path}'")
        sys.exit(1)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    if not text.strip():
        print("ERROR: No text extracted. PDF might be scanned/image-based.")
        sys.exit(1)

    print(f"✅ Loaded PDF: {len(reader.pages)} pages, {len(text):,} characters")
    return text


# ─────────────────────────────────────────────
# STEP 2 — Split text into chunks
# ─────────────────────────────────────────────
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    print(chunks)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────
# STEP 3 — Store chunks + embeddings in ChromaDB
# ─────────────────────────────────────────────
def store_in_chromadb(chunks):
    # PersistentClient automatically saves to disk at DB_PATH
    client     = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION)

    # Skip if already ingested (avoids duplicate embeddings)
    if collection.count() > 0:
        print(f"✅ ChromaDB already has {collection.count()} chunks — skipping ingestion")
        return collection

    print(f"⏳ Generating embeddings and storing in ChromaDB...")

    for i, chunk in enumerate(chunks):
        # Get embedding from Ollama
        response  = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
        embedding = response["embedding"]

        # Store chunk text + embedding together in ChromaDB
        collection.add(
            ids        = [str(i)],          # Unique ID for each chunk
            embeddings = [embedding],        # Vector
            documents  = [chunk]             # Original text (retrievable later)
        )

        # Progress indicator
        print(f"   {i + 1}/{len(chunks)} stored...", end="\r")

    print(f"\n✅ Stored {collection.count()} chunks in ChromaDB at '{DB_PATH}'")
    return collection


# ─────────────────────────────────────────────
# STEP 4 — Ask a question (Retrieval + Answer)
# ─────────────────────────────────────────────
def ask(question, collection):
    # Embed the question
    q_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]

    # Find top 3 most relevant chunks
    results = collection.query(
        query_embeddings = [q_embedding],
        n_results        = 3
    )

    top_chunks = results["documents"][0]  # List of matching chunk texts
    print(top_chunks)

    # Build a prompt with the retrieved context
    context = "\n\n".join(top_chunks)
    prompt  = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

    # Send to Ollama chat model
    response = ollama.chat(
        model    = CHAT_MODEL,
        messages = [{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  LOCAL RAG  —  Ollama + ChromaDB")
    print("=" * 50)

    # --- Ingestion ---
    text       = load_pdf(PDF_PATH)
    chunks     = split_text(text)
    collection = store_in_chromadb(chunks)

    print()
    print("Pipeline ready! Type your question (or 'quit' to exit)")
    print("-" * 50)

    # --- Query loop ---
    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        answer = ask(question, collection)
        print(f"\nBot: {answer}")


if __name__ == "__main__":
    main()
