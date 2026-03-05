# =============================================================================
# 01_build_rag.py
# =============================================================================
# This script does three things:
#   1. Loads our source documents (a PDF and a CSV)
#   2. Splits them into small chunks of text
#   3. Converts those chunks into embeddings and saves them into ChromaDB
#
# After running this script, you will have a "chroma_db" folder in your
# project directory that stores all the vector embeddings.
# Run this script ONCE before starting the API server.
# =============================================================================

import os
import csv

# LangChain tools for loading and splitting documents
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ChromaDB is our local vector database - no server needed, saves to disk
from langchain_community.vectorstores import Chroma

# Sentence Transformers gives us FREE local embeddings - no API key needed
from langchain_community.embeddings import SentenceTransformerEmbeddings


# =============================================================================
# CONFIGURATION - Change these paths if your files are in a different location
# =============================================================================

PDF_PATH = "data/SSP_359.pdf"          # The VW SSP 359 manual
CSV_PATH = "data/dtc.csv"  # The DTC codes spreadsheet
CHROMA_DB_PATH = "chroma_db"           # Where ChromaDB will save its files

# The embedding model to use - this runs 100% locally, no internet needed
# all-MiniLM-L6-v2 is small, fast, and works great for technical documents
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk settings - how we split the documents into pieces
# Think of chunks like index cards - each one holds a small piece of information
CHUNK_SIZE = 500        # Each chunk is ~500 characters long
CHUNK_OVERLAP = 50      # Chunks overlap by 50 chars so we don't lose context at the edges


# =============================================================================
# STEP 1: LOAD THE PDF
# =============================================================================

def load_pdf(path):
    """
    Loads a PDF and returns a list of Document objects.
    Each Document contains the text from one page of the PDF,
    plus metadata telling us which page it came from.
    """
    print(f"\n📄 Loading PDF: {path}")

    # PyPDFLoader reads the PDF page by page
    loader = PyPDFLoader(path)
    pages = loader.load()

    print(f"   ✅ Loaded {len(pages)} pages from the PDF")
    return pages


# =============================================================================
# STEP 2: LOAD THE CSV (DTC CODES)
# =============================================================================

def load_csv(path):
    """
    Loads the DTC CSV file and converts each row into a Document object.
    
    We do this manually (instead of using a LangChain CSV loader) so we can
    build a nicely formatted text string for each DTC code row.
    This makes it easier for the LLM to read and understand the content.
    """
    print(f"\n📊 Loading CSV: {path}")

    documents = []

    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # Build a human-readable text block from each CSV row
            # This format reads naturally when retrieved and passed to the LLM
            text = f"""
DTC Code: {row.get('Code', '')}
Description: {row.get('Description', '')}
System: {row.get('System', '')}
Affected Component: {row.get('Affected_Component', '')}
SSP Reference: {row.get('SSP_Reference', '')}
Common Causes: {row.get('Common_Causes', '')}
Symptoms: {row.get('Symptoms', '')}
Recommended Fix: {row.get('Recommended_Fix', '')}
""".strip()

            # Wrap the text in a Document object with useful metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source": path,
                    "type": "dtc_code",
                    "code": row.get('Code', ''),
                    "system": row.get('System', '')
                }
            )
            documents.append(doc)

    print(f"   ✅ Loaded {len(documents)} DTC codes from the CSV")
    return documents


# =============================================================================
# STEP 3: SPLIT DOCUMENTS INTO CHUNKS
# =============================================================================

def split_documents(documents):
    """
    Splits large documents into smaller chunks.
    
    Why do we need to split?
    - Embedding models have a token limit (they can't process huge blocks of text)
    - Smaller chunks = more precise search results
    - When a user asks a question, we retrieve the most relevant chunks,
      not the entire document
    
    RecursiveCharacterTextSplitter tries to split on natural boundaries
    like paragraphs and sentences before resorting to splitting mid-sentence.
    """
    print(f"\n✂️  Splitting documents into chunks...")
    print(f"   Chunk size: {CHUNK_SIZE} characters")
    print(f"   Chunk overlap: {CHUNK_OVERLAP} characters")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split on these characters in order (paragraphs first, then sentences, etc.)
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"   ✅ Created {len(chunks)} chunks total")
    return chunks


# =============================================================================
# STEP 4: CREATE EMBEDDINGS AND SAVE TO CHROMADB
# =============================================================================

def build_vectorstore(chunks):
    """
    Takes our text chunks and:
    1. Converts each chunk into a vector (embedding) using the sentence transformer model
    2. Saves all vectors into ChromaDB on disk
    
    What is an embedding?
    - An embedding is a list of numbers (e.g. [0.23, -0.11, 0.87, ...])
    - Similar pieces of text produce similar number lists
    - This lets us find relevant chunks by searching for similar vectors
    
    ChromaDB stores both the original text AND its vector embedding together,
    so we can retrieve the original text after a similarity search.
    """
    print(f"\n🔢 Creating embeddings and saving to ChromaDB...")
    print(f"   Embedding model: {EMBEDDING_MODEL}")
    print(f"   This may take a few minutes on first run (model downloads ~90MB)...")

    # Load the embedding model - this runs locally on your machine
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    # Create the ChromaDB vector store
    # - Chroma.from_documents() does everything: embeds each chunk AND saves to disk
    # - persist_directory tells it where to save the database files
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    print(f"   ✅ Vector store saved to: {CHROMA_DB_PATH}/")
    return vectorstore


# =============================================================================
# STEP 5: TEST THE VECTOR STORE WITH A SAMPLE QUERY
# =============================================================================

def test_query(vectorstore):
    """
    Runs a quick test to make sure the vector store is working correctly.
    We ask a sample question and print the top 3 most relevant chunks.
    
    This is a "similarity search" - it finds chunks whose embeddings are
    mathematically closest to the embedding of our test question.
    """
    print("\n🧪 Testing the vector store with a sample query...")

    test_question = "How does the supercharger boost pressure work at low RPM?"

    # similarity_search returns the k most relevant document chunks
    results = vectorstore.similarity_search(test_question, k=3)

    print(f"\n   Query: '{test_question}'")
    print(f"   Top {len(results)} results:\n")

    for i, result in enumerate(results):
        print(f"   --- Result {i+1} ---")
        # Show first 200 characters of each result so it's readable
        print(f"   Source: {result.metadata.get('source', 'unknown')}")
        print(f"   Preview: {result.page_content[:200]}...")
        print()


# =============================================================================
# MAIN - Run everything in order
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  RAG Vector Store Builder - 1.4L TSI Engine (SSP 359)")
    print("=" * 60)

    # --- Check that source files exist before we start ---
    if not os.path.exists(PDF_PATH):
        print(f"\n❌ ERROR: PDF not found at '{PDF_PATH}'")
        print("   Make sure SSP_359.pdf is inside the data/ folder")
        exit(1)

    if not os.path.exists(CSV_PATH):
        print(f"\n❌ ERROR: CSV not found at '{CSV_PATH}'")
        print("   Make sure dtc_ssp359_1.4tsi.csv is inside the data/ folder")
        exit(1)

    # --- Load documents ---
    pdf_docs = load_pdf(PDF_PATH)
    csv_docs = load_csv(CSV_PATH)

    # Combine both document lists into one
    all_documents = pdf_docs + csv_docs
    print(f"\n📚 Total documents loaded: {len(all_documents)}")

    # --- Split into chunks ---
    chunks = split_documents(all_documents)

    # --- Build the vector store ---
    vectorstore = build_vectorstore(chunks)

    # --- Run a quick test ---
    test_query(vectorstore)

    # --- Done! ---
    print("=" * 60)
    print("  ✅ Done! Your vector store is ready.")
    print(f"  📁 Database saved to: {CHROMA_DB_PATH}/")
    print()
    print("  Next step: Run the API server with:")
    print("  python 02_api_server.py")
    print("=" * 60)