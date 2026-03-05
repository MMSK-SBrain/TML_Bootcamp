# =============================================================================
# test_chromadb.py
# =============================================================================
# Run this script to verify your ChromaDB vector store is working correctly.
# It tests 4 things:
#   1. Can we connect to the database?
#   2. How many chunks are stored?
#   3. Does a PDF-related query return sensible results?
#   4. Does a DTC code query return sensible results?
#
# Run with:  python test_chromadb.py
# =============================================================================

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Must match exactly what you used in 01_build_rag.py
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── helpers ──────────────────────────────────────────────────────────────────

def divider(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)

def print_result(index, result):
    print(f"\n  Result {index + 1}")
    print(f"  Source : {result.metadata.get('source', 'unknown')}")

    # Show the DTC code label if this chunk came from the CSV
    if result.metadata.get("type") == "dtc_code":
        print(f"  DTC    : {result.metadata.get('code', '')}")

    # Print a short preview of the chunk text
    preview = result.page_content[:300].replace("\n", " ")
    print(f"  Preview: {preview}...")

# ── test queries ─────────────────────────────────────────────────────────────

TEST_QUERIES = [
    {
        "label": "SSP Technical Query",
        "question": "How does the supercharger magnetic clutch work?",
    },
    {
        "label": "Engine Specs Query",
        "question": "What is the maximum boost pressure of the TSI engine?",
    },
    {
        "label": "DTC Code Query",
        "question": "What causes a P0299 underboost fault code?",
    },
    {
        "label": "DTC Symptoms Query",
        "question": "What are the symptoms of a faulty knock sensor?",
    },
]

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    divider("ChromaDB Test — 1.4L TSI RAG Pipeline")

    # ── Test 1: Connect to the database ──────────────────────────────────────
    print("\n🔌 TEST 1: Connecting to ChromaDB...")

    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        print("  ✅ Connected successfully")
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        print("  Have you run 01_build_rag.py yet?")
        exit(1)

    # ── Test 2: Count stored chunks ───────────────────────────────────────────
    print("\n📦 TEST 2: Checking stored chunks...")

    try:
        count = vectorstore._collection.count()
        print(f"  ✅ {count} chunks found in the database")

        if count == 0:
            print("  ⚠️  Database is empty — re-run 01_build_rag.py")
            exit(1)
    except Exception as e:
        print(f"  ❌ Could not count chunks: {e}")
        exit(1)

    # ── Test 3 & 4: Run test queries ──────────────────────────────────────────
    print("\n🔍 TEST 3 & 4: Running similarity search queries...")

    all_passed = True

    for test in TEST_QUERIES:
        divider(test["label"])
        print(f"  Query: \"{test['question']}\"")

        try:
            results = vectorstore.similarity_search(test["question"], k=3)

            if not results:
                print("  ⚠️  No results returned")
                all_passed = False
                continue

            for i, result in enumerate(results):
                print_result(i, result)

            print(f"\n  ✅ Returned {len(results)} results")

        except Exception as e:
            print(f"  ❌ Query failed: {e}")
            all_passed = False

    # ── Summary ───────────────────────────────────────────────────────────────
    divider("Summary")

    if all_passed:
        print("  ✅ All tests passed — ChromaDB is ready!")
        print("  Next step: python 02_api_server.py")
    else:
        print("  ⚠️  Some tests had issues — check the output above")

    print("=" * 60 + "\n")