# =============================================================================
# 02_api_server.py
# =============================================================================
# This script runs a FastAPI web server that exposes a single endpoint:
#
#   POST /query
#   Body: { "question": "How does the supercharger work?" }
#   Returns: { "question": "...", "context": [...], "answer": "..." }
#
# The server does three things for every request:
#   1. Embeds the question and searches ChromaDB for relevant chunks
#   2. Bundles those chunks together as "context"
#   3. Sends the context + question to an LLM via OpenRouter and returns the answer
#
# Run with:
#   uvicorn 02_api_server:app --host 0.0.0.0 --port 8000 --reload
# =============================================================================

import os
from dotenv import load_dotenv

# FastAPI is our web framework - it handles incoming HTTP requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Pydantic defines the shape of our request and response data
from pydantic import BaseModel

# OpenAI-compatible client - OpenRouter uses the same API format as OpenAI
from openai import OpenAI

# ChromaDB and embeddings - same as in 01_build_rag.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# =============================================================================
# CONFIGURATION
# =============================================================================

# Load environment variables from .env file (if it exists)
# This is how we keep the API key out of the code
load_dotenv()

# Your OpenRouter API key - set this as an environment variable:
#   export OPENROUTER_API_KEY=sk-or-...
# Or put it in a .env file:
#   OPENROUTER_API_KEY=sk-or-...
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# The LLM model to use via OpenRouter
# Free options:  "meta-llama/llama-3.1-8b-instruct:free"
# Paid options:  "google/gemini-flash-1.5"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# Must match the paths used in 01_build_rag.py
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How many chunks to retrieve from ChromaDB for each query
# More chunks = more context for the LLM, but also more tokens used
NUM_CHUNKS = 4


# =============================================================================
# STARTUP - Load the vector store once when the server starts
# =============================================================================

print("\n🚀 Starting RAG API Server...")

# Warn if no API key is set - the server will still start but /query will fail
if not OPENROUTER_API_KEY:
    print("⚠️  WARNING: OPENROUTER_API_KEY is not set.")
    print("   Set it with: export OPENROUTER_API_KEY=your_key_here")

# Load the embedding model (runs locally, no API key needed)
print(f"🔢 Loading embedding model: {EMBEDDING_MODEL}")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Connect to the ChromaDB database we built with 01_build_rag.py
print(f"📦 Connecting to ChromaDB at: {CHROMA_DB_PATH}/")
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)

# Set up the OpenRouter client
# OpenRouter is OpenAI-compatible so we just change the base_url
# We use "not-set" as a fallback so the server starts even without a key
# The /query endpoint will catch the missing key and return a helpful error
openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY or "not-set",
    base_url="https://openrouter.ai/api/v1"
)

print(f"🤖 LLM model: {DEFAULT_MODEL}")
print("✅ Server ready!\n")


# =============================================================================
# FASTAPI APP SETUP
# =============================================================================

app = FastAPI(
    title="1.4L TSI RAG API",
    description="Ask questions about the VW 1.4L TSI engine using SSP 359 and DTC codes",
    version="1.0.0"
)

# CORS middleware lets any webpage or client call this API
# In production you would restrict this to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST AND RESPONSE MODELS
# =============================================================================

class QueryRequest(BaseModel):
    """The shape of the JSON body we expect in a POST /query request"""
    question: str                           # The user's question
    model: str = DEFAULT_MODEL              # Optionally override the LLM model
    num_chunks: int = NUM_CHUNKS            # Optionally override how many chunks to retrieve


class SourceChunk(BaseModel):
    """A single retrieved chunk of context from ChromaDB"""
    content: str                            # The text of the chunk
    source: str                             # Which file it came from
    metadata: dict                          # Any extra metadata (page number, DTC code, etc.)


class QueryResponse(BaseModel):
    """The shape of the JSON we send back"""
    question: str                           # Echo the original question
    answer: str                             # The LLM's answer
    model_used: str                         # Which LLM model was used
    context: list[SourceChunk]             # The chunks we retrieved from ChromaDB


# =============================================================================
# HELPER: RETRIEVE CONTEXT FROM CHROMADB
# =============================================================================

def retrieve_context(question: str, k: int) -> list[SourceChunk]:
    """
    Searches ChromaDB for the k most relevant chunks to the question.
    
    This is the "Retrieval" part of Retrieval-Augmented Generation (RAG).
    The embedding model converts the question into a vector, then ChromaDB
    finds the stored chunks whose vectors are most similar.
    """
    # similarity_search returns LangChain Document objects
    raw_results = vectorstore.similarity_search(question, k=k)

    # Convert to our SourceChunk response model
    chunks = []
    for doc in raw_results:
        chunks.append(SourceChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            metadata=doc.metadata
        ))

    return chunks


# =============================================================================
# HELPER: BUILD THE PROMPT
# =============================================================================

def build_prompt(question: str, context_chunks: list[SourceChunk]) -> str:
    """
    Combines the retrieved context chunks and the user's question into a
    single prompt string that we send to the LLM.
    
    This is the "Augmented Generation" part of RAG.
    We give the LLM the relevant facts BEFORE asking it to answer,
    so it doesn't have to rely on its training data alone.
    """

    # Join all context chunks into one block of text
    context_text = "\n\n---\n\n".join([chunk.content for chunk in context_chunks])

    prompt = f"""You are a helpful automotive technician assistant specialising in the Volkswagen 1.4L TSI engine with dual-charging (SSP 359).

Use ONLY the context provided below to answer the question.
If the answer is not in the context, say "I don't have enough information in the provided documents to answer that."
Be concise and technical in your response.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""

    return prompt


# =============================================================================
# HELPER: CALL THE LLM VIA OPENROUTER
# =============================================================================

def call_llm(prompt: str, model: str) -> str:
    """
    Sends the prompt to the specified LLM model via OpenRouter.
    Returns the model's response as a string.
    
    OpenRouter is a unified API gateway that gives access to many different
    LLM models (Llama, Gemini, GPT-4, etc.) through one endpoint.
    """
    response = openrouter_client.chat.completions.create(
        model=model,
        messages=[
            # The system message sets the overall behaviour of the model
            {
                "role": "system",
                "content": "You are an expert VW automotive technician. Answer questions using only the provided context."
            },
            # The user message contains the full prompt with context + question
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1000,
        temperature=0.2   # Low temperature = more factual, less creative responses
    )

    # Extract the text from the response object
    return response.choices[0].message.content


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/")
def root():
    """Health check endpoint - visit this in your browser to confirm the server is running"""
    return {
        "status": "running",
        "message": "1.4L TSI RAG API is online",
        "usage": "POST /query with JSON body: { 'question': 'your question here' }"
    }


@app.get("/health")
def health():
    """Returns the status of the server and its components"""
    chunk_count = vectorstore._collection.count()
    return {
        "status": "healthy",
        "chromadb": "connected",
        "chunks_stored": chunk_count,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": DEFAULT_MODEL,
        "api_key_set": bool(OPENROUTER_API_KEY)
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main endpoint - takes a question and returns an LLM answer grounded in the SSP documents.

    Example request body:
    {
        "question": "What happens when the supercharger magnetic clutch fails?"
    }
    """

    # Make sure we have an API key before trying to call the LLM
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY is not set on the server. Ask the instructor to set it."
        )

    # Make sure the question isn't empty
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    print(f"📥 Query received: {request.question}")

    # Step 1: Retrieve relevant chunks from ChromaDB
    context_chunks = retrieve_context(request.question, k=request.num_chunks)
    print(f"   📦 Retrieved {len(context_chunks)} chunks from ChromaDB")

    # Step 2: Build the prompt combining context + question
    prompt = build_prompt(request.question, context_chunks)

    # Step 3: Send to the LLM and get the answer
    print(f"   🤖 Calling LLM: {request.model}")
    answer = call_llm(prompt, request.model)
    print(f"   ✅ Answer received ({len(answer)} chars)")

    # Step 4: Return everything to the caller
    return QueryResponse(
        question=request.question,
        answer=answer,
        model_used=request.model,
        context=context_chunks
    )