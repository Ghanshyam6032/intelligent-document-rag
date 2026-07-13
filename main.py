import os
# ==========================================
# WINDOWS FAISS CRASH FIX (MUST BE AT THE TOP)
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import tempfile
import logging
import traceback
import gc  # OPTIMIZATION: Garbage collection for memory management
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. Configuration & Settings
# ==========================================
class Settings(BaseSettings):
    GROQ_API_KEY: str
    FAISS_INDEX_PATH: str = "./faiss_store"
    
    # Render OS environment variables directly read karega
    class Config:
        extra = "ignore"

try:
    settings = Settings()
except Exception as e:
    print(f"CRITICAL ERROR LOADING SETTINGS: {e}")
    sys.exit(1)

# ==========================================
# 2. Logger Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("rag_chatbot")

# ==========================================
# 3. Pydantic Schemas (Request/Response Models)
# ==========================================
class HealthResponse(BaseModel):
    status: str
    message: str
    vector_db_loaded: bool

class UploadResponse(BaseModel):
    status: str
    message: str
    filename: str

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str

# ==========================================
# 4. RAG Engine (LangChain & Vector DB Logic)
# ==========================================
class RAGEngine:
    def __init__(self):
        logger.info("Initializing FAST Embedding Model... (Lazy Loading Triggered)")
        self.embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        logger.info("Initializing Groq LLM...")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.GROQ_API_KEY,
            temperature=0.1
        )
        
        self.vector_db = None
        self.index_path = settings.FAISS_INDEX_PATH
        self.load_index()

    def load_index(self):
        """Loads the persistent FAISS index from disk if it exists."""
        if os.path.exists(self.index_path) and os.listdir(self.index_path):
            try:
                self.vector_db = FAISS.load_local(
                    self.index_path, 
                    self.embedding,
                    allow_dangerous_deserialization=True 
                )
                logger.info("FAISS index loaded successfully from disk.")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.vector_db = None

    def process_pdf(self, file_path: str):
        """
        Splits the PDF and creates a FAISS index efficiently.
        Optimized for Render: Uses lazy loading, batch embedding, and garbage collection.
        """
        logger.info("Loading PDF lazily to conserve RAM...")
        loader = PyPDFLoader(file_path)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        
        # OPTIMIZATION: Reset DB for the new upload to overwrite old data
        self.vector_db = None 
        
        # OPTIMIZATION: Process in batches of chunks to prevent OOM (Out of Memory) crashes
        batch_size = 100 
        current_batch = []
        batch_counter = 1

        try:
            # OPTIMIZATION: loader.lazy_load() yields pages one by one instead of loading 400 pages into RAM
            for page in loader.lazy_load():
                page_chunks = splitter.split_documents([page])
                current_batch.extend(page_chunks)

                # When batch reaches the limit, embed and index it
                if len(current_batch) >= batch_size:
                    self._process_and_index_batch(current_batch, batch_counter)
                    current_batch = []  # Clear the list
                    batch_counter += 1

            # Process any remaining chunks that didn't fill the last batch
            if current_batch:
                self._process_and_index_batch(current_batch, batch_counter)

            if self.vector_db is None:
                raise ValueError("The PDF appears to be empty or unreadable.")

            logger.info("Saving persistent FAISS index to disk...")
            self.vector_db.save_local(self.index_path)
            logger.info("PDF processing completed successfully.")

        except Exception as e:
            logger.error(f"Error during PDF processing: {str(e)}")
            raise e
        finally:
            # OPTIMIZATION: Final memory cleanup after full document processing
            gc.collect()

    def _process_and_index_batch(self, batch: list, batch_number: int):
        """Helper method to embed a batch and incrementally add to FAISS, then clear memory."""
        logger.info(f"Embedding and indexing batch {batch_number} ({len(batch)} chunks)...")
        
        if self.vector_db is None:
            # OPTIMIZATION: Initialize the FAISS index ONLY on the first batch
            self.vector_db = FAISS.from_documents(batch, self.embedding)
        else:
            # OPTIMIZATION: Incrementally add to existing index for subsequent batches
            self.vector_db.add_documents(batch)
            
        # OPTIMIZATION: Aggressive garbage collection to free RAM on Render immediately
        del batch
        gc.collect()

    def generate_answer(self, question: str) -> str:
        """Retrieves context and generates an answer."""
        if self.vector_db is None:
            raise ValueError("Vector database is empty. Please upload a document first.")

        docs = self.vector_db.similarity_search(question, k=5)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
        You are a highly accurate AI assistant analyzing a medical or technical document.
        Read the context below and answer the user's question based ONLY on this exact context.
        Do NOT use outside knowledge. Do NOT guess.
        If the context does not contain the exact answer, reply strictly with: "I couldn't find that information in the uploaded document."
        
        Context:
        {context}
        
        Question:
        {question}
        """

        response = self.llm.invoke(prompt)
        return response.content

# ==========================================
# LAZY LOADING LOGIC (RENDER FIX)
# ==========================================
_rag_engine_instance = None

def get_rag_engine() -> RAGEngine:
    """Returns the RAG engine instance, initializing it on the first call."""
    global _rag_engine_instance
    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine()
    return _rag_engine_instance

# ==========================================
# 5. FastAPI App & Endpoints
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # App starts instantly. No AI models are loaded here to prevent Render timeout.
    logger.info("FastAPI Application has started successfully. Port is ready!")
    yield

app = FastAPI(
    title="PDF Assistant API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
def health_check():
    """Health check endpoint. Responds instantly to satisfy Render's port check."""
    global _rag_engine_instance
    is_db_loaded = (_rag_engine_instance is not None) and (_rag_engine_instance.vector_db is not None)
    return HealthResponse(
        status="ok",
        message="API is running smoothly.",
        vector_db_loaded=is_db_loaded
    )

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDFs are accepted.")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Engine is initialized ONLY when a file is uploaded
        engine = get_rag_engine()
        engine.process_pdf(tmp_path)
        os.remove(tmp_path)

        return UploadResponse(
            status="success",
            message="Document successfully processed.",
            filename=file.filename
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        # Engine is loaded here if user chats without uploading (assuming DB exists on disk)
        engine = get_rag_engine()
        answer = engine.generate_answer(request.question)
        return ChatResponse(question=request.question, answer=answer)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to generate response.")
