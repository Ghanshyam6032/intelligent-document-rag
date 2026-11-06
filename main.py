import os
import sys
import tempfile
import logging
import traceback
import gc
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# ==========================================
# 1. Configuration & Settings
# ==========================================
class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    FAISS_INDEX_PATH: str = "./faiss_store"
    
    # Local .env file aur Modal environment variables dono support karega
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# ==========================================
# 2. Logger Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("rag_chatbot")

# ==========================================
# 3. Pydantic Schemas
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
# 4. Global Variables for Deferred Imports
# ==========================================
ChatGroq = None
HuggingFaceEmbeddings = None
FAISS = None
PyPDFLoader = None
RecursiveCharacterTextSplitter = None

def load_heavy_libraries():
    """Lazy import heavy AI libraries."""
    global ChatGroq, HuggingFaceEmbeddings, FAISS, PyPDFLoader, RecursiveCharacterTextSplitter
    if ChatGroq is None:
        logger.info("Importing AI libraries...")
        from langchain_groq import ChatGroq as CG
        from langchain_huggingface import HuggingFaceEmbeddings as HFE
        from langchain_community.vectorstores import FAISS as F
        from langchain_community.document_loaders import PyPDFLoader as PPL
        from langchain_text_splitters import RecursiveCharacterTextSplitter as RCTS
        
        ChatGroq = CG
        HuggingFaceEmbeddings = HFE
        FAISS = F
        PyPDFLoader = PPL
        RecursiveCharacterTextSplitter = RCTS
        logger.info("AI libraries imported successfully!")

# ==========================================
# 5. RAG Engine
# ==========================================
class RAGEngine:
    def __init__(self):
        load_heavy_libraries()
        
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing! Set it in .env or Modal Secrets.")

        logger.info("Initializing Embedding Model (all-MiniLM-L6-v2)...")
        self.embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        logger.info("Initializing Groq LLM (llama-3.3-70b-versatile)...")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.GROQ_API_KEY,
            temperature=0.0
        )
        
        self.vector_db = None
        self.index_path = settings.FAISS_INDEX_PATH
        self.load_index()

    def load_index(self):
        """Loads FAISS index from disk if available."""
        if os.path.exists(self.index_path) and os.listdir(self.index_path):
            try:
                self.vector_db = FAISS.load_local(
                    self.index_path, 
                    self.embedding,
                    allow_dangerous_deserialization=True 
                )
                logger.info("Existing FAISS index loaded from disk.")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.vector_db = None

    def process_pdf(self, file_path: str):
        """Processes PDF in micro-batches and saves FAISS index."""
        logger.info("Starting PDF parsing and chunking...")
        loader = PyPDFLoader(file_path)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=250,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.vector_db = None 
        batch_size = 20 
        current_batch = []
        batch_counter = 1

        try:
            for page in loader.lazy_load():
                page_chunks = splitter.split_documents([page])
                current_batch.extend(page_chunks)

                while len(current_batch) >= batch_size:
                    process_batch = current_batch[:batch_size]
                    self._process_and_index_batch(process_batch, batch_counter)
                    current_batch = current_batch[batch_size:]
                    batch_counter += 1

            if current_batch:
                self._process_and_index_batch(current_batch, batch_counter)

            if self.vector_db is None:
                raise ValueError("The uploaded PDF appears to be empty or unreadable.")

            os.makedirs(self.index_path, exist_ok=True)
            self.vector_db.save_local(self.index_path)
            logger.info(f"FAISS index saved at {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error during PDF processing: {str(e)}")
            raise e
        finally:
            gc.collect()

    def _process_and_index_batch(self, batch: list, batch_number: int):
        logger.info(f"Indexing batch {batch_number} ({len(batch)} chunks)...")
        if self.vector_db is None:
            self.vector_db = FAISS.from_documents(batch, self.embedding)
        else:
            self.vector_db.add_documents(batch)
        del batch
        gc.collect()

    def generate_answer(self, question: str) -> str:
        if self.vector_db is None:
            self.load_index()
            if self.vector_db is None:
                raise ValueError("Vector database is empty. Please upload a PDF document first.")

        try:
            docs = self.vector_db.max_marginal_relevance_search(
                question, 
                k=6, 
                fetch_k=30, 
                lambda_mult=0.85
            )
        except Exception as e:
            logger.warning(f"MMR search fallback to similarity search: {e}")
            docs = self.vector_db.similarity_search(question, k=6)
            
        context = "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)

        prompt = f"""You are a highly accurate and intelligent AI assistant analyzing a technical document.
Carefully read the retrieved document context provided below and answer the user's question.

Guidelines:
1. Base your answer ONLY on the provided context. Do NOT use outside knowledge.
2. You must synthesize, summarize, and combine information from multiple parts of the context if needed to form a complete answer. Do NOT require an exact sentence match.
3. If the provided context does not contain any relevant information to answer the question, you MUST reply exactly and only with: "I couldn't find that information in the uploaded document."
4. Never append the failure phrase if you are providing an answer.
5. Do not guess or hallucinate details.

<context>
{context}
</context>

Question: {question}

Answer:"""

        response = self.llm.invoke(prompt)
        return response.content.strip()

# ==========================================
# LAZY SINGLETON INSTANCE
# ==========================================
_rag_engine_instance = None

def get_rag_engine() -> RAGEngine:
    global _rag_engine_instance
    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine()
    return _rag_engine_instance

# ==========================================
# 6. FastAPI App & Endpoints
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI Application started!")
    yield

web_app = FastAPI(
    title="DocuMind AI - RAG Backend",
    version="1.0.0",
    lifespan=lifespan
)

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@web_app.get("/", response_model=HealthResponse)
def health_check():
    global _rag_engine_instance
    is_db_loaded = (
        (_rag_engine_instance is not None and _rag_engine_instance.vector_db is not None)
        or (os.path.exists(settings.FAISS_INDEX_PATH) and bool(os.listdir(settings.FAISS_INDEX_PATH)))
    )
    return HealthResponse(
        status="ok",
        message="DocuMind AI API is running on Modal.",
        vector_db_loaded=is_db_loaded
    )

@web_app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        engine = get_rag_engine()
        engine.process_pdf(tmp_path)

        # Commit volume changes if running on Modal
        if "MODAL_ENVIRONMENT" in os.environ:
            faiss_volume.commit()

        return UploadResponse(
            status="success",
            message="Document processed and indexed successfully.",
            filename=file.filename
        )
    except Exception as e:
        logger.error(f"Upload error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as cleanup_error:
                logger.error(f"Failed to delete temp file: {cleanup_error}")

@web_app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        engine = get_rag_engine()
        answer = engine.generate_answer(request.question)
        return ChatResponse(question=request.question, answer=answer)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to generate response.")


# ==========================================
# 7. MODAL.COM DEPLOYMENT CONFIGURATION
# ==========================================
try:
    import modal

    def download_embedding_model():
        """Build time par hi HuggingFace model cache kar lega taaki cold-start super fast ho."""
        from langchain_huggingface import HuggingFaceEmbeddings
        HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    modal_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "fastapi[standard]",
            "uvicorn",
            "python-multipart",
            "pydantic",
            "pydantic-settings",
            "python-dotenv",
            "langchain-core",
            "langchain-community",
            "langchain-text-splitters",
            "langchain-groq",
            "langchain-huggingface",
            "sentence-transformers",
            "faiss-cpu",
            "torch",
            "pypdf",
        )
        .run_function(download_embedding_model)
    )

    app = modal.App("documind-rag-backend", image=modal_image)
    faiss_volume = modal.Volume.from_name("documind-faiss-store", create_if_missing=True)

    @app.function(
        secrets=[modal.Secret.from_name("groq-secret")],
        volumes={"/data/faiss_store": faiss_volume},
        memory=2048,
        timeout=300,
        scaledown_window=300,
    )
    @modal.asgi_app()
    def fastapi_app():
        settings.FAISS_INDEX_PATH = "/data/faiss_store"
        settings.GROQ_API_KEY = os.environ.get("GROQ_API_KEY", settings.GROQ_API_KEY)
        return web_app

except ImportError:
    logger.info("Modal not installed; running in local FastAPI mode.")