import os
import re
# ==========================================
# STRICT MEMORY & CPU THREAD LOCKS (512MB-1GB RAM FIX)
# REQUIRED FOR STABLE RAILWAY/RENDER DEPLOYMENT
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import tempfile
import logging
import traceback
import gc  
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# ==========================================
# 1. Configuration & Settings
# ==========================================
class Settings(BaseSettings):
    GROQ_API_KEY: str
    FAISS_INDEX_PATH: str = "./faiss_store"
    DOCSTORE_PATH: str = "./doc_store" # Persists parent documents to disk, saving RAM
    
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
ParentDocumentRetriever = None
LocalFileStore = None
Document = None

def load_heavy_libraries():
    """Deferred importing to pass Railway port timeout checks and save initial RAM."""
    global ChatGroq, HuggingFaceEmbeddings, FAISS, PyPDFLoader, RecursiveCharacterTextSplitter
    global ParentDocumentRetriever, LocalFileStore, Document
    if ChatGroq is None:
        logger.info("Importing heavy AI libraries on strict memory diet...")
        
        # PyTorch limits set before import to prevent memory spikes
        import torch
        torch.set_num_threads(1)
        
        from langchain_groq import ChatGroq as CG
        from langchain_huggingface import HuggingFaceEmbeddings as HFE
        from langchain_community.vectorstores import FAISS as F
        from langchain_community.document_loaders import PyPDFLoader as PPL
        from langchain_text_splitters import RecursiveCharacterTextSplitter as RCTS
        from langchain.retrievers import ParentDocumentRetriever as PDR
        from langchain.storage import LocalFileStore as LFS
        from langchain.schema import Document as Doc
        
        ChatGroq = CG
        HuggingFaceEmbeddings = HFE
        FAISS = F
        PyPDFLoader = PPL
        RecursiveCharacterTextSplitter = RCTS
        ParentDocumentRetriever = PDR
        LocalFileStore = LFS
        Document = Doc
        logger.info("Heavy libraries imported successfully!")

# ==========================================
# 5. RAG Engine
# ==========================================
class RAGEngine:
    def __init__(self):
        load_heavy_libraries()
        
        logger.info("Initializing BGE-Small Embedding Model...")
        # UPGRADE: BAAI/bge-small-en-v1.5. Requires normalize_embeddings=True for optimal accuracy.
        # It perfectly balances extreme lightweight RAM usage with top-tier semantic retrieval.
        self.embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True} 
        )
        
        logger.info("Initializing Groq LLM...")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.GROQ_API_KEY,
            temperature=0.0 # Absolute zero for ChatGPT-like confident, strictly factual responses
        )
        
        self.vector_db = None
        self.retriever = None
        self.index_path = settings.FAISS_INDEX_PATH
        self.docstore_path = settings.DOCSTORE_PATH
        
        # Smart Text Splitters for ParentDocumentRetriever
        # Parents: Large context (1500 chars) ensuring complete tables, algorithms, and sections.
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200,
            separators=["\n\n", "\n", "(?<=\\. )", " "], is_separator_regex=True
        )
        # Children: Tiny context (400 chars) for laser-focused semantic matching.
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=50,
            separators=["\n\n", "\n", "(?<=\\. )", " "], is_separator_regex=True
        )
        
        self.load_index()

    def _init_retriever(self, force_new=False):
        """Initializes the ParentDocumentRetriever and underlying FAISS/FileStore."""
        os.makedirs(self.docstore_path, exist_ok=True)
        store = LocalFileStore(self.docstore_path)
        
        if self.vector_db is None or force_new:
            # Initialize empty FAISS dynamically by feeding a dummy doc and deleting it.
            # This avoids hardcoding vector dimensions and prevents crash errors.
            dummy = Document(page_content="init")
            self.vector_db = FAISS.from_documents([dummy], self.embedding)
            self.vector_db.delete([list(self.vector_db.docstore._dict.keys())[0]])

        # Configure PDR with MMR retrieval logic
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_db,
            docstore=store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 30, "lambda_mult": 0.85} # High precision MMR
        )

    def load_index(self):
        """Loads persistent FAISS and Document Store from disk after server restart."""
        if os.path.exists(self.index_path) and os.listdir(self.index_path):
            try:
                self.vector_db = FAISS.load_local(
                    self.index_path, 
                    self.embedding,
                    allow_dangerous_deserialization=True 
                )
                self._init_retriever(force_new=False)
            except Exception as e:
                logger.error(f"Failed to load FAISS: {e}")
                self.vector_db = None
                self.retriever = None

    def process_pdf(self, file_path: str):
        """Ultra-low memory PDF processor driving the ParentDocumentRetriever."""
        logger.info("Starting production-grade PDF processing...")
        loader = PyPDFLoader(file_path)
        
        # Completely reset index for new uploads
        self.vector_db = None 
        self._init_retriever(force_new=True)
        
        batch_size = 10 # Micro-batching pages to keep RAM utilization flat
        current_batch = []

        try:
            for page in loader.lazy_load():
                # Add source metadata for tracking
                page.metadata['source_file'] = os.path.basename(file_path)
                current_batch.append(page)

                if len(current_batch) >= batch_size:
                    logger.info(f"Adding batch of {len(current_batch)} pages to PDR...")
                    # PDR automatically splits parents, splits children, embeds children, and saves parents to disk.
                    self.retriever.add_documents(current_batch) 
                    current_batch = []
                    gc.collect()

            if current_batch:
                logger.info(f"Adding final batch of {len(current_batch)} pages to PDR...")
                self.retriever.add_documents(current_batch)

            # Persist FAISS (Child chunks). LocalFileStore already persisted Parents to disk automatically.
            self.vector_db.save_local(self.index_path)
            
        except Exception as e:
            logger.error(f"Error during PDF processing: {str(e)}")
            raise e
        finally:
            gc.collect()

    def generate_answer(self, question: str) -> str:
        if self.retriever is None:
            raise ValueError("Vector database is empty. Please upload a document first.")

        # RETRIEVAL: Retrieves highly relevant 400-char chunks, but returns the full 1500-char parents.
        # This completely eliminates cut-off sentences and fragmented context.
        docs = self.retriever.invoke(question)
        
        # Merge all retrieved parent documents naturally.
        context = "\n\n".join(doc.page_content.strip() for doc in docs)

        # GENERATION UPGRADE: Ultimate ChatGPT-Style Zero-Hallucination Prompt
        prompt = f"""
        You are a highly capable, professional AI expert analyzing a technical document. 
        Answer the user's question directly, comprehensively, and confidently, acting as the ultimate authority on this document.

        CRITICAL GENERATION RULES:
        1. NO AI-SPEAK: NEVER use meta-phrases like "Based on the provided context", "According to the document", "The text states", or "I will attempt to answer". Start your answer immediately.
        2. NO HALLUCINATION: You are strictly limited to the provided Source Material. Never invent facts, advantages, algorithms, or infer missing details.
        3. STRICT ABSENCE FALLBACK: If and ONLY if the answer cannot be logically deduced from the Source Material, reply EXACTLY with: "I couldn't find that information in the uploaded document." Do not append this sentence if you have already generated an answer.
        4. TOPIC ISOLATION: Never mix information from different topics. Compare concepts only if the document explicitly compares them.
        5. NATURAL SYNTHESIS: Read the Source Material and synthesize a single, cohesive, human-like response. Do not mention that you are reading excerpts.

        FORMATTING DIRECTIVES (Match Intent):
        - Definitions -> Clear, concise, standard paragraph.
        - Explanations -> Structured, well-formatted paragraphs.
        - Lists / Advantages -> Clean bullet points.
        - Algorithms / Steps -> Numbered step-by-step sequences.
        - Differences / Comparisons -> Markdown comparison table.
        - Examples -> Always include examples exactly as they appear in the document.

        Source Material:
        {context}
        
        User Question:
        {question}
        """

        response = self.llm.invoke(prompt)
        return response.content

# ==========================================
# LAZY LOADING LOGIC
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
    logger.info("FastAPI Application has started instantly! Port is bound.")
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
    global _rag_engine_instance
    is_db_loaded = (_rag_engine_instance is not None) and (_rag_engine_instance.retriever is not None)
    return HealthResponse(
        status="ok",
        message="API is running.",
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

        engine = get_rag_engine()
        engine.process_pdf(tmp_path)
        os.remove(tmp_path)

        return UploadResponse(
            status="success",
            message="Document processed.",
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
        engine = get_rag_engine()
        answer = engine.generate_answer(request.question)
        return ChatResponse(question=request.question, answer=answer)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to generate response.")
