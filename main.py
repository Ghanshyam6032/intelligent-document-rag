import os
# ==========================================
# STRICT MEMORY & CPU THREAD LOCKS (512MB-1GB RAM FIX)
# DO NOT REMOVE: REQUIRED FOR RAILWAY DEPLOYMENT
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
import gc  # Garbage collection
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
    
    class Config:
        extra = "ignore"

try:
    settings = Settings()
except Exception as e:
    print(f"CRITICAL ERROR LOADING SETTINGS: {e}")
    sys.exit(1)

# ==========================================
# 2. Logger Setup (Improved for Railway)
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

def load_heavy_libraries():
    """Deferred importing to pass Railway port timeout checks and save initial RAM."""
    global ChatGroq, HuggingFaceEmbeddings, FAISS, PyPDFLoader, RecursiveCharacterTextSplitter
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
        
        ChatGroq = CG
        HuggingFaceEmbeddings = HFE
        FAISS = F
        PyPDFLoader = PPL
        RecursiveCharacterTextSplitter = RCTS
        logger.info("Heavy libraries imported successfully!")

# ==========================================
# 5. RAG Engine
# ==========================================
class RAGEngine:
    def __init__(self):
        load_heavy_libraries()
        
        logger.info("Initializing FAST Embedding Model...")
        self.embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True} 
        )
        
        logger.info("Initializing Groq LLM...")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.GROQ_API_KEY,
            temperature=0.0 # Absolute zero for strict factual accuracy
        )
        
        self.vector_db = None
        self.index_path = settings.FAISS_INDEX_PATH
        self.load_index()

    def load_index(self):
        """Loads FAISS index from disk safely."""
        if os.path.exists(self.index_path) and os.listdir(self.index_path):
            try:
                self.vector_db = FAISS.load_local(
                    self.index_path, 
                    self.embedding,
                    allow_dangerous_deserialization=True 
                )
            except Exception as e:
                logger.error(f"Failed to load FAISS: {e}")
                self.vector_db = None

    def process_pdf(self, file_path: str):
        """Ultra-low memory PDF processor optimized for safe contextual retrieval."""
        logger.info("Starting ultra-low memory PDF parsing...")
        loader = PyPDFLoader(file_path)
        
        # OPTIMIZATION: Adjusted Chunking parameters for technical PDFs 
        # (Standard splitter, no semantic chunkers as requested)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        
        self.vector_db = None 
        
        batch_size = 20 # Micro-batching to prevent RAM overflow on Railway
        current_batch = []
        batch_counter = 1
        global_chunk_id = 0 

        try:
            for page in loader.lazy_load():
                page_chunks = splitter.split_documents([page])
                
                # Tag chunks with sequential IDs to maintain context order during retrieval
                for chunk in page_chunks:
                    chunk.metadata['chunk_id'] = global_chunk_id
                    global_chunk_id += 1
                    current_batch.append(chunk)

                while len(current_batch) >= batch_size:
                    process_batch = current_batch[:batch_size]
                    self._process_and_index_batch(process_batch, batch_counter)
                    current_batch = current_batch[batch_size:]
                    batch_counter += 1

            if current_batch:
                self._process_and_index_batch(current_batch, batch_counter)

            if self.vector_db is None:
                raise ValueError("The PDF appears to be empty.")

            self.vector_db.save_local(self.index_path)
            
        except Exception as e:
            logger.exception("Error during PDF processing")
            raise e
        finally:
            gc.collect()

    def _process_and_index_batch(self, batch: list, batch_number: int):
        """Embeds micro-batches and forces memory clear."""
        logger.info(f"Indexing batch {batch_number} ({len(batch)} chunks)...")
        if self.vector_db is None:
            self.vector_db = FAISS.from_documents(batch, self.embedding)
        else:
            self.vector_db.add_documents(batch)
        del batch
        gc.collect()

    def generate_answer(self, question: str) -> str:
        if self.vector_db is None:
            raise ValueError("Vector database is empty. Please upload a document first.")

        # OPTIMIZATION: Updated MMR Search Parameters
        # k=8, fetch_k=20, lambda_mult=0.85
        docs = self.vector_db.max_marginal_relevance_search(
            question, 
            k=8, 
            fetch_k=20, 
            lambda_mult=0.85 
        )
        
        # Sort retrieved chunks by their original document order (chunk_id) to merge them naturally
        docs.sort(key=lambda x: x.metadata.get('chunk_id', 0))
        
        # Combine smoothly
        context = "\n\n".join(doc.page_content.strip() for doc in docs)

        # OPTIMIZATION: Enhanced Zero-Hallucination Prompt
        prompt = f"""
        You are a highly capable AI expert. Answer the user's question directly, accurately, and naturally using ONLY the provided Source Material.

        CRITICAL RULES:
        1. DIRECT ANSWER: NEVER use introductory filler like "Based on the provided context", "According to the document", "The text states", or "I will attempt to answer". Start your answer immediately with the facts.
        2. NO HALLUCINATION: You are strictly limited to the Source Material. Do not invent facts, advantages, or details not explicitly stated. Do not use outside knowledge.
        3. STRICT ABSENCE FALLBACK: If and ONLY if the answer cannot be logically deduced from the Source Material, reply EXACTLY with: "I couldn't find that information in the uploaded document." Do not add apologies or extra text. Never append this sentence if you have already generated an answer.
        4. NATURAL FORMATTING: Use Markdown.
           - Definitions -> paragraph
           - Algorithms -> numbered steps
           - Advantages -> bullet points
           - Comparisons -> markdown table
           - Examples -> include document examples if available

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
    is_db_loaded = (_rag_engine_instance is not None) and (_rag_engine_instance.vector_db is not None)
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
        logger.exception("Exception occurred during PDF upload and processing.")
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
        logger.exception("Exception occurred during chat generation.")
        raise HTTPException(status_code=500, detail="Failed to generate response.")
