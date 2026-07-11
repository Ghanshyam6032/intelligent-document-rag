📄 Document Intelligence RAG Assistant
A premium AI-powered chatbot that allows users to upload PDF documents and ask context-aware questions. It uses Retrieval-Augmented Generation (RAG) with HuggingFace Embeddings and FAISS Vector Store to retrieve relevant document chunks, and the lightning-fast Groq API (Llama 3) to generate accurate, context-specific answers.

✨ Features
Upload and process PDF documents instantly

Chat with your documents to extract insights and summaries

Premium Glassmorphism UI with Dark Mode support

Markdown rendering with code block formatting and copy buttons

Real-time typing animations and toast notifications

Persistent local vector database storage

⚙️ Working Pipeline
PDF Upload → LangChain Text Splitting → HuggingFace Embeddings → FAISS Vector DB → User Query → Similarity Search → Groq LLM → Frontend Response

🛠️ Tech Stack
Python

FastAPI

LangChain & FAISS

Groq API (Llama 3)

HuggingFace (all-MiniLM-L6-v2)

HTML5

CSS3

Vanilla JavaScript

🌐 Live Demo
//pdf-rag-chatbot-frontend.onrender.com

📊 Languages Used
Python — 70%

HTML — 15%

JavaScript — 10%

CSS — 5%

👨‍💻 Developer
Ghanshyam Prajapati
