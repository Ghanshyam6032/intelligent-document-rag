from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

loader=PyPDFLoader(r'D:\AI\chatbot\Machine_Learning.pdf')
docs=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunk=splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_db = FAISS.from_documents(
    chunk,
    embeddings
)

model=ChatGroq(model='llama-3.3-70b-versatile')


print("------- Type 'exit' to end chat -------")
while True:
    user=input('\nuser:')

    if user=='exit':
       break
    docs=vector_db.similarity_search(user,k=3)
    context='\n'.join(doc.page_content for doc in docs)
    prompt=f"""
Answer the question using only the context below.

Context:
{context}

Question:
{user}
"""
    
    response=model.invoke(prompt)
    print('\nAI:',response.content)
    