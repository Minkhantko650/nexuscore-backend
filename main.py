from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from dotenv import load_dotenv
from database import engine, Base
import models
from fastapi import UploadFile, File
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from io import BytesIO
from pydantic import BaseModel
from enum import Enum

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Category(str, Enum):
    gameplay = "gameplay"
    technical = "technical"
    account = "account"
    billing = "billing"
    community = "community"
    updates = "updates"
    general = "general"
load_dotenv()

Base.metadata.create_all(bind=engine)

embedding_model = OpenAIEmbeddings(model = "text-embedding-3-large")
vector_db = Chroma(
    client= chromadb.HttpClient(host="localhost",port=8001),
    collection_name = "vector_database", 
    embedding_function= embedding_model
)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from chat import router
from auth import router as auth_router
from forum import router as forum_router
from community_search import router as community_router
app.include_router(router)
app.include_router(auth_router)
app.include_router(forum_router)
app.include_router(community_router)

class Document(BaseModel):
    text:str 
    metadata: dict={} 

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), category: Category = Category.general):
    pdf_bytes= await file.read()
    pdf_file= BytesIO(pdf_bytes)
    
    pdf_reader= PyPDF2.PdfReader(pdf_file)
    full_text=""
    for page in pdf_reader.pages:
        full_text+=page.extract_text()
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size = 1000, 
        chunk_overlap = 200, 
        length_function = len 
    )
    chunks = text_splitter.split_text(full_text)
    vector_db.add_texts(
        texts= chunks,
        metadatas=[{
            "source": file.filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "category": category
        } for i in range(len(chunks))]
    )
    return { 
            "status":"successs",
            "filename":file.filename,
            "total_chunks":len(chunks),
            "total_chars":len(full_text)
        }
    
@app.post("/reset-db")
async def reset_db():
    vector_db.delete_collection()
    return {"status": "cleared"}

@app.post("/document")
async def document(doc:Document):
    vector_db.add_texts(
        texts=[doc.text],
        metadatas=[doc.metadata]
    )
    

    

@app.get("/documents")
async def get_all_documents():
    results = vector_db.get()
    return results

@app.get("/search")
async def search_documents(query:str, limit: int = 5):
    docs = vector_db.similarity_search(query, k=limit)
    return {
        "query": query,
        "results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    }
    

    
