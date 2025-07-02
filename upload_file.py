import hashlib
import io
import os
from typing import List
import PyPDF2
import chromadb
from dotenv import load_dotenv
from fastapi import File, UploadFile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import chromadb.utils.embedding_functions as embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.getenv("API_KEY"))
collection = chroma_client.get_or_create_collection(name="poc-copiloto-v2", embedding_function=google_ef, metadata={"hnsw:space": "cosine"})

def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to the server.
    """
    try:
        contents = file.file.read()
        with open(file.filename, "wb") as f:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
            
            result = create_embeddings(pdf_reader.pages)
            
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
    
def create_embeddings(pages: List[PyPDF2.PageObject]):
    """
    Create embeddings from the text.
    """
    all_pages_text = ""
    for page in pages:
       all_pages_text += page.extract_text() + "\n\n"
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    chunks = text_splitter.split_text(all_pages_text)
    
    if not chunks:
        return {"error": "No text found in the PDF file."}
    
    ids = [f"{hashlib.sha256(chunk.encode()).hexdigest()}" for chunk in chunks]
    metadatas = [{"chunk_content": chunk[:200]} for chunk in chunks]
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    
def query_text(text: str):
    """
    Get embeddings for the given text.
    """
    embeddings = collection.query(
        query_texts=[text],
        n_results=5
    )
    
    LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))
    
    model = PromptTemplate(
        template="""
        Você é um assistente de IA que responde perguntas com base em informações extraídas de documentos PDF.
        Você recebeu as seguintes informações:
        {context}
        Agora, responda à pergunta: {question}
        Intrução: Responda de forma clara e concisa, utilizando as informações fornecidas. E não invente informações.
        """,
        input_variables=["context", "question"]
    )
    
    chain = model | LLM | StrOutputParser()
    docs = embeddings["documents"]
    # docs é uma lista de listas, pegue todos os textos e una em uma string
    context_str = ""
    for doc in docs:
        context_str += "\n\n".join(doc)

    result = chain.invoke({
        "context": context_str,
        "question": text
    })
    return result
    