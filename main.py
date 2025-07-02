from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import UploadFile, File
from upload_file import upload_file, query_text

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

class Question(BaseModel):
    text: str
    
@app.post("/query")
def query_question(question: Question):
    """
    Query the text and return the result.
    """
    result = query_text(question.text)
    return {
        "message": "Query received successfully",
        "result": result
    }
    
    
class FileSent(BaseModel):
    file_name: str
    
@app.post("/sent-file")
async def receive_file(file: UploadFile = File(...)):
    result = upload_file(file)
    return {
        "message": "File received successfully",
        "result": result
    }
    
    