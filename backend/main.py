from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi import HTTPException

from fastapi import UploadFile, File, HTTPException
import shutil
import uuid

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from prometheus_fastapi_instrumentator import Instrumentator
import traceback
import faiss
import os
import logging

import mlflow
import mlflow.sklearn
import numpy as np





drift_embeddings = []  # to track embeddings for drift
DRIFT_THRESHOLD = 0.6  # cosine distance threshold (adjustable)



mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("rag-qa")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # logs to console
        logging.FileHandler("app.log", mode='a')  # logs to file
    ]
)

logger = logging.getLogger(__name__)



app = FastAPI()
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and data
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
index = faiss.read_index("vector.index")


def load_docs():
    with open("doc_names.txt", "r") as f:
        names = [line.strip() for line in f]

    contents = []
    for name in names:
        with open(os.path.join("docs", name), "r", encoding="utf-8") as f:
            contents.append(f.read())
    return contents

def load_index():
    return faiss.read_index("vector.index")


with open("doc_names.txt", "r") as f:
    doc_names = [line.strip() for line in f]

docs = []
for name in doc_names:
    with open(os.path.join("docs", name), "r", encoding="utf-8") as f:
        docs.append(f.read())


class QueryRequest(BaseModel):
    question: str


@app.post("/answer")
def get_answer(request: QueryRequest):
    try:
        query = request.question.strip()
        logger.info(f"Received question: {query}")
        if not query:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        # Dynamically load latest index and docs
        index = load_index()
        docs = load_docs()

        if index.ntotal == 0:
            raise HTTPException(status_code=500, detail="FAISS index is empty or not loaded properly.")

        # Generate query embedding
        try:
            query_embedding = embed_model.encode([query])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding model error: {str(e)}")

        # Perform FAISS search
        D, I = index.search(query_embedding, k=1)
        if I[0][0] == -1:
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        # Load the top document safely
        try:
            top_doc = docs[I[0][0]]
        except IndexError:
            raise HTTPException(status_code=500, detail="Document index out of range.")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

        # Construct the prompt
        prompt = (
            f"Answer the question based only on the provided context.\n\n"
            f"Context:\n{top_doc}\n\n"
            f"Question: {query}\n\n"
            f"Give a complete and helpful sentence as your answer:"
        )

        # Run MLflow logging + QA pipeline
        with mlflow.start_run():
            mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
            mlflow.log_param("qa_model", "google/flan-t5-base")
            mlflow.log_param("top_k", 1)
            mlflow.log_param("query", query)
            mlflow.log_param("question_length", len(query))
            mlflow.log_param("context_length", len(top_doc))
            

            result = qa_pipeline(prompt, max_length=200, do_sample=False)
            answer = result[0]["generated_text"]

            mlflow.log_metric("context_length", len(top_doc))
            mlflow.log_text(answer, "answer.txt")

        return {"answer": answer}

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unhandled error: {str(e)}")  


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Ensure docs folder exists
        os.makedirs("docs", exist_ok=True)

        # Validate file extension
        extension = file.filename.split(".")[-1]
        if extension != "txt":
            raise HTTPException(status_code=400, detail="Only .txt files supported")

        # Save file with unique name
        unique_name = f"{uuid.uuid4().hex}.txt"
        file_path = os.path.join("docs", unique_name)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Embed the uploaded content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        embedding = embed_model.encode([content])
        index.add(embedding)

        # Update in-memory and disk
        docs.append(content)
        with open("doc_names.txt", "a") as f:
            f.write(unique_name + "\n")

        faiss.write_index(index, "vector.index")

        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("uploaded_file", unique_name)
            mlflow.log_artifact(file_path)

        return {"status": "success", "file": unique_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok"}
