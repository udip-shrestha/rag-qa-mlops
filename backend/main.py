from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from prometheus_fastapi_instrumentator import Instrumentator
import traceback
import faiss
import os
import logging
import subprocess
import mlflow
import mlflow.sklearn
import numpy as np
import shutil
import uuid




drift_embeddings = []  # to track embeddings for drift
DRIFT_THRESHOLD = 0.6  # cosine distance threshold (adjustable)

# query_log = []
# detector = DriftDetector()
# reference_queries = [
#     "how to train model",
#     "what is mlops",
#     "explain transformers"
# ]
# detector.update_reference(reference_queries)


def configure_mlflow():
    if os.getenv("GITHUB_ACTIONS") == "true":
        mlflow.set_tracking_uri("file:/tmp/mlruns")  # Writable during GitHub Actions
    else:
        mlflow.set_tracking_uri("http://mlflow:5000")  # Use MLflow server in prod/dev

    mlflow.set_experiment("rag-qa")

configure_mlflow()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # logs to console
        logging.FileHandler("app.log", mode='a')  # logs to file
    ]
)

logger = logging.getLogger(__name__)


# Load models and data
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

embed_model = None
qa_pipeline = None


if os.path.exists("vector.index"):
    index = faiss.read_index("vector.index")
else:
    dim = 384  # dimension of MiniLM embeddings
    index = faiss.IndexFlatL2(dim)
    print("No FAISS index found. Starting with empty index.")



def load_docs():
    contents = []
    if os.path.exists("doc_names.txt"):
        with open("doc_names.txt", "r") as f:
            names = [line.strip() for line in f]

        for name in names:
            path = os.path.join("docs", name)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    contents.append(f.read())
            else:
                print(f"Missing file: {path}")
    else:
        print("doc_names.txt not found in load_docs()")

    return contents

def load_index():
    return faiss.read_index("vector.index")


docs = []
if os.path.exists("doc_names.txt"):
    with open("doc_names.txt", "r") as f:
        doc_names = [line.strip() for line in f]

    for name in doc_names:
        doc_path = os.path.join("docs", name)
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                docs.append(f.read())
        else:
            print(f"⚠️ Missing doc file: {doc_path}")
else:
    print("No doc_names.txt found. Starting with empty docs.")



class QueryRequest(BaseModel):
    question: str


@app.post("/answer")
def get_answer(request: QueryRequest):
    try:
        query = request.question.strip()
        logger.info(f"Received question: {query}")
        if not query:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        

         # Drift Detection: track query and trigger retrain if needed
        # query_log.append(query)

        # if len(query_log) >= 10:
        #     drift = detector.check_drift(query_log[-10:])
        #     if drift:
        #         subprocess.Popen(["python", "retrain.py"])
        #         logger.info("Drift detected. Retraining triggered.")

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

@app.on_event("startup")
def startup_event():
    print(" FastAPI backend started and ready.")
