from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi import HTTPException

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import traceback
import faiss
import os
import logging

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

        # Check if FAISS index is loaded
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

        # Use QA model safely
        try:
            result = qa_pipeline(prompt, max_length=200, do_sample=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text generation error: {str(e)}")

        return {"answer": result[0]["generated_text"]}

    except HTTPException as http_err:
        raise http_err  # Propagate known HTTP errors
    except Exception as e:
        traceback.print_exc()  # Optional: print full traceback in terminal
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok"}
