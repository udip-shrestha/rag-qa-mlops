from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import os

app = FastAPI()

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


@app.post("/query")
def get_answer(request: QueryRequest):
    query = request.question
    query_embedding = embed_model.encode([query])
    D, I = index.search(query_embedding, k=1)
    top_doc = docs[I[0][0]]

    prompt = (
    f"Answer the question based only on the provided context.\n\n"
    f"Context:\n{top_doc}\n\n"
    f"Question: {query}\n\n"
    f"Give a complete and helpful sentence as your answer:"
)


    result = qa_pipeline(prompt, max_length=200, do_sample=False)
    return {"answer": result[0]["generated_text"]}
