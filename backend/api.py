from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import os

app = FastAPI()

# Load models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-xl")


# Load FAISS index and documents
index = faiss.read_index("backend/vector.index")
with open("backend/doc_names.txt", "r") as f:
    doc_names = [line.strip() for line in f]

docs = []
for name in doc_names:
    with open(f"backend/docs/{name}", "r", encoding="utf-8") as f:
        docs.append(f.read())

@app.get("/ask")
def ask_question(question: str = Query(..., min_length=3)):
    # Embed the question
    query_embedding = embed_model.encode([question])
    D, I = index.search(query_embedding, k=1)
    top_doc = docs[I[0][0]]

    # Prompt for LLM
    prompt = (
    f"You are an AI assistant that provides clear, detailed answers.\n"
    f"Based on the following document, answer the question in full sentences:\n\n"
    f"Document:\n{top_doc}\n\n"
    f"Question: {question}\n"
    f"Answer:"
)


    # Generate answer
    result = qa_pipeline(prompt, max_new_tokens=300, temperature=0.7, do_sample=True, return_full_text=True)


    answer = result[0]["generated_text"].strip()


    return {"question": question, "answer": answer}
