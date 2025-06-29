from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import os

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("vector.index")

# Load document names and content
with open("doc_names.txt", "r") as f:
    doc_names = [line.strip() for line in f]

docs = []
for name in doc_names:
    with open(os.path.join("docs", name), "r", encoding="utf-8") as f:
        docs.append(f.read())

# Ask user a question
query = input("Enter your question: ")

# Embed the query
query_embedding = embed_model.encode([query])
D, I = index.search(query_embedding, k=1)

# Retrieve top document
top_doc = docs[I[0][0]]

# Format prompt
context = top_doc.strip()
prompt = (
    f"You are an intelligent assistant. Read the context and answer the question clearly and completely.\n\n"
    f"Context: {context}\n\n"
    f"Question: {query}\n\n"
    f"Answer:"
)



# Load FLAN-T5 model
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")


# Generate answer
result = qa_pipeline(prompt, max_new_tokens=100, do_sample=False)



# Show result
print("\nGenerated Answer:")
print(result[0]["generated_text"])
