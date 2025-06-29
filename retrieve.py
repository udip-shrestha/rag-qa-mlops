from sentence_transformers import SentenceTransformer
import faiss

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("vector.index")

# Load document names
with open("doc_names.txt", "r") as f:
    doc_names = [line.strip() for line in f]

# Load actual document contents
docs = []
for name in doc_names:
    with open(f"docs/{name}", "r", encoding="utf-8") as f:
        docs.append(f.read())

# Ask a question
query = input(" Enter your question: ")

# Embed the question
query_embedding = model.encode([query])

# Search top 1 result
D, I = index.search(query_embedding, k=1)

# Get result
top_doc = docs[I[0][0]]

print("\n Most Relevant Document Snippet:")
print("-" * 50)
print(top_doc)
