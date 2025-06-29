from sentence_transformers import SentenceTransformer
import faiss
import os

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Folder with text files
doc_folder = "docs"
docs = []
doc_names = []

# Load and store content
for fname in os.listdir(doc_folder):
    if fname.endswith(".txt"):
        with open(os.path.join(doc_folder, fname), "r", encoding="utf-8") as f:
            text = f.read()
            docs.append(text)
            doc_names.append(fname)

# Generate embeddings
embeddings = model.encode(docs, convert_to_numpy=True)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and doc names
faiss.write_index(index, "vector.index")
with open("doc_names.txt", "w") as f:
    for name in doc_names:
        f.write(name + "\n")

print(" Embedded", len(docs), "documents and saved the index.")
