import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

print("Loading MIMIC dataset...")

# Load dataset
dataset = load_dataset("MLforHealthcare/mimic-cxr", split="train")

# Remove image column (prevents Pillow error)
if "image" in dataset.column_names:
    dataset = dataset.remove_columns(["image"])

# Select only first 500 samples
dataset = dataset.select(range(500))

print("Total samples selected:", len(dataset))

# ==============================
# Extract Report Text
# ==============================

reports = []

for sample in dataset:
    # Handle possible column names
    if "report" in sample:
        text = sample["report"]
    elif "reports" in sample:
        text = sample["reports"]
    elif "findings" in sample and "impression" in sample:
        text = sample["findings"] + " " + sample["impression"]
    else:
        continue

    if text and len(text) > 50:
        reports.append(text)

print("Valid reports collected:", len(reports))

# ==============================
# Create Embeddings
# ==============================

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = embed_model.encode(reports, convert_to_numpy=True)

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# ==============================
# Build FAISS Index
# ==============================

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Cosine similarity

index.add(embeddings)

print("FAISS index created.")
print("Total vectors stored:", index.ntotal)

# ==============================
# Save Index + Reports
# ==============================

faiss.write_index(index, "mimic_faiss_index.faiss")
np.save("mimic_reports.npy", np.array(reports))

print("FAISS index and reports saved successfully.")