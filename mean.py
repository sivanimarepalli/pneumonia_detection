import torch
import torchxrayvision as xrv
import numpy as np
import faiss
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from google import genai
from f1chexbert import F1CheXbert
from sklearn.metrics import f1_score

# =====================================
# 1️⃣ THE MONKEY-PATCH (Modern API)
# =====================================
def patched_get_label(self, impression):
    encoded = self.tokenizer(
        impression,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(self.device)
    
    with torch.no_grad():
        out = self.model(encoded['input_ids'], encoded['attention_mask'])
        labels = [torch.argmax(o, dim=1).item() for o in out]
    return labels

evaluator = F1CheXbert(device="cpu")
evaluator.get_label = patched_get_label.__get__(evaluator, F1CheXbert)

# =====================================
# 2️⃣ INITIALIZE MODELS & DATA
# =====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client = genai.Client(api_key="Your api_key")


print("Loading Models & Dataset...")
ds = load_dataset("MLforHealthcare/mimic-cxr", split="train", streaming=True)
dl_model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
dl_model.eval()

index = faiss.read_index("mimic_faiss_index.faiss")
reports_db = np.load("mimic_reports.npy", allow_pickle=True)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

pathology_cols = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", 
    "Support Devices", "No Finding"
]

# =====================================
# 3️⃣ OPTIMIZED EXECUTION LOOP
# =====================================
results_rag, results_baseline = [], []
num_samples = 10 

for i, example in enumerate(ds):
    if i >= num_samples: break
    print(f"\n--- Study {i+1} ---")
    
    try:
        # A. Get Human Ground Truth (1.0 = Positive, everything else = 0)
        gt_vector = np.array([1 if example.get(col) == 1.0 else 0 for col in pathology_cols])

        # B. Vision Model (Using a more sensitive 0.35 threshold to match human recall)
        img_pil = example['image'].convert("L")
        img_array = xrv.datasets.normalize(np.array(img_pil.resize((224, 224))), maxval=255)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            raw_preds = dl_model(img_tensor).cpu().numpy()[0]
        
        # We extract findings with higher sensitivity (0.35)
        findings_list = [p for p, s in zip(dl_model.pathologies, raw_preds) if s > 0.35]
        findings_str = ", ".join(findings_list) if findings_list else "No acute abnormalities"

        # C. RAG Retrieval
        query_vec = embed_model.encode([f"Findings: {findings_str}"])
        _, idx = index.search(query_vec, k=2)
        context = "\n---\n".join([reports_db[j] for j in idx[0]])

        # D. HIGH-ACCURACY PROMPT
        rgb_img = example['image'].convert("RGB")
        
        # We explicitly instruct the LLM to act as a "Validator" of the Vision findings
        rag_prompt = f"""
        Act as a Diagnostic Radiologist. Your goal is to match the Ground Truth findings exactly.
        
        OBSERVED PATHOLOGIES: {findings_str}
        
        REFERENCE TEMPLATES (Use ONLY for phrasing):
        {context}
        
        STRICT REPORTING PROTOCOL:
        1. Only mention the OBSERVED PATHOLOGIES listed above. 
        2. If a pathology (like Pleural Effusion) is in the REFERENCE TEMPLATES but NOT in the OBSERVED PATHOLOGIES, omit it.
        3. Use standard clinical terminology (e.g., 'Cardiomegaly' for heart enlargement).
        4. Be extremely precise. Avoid vague descriptions that might be mislabeled by a BERT classifier.
        """

        res_rag = client.models.generate_content(model="gemma-3-27b-it", contents=[rag_prompt, rgb_img])
        res_base = client.models.generate_content(model="gemma-3-27b-it", contents=["Generate a professional radiology report.", rgb_img])

        # E. Scoring
        l_rag = np.array(evaluator.get_label(res_rag.text))
        l_base = np.array(evaluator.get_label(res_base.text))

        f1_rag = f1_score(gt_vector, l_rag, average='macro')
        f1_base = f1_score(gt_vector, l_base, average='macro')

        results_rag.append(f1_rag)
        results_baseline.append(f1_base)

        print(f"Dataset F1 -> RAG: {f1_rag:.4f} | Base: {f1_base:.4f}")

    except Exception as e:
        print(f"Skipped Study {i+1}: {e}")

# =====================================
# 4️⃣ SUMMARY
# =====================================
if results_rag:
    print("\n" + "="*45)
    print("      FINAL HUMAN-LABEL DATASET ACCURACY")
    print("="*45)
    print(f"MEAN RAG ACCURACY:      {np.mean(results_rag):.4f}")
    print(f"MEAN BASELINE ACCURACY: {np.mean(results_baseline):.4f}")
    print("="*45)