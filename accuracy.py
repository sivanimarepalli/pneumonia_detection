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
# 1️⃣ THE CORE FIX: MODERN MONKEY-PATCH
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

print("Loading Models...")
ds = load_dataset("MLforHealthcare/mimic-cxr", split="train", streaming=True)
dl_model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
dl_model.eval()

index = faiss.read_index("mimic_faiss_index.faiss")
reports_db = np.load("mimic_reports.npy", allow_pickle=True)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_silver_vector(preds):
    vec = np.zeros(14)
    mapping = {
        "Atelectasis": 7, "Cardiomegaly": 1, "Consolidation": 5, "Edema": 4, 
        "Enlarged Cardiomediastinum": 0, "Fracture": 11, "Lung Lesion": 3, 
        "Lung Opacity": 2, "Pleural Effusion": 9, "Pneumonia": 6, "Pneumothorax": 8
    }
    for path, target_idx in mapping.items():
        if path in dl_model.pathologies:
            p_idx = dl_model.pathologies.index(path)
            # Use 0.5 for a strict Silver Standard comparison
            if preds[p_idx] > 0.5: vec[target_idx] = 1
    if np.sum(vec) == 0: vec[13] = 1 # 'No Finding'
    return vec.astype(int)

# =====================================
# 3️⃣ EXECUTION LOOP (OPTIMIZED PROMPT)
# =====================================
results_rag, results_baseline = [], []
num_samples = 10 

for i, example in enumerate(ds):
    if i >= num_samples: break
    print(f"\n--- Study {i+1} ---")
    
    try:
        # A. Vision Model Inference
        img_pil = example['image'].convert("L")
        img_array = xrv.datasets.normalize(np.array(img_pil.resize((224, 224))), maxval=255)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            raw_preds = dl_model(img_tensor).cpu().numpy()[0]
        
        silver_gt = get_silver_vector(raw_preds)
        findings_list = [p for p, s in zip(dl_model.pathologies, raw_preds) if s > 0.5]
        findings_str = ", ".join(findings_list) if findings_list else "No acute abnormalities"

        # B. RAG Retrieval
        query_vec = embed_model.encode([f"Findings: {findings_str}"])
        _, idx = index.search(query_vec, k=2)
        context = "\n---\n".join([reports_db[j] for j in idx[0]])

        # C. UPDATED HIGH-F1 PROMPT
        rgb_img = example['image'].convert("RGB")
        
        rag_prompt = f"""
        You are a senior clinical radiologist. Your task is to generate a report based EXCLUSIVELY on the provided findings.
        
        ACTUAL FINDINGS DETECTED: {findings_str}
        
        STYLE REFERENCE (Use for professional vocabulary ONLY):
        {context}
        
        INSTRUCTIONS:
        1. Accuracy is mandatory. Only report the ACTUAL FINDINGS DETECTED above.
        2. If the STYLE REFERENCE mentions a disease that is NOT in the ACTUAL FINDINGS, ignore it completely.
        3. Structure the report with 'Findings' and 'Impression' sections.
        4. Be clinical, concise, and do not speculate beyond the detected findings.
        """

        res_rag = client.models.generate_content(model="gemma-3-27b-it", contents=[rag_prompt, rgb_img])
        
        # Baseline Prompt
        res_base = client.models.generate_content(model="gemma-3-27b-it", 
            contents=["Analyze this chest X-ray and write a professional radiology report.", rgb_img])

        # D. Fidelity Scoring
        l_rag = np.array(evaluator.get_label(res_rag.text))
        l_base = np.array(evaluator.get_label(res_base.text))

        f1_rag = f1_score(silver_gt, l_rag, average='macro')
        f1_base = f1_score(silver_gt, l_base, average='macro')

        results_rag.append(f1_rag)
        results_baseline.append(f1_base)

        print(f"RAG Fidelity: {f1_rag:.4f} | Base Fidelity: {f1_base:.4f}")

    except Exception as e:
        print(f"Skipped Study {i+1}: {e}")

# =====================================
# 4️⃣ FINAL SUMMARY
# =====================================
if results_rag:
    print("\n" + "="*45)
    print("      FINAL SILVER STANDARD FIDELITY")
    print("="*45)
    print(f"MEAN RAG FIDELITY:      {np.mean(results_rag):.4f}")
    print(f"MEAN BASELINE FIDELITY: {np.mean(results_baseline):.4f}")
    print("-" * 45)
    gain = ((np.mean(results_rag) - np.mean(results_baseline)) / np.mean(results_baseline) * 100)
    print(f"FIDELITY IMPROVEMENT:   {gain:.2f}%")
    print("="*45)