import streamlit as st
import torch
import torchxrayvision as xrv
import numpy as np
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer
from google import genai

# =====================================
# 1️⃣ PAGE SETUP & UI STYLING
# =====================================
st.set_page_config(page_title="AI Radiology Pro", layout="wide", page_icon="🏥")

# Injecting CSS to force the report to wrap and look like a document (No Scrollers)
st.markdown("""
    <style>
    .report-card {
        background-color: white;
        padding: 30px;
        border-radius: 10px;
        border: 1px solid #d1d8e0;
        line-height: 1.6;
        color: #2f3542;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏥 RAG-Enhanced Radiology Reporting")
st.markdown("---")

# =====================================
# 2️⃣ ASSET LOADING
# =====================================
@st.cache_resource
def load_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
    dl_model.eval()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("mimic_faiss_index.faiss")
    reports_db = np.load("mimic_reports.npy", allow_pickle=True)
    return dl_model, embed_model, index, reports_db, device

dl_model, embed_model, index, reports_db, device = load_assets()
client = genai.Client(api_key="Your api_key")

# =====================================
# 3️⃣ THE INTERFACE
# =====================================
col_in, col_out = st.columns([1, 1.2], gap="large")

with col_in:
    st.subheader("📸 Upload Scan")
    file = st.file_uploader("Upload X-Ray", type=["jpg", "png", "jpeg"])
    
    # This is the "Extra Query" box
    user_query = st.text_area("❓ Extra Query / Clinical Context", 
                             placeholder="e.g., Check for pneumothorax specifically or 'What is the heart size?'",
                             help="The AI will prioritize this question in the final report.")
    
    if file:
        st.image(file, use_container_width=True)

with col_out:
    st.subheader("📋 AI Clinical Report")
    
    if file and st.button("🚀 Generate Final Report"):
        with st.spinner("Processing Vision-RAG Pipeline..."):
            # A. Vision Extraction
            img = Image.open(file).convert("L")
            img_tensor = torch.from_numpy(xrv.datasets.normalize(np.array(img.resize((224, 224))), 255)).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = dl_model(img_tensor).cpu().numpy()[0]
            
            detected = [p for p, s in zip(dl_model.pathologies, preds) if s > 0.35]
            findings_str = ", ".join(detected) if detected else "No acute abnormalities"
            
            # B. RAG Context
            query_vec = embed_model.encode([f"Findings: {findings_str}"])
            _, idx = index.search(query_vec, k=2)
            retrieved = "\n---\n".join([reports_db[j] for j in idx[0]])

            # C. THE "FIXED" PROMPT (Prioritizes the Extra Query)
            rag_prompt = f"""
            SYSTEM ROLE: Board-Certified Radiologist.
            
            USER'S SPECIFIC QUESTION (PRIORITY): {user_query if user_query else "General analysis required."}
            
            IMAGE DATA: {findings_str}
            PROFESSIONAL EXAMPLES: {retrieved}
            
            INSTRUCTIONS:
            1. Directly answer the USER'S SPECIFIC QUESTION first.
            2. Provide a structured 'Findings' and 'Impression' report.
            3. Use professional medical language.
            4. DO NOT wrap your output in backticks or code blocks. Just write plain text with Markdown bolding.
            """

            response = client.models.generate_content(
                model="gemma-3-27b-it", 
                contents=[rag_prompt, Image.open(file).convert("RGB")],
                config={'temperature': 0.0}
            )

            # D. THE "NO-SCROLLER" DISPLAY LOGIC
            # Clean the text of any Markdown code blocks that force scrolling
            clean_text = response.text.replace("```markdown", "").replace("```text", "").replace("```", "").strip()
            
            st.success(f"**Vision Model Detects:** {findings_str}")
            
            # The "Card" Look
            st.markdown(f'<div class="report-card">{clean_text}</div>', unsafe_allow_html=True)
            
            st.divider()
            st.download_button("💾 Download Report", clean_text, file_name="report.txt")
    else:
        st.info("Upload an image and click 'Generate' to see the structured report.")