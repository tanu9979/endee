"""JusticeMind — AI Legal Document Q&A Engine (Streamlit UI)."""

import streamlit as st

st.set_page_config(page_title="JusticeMind", page_icon="⚖️", layout="wide")

st.title("⚖️ JusticeMind")
st.caption("AI-Powered Legal Document Q&A Engine · Powered by Endee Vector DB + Groq LLM")

st.markdown("---")

col1, col2, col3 = st.columns(3)
col1.metric("Vector Database", "Endee (Hybrid)")
col2.metric("LLM", "Llama 3.1 8B")
col3.metric("Embeddings", "all-MiniLM-L6-v2")

st.markdown("---")

st.info("🚀 Full application coming soon. Upload legal documents and ask questions with cited answers.")

with st.sidebar:
    st.title("⚖️ JusticeMind")
    st.caption("Powered by Endee Vector DB")
    st.divider()
    st.markdown("### Features")
    st.markdown("- 📄 Upload PDFs & TXT files")
    st.markdown("- 🔍 Hybrid semantic + keyword search")
    st.markdown("- 🤖 Cited answers via Groq LLM")
    st.markdown("- ⚖️ Supports contracts, laws & judgements")
    st.divider()
    st.markdown("### Tech Stack")
    st.markdown("- **Vector DB:** Endee")
    st.markdown("- **Embeddings:** sentence-transformers")
    st.markdown("- **LLM:** llama-3.1-8b-instant")
    st.markdown("- **Backend:** FastAPI")
