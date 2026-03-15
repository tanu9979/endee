"""JusticeMind Streamlit frontend — upload legal documents and ask questions."""

import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="JusticeMind", page_icon="⚖️", layout="wide")

st.markdown("""<style>
.answer-box{background:#f8f9ff;border-left:4px solid #4f6ef7;padding:1rem 1.2rem;border-radius:6px;margin-bottom:1rem;font-size:.97rem;line-height:1.6}
.disclaimer-box{background:#fffbea;border-left:4px solid #f59e0b;padding:.6rem 1rem;border-radius:4px;color:#92400e;font-size:.88rem}
.badge-contract{background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:12px;font-size:.78rem}
.badge-legislation{background:#dcfce7;color:#166534;padding:2px 8px;border-radius:12px;font-size:.78rem}
.badge-judgement{background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:12px;font-size:.78rem}
</style>""", unsafe_allow_html=True)

for k, v in [("docs", []), ("answer", None), ("sources", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

DOC_TYPE_MAP = {"Contract / NDA": "contract", "Indian Law / Act": "legislation", "Court Judgement": "judgement"}
BADGE = {"contract": '<span class="badge-contract">Contract</span>',
         "legislation": '<span class="badge-legislation">Legislation</span>',
         "judgement": '<span class="badge-judgement">Judgement</span>'}

with st.sidebar:
    st.title("⚖️ JusticeMind")
    st.caption("Powered by Endee Vector DB")
    st.divider()
    st.subheader("📂 Upload Document")
    uploaded = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
    doc_type_label = st.selectbox("Document type", list(DOC_TYPE_MAP.keys()))

    if st.button("⬆️ Upload & Ingest") and uploaded:
        with st.spinner("Ingesting…"):
            try:
                r = requests.post(f"{API_BASE}/upload",
                                  files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                                  data={"doc_type": DOC_TYPE_MAP[doc_type_label]}, timeout=180)
                r.raise_for_status()
                d = r.json()
                st.success(f"✅ **{d['filename']}** — {d['chunk_count']} chunks ingested")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.divider()
    st.subheader("📚 Ingested Documents")
    if st.button("🔄 Refresh"):
        try:
            r = requests.get(f"{API_BASE}/documents", timeout=10)
            r.raise_for_status()
            st.session_state.docs = r.json().get("documents", [])
        except Exception as e:
            st.error(str(e))

    for doc in st.session_state.docs:
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{doc['filename']}** {BADGE.get(doc['doc_type'],'')} · {doc['chunk_count']} chunks",
                    unsafe_allow_html=True)
        if c2.button("🗑️", key=f"del_{doc['doc_id']}"):
            try:
                requests.delete(f"{API_BASE}/documents/{doc['doc_id']}", timeout=10).raise_for_status()
                st.session_state.docs = [d for d in st.session_state.docs if d["doc_id"] != doc["doc_id"]]
                st.rerun()
            except Exception as e:
                st.error(str(e))

st.header("Ask a Legal Question")
st.caption("Upload a document first, then ask questions about it")

scope = st.radio("Search scope", ["All documents", "By document type", "Specific document"], horizontal=True)
filter_doc_id = filter_doc_type = None

if scope == "By document type":
    filter_doc_type = DOC_TYPE_MAP[st.selectbox("Document type", list(DOC_TYPE_MAP.keys()), key="scope_type")]
elif scope == "Specific document":
    doc_names = {d["filename"]: d["doc_id"] for d in st.session_state.docs}
    if doc_names:
        filter_doc_id = doc_names[st.selectbox("Select document", list(doc_names.keys()))]
    else:
        st.info("No documents ingested yet.")

question = st.text_area("Your question", height=100,
                        placeholder="e.g. What are the termination clauses in this contract?")
col1, col2 = st.columns([3, 1])
ask_clicked = col1.button("⚖️ Ask JusticeMind", use_container_width=True)
top_k = col2.number_input("Top-K", 1, 10, 5)

if ask_clicked and question.strip():
    with st.spinner("Searching legal documents via Endee..."):
        try:
            payload: dict = {"question": question, "top_k": int(top_k)}
            if filter_doc_id:
                payload["doc_id"] = filter_doc_id
            elif filter_doc_type:
                payload["doc_type"] = filter_doc_type
            r = requests.post(f"{API_BASE}/ask", json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            st.session_state.answer = data["answer"]
            st.session_state.sources = data["sources"]
        except requests.HTTPError as e:
            st.error(f"API error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.error(str(e))

if st.session_state.answer:
    st.markdown(f'<div class="answer-box">{st.session_state.answer}</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer-box">⚠️ For informational purposes only. Not legal advice.</div>',
                unsafe_allow_html=True)
    if st.session_state.sources:
        st.subheader("📎 Retrieved Sources")
        for i, src in enumerate(st.session_state.sources):
            with st.expander(f"[{i+1}] {src['filename']} — chunk {src['chunk_idx']} (similarity: {src['similarity']})"):
                st.markdown(f"{BADGE.get(src['doc_type'], src['doc_type'])}  {src['excerpt']}",
                            unsafe_allow_html=True)
