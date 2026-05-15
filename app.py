"""
AI Study Buddy - All-in-One Streamlit (Ready for Cloud Deploy)
Skip FastAPI - langsung call RAG functions
"""
import streamlit as st
import os
import tempfile
from pathlib import Path

# Import RAG modules
from src.loader import load_document
from src.indexer import (
    index_document,
    get_collection_stats,
    detect_subject,
    collection
)
from src.generator import generate_answer_streaming
from src.memory import ChatHistory


# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] * { color: white !important; }

    /* HERO */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(102,126,234,0.4);
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    .hero-title { font-size: 2.8rem; font-weight: 800; color: white; margin: 0; position: relative; z-index: 1; }
    .hero-sub { font-size: 1rem; color: rgba(255,255,255,0.8); margin-top: 10px; position: relative; z-index: 1; }

    /* STAT CARDS */
    .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }
    .stat-card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.5);
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* CHAT */
    .chat-wrapper {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 24px;
        padding: 24px;
    }
    [data-testid="stChatMessage"] { background: transparent !important; border: none !important; }
    [data-testid="stChatMessage"][data-role="user"] .stMarkdown p {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 12px 18px !important;
        color: white !important;
        display: inline-block;
    }
    [data-testid="stChatMessage"][data-role="assistant"] .stMarkdown {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 14px 18px !important;
        color: rgba(255,255,255,0.9) !important;
    }

    /* CHAT INPUT */
    [data-testid="stChatInput"] {
        background: rgba(15,12,41,0.95) !important;
        border: 2px solid rgba(102,126,234,0.4) !important;
        border-radius: 50px !important;
    }
    textarea {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        caret-color: #ffffff !important;
    }
    textarea::placeholder {
        color: rgba(255,255,255,0.5) !important;
        -webkit-text-fill-color: rgba(255,255,255,0.5) !important;
    }
    [data-testid="stChatInput"] button svg { fill: white !important; }

    /* BUTTONS */
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04) !important;
        border: 2px dashed rgba(102,126,234,0.5) !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] section { background: transparent !important; }
    [data-testid="stFileUploader"] * { color: white !important; }
    [data-testid="stFileUploader"] section button {
        background: rgba(255,255,255,0.1) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }

    .doc-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 99px;
        font-size: 0.7rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    hr { border-color: rgba(255,255,255,0.1) !important; margin: 16px 0 !important; }
    .stMarkdown, p, span, label { color: rgba(255,255,255,0.85) !important; }
    h1, h2, h3, h4 { color: white !important; }
    [data-testid="stMetricValue"] { color: #a78bfa !important; font-weight: 800 !important; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ============ HELPERS ============
def get_chat_history():
    """Get or create chat history in session state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatHistory(max_turns=10)
    return st.session_state.chat_history


def list_documents():
    """List unique documents from ChromaDB."""
    try:
        all_data = collection.get()
        metadatas = all_data.get("metadatas", [])
        documents = {}
        for meta in metadatas:
            filename = meta.get("filename")
            if filename and filename not in documents:
                documents[filename] = {
                    "filename": filename,
                    "subject": meta.get("subject", "general"),
                    "total_pages": meta.get("total_pages", 0),
                }
        return list(documents.values())
    except:
        return []


# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:20px 0 10px">
        <div style="font-size:3rem">🎓</div>
        <div style="font-size:1.3rem; font-weight:800; color:white; margin-top:6px">AI Study Buddy</div>
        <div style="font-size:0.78rem; color:rgba(255,255,255,0.4); margin-top:4px">Powered by RAG + DeepSeek</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Stats
    stats = get_collection_stats()
    chat_history = get_chat_history()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chunks", stats["total_chunks"])
    with col2:
        st.metric("Turns", chat_history.turn_count)

    if chat_history.active_subject:
        st.info(f"🎯 Subject: **{chat_history.active_subject}**")

    st.divider()

    # Upload Materi
    st.markdown("**📚 Upload Materi**")
    uploaded_file = st.file_uploader(
        "Upload",
        type=["pdf", "docx", "txt", "md"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown(f"""
        <div style="background:rgba(102,126,234,0.15); border-radius:10px; padding:10px; margin:8px 0">
            <div style="font-size:0.85rem; font-weight:600; color:white">📄 {uploaded_file.name}</div>
            <div style="font-size:0.75rem; color:rgba(255,255,255,0.4)">{round(uploaded_file.size/1024, 1)} KB</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("⬆️ Upload & Index", use_container_width=True):
            with st.spinner("Indexing materi..."):
                try:
                    # Save to temp file
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    # Load & detect subject
                    document = load_document(tmp_path)
                    document["filename"] = uploaded_file.name  # use original name
                    subject = detect_subject(uploaded_file.name)

                    # Index
                    index_document(
                        document,
                        subject=subject,
                        grade_level="umum",
                        difficulty="intermediate"
                    )

                    # Cleanup temp
                    os.unlink(tmp_path)

                    st.success(
                        f"✅ **{uploaded_file.name}**\n\n"
                        f"Subject: `{subject}` · {document['total_pages']} hal"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.divider()

    # Daftar Materi
    st.markdown("**📖 Materi Tersedia**")
    docs = list_documents()
    if not docs:
        st.markdown("""
        <div style="text-align:center; padding:16px; color:rgba(255,255,255,0.35); font-size:0.85rem">
            📭 Belum ada materi
        </div>
        """, unsafe_allow_html=True)
    else:
        for doc in docs:
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1);
                            border-radius:10px; padding:10px; margin-bottom:6px">
                    <div style="font-size:0.82rem; font-weight:600; color:white;
                                white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:170px">
                        📄 {doc['filename']}
                    </div>
                    <span class="doc-badge">{doc['subject']}</span>
                    <span style="font-size:0.7rem; color:rgba(255,255,255,0.4); margin-left:4px">{doc['total_pages']} hal</span>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                if st.button("🗑", key=f"del_{doc['filename']}"):
                    collection.delete(where={"filename": doc['filename']})
                    st.rerun()

    st.divider()

    if st.button("🗑️ Clear History", use_container_width=True):
        chat_history.clear()
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style="text-align:center; margin-top:16px; color:rgba(255,255,255,0.25); font-size:0.72rem">
        AI Study Buddy v1.0<br>RAG + Streamlit
    </div>
    """, unsafe_allow_html=True)


# ============ MAIN ============

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-title">🎓 AI Study Buddy</div>
    <div class="hero-sub">Teman belajar AI yang siap bantu 24/7 — upload materi, langsung bisa tanya!</div>
</div>
""", unsafe_allow_html=True)

# Stats
doc_count = len(list_documents())
st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card"><div class="stat-number">{doc_count}</div><div class="stat-label">📚 Materi</div></div>
    <div class="stat-card"><div class="stat-number">{stats['total_chunks']}</div><div class="stat-label">🧩 Chunks</div></div>
    <div class="stat-card"><div class="stat-number">{chat_history.turn_count}</div><div class="stat-label">💬 Turns</div></div>
</div>
""", unsafe_allow_html=True)

# Chat
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "**Halo! Aku AI Study Buddy** 👋\n\n"
                "Aku siap bantu kamu belajar dari materi yang kamu upload!\n\n"
                "**Cara pakai:**\n"
                "1. 📚 Upload materi di sidebar kiri\n"
                "2. 💬 Tanya apapun tentang materinya\n"
                "3. 🎯 Sebut subject biar fokus: *'soal matematika tentang turunan'*\n\n"
                "Mau belajar apa hari ini? 😊"
            )
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

st.markdown('</div>', unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Tanya apapun tentang materi belajar lo..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎓"):
        try:
            # LANGSUNG call generator (no HTTP request!)
            full_response = st.write_stream(
                generate_answer_streaming(
                    question=prompt,
                    chat_history=chat_history
                )
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response or ""
            })
        except Exception as e:
            msg = f"❌ Error: {str(e)}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})