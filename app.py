"""
AI Study Buddy - Streamlit App
Premium Modern UI v2.1
"""
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st
import os
import tempfile
from pathlib import Path

from src.loader import load_document
from src.indexer import index_document, get_collection_stats, collection
from src.generator import generate_answer_streaming
from src.memory import ChatHistory

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STATE MANAGEMENT ──────────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

def set_page(page):
    st.session_state.current_page = page

# ─── CSS - PREMIUM DESIGN ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Geist:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after {
    font-family: 'Inter', -apple-system, sans-serif !important;
    box-sizing: border-box;
}

.stApp {
    background: #05080f;
    background: radial-gradient(circle at top left, rgba(0, 242, 255, 0.05) 0%, transparent 40%),
                radial-gradient(circle at bottom right, rgba(112, 0, 255, 0.05) 0%, transparent 40%),
                #05080f;
    min-height: 100vh;
}

#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none !important; }

header[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 99 !important;
}

/* Sembunyikan header default hanya di layar desktop */
@media (min-width: 993px) {
    header[data-testid="stHeader"] {
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"] { 
        display: none !important; 
    }
}

[data-testid="stSidebar"] {
    width: 280px !important;
    min-width: 260px !important;
    max-width: 85vw !important;
    background: rgba(13, 19, 34, 0.85) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Sidebar Brand */
.sb-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 20px 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 16px;
}
.sb-brand-icon {
    width: 40px;
    height: 40px;
    background: rgba(0, 242, 255, 0.1);
    border: 1px solid rgba(0, 242, 255, 0.3);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 0 15px rgba(0, 242, 255, 0.2);
}
.sb-brand-text {
    flex: 1;
}
.sb-name {
    font-family: 'Geist', sans-serif !important;
    font-size: 1.1rem;
    font-weight: 800;
    color: #e1fdff;
    line-height: 1.2;
}
.sb-powered {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem;
    color: #b9cacb !important;
    letter-spacing: 0.05em;
}

/* Brain Mode Button */
.btn-brain {
    width: 100%;
    background: linear-gradient(135deg, #00f2ff 0%, #7000ff 100%);
    color: #fff !important;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 700;
    font-family: 'Geist', sans-serif !important;
    letter-spacing: 0.5px;
    margin-bottom: 16px;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(112, 0, 255, 0.3);
}

/* Custom st.button styling for Sidebar nav */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: #dde2f8 !important;
    justify-content: flex-start !important;
    padding: 10px 16px !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255, 255, 255, 0.05) !important;
}
[data-testid="stSidebar"] .stButton > button p {
    font-size: 0.95rem !important;
}

/* Hero Section */
.main-hdr {
    background: rgba(13, 19, 34, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 40px 40px;
    text-align: center;
    margin-bottom: 24px;
    backdrop-filter: blur(24px);
    position: relative;
    overflow: hidden;
}
.hero-icon-wrapper {
    display: inline-flex;
    width: 64px;
    height: 64px;
    background: rgba(0, 242, 255, 0.05);
    border: 1px solid rgba(0, 242, 255, 0.2);
    border-radius: 16px;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    margin-bottom: 20px;
    box-shadow: 0 0 30px rgba(0, 242, 255, 0.15);
}
.main-title {
    font-family: 'Geist', sans-serif !important;
    font-size: 3rem;
    font-weight: 700;
    color: #e1fdff;
    letter-spacing: -0.02em;
    margin-bottom: 12px;
}
.main-title span {
    background: linear-gradient(135deg, #00f2ff, #7000ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-sub {
    font-size: 1rem;
    color: #b9cacb !important;
    line-height: 1.6;
}

/* Insight Cards */
.sc {
    background: rgba(25, 31, 47, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 16px;
    height: 100%;
}
.sc-icon {
    font-size: 1.2rem;
    margin-bottom: 16px;
    color: #00f2ff;
}
.sc-l {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem;
    color: #b9cacb;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}
.sc-n {
    font-family: 'Geist', sans-serif !important;
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff;
}

/* Doc Cards */
.doc-card {
    background: rgba(25, 31, 47, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.doc-card-icon {
    width: 36px;
    height: 36px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}
.doc-name {
    font-size: 0.85rem;
    font-weight: 600;
    color: #dde2f8;
}
.doc-meta {
    font-size: 0.75rem;
    color: #b9cacb;
    margin-top: 4px;
}
.doc-badge {
    background: rgba(255,255,255,0.1);
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono';
}

/* Chat Input Styling */
[data-testid="stBottomBlock"], [data-testid="stBottom"], .stAppBottomBlock {
    background: transparent !important;
}
[data-testid="stFileUploader"] label { 
    display: none !important; 
}
[data-testid="stChatInputContainer"] {
    background: rgba(13, 19, 34, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(24px) !important;
    padding: 8px !important;
}
[data-testid="stChatInputContainer"] button {
    background: #00f2ff !important;
    border-radius: 12px !important;
}
[data-testid="stChatInputContainer"] button svg { fill: #002022 !important; }





/* Chat UI Customization */
[data-testid="stChatMessage"] {
    background: transparent !important;
}
[data-testid="stChatMessage"]:has(.is-user) {
    flex-direction: row !important;
}
[data-testid="stChatMessage"]:has(.is-user) [data-testid="stChatMessageContent"] {
    background: rgba(25, 31, 47, 0.4);
    border: 1px solid rgba(255,255,255,0.05);
    padding: 12px 16px;
    border-radius: 0 16px 16px 16px;
}
[data-testid="stChatMessage"]:has(.is-ai) {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has(.is-ai) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, rgba(0, 242, 255, 0.1), rgba(112, 0, 255, 0.1));
    border: 1px solid rgba(0, 242, 255, 0.2);
    padding: 12px 16px;
    border-radius: 16px 0 16px 16px;
    text-align: right;
    align-items: flex-end;
}

/* ANIMATIONS */
@keyframes breathingBackground {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}
@keyframes pulseGlow {
    0% { box-shadow: 0 4px 15px rgba(112, 0, 255, 0.3); }
    50% { box-shadow: 0 4px 25px rgba(0, 242, 255, 0.6); }
    100% { box-shadow: 0 4px 15px rgba(112, 0, 255, 0.3); }
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Apply Animations */
.stApp {
    background-size: 200% 200% !important;
    animation: breathingBackground 15s ease infinite;
}
.hero-icon-wrapper, .sb-brand-icon {
    animation: float 4s ease-in-out infinite;
}
.btn-brain {
    animation: pulseGlow 3s infinite;
}
.sc {
    animation: slideUp 0.6s ease-out forwards;
}
.doc-card {
    animation: slideUp 0.5s ease-out forwards;
    animation-fill-mode: both;
}
.doc-card:nth-child(1) { animation-delay: 0.1s; }
.doc-card:nth-child(2) { animation-delay: 0.2s; }
.doc-card:nth-child(3) { animation-delay: 0.3s; }

.sc:hover, .doc-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 20px rgba(0, 242, 255, 0.15);
    border-color: rgba(0, 242, 255, 0.3);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}

/* Ubah teks tombol upload bawaan Streamlit */
[data-testid="stFileUploaderDropzone"] button {
    color: transparent !important;
    position: relative !important;
}
[data-testid="stFileUploaderDropzone"] button::after {
    content: 'Masukan file';
    color: #ffffff !important; /* warna terang untuk teks */
    display: block;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-weight: 600;
    width: 100%;
    text-align: center;
}

/* ================= RESPONSIVE MEDIA QUERIES (TABLET & HP) ================= */

/* Tablet & Mobile (<= 992px) */
@media (max-width: 992px) {
    header[data-testid="stHeader"] {
        visibility: visible !important;
        height: auto !important;
        min-height: 48px !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        background: rgba(13, 19, 34, 0.88) !important;
        border: 1px solid rgba(0, 242, 255, 0.35) !important;
        border-radius: 10px !important;
        color: #00f2ff !important;
        top: 8px !important;
        left: 8px !important;
        box-shadow: 0 0 14px rgba(0, 242, 255, 0.25) !important;
        z-index: 999999 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        color: #00f2ff !important;
    }
    .main-title {
        font-size: 2.2rem !important;
    }
    .main-hdr {
        padding: 28px 24px !important;
    }
    .sc {
        margin-bottom: 12px;
    }
}

/* Smartphone / HP (<= 640px) */
@media (max-width: 640px) {
    html, body, .stApp {
        overflow-x: hidden !important;
        width: 100% !important;
    }
    .main-title {
        font-size: 1.55rem !important;
        line-height: 1.25 !important;
    }
    .main-sub {
        font-size: 0.85rem !important;
        line-height: 1.5 !important;
    }
    .main-hdr {
        padding: 20px 14px !important;
        margin-bottom: 16px !important;
        border-radius: 12px !important;
    }
    .hero-icon-wrapper {
        width: 48px !important;
        height: 48px !important;
        font-size: 1.5rem !important;
        margin-bottom: 12px !important;
    }
    .sc {
        padding: 12px !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
    }
    .sc-n {
        font-size: 1.35rem !important;
    }
    .sc-l {
        font-size: 0.58rem !important;
    }
    .doc-card {
        padding: 10px 12px !important;
    }
    .doc-name {
        font-size: 0.8rem !important;
        word-break: break-word !important;
    }
    [data-testid="stChatMessage"] {
        max-width: 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    [data-testid="stChatMessageContent"] {
        max-width: 90% !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
        font-size: 0.9rem !important;
        padding: 10px 12px !important;
    }
    [data-testid="stChatInputContainer"] {
        padding: 4px 8px !important;
        border-radius: 12px !important;
    }
    .txt-theme {
        font-size: 1.15rem !important;
    }
}

</style>
""", unsafe_allow_html=True)

if st.session_state.theme_mode == "Light":
    st.markdown("""
    <style>
    :root {
        --background-color: #f8fafc !important;
        --secondary-background-color: #f1f5f9 !important;
        --text-color: #0f172a !important;
    }
    .stApp { background: #f8fafc !important; }
    .main-title { color: #0f172a !important; }
    .main-sub { color: #475569 !important; }
    [data-testid="stSidebar"] { background: rgba(241, 245, 249, 0.9) !important; border-right: 1px solid rgba(0,0,0,0.1) !important; }
    .sb-name { color: #0f172a !important; }
    .sb-powered { color: #475569 !important; }
    .sc { background: rgba(255, 255, 255, 0.9) !important; border-color: rgba(0,0,0,0.1) !important; }
    .sc-n { color: #0f172a !important; }
    .sc-l { color: #475569 !important; }
    .doc-card { background: rgba(255, 255, 255, 0.9) !important; border-color: rgba(0,0,0,0.1) !important; }
    .doc-name { color: #0f172a !important; }
    .doc-meta { color: #475569 !important; }
    .doc-card-icon { background: rgba(0,0,0,0.05) !important; }
    [data-testid="stSidebar"] .stButton > button { color: #0f172a !important; }
    [data-testid="stSidebar"] .stButton > button:hover { background: rgba(0,0,0,0.05) !important; }
    .main-hdr { background: rgba(255, 255, 255, 0.6) !important; border-color: rgba(0,0,0,0.1) !important; }
    [data-testid="stChatInputContainer"] { background: rgba(255, 255, 255, 0.8) !important; border-color: rgba(0,0,0,0.1) !important; }
    .txt-theme { color: #0f172a !important; }
    .sub-theme { color: #475569 !important; }
    [data-testid="stMetricValue"] { color: #0f172a !important; }
    [data-testid="stMetricLabel"] { color: #475569 !important; }
    [data-testid="stBottomBlock"], [data-testid="stBottom"], .stAppBottomBlock { background: #f8fafc !important; }
    [data-baseweb="input"] { background: rgba(0,0,0,0.05) !important; color: #0f172a !important; }
    [data-testid="stFileUploaderDropzone"] button::after { color: #0f172a !important; }
    [data-testid="stChatMessage"]:has(.is-user) [data-testid="stChatMessageContent"] { background: rgba(255,255,255,0.9); border-color: rgba(0,0,0,0.1); color: #0f172a; }
    [data-testid="stChatMessage"]:has(.is-ai) [data-testid="stChatMessageContent"] { background: linear-gradient(135deg, rgba(0,242,255,0.15), rgba(112,0,255,0.15)); border-color: rgba(0,242,255,0.4); color: #0f172a; }
    @media (max-width: 992px) {
        [data-testid="stSidebarCollapsedControl"] {
            background: rgba(241, 245, 249, 0.95) !important;
            border-color: rgba(0, 0, 0, 0.15) !important;
            color: #0f172a !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def get_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatHistory(max_turns=10)
    return st.session_state.chat_history

def list_documents():
    try:
        all_data = collection.get()
        docs = {}
        for meta in all_data.get("metadatas", []):
            fn = meta.get("filename")
            if fn and fn not in docs:
                docs[fn] = {
                    "filename": fn,
                    "subject": meta.get("subject", "general"),
                    "total_pages": meta.get("total_pages", 0),
                }
        return list(docs.values())
    except Exception as e:
        return []

if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    stats = get_collection_stats()
    chat_history = get_chat_history()

    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-brand-icon">🧠</div>
        <div class="sb-brand-text">
            <div class="sb-name">AI Study Buddy</div>
            <div class="sb-powered">Powered by RAG + Groq</div>
        </div>
    </div>
    <div style="padding: 0 16px;">
        <button class="btn-brain">⚡ Brain Mode</button>
    </div>
    """, unsafe_allow_html=True)

    nav_items = {
        "Dashboard": "💠",
        "Library": "📚",
        "Analytics": "📈"
    }

    for name, icon in nav_items.items():
        if st.button(f"{icon}  {name}", use_container_width=True):
            set_page(name)

    st.markdown('<div style="height: 40vh;"></div>', unsafe_allow_html=True)
    if st.button("🕒 Clear History", use_container_width=True):
        chat_history.clear()
        st.session_state.messages = []
        st.rerun()

# ─── TOP HEADER ────────────────────────────────────────────────────────────────
colA, colB, colC = st.columns([1, 2, 1])
with colA:
    st.markdown('<div class="txt-theme" style="font-family: Geist; font-size: 1.5rem; font-weight: 600; color: #fff; padding-top:4px;">Knowledge Center</div>', unsafe_allow_html=True)
with colB:
    st.text_input("Search", placeholder="🔍 Cari materi...", label_visibility="collapsed")
with colC:
    theme_icon = "🌞 Mode Terang" if st.session_state.theme_mode == "Dark" else "🌙 Mode Gelap"
    if st.button(theme_icon, use_container_width=True):
        st.session_state.theme_mode = "Light" if st.session_state.theme_mode == "Dark" else "Dark"
        st.rerun()

st.divider()

# ─── MAIN CONTENT ROUTING ──────────────────────────────────────────────────────
if st.session_state.current_page == "Dashboard":
    
    st.markdown("""
    <div class="main-hdr">
        <div class="hero-icon-wrapper">🧠</div>
        <div class="main-title">Halo! Aku AI Study Buddy <span>Anda.</span></div>
        <div class="main-sub">Teman belajar AI yang siap bantu 24/7 — upload materi, langsung bisa tanya!<br>Sistem terhubung, analisis mendalam, pemahaman lebih cepat.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="txt-theme" style="font-family: Geist; font-size: 1.2rem; font-weight: 600; color: #fff; margin-bottom: 16px;">📈 Study Insights</div>', unsafe_allow_html=True)
        docs_count = len(list_documents())
        
        c_in1, c_in2, c_in3 = st.columns(3)
        with c_in1:
            st.markdown(f"""
            <div class="sc">
                <div class="sc-icon">📂</div>
                <div class="sc-l">MATERI TERUNGGAH</div>
                <div class="sc-n">{docs_count}</div>
            </div>""", unsafe_allow_html=True)
        with c_in2:
            st.markdown(f"""
            <div class="sc">
                <div class="sc-icon">🧩</div>
                <div class="sc-l">TOTAL CHUNKS</div>
                <div class="sc-n">{stats.get('total_chunks', 0)}</div>
            </div>""", unsafe_allow_html=True)
        with c_in3:
            st.markdown(f"""
            <div class="sc">
                <div class="sc-icon">💬</div>
                <div class="sc-l">AI INTERACTIONS</div>
                <div class="sc-n">{chat_history.turn_count}</div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="txt-theme" style="font-family: Geist; font-size: 1.2rem; font-weight: 600; color: #fff; margin-bottom: 16px;">⏳ Recent <span style="float:right; font-size: 0.8rem; color: #b9cacb; margin-top:4px;">Lihat Semua</span></div>', unsafe_allow_html=True)
        docs = list_documents()[:3]
        if not docs:
            st.markdown('<div class="doc-card" style="justify-content:center; color:#b9cacb;">Belum ada dokumen</div>', unsafe_allow_html=True)
        for doc in docs:
            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-card-icon">📄</div>
                <div style="flex:1;">
                    <div class="doc-name">{doc['filename']}</div>
                    <div class="doc-meta">Diakses baru saja</div>
                </div>
                <div class="doc-badge">{doc['subject'].upper()}</div>
            </div>
            """, unsafe_allow_html=True)

elif st.session_state.current_page == "Library":
    st.markdown('<div class="txt-theme" style="font-family: Geist; font-size: 1.5rem; font-weight: 600; color: #fff; margin-bottom: 16px;">📚 Library & Upload</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="sub-theme" style="font-size: 0.9rem; margin-bottom: 8px; color: #b9cacb;">Unggah dokumen materi (.pdf, .docx, .txt, .md)</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Dokumen Baru", type=["pdf", "docx", "txt", "md"], label_visibility="collapsed")
        if uploaded_file:
            st.markdown(f"**File:** {uploaded_file.name} ({round(uploaded_file.size/1024,1)} KB)")
            subj = st.text_input("Subjek Dokumen (Opsional)")
            if st.button("Mulai Indexing"):
                with st.spinner("Mengindeks materi..."):
                    try:
                        suffix = Path(uploaded_file.name).suffix
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        try:
                            document = load_document(tmp_path)
                            document["filename"] = uploaded_file.name
                            index_document(document, subject=subj, grade_level="umum", difficulty="intermediate")
                            st.success("✅ Berhasil diunggah dan diindeks!")
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown('<h3 class="txt-theme" style="color: #fff;">Dokumen Tersedia</h3>', unsafe_allow_html=True)
    docs = list_documents()
    for doc in docs:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f'<span class="txt-theme" style="color: #fff;">📄 **{doc["filename"]}** — {doc["subject"]} ({doc["total_pages"]} hal)</span>', unsafe_allow_html=True)
        with col2:
            if st.button("Hapus", key=f"del_{doc['filename']}"):
                collection.delete(where={"filename": doc['filename']})
                st.rerun()



elif st.session_state.current_page == "Analytics":
    st.markdown('<div class="txt-theme" style="font-family: Geist; font-size: 1.5rem; font-weight: 600; color: #fff; margin-bottom: 16px;">📈 Study Analytics</div>', unsafe_allow_html=True)
    
    import pandas as pd
    
    docs = list_documents()
    total_docs = len(docs)
    total_chunks = get_collection_stats().get("total_chunks", 0)
    
    # Hitung interaksi berdasarkan panjang pesan (bagi 2 karena 1 pasang user-assistant)
    interactions = len(st.session_state.get("messages", [])) // 2
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Dokumen", value=f"{total_docs} File")
    with col2:
        st.metric(label="Materi Terindeks", value=f"{total_chunks} Chunks")
    with col3:
        st.metric(label="Tanya Jawab AI", value=f"{interactions} Sesi")
        
    st.divider()
    
    st.markdown('<div class="sub-theme" style="font-size: 1.1rem; color: #b9cacb; margin-bottom: 12px;">Distribusi Topik Materi</div>', unsafe_allow_html=True)
    
    if total_docs > 0:
        subjects = {}
        for d in docs:
            subj = d.get('subject', '').strip()
            if not subj:
                subj = "Umum"
            subj = subj.capitalize()
            subjects[subj] = subjects.get(subj, 0) + 1
            
        chart_data = pd.DataFrame({
            "Topik": list(subjects.keys()),
            "Jumlah Dokumen": list(subjects.values())
        }).set_index("Topik")
        
        st.bar_chart(chart_data, color="#00f2ff")
    else:
        st.info("Belum ada dokumen yang diunggah. Unggah materi di menu Library untuk melihat analitik.")




# ─── CHAT AREA ─────────────────────────────────────────────────────────────────
if st.session_state.current_page == "Dashboard":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Only show chat messages if there are any
    if st.session_state.messages:
        st.markdown('<div style="margin-top: 40px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 20px;">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            is_user = msg["role"] == "user"
            avatar = "🧑‍🎓" if is_user else "🧠"
            span_class = "is-user" if is_user else "is-ai"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(f"<span class='{span_class}'></span>" + msg["content"], unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Tanya apapun tentang materi belajar lo..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant", avatar="🧠"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in generate_answer_streaming(st.session_state.messages[-1]["content"], chat_history=get_chat_history()):
                    full_response += chunk
                    response_placeholder.markdown("<span class='is-ai'></span>" + full_response + "▌", unsafe_allow_html=True)
                response_placeholder.markdown("<span class='is-ai'></span>" + full_response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error: {e}")
