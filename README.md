# 🧠 AI Study Buddy

![AI Study Buddy Banner](https://img.shields.io/badge/AI_Study_Buddy-Production_Grade_RAG-00f2ff?style=for-the-badge&logo=openai&logoColor=white)

AI Study Buddy is an advanced, production-grade **Retrieval-Augmented Generation (RAG)** web application designed to act as a highly personalized virtual tutor. Unlike generic AI wrappers, this application allows users to upload their own educational materials (PDF, Word, TXT, MD) and interact with an intelligent tutor that strictly answers based on the provided context with accurate citations.

---

## 🌟 Key Features (Unique Selling Points)

### 1. 🏗️ Production-Grade RAG Pipeline
This project implements an advanced search and retrieval pipeline to guarantee high-accuracy answers:
*   **Hybrid Search:** Combines Vector Embeddings (Semantic Search) with BM25 (Keyword Search) using **Reciprocal Rank Fusion (RRF)** to fetch the most relevant documents.
*   **Cohere Reranking:** Implements a Machine Learning Reranker to re-evaluate and sort the top chunks, drastically reducing AI hallucinations.
*   **Vector Database:** Powered by local **ChromaDB** for fast, persistent, and secure document embedding storage.

### 2. 🧠 Conversational Memory & Query Rewriting
Most RAG applications fail at follow-up questions (e.g., *"What is the formula for that?"*). AI Study Buddy solves this by leveraging a **Standalone Query Generator (Query Rewriting)**. The LLM reads the chat history and automatically rewrites ambiguous follow-up questions into fully contextualized standalone queries before searching the database, ensuring 100% context retention.

### 3. 🎯 Strict Anti-Hallucination & Citations
Through rigorous System Prompt engineering, the AI acts strictly as an educational tutor. It forces the generation of **Source Citations** (e.g., `[1] Physics_Material.pdf, page 4`) at the end of its answers. If the answer does not exist in the provided documents, it is programmed to transparently admit it instead of hallucinating.

### 4. 💅 Premium & Dynamic UI/UX
Built with Streamlit but completely transformed via custom CSS injection:
*   **Glassmorphism Design:** Modern aesthetic with breathing background animations.
*   **Dynamic Theme Toggle:** Native Light / Dark Mode seamlessly implemented without reloading the app.
*   **Analytics Dashboard:** Real-time metrics powered by Pandas, displaying Total Documents, Indexed Chunks, AI Interactions, and a dynamic Subject Distribution chart.
*   **Asymmetric Chat Interface:** Custom chat bubbles with AI aligning right and Users aligning left, mirroring professional chat applications.

---

## 🛠️ Technology Stack

*   **Frontend:** Streamlit, Custom CSS (Glassmorphism), HTML Injection
*   **Language Model:** Groq (LLaMA 3) / OpenAI API (Ultra-fast inference)
*   **RAG Engine:** LangChain
*   **Vector Database:** ChromaDB
*   **Reranker:** Cohere API
*   **Data Processing:** Pandas, NumPy, PyPDF2, docx2txt

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nikoardyan/ai-study-buddy.git
   cd ai-study-buddy
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your API Keys:
   ```env
   GROQ_API_KEY=your_groq_api_key
   COHERE_API_KEY=your_cohere_api_key
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## 📸 Screenshots
*(Add your screenshots here)*
- Dashboard & Chat Interface
- Light / Dark Mode Toggle
- Analytics & Document Library

---
*Built with passion for the future of personalized education.* 🎓
