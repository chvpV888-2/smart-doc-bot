# 🤖 Smart Doc Bot: Hybrid RAG PDF Assistant

A high-performance "Chat with PDF" application built with **Streamlit**, **LangChain**, and a **Hybrid AI Architecture**. This bot allows users to upload PDF documents and ask questions in natural language, receiving instant, context-aware answers.

## 🚀 Live Demo
[Check out the live app here!](https://smart-doc-bot-wsfbptdmhkce5iwzlyauib.streamlit.app)

## ✨ Features
- **Hybrid AI Pipeline:** Uses Google Gemini for high-accuracy embeddings and Groq (Llama 3.3) for lightning-fast inference.
- **Conversational Memory:** Remembers the context of your chat within a session for seamless follow-up questions.
- **Efficient Vector Search:** Implements FAISS (Facebook AI Similarity Search) for local, high-speed document retrieval.
- **Modern UI:** A clean, responsive interface built with Streamlit.

## 🛠️ Technical Stack
- **Frontend:** Streamlit
- **Orchestration:** LangChain
- **Embeddings:** Google Generative AI (`models/gemini-embedding-001`)
- **LLM:** Groq (`llama-3.3-70b-versatile`)
- **Vector Store:** FAISS
- **Environment:** Python 3.10+

## ⚙️ How It Works (RAG Architecture)
1. **Extraction:** The bot reads the uploaded PDF and extracts raw text.
2. **Chunking:** Text is split into manageable segments using `RecursiveCharacterTextSplitter`.
3. **Embedding:** Google Gemini converts text chunks into mathematical vectors.
4. **Storage:** Vectors are stored in a FAISS index for similarity searching.
5. **Retrieval:** When a user asks a question, the bot finds the most relevant chunks.
6. **Generation:** Groq processes the question + chunks to provide a concise, factual answer.

## 📝 Setup Instructions

### 1. Clone the repository
```bash
git clone [https://github.com/chvpV888-2/smart-doc-bot.git](https://github.com/chvpV888-2/smart-doc-bot.git)
cd smart-doc-bot
