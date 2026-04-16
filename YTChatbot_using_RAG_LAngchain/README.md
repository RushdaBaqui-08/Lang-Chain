# 🎥 YouTube Chatbot using RAG (LangChain + Groq + FAISS)

## 🚀 Overview

This project builds a **Retrieval-Augmented Generation (RAG)** based chatbot that can answer questions from a YouTube video.

Instead of relying on general knowledge, the chatbot:

* Extracts transcript from a YouTube video
* Converts it into embeddings
* Stores it in a vector database (FAISS)
* Retrieves relevant context
* Uses an LLM (Groq) to generate accurate answers

---

## 🧠 How It Works

1. **Transcript Extraction**

   * Fetch subtitles using `youtube-transcript-api`

2. **Text Chunking**

   * Split transcript into smaller chunks using `RecursiveCharacterTextSplitter`

3. **Embeddings**

   * Convert text into vectors using HuggingFace models

4. **Vector Storage**

   * Store embeddings in FAISS for fast similarity search

5. **Retrieval**

   * Fetch top relevant chunks based on user query

6. **Augmentation**

   * Combine retrieved context + user question into a prompt

7. **Generation**

   * Use Groq LLM to generate final answer

---

## 🏗️ Tech Stack

* **LangChain** – Framework for building LLM pipelines
* **Groq (LLaMA 3)** – Fast LLM inference
* **HuggingFace** – Embedding models
* **FAISS** – Vector database
* **YouTube Transcript API** – Data source

---

## 📂 Project Structure

```
.
├── YTChatbot_using_RAG_langchain.ipynb
├── .env
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/RushdaBaqui-08/Lang-Chain.git
cd Lang-Chain
```

### 2. Install dependencies

```bash
pip install youtube-transcript-api langchain-community langchain-groq faiss-cpu tiktoken python-dotenv sentence-transformers
```

### 3. Add your API key

Create a `.env` file and add:

```
GROQ_API_KEY=your_api_key_here
```

### 4. Run the notebook

Open the notebook and run all cells.

---

## 💬 Example Queries

* "What is this video about?"
* "Who is Demis?"
* "Explain the main topic discussed"
* "Is nuclear fusion discussed?"

---

## 🎯 Key Features

* ✅ Context-aware answers (no hallucination)
* ✅ Uses real video data
* ✅ Fast retrieval with FAISS
* ✅ Open-source (no OpenAI dependency)
* ✅ Modular LangChain pipeline

---

## 🧠 Learnings

This project demonstrates:

* How RAG works end-to-end
* Importance of retrieval quality
* How to reduce hallucination in LLMs
* Building scalable AI pipelines

---

## ⭐ Acknowledgements

* LangChain
* HuggingFace
* Groq
* FAISS
