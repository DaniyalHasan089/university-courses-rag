# 📚 University Chatbot (Offline LLM with PDF Knowledge Base)

A fully **offline**, locally-run university chatbot powered by your own **PDF notes, lectures, and handouts**. This project allows you to interact with your academic material through a **chat interface**, without needing internet or cloud services.

Built using:
- 🧠 **Phi-3 Mini** (via Ollama)
- 🔗 **LangChain** + **ChromaDB**
- 📄 **PDF parsing**
- 💬 **Gradio UI**
- 💻 Runs on **CPU-only machines**

---

## 🎯 Features

- 🗂️ Ask questions from any uploaded course material (PDF)
- 🔍 Semantic search & context-aware answers
- ⚡ 100% offline: No API keys, no cloud, no OpenAI
- 🧠 Uses local LLMs (e.g. Phi-3 from Ollama)
- 🎓 Perfect for studying, revision, and Q&A
- 💬 Simple chat interface (built with Gradio)

---

## 📁 Project Structure

```bash
Uni_LLM/
├── pdfs/                     # Folder with all university PDFs
├── embeddings/              # Auto-generated vector DB storage
├── university_chatbot.py    # Main chatbot script
├── requirements.txt         # Python dependencies
└── README.md                # This file

