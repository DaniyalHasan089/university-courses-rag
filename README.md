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
```

## 🚀 Getting Started
1. Install Requirements
Make sure you have Python 3.10+ installed. Then run:

```bash
pip install -r requirements.txt
```

2. Install and Run Ollama
Install Ollama from: https://ollama.com

Then open a terminal and run:

```bash
ollama run phi3
This will download and run the Phi-3 Mini language model locally.
```

3. Add Your PDFs
Place your university/course PDFs in the pdfs/ folder. Example:

```bash
pdfs/
├── data_structures.pdf
├── calculus_notes.pdf
└── machine_learning.pdf
```

4. Start the Chatbot
Run the chatbot:
```bash
python university_chatbot.py
A Gradio interface will open in your browser, allowing you to chat with your academic PDFs.
```

💬 Example Use Cases
"Explain the concept of Big-O from the data structures PDF."

"What is in chapter 3 of my machine learning notes?"

"Summarize calculus_notes.pdf."

---

🛠️ Tech Stack
Tool	Purpose
Python	Core programming language
LangChain	RAG pipeline and logic
langchain-community	PDF loader, embeddings, and chains
ChromaDB	Vector store
Ollama + Phi-3	Local LLM model (CPU support)
Sentence Transformers	Embeddings
Gradio	Chat UI
PyPDF	PDF parsing

---

👨‍💻 Author
Daniyal – Final year CS student passionate about AI, education, and open tools that empower learners.

---

📜 License
This project is licensed under the MIT License. You are free to use, modify, and distribute it.

---

📌 Notes
You can replace Phi-3 with other models like mistral, llama3, etc. supported by Ollama.

