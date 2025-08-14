# 🎓 University AI Assistant

A simple and powerful Retrieval-Augmented Generation (RAG) chatbot for university course assistance. Ask questions about your course materials using free AI models via OpenRouter.

## ✨ Features

- **🆓 Completely Free**: Uses only free AI models - no payment required
- **📚 PDF Processing**: Automatically processes and indexes your PDF course materials
- **🔍 Smart Search**: Advanced vector search finds relevant content from your documents
- **💬 Clean Interface**: Simple Streamlit-based chat interface
- **📊 Source Citations**: See which documents were used for each answer
- **💾 Persistent Storage**: Your processed documents are saved for future use

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Free OpenRouter API key from [openrouter.ai/keys](https://openrouter.ai/keys)

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your API key**
   - Create a `.env` file in the project folder
   - Add your OpenRouter API key:
   ```env
   OPENROUTER_API_KEY=your_api_key_here
   ```

3. **Add your course materials**
   - Place your PDF files in the `pdfs/` folder
   - The app will automatically process them on first run

4. **Run the application**
   ```bash
   streamlit run university_chatbot.py
   ```
   Or on Windows: double-click `run_streamlit.bat`

## 📁 Project Structure

```
Uni_LLM/
├── university_chatbot.py    # Main application
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── run_streamlit.bat      # Windows launcher
├── .env                   # Your API key (create this)
├── .gitignore            # Git ignore rules
├── README.md             # This documentation
├── pdfs/                 # Your course materials
│   └── Software Requirements Engineering/
│       ├── Week No. 1 SRE.pdf
│       └── ... (your PDF files)
└── db/                   # Vector database (auto-created)
```

## 🤖 Available AI Models

All models are **completely free** to use:

- **Microsoft Phi-3 Mini** - Fast and efficient, great for general questions
- **Meta Llama 3.1 8B** - Powerful open-source model with excellent reasoning
- **Google Gemma 2 9B** - Google's optimized model, good balance of speed and quality
- **Mistral 7B** - European model with strong multilingual capabilities

## 💡 How to Use

1. **First Time Setup**:
   - The app will automatically process your PDF files
   - This creates a searchable database of your course content
   - Processing happens only once - future startups are instant

2. **Asking Questions**:
   - Type your question in the text box
   - The AI will search your course materials and provide answers
   - Sources are shown so you can verify the information

3. **Example Questions**:
   - "What are the main principles covered in Week 1?"
   - "Explain the difference between functional and non-functional requirements"
   - "Summarize the key points from the software engineering lecture"
   - "What are the challenges mentioned in requirements elicitation?"

## ⚙️ Settings

- **AI Model**: Choose which free model to use
- **Advanced Settings**: Adjust how documents are processed (usually not needed)
- **Refresh Models**: Update the list of available models

## 🔧 Troubleshooting

### Common Issues

1. **"API key not configured"**
   - Make sure you created a `.env` file
   - Check that your API key is correct
   - Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys)

2. **"No PDF documents found"**
   - Place your PDF files in the `pdfs/` folder
   - Make sure files are not password-protected

3. **App runs but no models available**
   - Check your internet connection
   - Verify your API key is valid
   - Click the "🔄 Refresh Models" button

### Performance Tips

- **Smaller PDFs**: Process faster and use less memory
- **Clear File Names**: Help identify sources in answers
- **Regular Cleanup**: Delete old database (`db/` folder) if you change course materials significantly

## 🛠️ Technical Details

- **Framework**: Streamlit for web interface
- **Vector Database**: ChromaDB for document storage and search
- **Embeddings**: Sentence Transformers for text understanding
- **LLM Access**: OpenRouter API for free model access
- **Document Processing**: LangChain for PDF handling

## 🆓 Cost Information

- **API Usage**: Completely free with OpenRouter's free tier
- **No Payment Required**: All selected models are free to use
- **No Hidden Costs**: No credit cards or subscriptions needed

## 📞 Need Help?

- Check the troubleshooting section above
- All models are free - no payment issues to worry about
- The app works offline once your documents are processed

---

**Start learning with AI assistance today! 🚀**

*Simple setup, powerful results, completely free.*