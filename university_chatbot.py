import os
import streamlit as st
import requests
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Fix SQLite version compatibility issue for ChromaDB
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb

# Try to import HuggingFaceEmbeddings with fallback
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        # Final fallback - create a simple embedding class
        class HuggingFaceEmbeddings:
            def __init__(self, model_name="all-MiniLM-L6-v2"):
                try:
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(model_name)
                except ImportError:
                    st.error("Required embedding packages not available. Please install sentence-transformers.")
                    st.stop()
            
            def embed_documents(self, texts):
                return self.model.encode(texts).tolist()
            
            def embed_query(self, text):
                return self.model.encode([text])[0].tolist()
import json
import time
from datetime import datetime
from config import Config

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

class OpenRouterLLM:
    """Custom OpenRouter LLM integration"""
    
    def __init__(self, api_key: str, model: str = None):
        self.api_key = api_key
        self.model = model or Config.DEFAULT_MODEL
        self.base_url = Config.OPENROUTER_BASE_URL
        
    def invoke(self, prompt: str, context: str = "") -> str:
        """Invoke the OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",  # Streamlit default
            "X-Title": "University AI Assistant"
        }
        
        system_message = Config.get_system_prompt()
        
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        if context:
            messages.append({
                "role": "user", 
                "content": f"Context from course materials:\n{context}\n\nQuestion: {prompt}"
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": Config.DEFAULT_TEMPERATURE,
            "max_tokens": Config.DEFAULT_MAX_TOKENS
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            return f"Error calling OpenRouter API: {str(e)}"
        except KeyError as e:
            return f"Unexpected API response format: {str(e)}"

class UniversityRAGSystem:
    """Main RAG system for university course assistance"""
    
    def __init__(self):
        self.vectordb = None
        self.llm = None
        
    def load_all_pdfs(self, folder_path: str) -> List[Document]:
        """Load all PDF documents from a folder"""
        documents = []
        pdf_files = []
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_path in enumerate(pdf_files):
            status_text.text(f"Loading {os.path.basename(pdf_path)}...")
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                # Add metadata
                for doc in docs:
                    doc.metadata['source_file'] = os.path.basename(pdf_path)
                    doc.metadata['full_path'] = pdf_path
                documents.extend(docs)
            except Exception as e:
                st.warning(f"Error loading {pdf_path}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(pdf_files))
        
        status_text.text(f"Loaded {len(documents)} pages from {len(pdf_files)} PDF files")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        return documents

    def build_vector_store(self, docs: List[Document], chunk_size: int = None, chunk_overlap: int = None):
        """Build and persist vector store"""
        try:
            chunk_size = chunk_size or Config.DEFAULT_CHUNK_SIZE
            chunk_overlap = chunk_overlap or Config.DEFAULT_CHUNK_OVERLAP
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            
            st.info("Splitting documents into chunks...")
            chunks = splitter.split_documents(docs)
            
            if not chunks:
                st.error("No chunks were created from the documents. Please check your PDF files.")
                return
            
            st.info(f"Creating embeddings for {len(chunks)} chunks...")
            progress_bar = st.progress(0)
            
            # Create vector store in batches to show progress
            batch_size = 50
            all_chunks_processed = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                all_chunks_processed.extend(batch)
                progress_bar.progress(min(1.0, (i + batch_size) / len(chunks)))
            
            # Clean up any existing database first
            import shutil
            if os.path.exists(Config.DB_PERSIST_DIRECTORY):
                try:
                    shutil.rmtree(Config.DB_PERSIST_DIRECTORY)
                    st.info("Cleaned up existing database...")
                except Exception as e:
                    st.warning(f"Could not clean existing database: {e}")
            
            # Create fresh ChromaDB client and collection
            try:
                client = chromadb.PersistentClient(path=Config.DB_PERSIST_DIRECTORY)
                
                # Try to delete collection if it exists
                try:
                    client.delete_collection(name="documents")
                except:
                    pass  # Collection doesn't exist, which is fine
                
                # Create new collection with better error handling
                collection = client.create_collection(
                    name="documents",
                    embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=Config.EMBEDDING_MODEL
                    )
                )
                
            except Exception as e:
                st.error(f"Failed to create ChromaDB collection: {str(e)}")
                progress_bar.empty()
                return
            
            # Prepare documents for ChromaDB with better validation
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(all_chunks_processed):
                if chunk.page_content and chunk.page_content.strip():
                    documents.append(chunk.page_content)
                    # Ensure metadata is serializable
                    metadata = {}
                    for key, value in chunk.metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                        else:
                            metadata[key] = str(value)
                    metadatas.append(metadata)
                    ids.append(f"doc_{i}")
            
            if not documents:
                st.error("No valid document content found after processing.")
                progress_bar.empty()
                return
            
            # Add documents to ChromaDB in smaller batches to avoid memory issues
            st.info("Adding documents to vector database...")
            batch_size = 100
            
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                batch_docs = documents[i:end_idx]
                batch_metas = metadatas[i:end_idx]
                batch_ids = ids[i:end_idx]
                
                try:
                    collection.add(
                        documents=batch_docs,
                        metadatas=batch_metas,
                        ids=batch_ids
                    )
                    progress_bar.progress(min(1.0, end_idx / len(documents)))
                except Exception as e:
                    st.error(f"Error adding batch {i//batch_size + 1}: {str(e)}")
                    progress_bar.empty()
                    return
            
            # Store the collection
            self.vectordb = collection
            
            progress_bar.empty()
            st.success(f"✅ Vector store created successfully with {len(documents)} chunks!")
            
        except Exception as e:
            st.error(f"Error building vector store: {str(e)}")
            st.error("Please try rebuilding the database or check your PDF files.")
            if 'progress_bar' in locals():
                progress_bar.empty()
        
    def load_vector_store(self):
        """Load existing vector store"""
        try:
            if not os.path.exists(Config.DB_PERSIST_DIRECTORY):
                return False
                
            client = chromadb.PersistentClient(path=Config.DB_PERSIST_DIRECTORY)
            
            # Check if collection exists
            try:
                self.vectordb = client.get_collection(name="documents")
                # Verify the collection has data
                count = self.vectordb.count()
                if count == 0:
                    st.warning("Database exists but is empty. Will rebuild from PDFs.")
                    return False
                st.info(f"Loaded existing database with {count} documents")
                return True
            except Exception as collection_error:
                st.warning(f"Database exists but collection is invalid: {str(collection_error)}")
                return False
                
        except Exception as e:
            st.warning(f"Could not access database: {str(e)}. Will rebuild from PDFs.")
            return False
    
    def setup_llm(self, api_key: str, model: str):
        """Setup OpenRouter LLM"""
        self.llm = OpenRouterLLM(api_key, model)
    
    def query(self, question: str, k: int = None) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.vectordb or not self.llm:
            return {"error": "System not properly initialized"}
        
        try:
            k = k or Config.DEFAULT_RETRIEVAL_K
            
            # Query ChromaDB directly
            results = self.vectordb.query(
                query_texts=[question],
                n_results=k
            )
            
            # Convert results to document-like objects
            docs = []
            if results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    doc_obj = type('Document', (), {
                        'page_content': doc_text,
                        'metadata': metadata or {}
                    })()
                    docs.append(doc_obj)
            
            # Prepare context
            context = "\n\n".join([f"Source: {doc.metadata.get('source_file', 'Unknown')}\n{doc.page_content}" for doc in docs])
            
            # Generate response
            response = self.llm.invoke(question, context)
            
            return {
                "response": response,
                "sources": docs,
                "context": context
            }
            
        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}

def get_available_models(api_key: str = ""):
    """Get available models from OpenRouter API or fallback"""
    return Config.fetch_available_models(api_key)

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = UniversityRAGSystem()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    
    # Header
    st.title("🎓 University AI Assistant")
    st.markdown("*Ask questions about your course materials*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Use API key from environment
        api_key = Config.OPENROUTER_API_KEY
        
        # Model selection
        models = get_available_models(api_key)
        
        if not models:
            st.error("Unable to load models. Using fallback models.")
            models = Config.FALLBACK_MODELS
        
        selected_model_name = st.selectbox(
            "AI Model",
            options=list(models.keys()),
            help="All models are free to use - no credits required!"
        )
        
        selected_model = models[selected_model_name]
        
        # Show refresh button
        if st.button("🔄 Refresh Models", use_container_width=True):
            st.rerun()
        
        # System status
        st.divider()
        if st.session_state.system_ready:
            st.success("✅ System Ready")
            # Show simple stats
            try:
                doc_count = st.session_state.rag_system.vectordb.count()
                st.info(f"📚 {doc_count} documents loaded")
            except:
                st.info("📚 Documents loaded")
        else:
            st.warning("⚠️ Setting up system...")
        
        # Advanced settings (collapsible)
        with st.expander("🔧 Advanced Settings"):
            chunk_size = st.slider("Chunk Size", 200, 1000, Config.DEFAULT_CHUNK_SIZE)
            chunk_overlap = st.slider("Chunk Overlap", 0, 200, Config.DEFAULT_CHUNK_OVERLAP)
            retrieval_k = st.slider("Retrieved Chunks", 1, 10, Config.DEFAULT_RETRIEVAL_K)
            
            if st.button("🔄 Rebuild Database", help="Clear and rebuild the vector database"):
                if os.path.exists(Config.DB_PERSIST_DIRECTORY):
                    import shutil
                    shutil.rmtree(Config.DB_PERSIST_DIRECTORY)
                    st.success("Database cleared. Restart to rebuild.")
                    st.rerun()
    
    # Main content area
    # System initialization
    if not st.session_state.system_ready:
        st.info("🚀 Setting up your AI assistant...")
        
        if not api_key:
            st.error("OpenRouter API key not configured. Please check your .env file.")
            st.info("💡 Add OPENROUTER_API_KEY=your_key_here to your .env file")
            return
        
        # Setup LLM
        st.session_state.rag_system.setup_llm(api_key, selected_model)
        
        # Load or build vector store
        if os.path.exists(Config.DB_PERSIST_DIRECTORY):
            with st.spinner("Loading your documents..."):
                if st.session_state.rag_system.load_vector_store():
                    st.session_state.system_ready = True
                    st.success("✅ Ready! You can now ask questions about your course materials.")
                    st.rerun()
        else:
            if os.path.exists(Config.PDF_FOLDER):
                with st.spinner("Processing your PDF files... This may take a moment."):
                    docs = st.session_state.rag_system.load_all_pdfs(Config.PDF_FOLDER)
                    if docs:
                        st.session_state.rag_system.build_vector_store(docs, chunk_size, chunk_overlap)
                        st.session_state.system_ready = True
                        st.success("✅ Setup complete! You can now ask questions.")
                        st.rerun()
                    else:
                        st.error(f"No PDF documents found in {Config.PDF_FOLDER} folder")
            else:
                st.error(f"Please create a {Config.PDF_FOLDER} folder and add your PDF files.")
    
    else:
        # Chat interface
        st.markdown("## 💬 Ask me anything about your course materials")
        
        # Query input at the top
        with st.form("query_form", clear_on_submit=True):
            question = st.text_area(
                "Your Question",
                placeholder="e.g., What are the main principles of software requirements engineering?",
                height=80,
                label_visibility="collapsed"
            )
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                submit_button = st.form_submit_button("🚀 Ask Question", use_container_width=True, type="primary")
            with col2:
                clear_button = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)
            with col3:
                if st.form_submit_button("📥 Export", use_container_width=True) and st.session_state.chat_history:
                    chat_export = []
                    for q, a, _ in st.session_state.chat_history:
                        chat_export.append({"question": q, "answer": a, "timestamp": datetime.now().isoformat()})
                    
                    st.download_button(
                        "Download",
                        data=json.dumps(chat_export, indent=2),
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        if submit_button and question.strip():
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.rag_system.query(question, retrieval_k)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.chat_history.append((
                        question,
                        result["response"],
                        result["sources"]
                    ))
                    st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            for i, (question, answer, sources) in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    st.markdown(f"**❓ You:** {question}")
                    st.markdown(f"**🤖 Assistant:** {answer}")
                    
                    if sources:
                        with st.expander(f"📚 View sources ({len(sources)} documents)", expanded=False):
                            for j, source in enumerate(sources):
                                st.markdown(f"**📄 {source.metadata.get('source_file', 'Unknown')}**")
                                st.markdown(f"```\n{source.page_content[:300]}...\n```")
                    
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("---")
        else:
            st.markdown("👋 **Welcome!** Ask me anything about your course materials to get started.")

if __name__ == "__main__":
    main()
