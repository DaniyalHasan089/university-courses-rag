import os
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def load_all_pdfs(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())
    return documents

def build_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="./db")
    vectordb.persist()
    return vectordb

def setup_qa_chain():
    vectordb = Chroma(persist_directory="./db", embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
    llm = Ollama(model="phi3")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(search_kwargs={"k": 3}))
    return qa

qa_chain = None

def chat(query):
    if not query.strip():
        return "Please ask a valid question."
    result = qa_chain({"query": query})
    return result["result"]

def main():
    global qa_chain
    if not os.path.exists("./db"):
        print("Loading and processing PDFs...")
        docs = load_all_pdfs("pdfs")
        build_vector_store(docs)
    qa_chain = setup_qa_chain()

    with gr.Blocks() as demo:
        gr.Markdown("## University Course Assistant (Offline)")
        chatbot = gr.Chatbot()
        with gr.Row():
            msg = gr.Textbox(placeholder="Ask something about your university material...", show_label=False)
            send_btn = gr.Button("Send")

        def respond(user_input, chat_history):
            response = chat(user_input)
            chat_history.append((user_input, response))
            return "", chat_history

        send_btn.click(fn=respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(fn=respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

    demo.launch()

if __name__ == "__main__":
    main()
