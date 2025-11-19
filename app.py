import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
# Removed deprecated load_qa_chain import
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from PIL import Image
import pandas as pd

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Page configuration
st.set_page_config(
    page_title="Multi-Modal AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

def extract_text_from_pdf(pdf_files):
    """Extract text from uploaded PDF files"""
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_csv(csv_files):
    """Extract text from uploaded CSV files"""
    text = ""
    for csv in csv_files:
        df = pd.read_csv(csv)
        text += df.to_string() + "\n\n"
        text += f"CSV Summary:\n"
        text += f"Columns: {', '.join(df.columns.tolist())}\n"
        text += f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
    return text

def get_text_chunks(text):
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    """Create FAISS vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def query_with_context(question, context_docs):
    """Query documents with context using Gemini directly"""
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    
    # Combine document content
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say "I cannot find the answer in the uploaded documents."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    response = model.invoke(prompt)
    return response.content

def process_image_query(image_file, query):
    """Process image with Gemini Vision"""
    image = Image.open(image_file)
    model = genai.GenerativeModel('gemini-pro-vision')
    
    if query:
        response = model.generate_content([query, image])
    else:
        response = model.generate_content(["Describe this image in detail.", image])
    
    return response.text

def handle_document_query(user_question):
    """Handle queries on uploaded documents"""
    if st.session_state.vector_store is None:
        return "Please upload and process documents first."
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    # Load vector store
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Search for relevant documents
    docs = vector_store.similarity_search(user_question, k=3)
    
    # Get answer using our custom function
    response = query_with_context(user_question, docs)
    
    return response

def main():
    st.title("🤖 Multi-Modal AI Chatbot with LangChain & FAISS")
    st.markdown("Upload PDFs, CSVs, Images and chat with your documents using Gemini AI + Vector Search")
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 Upload Files")
        
        # PDF Upload
        pdf_files = st.file_uploader(
            "Upload PDF Files",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        # CSV Upload
        csv_files = st.file_uploader(
            "Upload CSV Files",
            type=["csv"],
            accept_multiple_files=True
        )
        
        # Image Upload
        image_files = st.file_uploader(
            "Upload Image Files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        # Process documents button
        if st.button("🔄 Process Documents"):
            if pdf_files or csv_files:
                with st.spinner("Processing documents..."):
                    raw_text = ""
                    
                    # Extract text from PDFs
                    if pdf_files:
                        raw_text += extract_text_from_pdf(pdf_files)
                        st.session_state.processed_files.extend([f.name for f in pdf_files])
                    
                    # Extract text from CSVs
                    if csv_files:
                        raw_text += extract_text_from_csv(csv_files)
                        st.session_state.processed_files.extend([f.name for f in csv_files])
                    
                    # Create chunks and vector store
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = create_vector_store(text_chunks)
                    
                    st.success("✅ Documents processed successfully!")
                    st.info(f"Created {len(text_chunks)} text chunks in FAISS vector store")
            else:
                st.warning("Please upload at least one document.")
        
        # Show processed files
        if st.session_state.processed_files:
            st.subheader("Processed Files:")
            for file in st.session_state.processed_files:
                st.text(f"📄 {file}")
        
        # Clear history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    st.subheader("💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "image" in message:
                st.image(message["image"], width=300)
    
    # Chat input
    user_input = st.chat_input("Ask a question about your documents or images...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process based on uploaded files
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ""
                
                # Handle image queries
                if image_files:
                    for img in image_files:
                        img_response = process_image_query(img, user_input)
                        response += f"\n\n**Analysis of {img.name}:**\n{img_response}"
                        
                        # Display image in chat
                        st.image(img, width=300, caption=img.name)
                
                # Handle document queries using FAISS
                if (pdf_files or csv_files) and st.session_state.vector_store:
                    doc_response = handle_document_query(user_input)
                    if response:
                        response += f"\n\n**Document Analysis:**\n{doc_response}"
                    else:
                        response = doc_response
                
                # If no files uploaded, use general Gemini chat
                if not response:
                    model = genai.GenerativeModel('gemini-1.5-pro')
                    chat_response = model.generate_content(user_input)
                    response = chat_response.text
                
                st.markdown(response)
                
                # Add assistant response to chat
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
    
    # Information section
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Upload Files**: Use the sidebar to upload PDF, CSV, or image files
        2. **Process Documents**: Click 'Process Documents' to create FAISS vector embeddings
        3. **Ask Questions**: Type your questions in the chat input
        4. **Get Answers**: The AI will search the vector store and provide detailed answers
        
        **Features:**
        - 📄 PDF text extraction and Q&A
        - 📊 CSV data analysis
        - 🖼️ Image analysis with Gemini Vision
        - 💾 FAISS vector storage for semantic search
        - 🔗 LangChain integration for advanced Q&A
        - 💬 Conversational chat interface
        
        **Technology Stack:**
        - LangChain for document processing
        - FAISS for vector similarity search
        - Google Gemini AI for responses
        - Streamlit for the UI
        """)

if __name__ == "__main__":
    main()