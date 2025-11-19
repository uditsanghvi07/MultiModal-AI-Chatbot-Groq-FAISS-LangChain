# 🧠 Multi-Modal AI Chatbot using Groq, FAISS, LangChain & Streamlit
This project is a powerful and flexible multi-modal AI assistant that allows users to interact with multiple data types — PDF documents, CSV files, and images — using a unified conversational interface. Built with Groq LLaMA 3.3 70B, FAISS vector search, and HuggingFace embeddings, the system supports fast semantic search, document-aware question-answering, and general conversational AI.

The chatbot extracts knowledge from uploaded documents, converts them into meaningful text chunks, embeds them using sentence-transformers/all-MiniLM-L6-v2, and stores them locally in FAISS for efficient similarity search. When a user asks a question, the system retrieves the most relevant context chunks and generates an accurate, context-aware answer using Groq’s high-performance LLMs. This approach combines the stability of local vector search with the intelligence of state-of-the-art large language models.

Alongside document analysis, the chatbot also supports image uploads. Although current Groq vision models provide limited native image processing, the system accepts image files, integrates them into the chat flow, and uses fallback text-based responses to maintain smooth interaction. When no documents or images are provided, the assistant seamlessly switches to a general conversational mode powered by Groq.

The entire application is built using Streamlit, providing a clean and interactive chat interface, sidebar file uploads, document processing controls, and persistent chat history. It is lightweight, easy to deploy, and suitable for personal, academic, or enterprise document-analysis workflows.

This project showcases a modern RAG (Retrieval-Augmented Generation) pipeline using free and fast components, making it ideal for data exploration, summarization, multi-file analysis, and conversational insights.


<img width="1920" height="1080" alt="Screenshot (875)" src="https://github.com/user-attachments/assets/c8bb4d46-c5f3-4e54-9d36-a2239715a2f4" />


⭐ Key Features


PDF extraction and detailed Q&A

CSV data summarization and semantic search

Image upload with fallback Groq text reasoning

FAISS vector database for fast local retrieval

Multi-modal chat interface with memory

Built using Groq LLaMA 3.3 70B and HuggingFace embeddings

Fully powered by LangChain and Streamlit
