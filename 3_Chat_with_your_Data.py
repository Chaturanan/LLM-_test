"""
Chat with your Data

Upload PDF documents and ask questions about them. The AI will search through
your documents and provide answers based on the content.
"""


# =========================================================
# IMPORTS (Libraries we need)
# =========================================================

# Streamlit: Framework for building web apps with Python
import streamlit as st

# os: For file operations and environment variables
import os

# ChatOpenAI: Connects to OpenAI's GPT models (like ChatGPT)
# OpenAIEmbeddings: Converts text to vectors (numbers) for similarity search
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ChatPromptTemplate: Template for formatting messages to the AI
from langchain_core.prompts import ChatPromptTemplate

# PyPDFLoader: Loads and reads PDF files
from langchain_community.document_loaders import PyPDFLoader

# FAISS: Fast similarity search database (stores document chunks as vectors)
from langchain_community.vectorstores import FAISS

# RecursiveCharacterTextSplitter: Splits long documents into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Chat with Documents",
    page_icon="📚",
    layout="wide"  # Use full width of browser
)

st.title("📚 Chat with your Data")
st.caption("Ask questions about your PDF documents")


# =========================================================
# SESSION STATE
# =========================================================

if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""  # Store OpenAI API key

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # Store document database

if "llm" not in st.session_state:
    st.session_state.llm = None  # Store language model instance

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []  # Store chat history

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []  # Track which files we've processed


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.subheader("🔑 API Keys")
    
    if st.session_state.openai_key:
        st.success("✅ OpenAI Connected")
        if st.button("Change API Keys"):
            # Reset everything to start fresh
            st.session_state.openai_key = ""
            st.session_state.vector_store = None
            st.session_state.rag_messages = []
            st.rerun()
    else:
        st.warning("⚠️ Not Connected")


# =========================================================
# API KEY INPUT
# =========================================================

if not st.session_state.openai_key:
    # Show input form for API key
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",  # Hide the key as user types
        placeholder="sk-proj-..."
    )
    
    if st.button("Connect"):
        # Validate key format
        if api_key and api_key.startswith("sk-"):
            st.session_state.openai_key = api_key
            st.rerun()
        else:
            st.error("❌ Invalid API key format")
    
    st.stop()  # Don't show rest of app until connected


# =========================================================
# PDF UPLOAD AND PROCESSING
# =========================================================

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True  # Allow multiple PDFs
)

if uploaded_files:
    current_files = [f.name for f in uploaded_files]
    
    # Check if we need to process (avoid reprocessing on every rerun)
    if st.session_state.processed_files != current_files:
        
        with st.spinner("Processing documents..."):
            # Load PDFs
            documents = []
            os.makedirs("tmp", exist_ok=True)
            
            for file in uploaded_files:
                # Save to temporary file (PyPDFLoader needs file path)
                file_path = os.path.join("tmp", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                
                # Load PDF
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            
            # Split into chunks (long documents → smaller pieces)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,      # Each chunk max 1500 characters
                chunk_overlap=200     # 200 character overlap between chunks
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create vector store (searchable database)
            # Converts text chunks to vectors (numbers) for similarity search
            embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_key)
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
            
            # Create language model
            st.session_state.llm = ChatOpenAI(
                model="gpt-4o-mini",  # Use GPT-4o-mini (fast and cheap)
                temperature=0,  # 0 = deterministic, 1 = creative
                api_key=st.session_state.openai_key
            )
            
            # Reset chat and update processed files
            st.session_state.rag_messages = []
            st.session_state.processed_files = current_files
            
            st.success(f"✅ Processed {len(uploaded_files)} document(s)!")


# =========================================================
# CHAT INTERFACE
# =========================================================

if st.session_state.vector_store:
    
    # Display chat history
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Handle user input
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        # Save user message
        st.session_state.rag_messages.append({
            "role": "user",
            "content": user_input
        })
        
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                # Retrieve relevant documents
                retriever = st.session_state.vector_store.as_retriever()
                docs = retriever.invoke(user_input)
                
                # Combine documents into context
                context = "\n\n---\n\n".join(doc.page_content for doc in docs[:5])
                
                # Generate answer
                if not context.strip():
                    response_text = "I couldn't find relevant information in the documents."
                else:
                    # Create prompt for answering
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "Answer the question using ONLY the provided context. Be concise and accurate."),
                        ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:")
                    ])
                    
                    # Get answer from AI
                    response = st.session_state.llm.invoke(
                        prompt.format_messages(question=user_input, context=context)
                    )
                    
                    response_text = response.content
                
                st.write(response_text)
                
                # Save assistant response
                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": response_text
                })

else:
    st.info("📄 Please upload PDF documents to start chatting.")
