import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import time 
import tempfile # To store uploaded PDFs on disk temporarily

import streamlit as st # Main streamlit library 
 
 
from dotenv import load_dotenv # to read .env file

# Langchain core classes & utilities
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

## Langchain LLM and chaining utilities
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# Text splitting & embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# vector store
from langchain_community.vectorstores import FAISS 

## PDF file loader (loads a single PDF into docs)
from langchain_community.document_loaders import PyPDFLoader


# Loading environment variables ( HF_TOKEN, GROQ_API_KEY)
load_dotenv()


# Streamlit page setup

st.set_page_config(
    page_title = " 📄 RAG Q&A with PDF & Chat History",
    layout = "wide",
    initial_sidebar_state= "expanded"
)
st.title(" 📄 RAG Q&A with PDF uploades and Chat History")

st.sidebar.header(" ⚙️ Configuration")
st.sidebar.write(
    " - Enter your Groq API Key \n "
    " - Upload PDFs on the main page \n "
    " - Ask questions and see chat history"
)

## API Keys & embedding setup

api_key = st.sidebar.text_input("Groq API Key", type="password")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN","") # for HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# only proceed if the user has entered their Groq Key

if not api_key:
    st.warning(" 🔑 Please enter your Groq API Key in the sidebar to continue. ")
    st.stop()

# Instantiate the Groq LLM
llm = ChatGroq(groq_api_key=api_key, model_name = "gemma2-9b-it")


# File Uploader 
uploaded_files = st.file_uploader(
    "📁 Choose PDF file(s)",
    type="pdf",
    accept_multiple_files=True,
    )

# A placeholder to collect all documents

all_docs= []

if uploaded_files:
    # Show progress spinner while loading
    with st.spinner(" 🔁 Loading and splitting PDFs ..."):
        for pdf in uploaded_files:
            # Write to a temp file so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.getvalue())
                pdf_path = tmp.name

            # Load the pdf into a list of Document objects
            loader = PyPDFLoader(pdf_path)
            docs =loader.load()
            all_docs.extend(docs)

# Split docs into chunks for embedding

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)
splits = text_splitter.split_documents(all_docs)
st.write(f"🔍 Loaded {len(all_docs)} documents, produced {len(splits)} chunks.")
if not splits:
    st.warning("No text chunks found – please upload at least one valid PDF.")
    st.stop()

# Build or load the Chroma vector store (caching for performance)

@st.cache_resource
def get_vectorstore(_splits):
    return FAISS.from_documents(_splits, embeddings)  # no persist_directory

vectorstore = get_vectorstore(splits)
retriever = vectorstore.as_retriever()
vectorstore = get_vectorstore(splits)
retriever = vectorstore.as_retriever()

# Build a history-aware retriever that uses past chat to refine searches

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system","Given the chat history and the latest user question, decide what to retrieve."),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

# QA Chain: "stuff" all retrieved docs into the LLM

qa_prompt  = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant. Use the retrieved context to answer."
                "If you don't know, say so. Keep it under three sentences.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



# Session state for chat history

if "chathistory" not in st.session_state:
    st.session_state.chathistory = {}

def get_history(session_id: str):
   
    if session_id not in st.session_state.chathistory:
        st.session_state.chathistory[session_id] = ChatMessageHistory()
    return st.session_state.chathistory[session_id]

# Wrap the RAG chain so it automatically logs history

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key = "input",
    history_messages_key = "chat_history",
    output_messages_key= "answer",
)

# Chat UI

session_id = st.text_input(" 🆔 Session ID",value="default_session")
user_question =  st.chat_input("✍🏻  You question here...")

if user_question:
    history = get_history(session_id)
    result =  conversational_rag.invoke(
        {"input": user_question},
        config = {"configurable": {"session_id": session_id}},
    )
    answer = result["answer"]

    # Display in streamlit new chat format

    st.chat_message("user").write(user_question)
    st.chat_message("assistant").write(answer)

    # show full history below

    with st.expander(" 📖 Full chat history"):
        for msg in history.messages:
            role = getattr(msg, "role",msg.type) #human and assitant
            content = msg.content
            st.write(f"**{role.title()}: ** {content}")
else:
    st.info("ℹ️ Upload one or more PDFs above to begin.")
