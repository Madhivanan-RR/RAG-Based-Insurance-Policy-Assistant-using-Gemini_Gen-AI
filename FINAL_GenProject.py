import json
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# Page configuration
st.set_page_config(
    page_title="Insurance Policy Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title with icon
st.markdown("## :material/shield: Insurance Policy Assistant")
st.caption("Free, local, and secure RAG app using Ollama + local embeddings. No API keys required.")


@st.cache_resource(show_spinner="Loading policy data...")
def load_knowledge_df():
    kb_path = Path(__file__).with_name("knowledge_base (1).json")
    with kb_path.open("r", encoding="utf-8") as file:
        df = pd.read_json(file)

    if df.empty:
        raise ValueError("Knowledge base is empty.")

    return df


@st.cache_resource(show_spinner="Building local RAG index...")
def load_rag_chain():
    df = load_knowledge_df()

    docs = [
        Document(
            page_content=row["content"],
            metadata={
                "doc_id": row.get("doc_id", ""),
                "title": row.get("title", ""),
                "category": row.get("category", "general"),
            },
        )
        for _, row in df.iterrows()
    ]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        collection_name="insurance_policy_docs",
        persist_directory="./chroma_store",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        "You are an expert insurance assistant. Use ONLY the policy context below.\n"
        "If the answer is not in the context, say: 'That information is not available in the provided policy documents.'\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )

    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")

    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('doc_id', 'N/A')} - {doc.metadata.get('title', 'Untitled')}]: {doc.page_content}"
            for doc in docs
        )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# Sidebar Configuration
with st.sidebar:
    st.header(":material/settings: Configuration")
    st.divider()
    
    st.subheader("App Status")
    col1, col2 = st.columns(2)
    with col1:
        st.badge("Ollama", icon=":material/check_circle:")
    with col2:
        st.badge("Chroma DB", icon=":material/check_circle:")
    
    st.divider()
    st.subheader("Features")
    st.markdown("""
    :green-badge[Free] :green-badge[Local] :green-badge[Secure]
    
    ✓ No external API keys needed  
    ✓ 100% offline & private  
    ✓ Runs on local Ollama  
    ✓ Vector embeddings locally stored  
    """)
    
    st.divider()
    st.caption("Insurance Policy RAG System v1.0.0")


# Initialize the RAG chain
try:
    chain = load_rag_chain()
except Exception as e:
    st.error(
        "Could not initialize the local model stack. Please make sure Ollama is installed and the model 'llama3.2' is available.\n\n"
        f"Details: {e}"
    )
    st.info(
        "Install Ollama from https://ollama.com and run: 'ollama pull llama3.2'"
    )
    st.stop()


# Main content area
st.divider()

# FAQ Section with quick questions
st.subheader(":material/help: Quick answers - common questions")
st.caption("Click any question below to get an instant answer from the policy documents.")

# Create columns for FAQ buttons
faq_questions = [
    "What is covered under hospitalization?",
    "What are the waiting periods for pre-existing diseases?",
    "How do I file a cashless claim?",
    "What is the grace period for renewal?",
    "Are cosmetic surgeries covered?",
]

# Display FAQ as expandable cards
col1, col2 = st.columns(2)

with col1:
    if st.button(
        f":material/help: {faq_questions[0]}",
        key="faq_1",
        use_container_width=True,
    ):
        st.session_state.faq_selected = faq_questions[0]

    if st.button(
        f":material/help: {faq_questions[1]}",
        key="faq_2",
        use_container_width=True,
    ):
        st.session_state.faq_selected = faq_questions[1]

    if st.button(
        f":material/help: {faq_questions[2]}",
        key="faq_3",
        use_container_width=True,
    ):
        st.session_state.faq_selected = faq_questions[2]

with col2:
    if st.button(
        f":material/help: {faq_questions[3]}",
        key="faq_4",
        use_container_width=True,
    ):
        st.session_state.faq_selected = faq_questions[3]

    if st.button(
        f":material/help: {faq_questions[4]}",
        key="faq_5",
        use_container_width=True,
    ):
        st.session_state.faq_selected = faq_questions[4]

st.divider()

# Chat section header
st.subheader(":material/chat: Ask your own question")
st.caption("Or type your own custom question about the insurance policy.")

# Handle FAQ selection
if "faq_selected" in st.session_state and st.session_state.faq_selected:
    faq_question = st.session_state.faq_selected
    
    st.chat_message("user").write(faq_question)
    
    with st.spinner("Searching the policy documents..."):
        try:
            answer = chain.invoke(faq_question)
            st.chat_message("assistant").write(answer)
            st.session_state.faq_selected = None  # Reset
        except Exception as e:
            st.chat_message("assistant").error(
                f"Local model request failed. Please check Ollama and try again.\n\n{e}"
            )

# Custom question input
question = st.chat_input("Ask a question about your insurance policy...", key="custom_question")
if question:
    st.chat_message("user").write(question)
    with st.spinner("Searching the policy documents..."):
        try:
            answer = chain.invoke(question)
            st.chat_message("assistant").write(answer)
        except Exception as e:
            st.chat_message("assistant").error(
                f"Local model request failed. Please check Ollama and try again.\n\n{e}"
            )

st.divider()
st.caption("All data is processed locally on your machine. No information is sent to external servers.")
