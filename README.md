# RAG-Based-Insurance-Policy-Assistant-using-Gemini_Gen-AI
A free, local, and secure RAG application built using Streamlit, LangChain, Chroma DB, and Meta Llama 3.2 via Ollama.

Here is a summary of your completed **RAG-based Insurance Policy Assistant** project, covering its architecture, components, and implementation features.

---

### Project Summary

* **Project Title:** RAG-Based Insurance Policy Assistant.

* **Domain:** Insurance / InsurTech / Customer Support Analytics.

* **Core Technology Stack:** Python, Streamlit, LangChain, Chroma DB, HuggingFace Embeddings, and Meta Llama 3.2 via Ollama.

* **Deployment Model:** 100% Free, Local, and Offline (No external cloud API keys required).

---

### System Architecture & Pipeline Steps

1. **Data Ingestion:** Loads insurance policy documents, FAQs, and clause details from a local JSON dataset (`knowledge_base (1).json`) using `pandas` and `pathlib`.


2. **Text Chunking & Document Wrapping:** Wraps structured JSON entries into standard LangChain `Document` objects containing both text content and metadata attributes (such as `doc_id`, `title`, and `category`).


3. **Local Embedding Generation:** Converts policy text into 384-dimensional dense vector embeddings using the open-source `sentence-transformers/all-MiniLM-L6-v2` model locally.


4. **Vector Database Storage:** Stores vector representations and metadata in a persistent local Chroma DB database (`./chroma_store`).


5. **Context Retrieval:** Uses semantic similarity search to fetch the top 3 (`k=3`) most relevant policy clauses when a user asks a question.


6. **Grounded Response Generation:** Passes retrieved policy context into a local `llama3.2` model via Ollama using a strict prompt template that prevents hallucinations.


7. **User Interface:** Renders a responsive Streamlit dashboard featuring system status badges, interactive 2-column FAQ shortcut buttons, and a free-form chat input interface.



---

### Project Deliverables Achieved

* **Fully Functional RAG Pipeline:** Successfully implemented end-to-end local document search and contextual question answering.


* **Clean & Modern UI:** Built with custom layouts, status check indicators, and interactive FAQ cards.
* **Cost & Security Compliance:** Ensures policy data remains completely local on your computer with zero third-party tracking or cloud token costs.


### GitHub Repository Summary

**Project Name:** RAG-Based Insurance Policy Assistant

**Repository Goal:** Provide a free, local, and secure Retrieval-Augmented Generation (RAG) system to analyze insurance policy documents and answer customer queries using open-source models without external API keys.

#### Key Features

* **100% Offline & Private:** Runs entirely on local hardware using Ollama and local HuggingFace embedding models, ensuring zero data leakage and no cloud API costs.


* **Local Vector Store:** Uses Chroma DB to persist text embeddings locally for fast similarity search.
* **Grounded Answers:** Employs strict prompt engineering to answer strictly from retrieved policy context and prevent hallucinations.
* **Interactive UI:** Built with Streamlit, featuring local system status badges, shortcut buttons for common FAQs, and a real-time chat input interface.

Here is your consolidated, professional package containing a complete **GitHub README.md** ready to copy/paste, along with a targeted **Interview Talking Points & Q&A guide** tailored for data science and AI roles.

---

# Part 1: Production-Ready `README.md` (For GitHub)

```markdown
# 🛡️ RAG-Based Insurance Policy Assistant

An end-to-end, local, private, and zero-cost Retrieval-Augmented Generation (RAG) system built to parse complex insurance policy documents, retrieve clause-level context, and generate precise, grounded answers. 

Built using **Streamlit**, **LangChain (LCEL)**, **Chroma DB**, **HuggingFace Embeddings**, and **Meta Llama 3.2 via Ollama**.

---

## 📌 Project Overview & Domain

* **Domain:** InsurTech / Insurance / Customer Support Analytics
* **Problem:** Insurance policy documents contain complex legal language, strict coverage exclusions, and multi-tier waiting periods. Customers and support agents often struggle to extract accurate clause-level answers quickly.
* **Solution:** A privacy-compliant local RAG engine that ingests policy documents into an in-memory vector store, retrieves relevant clause chunks via dense semantic similarity search, and synthesizes human-readable answers strictly backed by policy context.

---

## 🏗️ System Architecture & Pipeline


```

[ Local Policy JSON ]
│
▼
[ Chunking & Metadata Enrichment (LangChain Document Objects) ]
│
▼
[ HuggingFace Embeddings (all-MiniLM-L6-v2 - 384d Vectors) ]
│
▼
[ Chroma DB Vector Store (Persistent Local Directory ./chroma_store) ]
│
▼
[ Dense Vector Similarity Search (Top-k Retrieval: k=3) ]
│
▼
[ Grounded System Prompt Guardrails (Zero Hallucination Constraint) ]
│
▼
[ Meta Llama 3.2 LLM (Local Execution via Ollama) ]
│
▼
[ Interactive UI Dashboard (Streamlit Frontend + FAQ Buttons) ]

```

---

## 🚀 Key Features

* **100% Privacy & Zero API Cost:** Processes all data, vector embeddings, and LLM inferences locally without sending sensitive policy metrics or user queries to third-party cloud APIs.
* **Grounded Generation (Anti-Hallucination):** Uses strict prompt guardrails requiring the assistant to explicitly reply with *"That information is not available in the provided policy documents."* when context is missing.
* **Streamlit UI Layout:** Features real-time status badges, interactive one-click FAQ buttons, and a chat interface.
* **Persistent Vector Indexing:** Stores document embeddings in `./chroma_store` to eliminate re-indexing latency on application restarts.

---

## 🛠️ Technology Stack

| Layer | Component / Tool | Usage |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Chat interface, status badges, and interactive layout |
| **Framework** | LangChain (LCEL) | End-to-end orchestration chain (`Retriever \| Prompt \| LLM \| Parser`) |
| **Embedding Engine** | HuggingFace (`all-MiniLM-L6-v2`) | Generates 384-dimensional dense semantic vectors |
| **Vector Store** | Chroma DB | Local vector indexing and cosine/L2 distance search |
| **Generator LLM** | Meta Llama 3.2 (via Ollama) | Local open-weights model for response synthesis |
| **Data Processing** | Pandas, Pathlib | Ingests and parses JSON policy documents |

---

## 📋 Prerequisites & Local Setup

### 1. Install & Launch Ollama
Download and install Ollama from [ollama.com](https://ollama.com).

Open your terminal and pull the Llama 3.2 model:
```bash
ollama pull llama3.2

```

---

### 2. Repository Setup & Environment Activation

Clone this repository and create a Python virtual environment:

```bash
git clone [https://github.com/YOUR_USERNAME/insurance-policy-assistant-rag.git](https://github.com/YOUR_USERNAME/insurance-policy-assistant-rag.git)
cd insurance-policy-assistant-rag

python3 -m venv .venv
source .venv/bin/activate

```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

---

### 4. Run the Streamlit Application

```bash
streamlit run "Test 4.py"

```

---

## 📂 Project Structure

```text
├── chroma_store/             # Local persistent Chroma vector database
├── knowledge_base (1).json   # Raw insurance policy documents & FAQ dataset
├── Test 4.py                # Main Streamlit app and LCEL RAG pipeline logic
├── requirements.txt          # Project python dependencies
├── .gitignore                # Environment and cache ignore configuration
└── README.md                 # Project documentation

```

```

---

# Part 2: Interview Playbook (Technical Defense & Q&A)

### 1. Elevator Pitch (How to introduce this project in 60 seconds)

> "I developed a privacy-focused, zero-cloud-cost **Retrieval-Augmented Generation (RAG) assistant** designed for the insurance domain to query policy documents in plain English. 
> 
> To solve data privacy and API cost concerns, I built an end-to-end pipeline running entirely on local hardware. The system ingests policy documents, converts them into 384-dimensional dense vectors using **HuggingFace's `all-MiniLM-L6-v2`**, indexes them into **Chroma DB**, and retrieves context for user queries. The context is passed to a local **Meta Llama 3.2 model running via Ollama**. I also configured strict prompt guardrails to eliminate hallucinations when policy coverage details are missing, and built the application using **Streamlit**."

---

### 2. Strategic Answers to Common Interview Questions

#### Q1: Why did you choose a local RAG pipeline over cloud APIs like Google Gemini or OpenAI?
* **Answer:** In commercial insurance and financial analytics, customer queries and policy documents often contain proprietary rules or PII. Running locally via **Ollama** and **Chroma DB** guarantees total privacy and zero external data exposure. It also eliminates API rate limits, network latency dependencies, and recurring token billing costs.

#### Q2: How did you mitigate LLM hallucinations in insurance domain queries?
* **Answer:** I used a two-pronged strategy:
  1. **Strict Contextual Prompt Engineering:** I engineered the system prompt to instruct the model to act solely as a retriever-synthesizer and explicitly answer *"That information is not available in the provided policy documents."* if the retrieved top-$k$ context lacks the answer.
  2. **Temperature Control & Top-$k$ Tuning:** Set the LLM temperature to `0.0` for deterministic outputs and restricted retrieval to top-3 ($k=3$) high-relevance chunks.

#### Q3: Why did you select Chroma DB over FAISS or Pinecone?
* **Answer:** Chroma DB offers built-in persistence (`persist_directory="./chroma_store"`), seamless native integration with LangChain's latest ecosystem (`langchain-chroma`), and native metadata filtering capabilities without requiring complex C++ bindings or external cloud infrastructure like Pinecone.

#### Q4: How would you scale this application to handle thousands of multi-page PDF policies in production?
* **Answer:** 
  1. **Document Chunking & Hierarchy:** Transition from raw JSON ingestion to parent-document chunking strategies (e.g., using `RecursiveCharacterTextSplitter` with chunk sizes of ~500–1000 tokens and 100-token overlaps).
  2. **Hybrid Search & Re-ranking:** Implement a hybrid retrieval system combining dense vector search (Chroma) with sparse keyword search (BM25), followed by a **Cross-Encoder re-ranker** (such as `ms-marco-MiniLM-L-6-v2`) to re-order top chunks before feeding them into the LLM context window.
  3. **Async / Microservice Architecture:** Decouple the UI frontend from the vector ingestion engine by deploying Chroma and Ollama on dedicated containerized GPU instances (via Docker/Kubernetes) communicating through FastAPI endpoints.

```
