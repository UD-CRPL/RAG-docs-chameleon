RAG-docs-chameleon
--------------------------------------------------------------------------------------------------
Overview
--------------------------------------------------------------------------------------------------
This repository provides a primary Retrieval-Augmented Generation (RAG) pipeline for answering questions about Chameleon Cloud documentation. The repo includes files that build a vector store by indexing, ask users questions, and retrieve information from provided sources. 

Prerequisites
--------------------------------------------------------------------------------------------------
1. Operating System: The code is written for Linux operating systems.
2. Python: Ensure you have Python 3.9 or higher installed.
3. Virtual Environment: It is necessary to use a virtual environment (venv) to manage dependencies.

Setup
-------------------------------------------------------------------------------------------------
1. Clone the Repository:

    git clone <repository_url>

    cd RAG-docs-chameleon

2. Create and activate a virtual environment:
 
    python -m venv <env_name>

    source env_name/bin/activate
    
3. Run requirements.txt:

    pip install -r requirements.txt

4. Install Llama 3.1:

    curl -fsSL https://ollama.com/install.sh | sh

    ollama pull llama3.1

    pip install -qU langchain-ollama
    
Execution
-------------------------------------------------------------------------------------------------
Using Command Line Interface (CLI):
You can run rag.py script using the command line:

1. Basic Execusion:
 
    python rag.py

2. Using streamlit:

    pip install streamlit
 
    streamlit run web_rag.py

Notes
-------------------------------------------------------------------------------------------------
We ran rag.py on a Chameleon Cloud KVM-based VM with an NVIDIA H100 GPU.  
