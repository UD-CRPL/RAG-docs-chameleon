This repository provides a primary Retrieval-Augmented Generation (RAG) pipeline for answering questions about Chameleon Cloud documentation. The repo includes files that build a vector store by indexing, ask users questions, and retrieve information from provided sources. 

Prerequisites
----------
1. Operating System: The code is written for Linux operating systems.
2. Python: Ensure you have Python 3.11 or higher installed.
3. Virtual Environment: It is necessary to use a virtual environment (venv) to manage dependencies.

Setup
-----
1. Clone the Repository:

    ```
        git clone <repository_url>
    ```

2. Create and activate a virtual environment using Conda:

    ``` 
        # Download and install Conda
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh
        
        # Create a new environment named 'rag' with Python 3.11
        conda create --name rag python=3.11

        # Activate your new environment
        conda activate rag
    ```

3. Install `faiss-gpu` from the pytorch channel:

    
    ```
        conda install -c pytorch faiss-gpu
    ```

4. Run requirements.txt:

    ```
        pip install -r requirements.txt
    ```

5. Install Llama 3.1:

    ```
        curl -fsSL https://ollama.com/install.sh | sh

        sudo systemctl start ollama

        ollama pull llama3.1
    ```

6. Add a `.env` file in the root directory with the following content:

    ```
    HUGGINGFACE_API_KEY=<your_huggingface_api_key>
    ```

Execution
---------

You can run rag.py script using the command line:

1. Basic Execusion:

``` 
    python rag.py
```

2. Using streamlit:

```
    pip install streamlit
 
    streamlit run web_rag.py
```

Hardware Requirements
---------------------

We ran rag.py on a Chameleon Cloud KVM-based VM with one NVIDIA H100 GPU.  
