import os
import shutil
os.environ['USER_AGENT'] = 'myagent'
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from loader import loader_docs
from langchain_ollama import ChatOllama


#load huggingface token from .env
def load_token():
    api_key = "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"
    if not api_key: 
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")
    return api_key


#load the document
loader_docs()

#splitting the texts
def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1024,
            chunk_overlap = 150, 
            separators= ["\n### ", "\n#### ", "\n", " ", ""]
    )
    return text_splitter.split_documents(docs)



#creating vectorstore using huggingface embedding model
def create_vectorstore(chunks, save_path="vect_store"):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print(f'removed the old vectorstore at {save_path}')

    embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/BAAI/bge-base-en-v1.5")
    vectorstore = FAISS.from_documents(documents= chunks, embedding= embeddings_model)
    vectorstore.save_local(save_path)
    print(f'new vectorstore saved at {save_path}')

    return vectorstore



def create_llm_chain():
    llm = ChatOllama(
        model="llama3.1",
        temperature=0
)

    chat_model = llm

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that helps answer the questions about Chameleon Cloud documentation. Keep the answer short and precise — a maximum of 5 sentences and be precise."),
        ("user", "Question:{question}")
])

    return prompt | chat_model


def main():
    load_token()
    docs = loader_docs()
    chunks = split_docs(docs)
    print(f"Number of chunks: {len(chunks)}")
    vectorstore = create_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'fetch_k': 20})
    embed_model_name = "BAAI/bge-large-en"
    #return retriever, embed_model_name
    chain = create_llm_chain()
    
    print("________________________________________________________________________________________________________________")
    print("Type 'exit' to quit.")
    while True: 
        query = input("\n Please enter a question: ")
        if query == "exit":
            break
        instructional_query = f"A question regarding the Chameleon Cloud testbed: {query}"
        # retrieved_docs= retriever.invoke(instructional_query)
        # context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        response = chain.invoke({"question": query})
        print(response.content)
        #print(retrieved_docs)
        # for i, doc in enumerate(retrieved_docs):
        #     print(f"\n================== Match {i+1} ===================")
        #     print(doc.page_content)
        #     print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    main()

