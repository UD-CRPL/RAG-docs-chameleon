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
# New import for the reranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


#load huggingface token from .env
def load_token():
    # It's better to load this from an environment variable for security
    # For example: os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    api_key = "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"
    if not api_key:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    return api_key


#load the document
# This function is called but its return value isn't used in main.
# Assuming it's for pre-loading or caching.
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

    embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
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
        ("system", "You are an assistant that helps answer the questions about Chameleon Cloud documentation. Use the following context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}"),
        ("user", "Question:{question}")
])

    return prompt | chat_model


def main():
    load_token()
    docs = loader_docs()
    chunks = split_docs(docs)
    print(f"Number of chunks: {len(chunks)}")
    vectorstore = create_vectorstore(chunks)

    # Standard retriever
    retriever = vectorstore.as_retriever(search_kwargs={'k': 20}) # Retrieve more documents initially

    # Initialize the reranker model
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
    compressor = CrossEncoderReranker(model=model, top_n=5) # Rerank and get top 5

    # Create the compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    chain = create_llm_chain()

    print("________________________________________________________________________________________________________________")
    print("Type 'exit' to quit.")
    while True:
        query = input("\n Please enter a question: ")
        if query == "exit":
            break
        
        # The instructional query is a good idea for improving retrieval
        instructional_query = f"A question regarding the Chameleon Cloud testbed: {query}"
        
        # Use the compression retriever to get reranked documents
        retrieved_docs = compression_retriever.invoke(instructional_query)
        
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # Pass the context to the chain
        response = chain.invoke({"context": context, "question": query})
        
        print("\nResponse:")
        print(response.content)
        
        print("\n\n--- Retrieved Documents ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n================== Match {i+1} (Score: {doc.metadata.get('relevance_score', 'N/A')}) ===================")
            print(doc.page_content)
            print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    main()
