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

    embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
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
    system_prompt = """
        ## ROLE ##
        You are an expert Q&A assistant for Chameleon Cloud, a testbed for computer science research.

        ## TASK ##
        Your primary goal is to provide a comprehensive and helpful answer by synthesizing information from ALL relevant context sources provided. You must accurately interpret the user's intent to deliver the most useful response.

        ## INSTRUCTIONS ##
        - First, understand the user's question to determine their underlying intent (e.g., are they asking for a definition, a step-by-step guide, or troubleshooting help?).
        - Scrutinize all provided context sources to gather relevant information.
        - Synthesize a single, cohesive answer from the different sources. Do not simply list information from each source separately.
        - If the answer is not present in the context, you MUST respond with the single phrase: "I don't know."
        - Do not use any information outside of the provided context. Do not make up answers.
        - After your answer, list all the sources you used to construct it.

        ## OUTPUT FORMAT ##
        <A comprehensive, synthesized answer that directly addresses the user's intent.>

        ---
        ### Read More:
        * **[Title of Source 1]:** <URL from metadata>
        * **[Title of Source 2]:** <URL from metadata>
        * **[Title of Source n]:** <URL from metadata>

        {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
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
