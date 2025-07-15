import os
from dotenv import load_dotenv
os.environ['USER_AGENT'] = 'myagent'
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

#load huggingface token from .env
def load_token():
    load_dotenv()
    api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not api_key: 
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")
    return api_key


#load the source (FAQs)
def load_docs():
    loader = WebBaseLoader(["https://chameleoncloud.org/learn/frequently-asked-questions/"])
    return loader.load()

#splitting the texts
def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 200, 
            separators= ["\n# ", "\n## ", "\n### ", "\n#### ", "\n", " ", ""]
    )
    return text_splitter.split_documents(docs)

#creating vectorstore using huggingface embedding model
def create_vectorestore(chunks):
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents= chunks, embedding= embeddings_model)


def create_llm_chain():
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-0528",
        task="conversational",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto",  # or explicitly set "fireworks-ai" if needed
    )

    chat_model = ChatHuggingFace(llm = llm_endpoint)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistance that helps answer the questions about Chameleon Cloud documentations.Use the provided context to answer the questions. IMPORTANT: If you are unsure of the answer, say'I don't know' and do not make up the answer. Keep the answer short a maximum of 5 sentences and be percise."), 
        ("user", "Question:{question}\nContext: {context}")
     ])
        
    return prompt | chat_model


def main():
    load_token()
    docs = load_docs()
    chunks = split_docs(docs)
    print(f"Number of chunks: {len(chunks)}")
    vectorstore = create_vectorestore(chunks)
    retriever = vectorstore.as_retriever()
    chain = create_llm_chain()

    print("________________________________________________________________________________________________________________")
    print("Type 'exit' to quit.")
    while True: 
        query = input("Please enter a question: ")
        if query == "exit":
            break
        retrieved_docs= retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        response = chain.invoke({"question": query, "context": context})
        print(response.content)

if __name__ == "__main__":
    main()

