import os
import shutil
from dotenv import load_dotenv
os.environ['USER_AGENT'] = 'myagent'
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loader import loader_docs

load_dotenv()

TEJAS_API_BASE = "https://ai.tejas.tacc.utexas.edu/v1"
TEJAS_API_KEY = os.environ.get("TEJAS_API_KEY")
EMBED_MODEL = "E5-Mistral-7B-Instruct"
VECT_STORE_PATH = "vect_store"


def get_embeddings_model():
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=TEJAS_API_KEY,
        openai_api_base=TEJAS_API_BASE,
        check_embedding_ctx_length=False,
        chunk_size=16,
    )


#splitting the texts
def split_docs(docs, chunk_size=512, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n### ", "\n#### ", "\n", " ", ""]
    )
    return text_splitter.split_documents(docs)


#creating vectorstore using huggingface embedding model
def create_vectorstore(chunks, save_path=VECT_STORE_PATH):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    vectorstore = FAISS.from_documents(documents=chunks, embedding=get_embeddings_model())
    vectorstore.save_local(save_path)
    return vectorstore


def load_vectorstore(save_path=VECT_STORE_PATH, k=6, fetch_k=20, search_type='mmr'):
    vectorstore = FAISS.load_local(save_path, get_embeddings_model(), allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_type=search_type, search_kwargs={'k': k, 'fetch_k': fetch_k})


def create_llm_chain():
    llm = ChatOpenAI(
        model="Meta-Llama-3.3-70B-Instruct",
        temperature=0,
        openai_api_key=TEJAS_API_KEY,
        openai_api_base=TEJAS_API_BASE,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant that answers questions about Chameleon Cloud based on its official documentation. "
            "Use only the provided context to answer. "
            "For simple questions, give a concise answer. For complex questions, give a thorough, step-by-step answer. "
            "Do not include URLs or source citations in your answer — relevant documentation links will be shown separately. "
            "If the answer is not in the context, say 'I don't know' and do not make up an answer."
        )),
        ("user", "Question: {question}\nContext: {context}")
    ])

    return prompt | llm


def main():
    if not os.path.exists(VECT_STORE_PATH):
        print("No vector store found. Building index from docs...")
        docs = loader_docs()
        chunks = split_docs(docs)
        print(f"Number of chunks: {len(chunks)}")
        create_vectorstore(chunks)

    retriever = load_vectorstore()
    chain = create_llm_chain()

    print("Type 'exit' to quit.")
    while True:
        query = input("\nPlease enter a question: ")
        if query == "exit":
            break
        instructional_query = f"A question regarding the Chameleon Cloud testbed: {query}"
        retrieved_docs = retriever.invoke(instructional_query)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        response = chain.invoke({"question": query, "context": context})
        print(response.content)
        for i, doc in enumerate(retrieved_docs):
            print(f"\n================== Match {i+1} ===================")
            print(doc.page_content)
            print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    main()
