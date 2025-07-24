import os
import shutil
#hide TensorFlow/XLA-related warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dotenv import load_dotenv
os.environ['USER_AGENT'] = 'myagent'
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from loader import loader_docs


#load huggingface token from .env
def load_token():
    load_dotenv()
    api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not api_key: 
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")
    return api_key


#load the document
loader_docs()

#splitting the texts
def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 400, 
            separators= ["\n### ", "\n#### ", "\n", " ", ""]
    )
    return text_splitter.split_documents(docs)

#creating vectorstore using huggingface embedding model
def create_vectorstore(chunks, save_path="vect_store_faiss"):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print(f'removed the old vectorstore at {save_path}')

    embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
    vectorstore = FAISS.from_documents(documents= chunks, embedding= embeddings_model)
    vectorstore.save_local(save_path)
    print(f'new vectorstore saved at {save_path}')

    return vectorstore

def create_llm_chain():
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-0528",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )

    chat_model = ChatHuggingFace(llm = llm_endpoint)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistance that helps answer the questions about Chameleon Cloud documentations.Use the provided context to answer the question directly and include a citation for the answer from the context provided. 'For example, this information comes from the FAQs and here is the link'.Do not include internal thoughts like '</think>'. IMPORTANT: If the answer is not clearly in the context, say'I don't know' and do not make up the answer. Keep the answer short a maximum of 5 sentences and be percise."), 
        ("user", "Question:{question}\nContext: {context}")
     ])
        
    return prompt | chat_model


def main():
    load_token()
    docs = loader_docs()
    chunks = split_docs(docs)
    print(f"Number of chunks: {len(chunks)}")
    vectorstore = create_vectorstore(chunks)
    retriever = vectorstore.as_retriever()
    embed_model_name = "BAAI/bge-large-en"
    return retriever, embed_model_name
    #chain = create_llm_chain()
    
'''
    print("chunk 30: \n")
    print(chunks[29].page_content)
    print("--------------------------------------------------------------------------------------------")
    print("chunk 31: \n")
    print(chunks[30].page_content)
    print("--------------------------------------------------------------------------------------------")
    print("chunk 32: \n")
    print(chunks[31].page_content)
    print("--------------------------------------------------------------------------------------------")
    

    print("________________________________________________________________________________________________________________")
    print("Type 'exit' to quit.")
    while True: 
        query = input("Please enter a question: ")
        if query == "exit":
            break
        retrieved_docs= retriever.invoke(query)
        #context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        #response = chain.invoke({"question": query, "context": context})
        #print(response.content)
        #print(retrieved_docs)
        for i, doc in enumerate(retrieved_docs):
            print(f"\n================== Match {i+1} ===================")
            print(doc.page_content)
           # print(f"Metadata: {doc.metadata}")
'''

if __name__ == "__main__":
    main()

