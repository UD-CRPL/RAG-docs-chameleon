import os
from dotenv import load_dotenv
os.environ['USER_AGENT'] = 'myagent'
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#load the source (FAQs)
def load_docs():
    loader = WebBaseLoader(["https://chameleoncloud.org/learn/frequently-asked-questions/"])
    return loader.load()



#splitting the texts
def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 900,
            chunk_overlap = 200,
            separators= ["\n## ", "\n### ", "\n#### ", "\n", " ", ""]
    )
    return  text_splitter.split_documents(docs)


#creating vectorstore
def create_vectorstore(chunks):
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents= chunks, embedding= embeddings_model)



def main():
    docs = load_docs()
    #print(docs)
    chunks = split_docs(docs)
    #print(chunks)
    create_vectorstore(chunks)
    print(create_vectorstore)
"""
    for chunk in chunks:
        print(f"chunk {chunks.index(chunk)} size: {len(chunk.page_content)}\n")

    print("chunk 1: \n")
    print(chunks[0].page_content)
    print("--------------------------------------------------------------------------------------------")
    print("chunk 2: \n")
    print(chunks[1].page_content)
    print("--------------------------------------------------------------------------------------------")
    print("chunk 3: \n")
    print(chunks[2].page_content)
    print("--------------------------------------------------------------------------------------------")
    print("chunk 4: \n")
    print(chunks[3].page_content)
    print("--------------------------------------------------------------------------------------------")
    print("chunk 5: \n")
    print(chunks[4].page_content)
    print("--------------------------------------------------------------------------------------------")
    print("chunk 6: \n")
    print(chunks[5].page_content)
    print("--------------------------------------------------------------------------------------------")
    print("chunk 7: \n")
    print(chunks[6].page_content)
    print("--------------------------------------------------------------------------------------------")
    print("chunk 8: \n")
    print(chunks[7].page_content)
    print("--------------------------------------------------------------------------------------------")
    print("chunk 9: \n")
    print(chunks[8].page_content)
    print("chunk 1: \n")
    print("--------------------------------------------------------------------------------------------")
    print("chunk 10: \n")
    print(chunks[9].page_content)
    print("--------------------------------------------------------------------------------------------")
    print("chunk 11: \n")
    print(chunks[10].page_content)

"""
if __name__ == "__main__":
    main()
