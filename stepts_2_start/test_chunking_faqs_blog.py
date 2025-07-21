import os
from dotenv import load_dotenv
os.environ['USER_AGENT'] = 'myagent'
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#load the source (FAQs)
def load_docs():
    loader = WebBaseLoader(["https://chameleoncloud.org/learn/frequently-asked-questions/", 
                            "https://chameleoncloud.org/blog/2025/07/01/chameleon-changelog-for-june-2025/"])
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
    print(f"Number of chunks: {len(chunks)}")
    database = create_vectorstore(chunks)
    
    #Using a similarity search to see the top three matches.
    """ query = "What are Chameleon changelog for June 2025?"
    results = database.similarity_search(query, k=3)

    for i, res in enumerate(results):
        print(f"\n======================================= Match {i+1} ============================================")
        print(res.page_content)
        print(f"Metadata: {res.metadata}")  """





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
    print("chunk 68: \n")
    print(chunks[67].page_content)
    print("--------------------------------------------------------------------------------------------")
    print("chunk 69: \n")
    print(chunks[68].page_content)
    print("--------------------------------------------------------------------------------------------")
    print("chunk 70: \n")
    print(chunks[69].page_content)
    print("--------------------------------------------------------------------------------------------")
    print("chunk 71: \n")
    print(chunks[70].page_content)


if __name__ == "__main__":
    main()
