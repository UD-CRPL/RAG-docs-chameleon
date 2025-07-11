from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = WebBaseLoader(["https://chameleoncloud.readthedocs.io/en/latest/", "https://chameleoncloud.org/learn/frequently-asked-questions/"])
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, separators= ["\n# ", "\n## ", "\n### ", "\n#### ", "\n", " ", ""])
chunks = text_splitter.split_documents(docs)
print("Number of chunks: ", len(chunks))

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chunks_embedding = embeddings_model.embed_documents([chunks[0].page_content])
database = FAISS.from_documents(documents= chunks, embedding= embeddings_model)

retriever = database.as_retriever(search_kwargs={"k":3})
docs = retriever.invoke("What is Chameleon Cloud?")

#print the 200 characher from wikipedia

def print_docs(docs):
    for doc in docs:
        print(doc.page_content[:500])
        print("-"*100+"\n")

    
print_docs(docs)
len(docs)



