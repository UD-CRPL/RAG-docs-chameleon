from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = WebBaseLoader(["https://chameleoncloud.readthedocs.io/en/latest/","https://chameleoncloud.readthedocs.io/en/latest/getting-started/index.html", "https://chameleoncloud.org/learn/frequently-asked-questions/"])
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, separators= ["\n# ", "\n## ", "\n### ", "\n#### ", "\n", " ", ""])
chunks = text_splitter.split_documents(docs)
print("Number of chunks: ", len(chunks))

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chunks_embedding = embeddings_model.embed_documents([chunks[0].page_content])
#print(chunks_embedding[0])



database = FAISS.from_documents(documents= chunks, embedding= embeddings_model)
#check what this line is doing?
database.index.ntotal

query = "what is Chamelelon?"
#k=3: top three related topics
docs = database.similarity_search(query,k = 3)

#print the chunks

for doc in docs:
    print("-"*80)
    print(doc.page_content)
    print("\n"*2)



#save the vectore database
#database.save_local("vectore_db")

#loading the vectore database
#vectore_store = FAISS.load_local("vectore_db", embeddings_model)
