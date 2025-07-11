from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


loader = WebBaseLoader(["https://chameleoncloud.readthedocs.io/en/latest/","https://chameleoncloud.readthedocs.io/en/latest/getting-started/index.html", "https://chameleoncloud.org/learn/frequently-asked-questions/"])
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, separators= ["\n# ", "\n## ", "\n### ", "\n#### ", "\n", " ", ""])
chunks = text_splitter.split_documents(docs)
print("Number of chunks: ", len(chunks))

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#embeddings_model = HuggingFaceEmbeddings(model_name="NovaSearch/stella_en_1.5B_v5")
chunks_embedding = embeddings_model.embed_documents([chunks[0].page_content])
print(chunks_embedding[0])
