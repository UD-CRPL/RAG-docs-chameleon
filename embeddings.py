from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

loader = WebBaseLoader(["https://chameleoncloud.readthedocs.io/en/latest/","https://chameleoncloud.readthedocs.io/en/latest/getting-started/index.html", "https://chameleoncloud.org/learn/frequently-asked-questions/"])
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, separators= ["\n# ", "\n## ", "\n### ", "\n#### ", "\n", " ", ""])
chunks = text_splitter.split_documents(docs)
print("Number of chunks: ", len(chunks))

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
chunks_embedding = embeddings_model.embed_documents([chunk.page_content for chunk in chunks])
len(chunks_embedding[0])
