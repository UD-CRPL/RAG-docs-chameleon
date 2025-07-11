from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = WebBaseLoader(["https://chameleoncloud.readthedocs.io/en/latest/","https://chameleoncloud.readthedocs.io/en/latest/getting-started/index.html", "https://chameleoncloud.org/learn/frequently-asked-questions/"])
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, separators= ["\n# ", "\n## ", "\n### ", "\n#### ", "\n", " ", ""])
chunks = text_splitter.split_documents(docs)

for chunk in chunks:
    print(f"chunk {chunks.index(chunk)} size: {len(chunk.page_content)}\n")

#end of chunk 4
print("end of chunk 4:")
print(chunks[3].page_content[-300:])

#start of chunk 5
print("\nstart of chunk 5:")
print(chunks[4].page_content[300:])
