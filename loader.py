from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(["https://chameleoncloud.readthedocs.io/en/latest/","https://chameleoncloud.readthedocs.io/en/latest/getting-started/index.html", "https://chameleoncloud.org/learn/frequently-asked-questions/"])
docs = loader.load()

print(docs)
