from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever()
docs = retriever.invoke("Chameleon")
#print the 200 characher from wikipedia
docs[0].page_content[:200]

