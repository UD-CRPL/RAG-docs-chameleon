from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever()
docs = retriever.invoke("Chameleon Cloud")
#print the 200 characher from wikipedia
print(docs[0].page_content[:400])


