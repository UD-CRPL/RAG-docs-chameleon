import json
from rag_jinna import main

retriever, embed_model_name = main()

result  = []

count = 0
while count < 10:
    query = input(f"\n Question{count+1}: ")
    if query.lower() == "exit":
        break

    retrieved_docs = retriever.invoke(query)

    result.append({
        "query": query,
        "embedding model": embed_model_name,
        "matches": [{
            "match_number": i + 1,
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown")}
                    for i, doc in enumerate(retrieved_docs) 
                    ]
        })

    for i, doc in enumerate(retrieved_docs):
        print(f"\n================== Match {i+1} ===================")
        print(doc.page_content)
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
    
    count +=1


with open('res_jinna.json','w') as result_file:
    json.dump(result, result_file, indent = 2)

print(f"\n results save to json file")
