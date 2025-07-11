import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

#for API key in .env
load_dotenv()

api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not api_key: 
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")



#loading the source(FAQs)
loader = WebBaseLoader(["https://chameleoncloud.org/learn/frequently-asked-questions/"])
docs = loader.load()

#splitting the texts
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 200, separators= ["\n# ", "\n## ", "\n### ", "\n#### ", "\n", " ", ""])
chunks = text_splitter.split_documents(docs)

#openAI embedding model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
database = FAISS.from_documents(documents= chunks, embedding= embeddings_model)

retriever = database.as_retriever()

llm_endpoint = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="conversational",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # or explicitly set "fireworks-ai" if needed
)

chat_model = ChatHuggingFace(llm = llm_endpoint)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistance that helps answer the questions about Chameleon Cloud documentations.Use the provided context to answer the questions. IMPORTANT: If you are unsure of the answer, say'I don't know' and do not make up the answer. Keep the answer short a maximum of 5 sentences and be percise."), 
    ("user", "Question:{question}\nContext: {context}")
    ])
        
chain = prompt | chat_model


query ="What is Chameleon Cloud?"

#RAG

#Retrieval
docs = retriever.invoke(query)
docs_content = "\n\n".join(doc.page_content for doc in docs) 

# Augmented + Generation
response = chain.invoke({
    "question": query, 
    "context": docs_content
    })

print(response.content)



