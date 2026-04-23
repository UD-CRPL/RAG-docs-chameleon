import os
import json
from dotenv import load_dotenv
os.environ['USER_AGENT'] = 'myagent'
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loader import loader_docs

load_dotenv()

TEJAS_API_BASE = "https://ai.tejas.tacc.utexas.edu/v1"
TEJAS_API_KEY = os.environ.get("TEJAS_API_KEY")
EMBED_MODEL = "E5-Mistral-7B-Instruct"
VECT_STORE_PATH = "vect_store"


def get_embeddings_model():
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=TEJAS_API_KEY,
        openai_api_base=TEJAS_API_BASE,
        check_embedding_ctx_length=False,
        chunk_size=16,
    )


PARENT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""]
)

CHILD_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n### ", "\n#### ", "\n", " ", ""]
)


def create_vectorstore(docs, save_path=VECT_STORE_PATH):
    if os.path.exists(save_path):
        for f in os.listdir(save_path):
            os.remove(os.path.join(save_path, f))

    parent_chunks = PARENT_SPLITTER.split_documents(docs)

    parents = {}
    child_docs = []
    for i, parent in enumerate(parent_chunks):
        parent_id = str(i)
        parents[parent_id] = {
            "content": parent.page_content,
            "source": parent.metadata.get("source", ""),
            "source_type": parent.metadata.get("source_type", "other"),
        }
        children = CHILD_SPLITTER.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent_id
            child_docs.append(child)

    with open(os.path.join(save_path, "parents.json"), "w") as f:
        json.dump(parents, f)

    vectorstore = FAISS.from_documents(documents=child_docs, embedding=get_embeddings_model())
    vectorstore.save_local(save_path)
    return vectorstore


def load_parents(save_path=VECT_STORE_PATH):
    with open(os.path.join(save_path, "parents.json")) as f:
        return json.load(f)


def load_vectorstore(save_path=VECT_STORE_PATH, k=20, fetch_k=80, search_type='mmr'):
    vectorstore = FAISS.load_local(save_path, get_embeddings_model(), allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_type=search_type, search_kwargs={'k': k, 'fetch_k': fetch_k})


def diversify_sources(chunks, k=6, max_per_type=2):
    """
    Select up to k chunks with unique parent IDs, capping each source_type at
    max_per_type. Chunks are assumed ordered by MMR score so relevance is
    preserved while enforcing breadth.
    """
    seen_parents = set()
    type_counts: dict[str, int] = {}
    selected = []
    overflow = []

    for chunk in chunks:
        parent_id = chunk.metadata.get('parent_id')
        if not parent_id or parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)
        src_type = chunk.metadata.get('source_type', 'other')
        if type_counts.get(src_type, 0) < max_per_type:
            selected.append(chunk)
            type_counts[src_type] = type_counts.get(src_type, 0) + 1
        else:
            overflow.append(chunk)

    for chunk in overflow:
        if len(selected) >= k:
            break
        selected.append(chunk)

    return selected[:k]


def build_context(
    question: str,
    retriever,
    parents: dict,
    k: int = 6,
    max_per_type: int = 2,
) -> tuple[list[str], str]:
    """Retrieve and assemble context for a question."""
    instructional_query = f"A question regarding the Chameleon Cloud testbed: {question}"
    chunks = retriever.invoke(instructional_query)
    selected = diversify_sources(chunks, k=k, max_per_type=max_per_type)
    sources = [c.metadata['source'] for c in selected if c.metadata.get('source')]
    context = "\n\n---\n\n".join(
        parents[c.metadata['parent_id']]['content']
        for c in selected
        if c.metadata.get('parent_id') in parents
    )
    return sources, context


def create_llm_chain():
    llm = ChatOpenAI(
        model="Meta-Llama-3.3-70B-Instruct",
        temperature=0,
        openai_api_key=TEJAS_API_KEY,
        openai_api_base=TEJAS_API_BASE,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant that answers questions about Chameleon Cloud based on its official documentation. "
            "Use only the provided context to answer. "
            "For simple questions, give a concise answer. For complex questions, give a thorough, step-by-step answer. "
            "Do not include URLs or source citations in your answer — relevant documentation links will be shown separately. "
            "If the answer is not in the context, say 'I don't know' and do not make up an answer."
        )),
        MessagesPlaceholder(variable_name="history"),
        ("user", "Question: {question}\nContext: {context}")
    ])

    return prompt | llm


def main():
    if not os.path.exists(VECT_STORE_PATH):
        print("No vector store found. Building index from docs...")
        docs = loader_docs()
        create_vectorstore(docs)

    retriever = load_vectorstore()
    parents = load_parents()
    chain = create_llm_chain()

    print("Type 'exit' to quit.")
    while True:
        query = input("\nPlease enter a question: ")
        if query == "exit":
            break
        sources, context = build_context(query, retriever, parents)
        response = chain.invoke({"question": query, "context": context, "history": []})
        print(response.content)
        print("\nSources:")
        for src in sources:
            print(f"  {src}")


if __name__ == "__main__":
    main()
