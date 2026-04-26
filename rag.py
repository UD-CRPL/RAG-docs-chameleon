import os
import json
from dotenv import load_dotenv
os.environ['USER_AGENT'] = 'myagent'
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from loader import loader_docs

load_dotenv()

TEJAS_API_BASE = "https://ai.tejas.tacc.utexas.edu/v1"
TEJAS_API_KEY = os.environ.get("TEJAS_API_KEY")
VECT_STORE_PATH = "vect_store"


def get_embeddings_model():
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
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
    os.makedirs(save_path, exist_ok=True)

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
        title = parent.metadata.get("title", "")
        src_type = parent.metadata.get("source_type", "")
        header = f"[{src_type}: {title}]\n" if title else f"[{src_type}]\n" if src_type else ""

        children = CHILD_SPLITTER.split_documents([parent])
        for child in children:
            child.page_content = header + child.page_content
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


MIN_RERANKER_SCORE = 0.5

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
    return _reranker


def load_vectorstore(save_path=VECT_STORE_PATH):
    return FAISS.load_local(save_path, get_embeddings_model(), allow_dangerous_deserialization=True)


def build_context(
    question: str,
    vectorstore,
    parents: dict,
    k: int = 6,
    mmr_k: int = 20,
    fetch_k: int = 50,
) -> tuple[list[str], str, list[dict]]:
    """Retrieve and assemble context for a question.

    Returns (sources, context, debug_candidates) where debug_candidates is a list
    of all reranked chunks with their scores, sorted best-first.
    """
    instructional_query = f"A question regarding the Chameleon Cloud testbed: {question}"

    # Pull candidates from each source type separately so the reranker sees a
    # balanced pool — blog chunks outnumber readthedocs chunks after splitting,
    # so a single shared pool would be blog-dominated before reranking begins.
    all_scored = vectorstore.similarity_search_with_relevance_scores(
        instructional_query, k=fetch_k
    )
    by_type: dict[str, list] = {}
    for doc, score in all_scored:
        src = doc.metadata.get('source_type', 'other')
        by_type.setdefault(src, []).append(doc)

    per_type_k = max(k, mmr_k // max(len(by_type), 1))
    candidates = []
    for docs in by_type.values():
        candidates.extend(docs[:per_type_k])

    # Cross-encoder reranking: score each (query, chunk) pair together
    reranker = get_reranker()
    pairs = [(instructional_query, c.page_content) for c in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    seen_urls = set()
    selected = []
    selected_urls = set()
    for doc, score in ranked:
        if score < MIN_RERANKER_SCORE:
            break
        url = doc.metadata.get('source', '')
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        if len(selected) < k:
            selected.append(doc)
            selected_urls.add(url)

    # One entry per unique URL — keep the highest-scoring chunk for each URL.
    # ranked is already sorted descending so the first occurrence wins.
    _seen_debug_urls: set[str] = set()
    debug_candidates = []
    for doc, score in ranked:
        url = doc.metadata.get('source', '')
        if url in _seen_debug_urls:
            continue
        _seen_debug_urls.add(url)
        src = doc.metadata.get('source_type', 'other')
        debug_candidates.append({
            "url":      url,
            "score":    round(float(score), 4),
            "selected": url in selected_urls,
            "above_threshold": score >= MIN_RERANKER_SCORE,
            "source_type": src,
            "chunk":    doc.page_content,
        })

    SOURCE_TYPE_LABELS = {
        "readthedocs":   "Chameleon Docs",
        "python_chi":    "Python CHI Docs",
        "blog":          "Blog",
        "forum":         "Community Forum",
        "gitbook":       "CHI@Edge/Trovi Docs",
        "chameleon_org": "Chameleon Cloud",
    }
    PRIMARY_TYPES = {"readthedocs", "python_chi"}

    sources = [c.metadata['source'] for c in selected if c.metadata.get('source')]

    # Split selected chunks into primary (official docs) and supplementary (blog etc.)
    # selected is already ordered by reranker score descending, so score order is preserved.
    primary_parts = []
    supp_parts    = []
    for c in selected:
        parent = parents.get(c.metadata.get('parent_id', ''))
        if not parent:
            continue
        src_type = parent.get('source_type', 'other')
        label    = SOURCE_TYPE_LABELS.get(src_type, src_type.replace('_', ' ').title())
        url      = parent.get('source', '')
        passage  = f"[{label}: {url}]\n{parent['content']}"
        if src_type in PRIMARY_TYPES:
            primary_parts.append(passage)
        else:
            supp_parts.append(passage)

    sections = []
    if primary_parts:
        n = len(primary_parts)
        sections.append(
            f"=== PRIMARY DOCUMENTATION ({n} passage{'s' if n > 1 else ''}) ===\n\n"
            + "\n\n---\n\n".join(primary_parts)
        )
    if supp_parts:
        n = len(supp_parts)
        sections.append(
            f"=== SUPPLEMENTARY CONTEXT ({n} passage{'s' if n > 1 else ''}) ===\n\n"
            + "\n\n---\n\n".join(supp_parts)
        )
    context = "\n\n".join(sections)
    return sources, context, debug_candidates


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
            "The context is divided into two sections:\n"
            "- PRIMARY DOCUMENTATION: official Chameleon Docs and Python CHI Docs — treat these as authoritative. "
            "Ground your answer here first.\n"
            "- SUPPLEMENTARY CONTEXT: blog posts — use these only to add practical examples or fill specific gaps "
            "not covered by the primary documentation. Do not let them override the official docs.\n"
            "If the primary section is absent or thin, you may rely on supplementary context but be appropriately cautious. "
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

    vectorstore = load_vectorstore()
    parents = load_parents()
    chain = create_llm_chain()

    print("Type 'exit' to quit.")
    while True:
        query = input("\nPlease enter a question: ")
        if query == "exit":
            break
        sources, context, _ = build_context(query, vectorstore, parents)
        response = chain.invoke({"question": query, "context": context, "history": []})
        print(response.content)
        print("\nSources:")
        for src in sources:
            print(f"  {src}")


if __name__ == "__main__":
    main()
