import os
from dotenv import load_dotenv
import streamlit as st
from rag import load_vectorstore, load_pages, create_llm_chain, VECT_STORE_PATH

st.set_page_config(
    page_title="Chameleon Docs Assistant",
    page_icon="https://chameleoncloud.org/static/images/favicon.ico",
    layout="centered",
)

load_dotenv()

st.markdown("""
<style>
    :root { --chameleon-blue: #239ff0; }

    .chameleon-header {
        background-color: #ffffff;
        border-bottom: 3px solid #239ff0;
        padding: 12px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
    }
    .chameleon-header a {
        color: #239ff0;
        text-decoration: none;
        font-weight: 500;
        font-size: 14px;
        margin-left: 20px;
    }
    .chameleon-header a:hover { text-decoration: underline; }

    /* Example question chips */
    div[data-testid="column"] div.stButton > button {
        background-color: #f0f8ff;
        color: #239ff0;
        border: 1px solid #239ff0;
        border-radius: 16px;
        font-weight: 500;
        font-size: 13px;
        padding: 4px 12px;
        white-space: normal;
        height: auto;
    }
    div[data-testid="column"] div.stButton > button:hover {
        background-color: #239ff0;
        color: white;
    }

    /* Submit button */
    div.stButton.submit > button,
    button[kind="primary"] {
        background-color: #239ff0;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
    }

    /* Sources box */
    .sources-box {
        background: #f8f8f8;
        border-radius: 4px;
        padding: 10px 16px;
        font-size: 13px;
        margin-top: 4px;
    }

    header[data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 0 !important; }
</style>

<div class="chameleon-header">
    <a href="https://chameleoncloud.org" target="_blank">
        <img src="https://chameleoncloud.org/static/images/logo.png" height="36" alt="Chameleon Cloud">
    </a>
    <div>
        <a href="https://chameleoncloud.org" target="_blank">Home</a>
        <a href="https://chameleoncloud.readthedocs.io/en/latest/" target="_blank">Documentation</a>
        <a href="https://chameleoncloud.org/learn/frequently-asked-questions/" target="_blank">FAQ</a>
        <a href="https://chameleoncloud.org/user/help/" target="_blank">Help Desk</a>
        <a href="https://www.chameleoncloud.org/login/" target="_blank">Login</a>
    </div>
</div>
""", unsafe_allow_html=True)

st.title("Chameleon Docs Assistant")
st.caption("Ask questions about Chameleon Cloud — powered by RAG over the official documentation.")

if not os.path.exists(VECT_STORE_PATH):
    st.error("Vector store not found. Please run `python build_index.py` first to build the index.")
    st.stop()

if 'retriever' not in st.session_state:
    with st.spinner("Loading index..."):
        st.session_state.retriever = load_vectorstore()

if 'chain' not in st.session_state:
    st.session_state.chain = create_llm_chain()

if 'pages' not in st.session_state:
    st.session_state.pages = load_pages()

if 'history' not in st.session_state:
    st.session_state.history = []  # list of {"question": ..., "answer": ..., "sources": [...]}

if 'pending_question' not in st.session_state:
    st.session_state.pending_question = ""


# --- Example questions ---
EXAMPLES = [
    "How do I reserve a bare metal node?",
    "How do I use Jupyter notebooks on Chameleon?",
    "What GPU hardware is available?",
    "How do I create a custom disk image?",
    "How do I set up a private network between nodes?",
]

if not st.session_state.history:
    st.markdown("**Try asking:**")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        with col:
            if st.button(example, key=f"ex_{example}"):
                st.session_state.pending_question = example
                st.rerun()


# --- Chat history ---
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        if entry["sources"]:
            with st.expander("Further Reading"):
                sources_md = "\n".join(f"- [{s}]({s})" for s in entry["sources"])
                st.markdown(sources_md)


# --- Input ---
question = st.chat_input("Ask a question about Chameleon Cloud...")

# Handle example button click
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = ""


def get_sources(docs):
    seen = set()
    sources = []
    for doc in docs:
        src = doc.metadata.get('source')
        if src and src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        instructional_query = f"A question regarding the Chameleon Cloud testbed: {question}"
        chunks = st.session_state.retriever.invoke(instructional_query)

        # Use full pages instead of chunks for richer context
        seen_sources = []
        for chunk in chunks:
            src = chunk.metadata.get('source')
            if src and src not in seen_sources:
                seen_sources.append(src)
        context = "\n\n---\n\n".join(
            st.session_state.pages[src] for src in seen_sources if src in st.session_state.pages
        )

        # Stream the response
        def token_stream():
            for chunk in st.session_state.chain.stream({"question": question, "context": context}):
                if chunk.content:
                    yield chunk.content

        response_text = st.write_stream(token_stream())

        sources = seen_sources
        if sources:
            with st.expander("Further Reading"):
                st.markdown("\n".join(f"- [{s}]({s})" for s in sources))

    st.session_state.history.append({
        "question": question,
        "answer": response_text,
        "sources": sources,
    })


# --- Footer ---
st.markdown("""
<hr style="margin-top: 48px; border-color: #e0e0e0;">
<div style="text-align:center; color:#888; font-size:13px; padding: 12px 0;">
    Chameleon Docs Assistant &nbsp;|&nbsp;
    <a href="https://chameleoncloud.org" target="_blank" style="color:#239ff0;">chameleoncloud.org</a>
    &nbsp;|&nbsp; Powered by <a href="https://ai.tejas.tacc.utexas.edu" target="_blank" style="color:#239ff0;">Tejas AI</a>
</div>
""", unsafe_allow_html=True)
