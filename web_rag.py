import os
from urllib.parse import urlparse
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag import load_vectorstore, load_pages, create_llm_chain, build_context, VECT_STORE_PATH

st.set_page_config(
    page_title="Chameleon Docs Assistant",
    page_icon="https://chameleoncloud.org/static/images/favicon.ico",
    layout="centered",
)

load_dotenv()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide default Streamlit chrome */
    header[data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 0 !important; max-width: 780px; }

    /* ── Top navigation bar ── */
    .cc-navbar {
        position: sticky;
        top: 0;
        z-index: 100;
        background: #ffffff;
        border-bottom: 1px solid #e8edf2;
        padding: 0 28px;
        height: 56px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0;
    }
    .cc-navbar-links {
        display: flex;
        gap: 4px;
    }
    .cc-navbar-links a {
        color: #4a5568;
        text-decoration: none;
        font-size: 13px;
        font-weight: 500;
        padding: 6px 10px;
        border-radius: 6px;
        transition: background 0.15s, color 0.15s;
    }
    .cc-navbar-links a:hover {
        background: #f0f8ff;
        color: #239ff0;
    }

    /* ── Hero (empty state) ── */
    .cc-hero {
        text-align: center;
        padding: 56px 24px 32px;
    }
    .cc-hero h1 {
        font-size: 26px;
        font-weight: 600;
        color: #1a202c;
        margin: 0 0 10px;
        letter-spacing: -0.3px;
    }
    .cc-hero p {
        font-size: 15px;
        color: #718096;
        margin: 0 0 32px;
    }
    .cc-hero-label {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #a0aec0;
        margin-bottom: 12px;
    }

    /* ── Example chips ── */
    div[data-testid="column"] div.stButton > button {
        width: 100%;
        background: #ffffff;
        color: #2d3748;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        padding: 10px 14px;
        white-space: normal;
        height: auto;
        text-align: left;
        line-height: 1.4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        transition: border-color 0.15s, box-shadow 0.15s, color 0.15s;
    }
    div[data-testid="column"] div.stButton > button:hover {
        border-color: #239ff0;
        color: #239ff0;
        box-shadow: 0 2px 8px rgba(35,159,240,0.15);
    }

    /* ── Sources card ── */
    .cc-sources {
        margin-top: 12px;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }
    .cc-sources-header {
        background: #f7fafc;
        border-bottom: 1px solid #e2e8f0;
        padding: 8px 14px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #a0aec0;
    }
    .cc-sources-list {
        padding: 8px 14px;
        margin: 0;
        list-style: none;
    }
    .cc-sources-list li {
        padding: 5px 0;
        border-bottom: 1px solid #f0f4f8;
        font-size: 13px;
    }
    .cc-sources-list li:last-child { border-bottom: none; }
    .cc-sources-list a {
        color: #239ff0;
        text-decoration: none;
        font-weight: 500;
    }
    .cc-sources-list a:hover { text-decoration: underline; }
    .cc-source-type {
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #718096;
        background: #edf2f7;
        border-radius: 4px;
        padding: 1px 5px;
        margin-right: 6px;
        vertical-align: middle;
    }

    /* ── Footer ── */
    .cc-footer {
        text-align: center;
        color: #a0aec0;
        font-size: 12px;
        padding: 32px 0 16px;
        border-top: 1px solid #f0f4f8;
        margin-top: 48px;
    }
    .cc-footer a { color: #239ff0; text-decoration: none; }
    .cc-footer a:hover { text-decoration: underline; }
</style>

<div class="cc-navbar">
    <a href="https://chameleoncloud.org" target="_blank">
        <img src="https://chameleoncloud.org/static/images/logo.png" height="28" alt="Chameleon Cloud">
    </a>
    <div class="cc-navbar-links">
        <a href="https://chameleoncloud.org" target="_blank">Home</a>
        <a href="https://chameleoncloud.readthedocs.io/en/latest/" target="_blank">Docs</a>
        <a href="https://chameleoncloud.org/learn/frequently-asked-questions/" target="_blank">FAQ</a>
        <a href="https://chameleoncloud.org/user/help/" target="_blank">Help Desk</a>
        <a href="https://www.chameleoncloud.org/login/" target="_blank">Login</a>
    </div>
</div>
""", unsafe_allow_html=True)

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
    st.session_state.history = []

if 'pending_question' not in st.session_state:
    st.session_state.pending_question = ""


EXAMPLES = [
    "How do I reserve a bare metal node?",
    "How do I use Jupyter notebooks on Chameleon?",
    "What GPU hardware is available?",
    "How do I create a custom disk image?",
    "How do I set up a private network between nodes?",
    "How do I use object storage on Chameleon?",
]

SOURCE_TYPE_LABELS = {
    "readthedocs": "Docs",
    "python_chi":  "Python CHI",
    "blog":        "Blog",
    "forum":       "Forum",
    "gitbook":     "CHI@Edge",
    "chameleon_org": "Chameleon",
}


def _source_label(url: str) -> tuple[str, str]:
    """Return (type_badge, readable_title) for a source URL."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    segments = [s for s in parsed.path.rstrip("/").split("/") if s]

    def last_segment():
        raw = segments[-1] if segments else ""
        return raw.replace(".html", "").replace("-", " ").replace("_", " ").title()

    if "readthedocs.io" in host and "python-chi" not in host:
        meaningful = [s for s in segments
                      if s not in ("en", "latest", "index.html", "contents.html", "index")]
        label = (meaningful[-1].replace(".html", "").replace("-", " ")
                               .replace("_", " ").title()) if meaningful else "Documentation"
        return "Docs", label

    if "python-chi" in host:
        return "Python CHI", last_segment() or "API Reference"

    if "blog.chameleoncloud.org" in host:
        return "Blog", last_segment() or "Blog Post"

    if "forum.chameleoncloud.org" in host:
        return "Forum", "Community Discussion"

    if "gitbook.io" in host:
        return "CHI@Edge", last_segment() or "CHI@Edge Docs"

    if "chameleoncloud.org" in host:
        return "Chameleon", last_segment() or "Chameleon Cloud"

    return "", last_segment() or url


def render_sources(sources: list[str]):
    items = ""
    for url in sources:
        badge, title = _source_label(url)
        badge_html = f'<span class="cc-source-type">{badge}</span>' if badge else ""
        items += f'<li>{badge_html}<a href="{url}" target="_blank">{title}</a></li>\n'
    st.markdown(
        f'<div class="cc-sources">'
        f'<div class="cc-sources-header">Further Reading</div>'
        f'<ul class="cc-sources-list">{items}</ul>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Empty state hero ──
if not st.session_state.history:
    st.markdown("""
    <div class="cc-hero">
        <h1>Chameleon Docs Assistant</h1>
        <p>Ask anything about the Chameleon Cloud testbed — reservations, networking,<br>
        disk images, hardware, Python CHI, and more.</p>
        <div class="cc-hero-label">Try asking</div>
    </div>
    """, unsafe_allow_html=True)

    col_pairs = [EXAMPLES[i:i+3] for i in range(0, len(EXAMPLES), 3)]
    for row in col_pairs:
        cols = st.columns(len(row))
        for col, example in zip(cols, row):
            with col:
                if st.button(example, key=f"ex_{example}"):
                    st.session_state.pending_question = example
                    st.rerun()


# ── Chat history ──
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        if entry["sources"]:
            render_sources(entry["sources"])


# ── Input ──
question = st.chat_input("Ask a question about Chameleon Cloud...")

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = ""


if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        seen_sources, context = build_context(
            question, st.session_state.retriever, st.session_state.pages
        )

        history_messages = []
        for entry in st.session_state.history[-3:]:
            history_messages.append(HumanMessage(content=entry["question"]))
            history_messages.append(AIMessage(content=entry["answer"]))

        def token_stream():
            for chunk in st.session_state.chain.stream(
                {"question": question, "context": context, "history": history_messages}
            ):
                if chunk.content:
                    yield chunk.content

        response_text = st.write_stream(token_stream())

        if seen_sources:
            render_sources(seen_sources)

    st.session_state.history.append({
        "question": question,
        "answer": response_text,
        "sources": seen_sources,
    })


# ── Footer ──
st.markdown("""
<div class="cc-footer">
    Chameleon Docs Assistant &nbsp;·&nbsp;
    <a href="https://chameleoncloud.org" target="_blank">chameleoncloud.org</a>
    &nbsp;·&nbsp; Powered by <a href="https://ai.tejas.tacc.utexas.edu" target="_blank">Tejas AI</a>
</div>
""", unsafe_allow_html=True)
