import json
import os
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag import load_vectorstore, load_parents, create_llm_chain, build_context, VECT_STORE_PATH, MIN_RERANKER_SCORE
from feedback_store import FeedbackStore, FeedbackRecord, hash_response, FAILURE_CATEGORIES

LOG_PATH = os.path.join(os.path.dirname(__file__), "session_log.jsonl")


def log_query(question: str, sources: list[str], context: str, answer: str, latency_s: float):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "sources": sources,
        "context": context,
        "answer": answer,
        "latency_s": round(latency_s, 2),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

FEEDBACK_FORM_URL = "https://forms.gle/YOUR_FORM_ID"

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

    /* ── Disclaimer banner ── */
    .cc-disclaimer {
        background: #fffbeb;
        border-bottom: 1px solid #f6d860;
        padding: 9px 28px;
        font-size: 13px;
        color: #78580a;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        flex-wrap: wrap;
    }
    .cc-disclaimer-text {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .cc-disclaimer-icon {
        font-size: 15px;
        flex-shrink: 0;
    }
    .cc-disclaimer a {
        color: #b45309;
        font-weight: 600;
        text-decoration: underline;
        white-space: nowrap;
    }
    .cc-disclaimer a:hover { color: #92400e; }

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

st.markdown(f"""
<div class="cc-disclaimer">
    <div class="cc-disclaimer-text">
        <span class="cc-disclaimer-icon">⚠️</span>
        <span><strong>Testing purposes only.</strong>
        This assistant is experimental and may produce inaccurate or incomplete answers.
        Always verify important information against the
        <a href="https://chameleoncloud.readthedocs.io/en/latest/" target="_blank">official documentation</a>
        or reach out to us at the <a href="https://chameleoncloud.org/user/help">Help Desk.</a></span>
    </div>
</div>
""", unsafe_allow_html=True)

if not os.path.exists(os.path.join(VECT_STORE_PATH, "index.faiss")):
    st.error("Vector store not found or incomplete. Please run `python build_index.py` first to build the index.")
    st.stop()

if 'vectorstore' not in st.session_state:
    with st.spinner("Loading index..."):
        st.session_state.vectorstore = load_vectorstore()
        st.session_state.parents = load_parents()

if 'chain' not in st.session_state:
    st.session_state.chain = create_llm_chain()


if 'history' not in st.session_state:
    st.session_state.history = []

if 'pending_question' not in st.session_state:
    st.session_state.pending_question = ""

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'feedback_state' not in st.session_state:
    st.session_state.feedback_state = {}

if 'feedback_store' not in st.session_state:
    st.session_state.feedback_store = FeedbackStore()
    # Pre-populate already-rated responses so the UI shows confirmation after a page refresh
    rated = st.session_state.feedback_store.get_rated_hashes(st.session_state.session_id)
    st.session_state._rated_hashes = rated


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


def render_feedback_ui(msg_index: int, entry: dict) -> None:
    """Render thumbs-up/thumbs-down feedback UI for one assistant response.

    State machine (st.session_state.feedback_state[msg_index]):
      None / absent → show 👍 👎 buttons
      'pending_neg'  → show failure-category form
      'positive'     → show confirmation (locked)
      'negative'     → show confirmation (locked)
    """
    store = st.session_state.feedback_store
    response_hash = hash_response(entry["answer"])
    state = st.session_state.feedback_state.get(msg_index)

    # Recover confirmed state from DB if feedback_state was cleared by a page refresh
    if state is None and response_hash in st.session_state.get("_rated_hashes", set()):
        state = "positive"  # treat as confirmed; exact rating not needed for UI
        st.session_state.feedback_state[msg_index] = state

    if state in ("positive", "negative"):
        st.caption("✓ Thanks for your feedback")
        return

    if state is None:
        col1, col2, _ = st.columns([1, 1, 10])
        with col1:
            if st.button("👍", key=f"fb_pos_{msg_index}", help="Helpful"):
                if not store.already_rated(response_hash, st.session_state.session_id):
                    store.save(FeedbackRecord(
                        session_id=st.session_state.session_id,
                        question=entry["question"],
                        response_hash=response_hash,
                        rating="positive",
                    ))
                    st.session_state._rated_hashes.add(response_hash)
                st.session_state.feedback_state[msg_index] = "positive"
                st.rerun()
        with col2:
            if st.button("👎", key=f"fb_neg_{msg_index}", help="Not helpful"):
                st.session_state.feedback_state[msg_index] = "pending_neg"
                st.rerun()

    elif state == "pending_neg":
        with st.form(key=f"fb_form_{msg_index}"):
            st.caption("What went wrong? (select all that apply)")
            selected = st.multiselect(
                "Categories",
                FAILURE_CATEGORIES,
                label_visibility="collapsed",
            )
            comment = st.text_area("Additional comments (optional)", height=80)
            if st.form_submit_button("Submit feedback"):
                if not store.already_rated(response_hash, st.session_state.session_id):
                    store.save(FeedbackRecord(
                        session_id=st.session_state.session_id,
                        question=entry["question"],
                        response_hash=response_hash,
                        rating="negative",
                        failure_categories=selected,
                        comment=comment or None,
                    ))
                    st.session_state._rated_hashes.add(response_hash)
                st.session_state.feedback_state[msg_index] = "negative"
                st.rerun()


# ── Empty state hero ──
if not st.session_state.history:
    st.markdown("""
    <div class="cc-hero">
        <h1>Chameleon Docs Assistant</h1>
        <p>Ask anything about the Chameleon Cloud testbed — reservations, networking,<br>
        disk images, hardware, python-chi, and more.</p>
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
for msg_index, entry in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        if entry["sources"]:
            render_sources(entry["sources"])
        render_feedback_ui(msg_index, entry)


# ── Input ──
question = st.chat_input("Ask a question about Chameleon Cloud...")

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = ""


if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        t0 = time.time()
        seen_sources, context, debug_candidates = build_context(
            question, st.session_state.vectorstore, st.session_state.parents
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

        render_feedback_ui(
            len(st.session_state.history),  # index it will occupy after append
            {"question": question, "answer": response_text, "sources": seen_sources},
        )

        with st.expander("Retrieval debug"):
            n_total    = len(debug_candidates)
            n_above    = sum(1 for c in debug_candidates if c["above_threshold"])
            n_selected = sum(1 for c in debug_candidates if c["selected"])

            # ── Reranker score summary by source type ──────────────────────
            src_stats: dict[str, list] = {}
            for c in debug_candidates:
                src = "Chameleon Docs" if ("readthedocs" in c["url"] or "python-chi" in c["url"]) else "Blog"
                src_stats.setdefault(src, []).append(c["score"])

            st.markdown("**Reranker scores by source**")
            for src, scores in sorted(src_stats.items()):
                avg = sum(scores) / len(scores)
                st.markdown(f"- **{src}**: n={len(scores)}  avg={avg:.3f}  top={max(scores):.3f}")

            st.divider()

            # ── Candidate list ──────────────────────────────────────────────
            st.markdown(
                f"**{n_total} unique candidates → {n_above} above threshold → {n_selected} sent to model** "
                f"(threshold `{MIN_RERANKER_SCORE}`)"
            )
            st.caption("✅ sent to model · ⬜ above threshold, not selected · ❌ below threshold")
            for c in debug_candidates:
                if c["selected"]:
                    icon = "✅"
                elif c["above_threshold"]:
                    icon = "⬜"
                else:
                    icon = "❌"
                slug = c["url"].rstrip("/").split("/")[-1].replace(".html", "").replace("-", " ")
                with st.expander(f"{icon} `{c['score']:+.3f}` — {slug}", expanded=False):
                    st.caption(c["url"])
                    st.text(c["chunk"])

            st.divider()

            # ── Context sent to model ───────────────────────────────────────
            st.markdown("**Context sent to model:**")
            st.text_area("context", context, height=300, label_visibility="collapsed")

    log_query(question, seen_sources, context, response_text, time.time() - t0)

    st.session_state.history.append({
        "question": question,
        "answer": response_text,
        "sources": seen_sources,
    })


# ── Footer ──
st.markdown(f"""
<div class="cc-footer">
    Chameleon Docs Assistant &nbsp;·&nbsp;
    <a href="https://chameleoncloud.org" target="_blank">chameleoncloud.org</a>
    &nbsp;·&nbsp; Powered by <a href="https://ai.tejas.tacc.utexas.edu" target="_blank">Tejas AI</a>
    &nbsp;·&nbsp; <a href="{FEEDBACK_FORM_URL}" target="_blank">Submit feedback</a>
</div>
""", unsafe_allow_html=True)
