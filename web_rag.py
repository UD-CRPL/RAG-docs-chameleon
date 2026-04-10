import os
from dotenv import load_dotenv
import streamlit as st
from rag import load_vectorstore, create_llm_chain, VECT_STORE_PATH

st.set_page_config(
    page_title="Chameleon Docs Assistant",
    page_icon="https://chameleoncloud.org/static/images/favicon.ico",
    layout="wide",
)

load_dotenv()

# --- Branding ---
st.markdown("""
<style>
    /* Brand color variables */
    :root { --chameleon-blue: #239ff0; }

    /* Header bar */
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

    /* Answer box */
    .answer-box {
        background: #f0f8ff;
        border-left: 4px solid #239ff0;
        border-radius: 4px;
        padding: 16px;
    }

    /* Button styling */
    div.stButton > button {
        background-color: #239ff0;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #1a8fd8;
        color: white;
    }

    /* Hide default Streamlit header */
    header[data-testid="stHeader"] { display: none; }
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

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ask a Question")
    question = st.text_area("Enter your question:", height=120)

    if st.button("Get Answer"):
        if question:
            with st.spinner("Searching docs and generating answer..."):
                instructional_query = f"A question regarding the Chameleon Cloud testbed: {question}"
                docs = st.session_state.retriever.invoke(instructional_query)
                docs_content = "\n\n".join(doc.page_content for doc in docs)
                response = st.session_state.chain.invoke({
                    "question": question,
                    "context": docs_content
                })
                st.session_state.last_response = response.content
                st.session_state.last_context = docs
        else:
            st.warning("Please enter a question.")

with col2:
    st.subheader("Answer")
    if 'last_response' in st.session_state:
        st.markdown(
            f'<div class="answer-box">{st.session_state.last_response}</div>',
            unsafe_allow_html=True
        )

        st.markdown("#### Further Reading")
        seen = set()
        for doc in st.session_state.last_context:
            source = doc.metadata.get('source')
            if source and source not in seen:
                seen.add(source)
                st.markdown(f"- [{source}]({source})")

st.markdown("""
<hr style="margin-top: 48px; border-color: #e0e0e0;">
<div style="text-align:center; color:#888; font-size:13px; padding: 12px 0;">
    Chameleon Docs Assistant &nbsp;|&nbsp;
    <a href="https://chameleoncloud.org" target="_blank" style="color:#239ff0;">chameleoncloud.org</a>
    &nbsp;|&nbsp; Powered by <a href="https://ai.tejas.tacc.utexas.edu" target="_blank" style="color:#239ff0;">Tejas AI</a>
</div>
""", unsafe_allow_html=True)
