import os
from dotenv import load_dotenv
import streamlit as st
from loader import loader_docs
from rag import split_docs 
from rag import create_vectorstore
from rag import create_llm_chain

st.set_page_config(page_title="RAG Chameleon Web", page_icon= "🙂", layout = "wide")
st.title("RAG Chameleon Question-Answering Demo")

# Load environment varialbes
load_dotenv()
api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not api_key:
    st.error("api key is not set in the environment variable")
    st.stop()

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None


#url = st.text_input("Enter a URL to load documents from:", value="")

if st.button("Please click her to start the RAG System"):
    with st.spinner("Loading and processing documents..."):
        docs = loader_docs()
        chunks = split_docs(docs)
        vectorstore= create_vectorstore(chunks)
        st.session_state.vectorstore = vectorstore
        st.success("RAG system initialized successfuly!")


if st.session_state.vectorstore is not None:
    chain = create_llm_chain()
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ask a Question")
        question = st.text_area("Enter your question:")

        if st.button("Get Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    retriever = st.session_state.vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'fetch_k': 20})
                    instructional_query = f"A question regarding the Chameleon Cloud testbed: {question}"
                    docs= retriever.invoke(instructional_query)
                    docs_content = "\n\n".join(doc.page_content for doc in docs)
                    response = chain.invoke({
                        "question": question,
                        "context": docs_content
                        })
                    st.session_state.last_response = response.content
                    st.session_state.last_context = docs

            else:
                st.wasrning("Please enter a question:")

        with col2:
            st.subheader("Answer")
            if 'last_response' in st.session_state:
                st.write(st.session_state.last_response)

                with st.expander("Show Retrieved Context"):
                    for i, doc in enumerate(st.session_state.last_context, 1):
                        st.markdown(f"**Relevant Document {i}:**")
                        st.markdown(doc.page_content)
                        st.markdown("---")


