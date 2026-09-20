import streamlit as st
from api_client import ask_question, check_health

st.set_page_config(page_title="Egyptian Civil Code RAG Assistant", page_icon="⚖️", layout="centered")

st.title("⚖️ Egyptian Civil Code Assistant")
st.caption("Ask a question about the Egyptian Civil Code (in Arabic or English). "
           "Answers are grounded strictly in retrieved text from the code, with sources cited.")

with st.sidebar:
    st.subheader("Backend status")
    try:
        health = check_health()
        st.success(f"Connected — {health['num_chunks']} chunks indexed ({health['embedding_backend']})")
    except Exception as e:
        st.error(f"Could not reach the backend: {e}")
        st.info("Make sure the FastAPI backend is running and BACKEND_URL is set correctly.")

    top_k = st.slider("Number of retrieved chunks", min_value=1, max_value=10, value=3)

question = st.text_input("Your question", placeholder="متى يبدأ سريان القانون المدني؟")
submit = st.button("Ask", type="primary")

if submit:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                result = ask_question(question, top_k=top_k)
                st.subheader("Answer")
                st.write(result["answer"])
                st.caption(f"Generation backend: {result['generation_backend']}")

                st.subheader("Sources")
                for i, src in enumerate(result["sources"], 1):
                    label = f"Article {src['article']}" if src.get("article") else f"Page {src.get('page')}"
                    with st.expander(f"{i}. {label}"):
                        if src.get("section"):
                            st.caption(src["section"])
                        st.write(src["text_preview"] + "...")
            except Exception as e:
                st.error(f"Something went wrong while contacting the backend: {e}")

st.divider()
st.caption("This assistant answers only from the retrieved text of the Egyptian Civil Code "
           "and will say so explicitly when an answer isn't found in the retrieved context.")
