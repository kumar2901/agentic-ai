import streamlit as st
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage

from paper_pilot.chains import build_rag_chain, build_summary_and_keyword_chain
from paper_pilot.models import get_fallback_model, get_model
from paper_pilot.schemas import Keywords, PaperSummary, RagAnswer
from paper_pilot.vectorstore import vector_store

st.header("Research Paper Summary Tool")

try:
    model = get_model()
    fallback = get_fallback_model()
except ValueError as e:
    st.error(f"Error initializing model: {e}")
    st.stop()


# get chain
chain = build_summary_and_keyword_chain(model, fallback)


# load vector store data
@st.cache_resource(show_spinner="Loading your paper library...")
def load_store() -> Chroma:
    """Open chroma + the embedding model once, Not on every streamlit rerun"""
    return vector_store()


# Initialize session history as momory of the model
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize summary history (what user sees) as memory of the model
if "summary_history" not in st.session_state:
    st.session_state.summary_history = []

tab_summary, tab_ask = st.tabs(["Summarise", "Ask my papers"])


# Tab 1: Summarise a paper from the model's general knowledge


def render_summary_tab() -> None:
    # create boxes for user input
    paper_input = st.selectbox(
        "Select Research Paper Name",
        [
            "Attention Is All You Need",
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "GPT-3: Language Models are Few-Shot Learners",
            "Diffusion Models Beat GANs on Image Synthesis",
        ],
    )

    style_input = st.selectbox(
        "Select Explanation Style",
        ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
    )

    length_input = st.selectbox(
        "Select Explanation Length",
        ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"],
    )

    if not st.button("Summarise"):
        return

    inputs = {
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input,
        "chat_history": st.session_state.chat_history,
    }
    with st.spinner("Thinkning..."):
        try:
            output = chain.invoke(inputs)
        except Exception as ex:
            st.error(f"Coudl not reach LLM: {ex}")
            st.stop()

    summary = output["summary"]
    keywords = output["keywords"]
    if not isinstance(summary, PaperSummary) or not isinstance(keywords, Keywords):
        st.error("The model returned an unexpected output format. Please try again...")
        st.stop()

    st.subheader(summary.title)
    st.caption("Keywords : " + ", ".join(keywords.keywords))

    st.markdown("### Key Ideas")
    st.write(summary.key_ideas)
    if summary.equations:
        st.markdown("### Equations")
        for eq in summary.equations:
            st.latex(eq)

    st.info(f"**Analogy**: {summary.analogy}")
    st.progress(
        summary.difficulty_level / 5, text=f"Difficulty Level: {summary.difficulty_level}/5"
    )
    # raw data, handy while learning how to use the model and debug
    with st.expander("Show Raw Summary Data"):
        st.json(summary.model_dump())

    # save history as text so the next prompt has context of the previous summary
    st.session_state.chat_history.append(
        HumanMessage(content=f"Summarise paper input {paper_input}.")
    )
    st.session_state.chat_history.append(AIMessage(content=summary.model_dump_json()))

    # save the tidy copy for the history panel
    st.session_state.summary_history.append({**summary.model_dump(), "keywords": keywords.keywords})


def render_ask_tab() -> None:
    """Ansers questions ONLY from the ingested PDFs (RAG)."""
    try:
        store = load_store()
    except Exception as ex:
        st.error(f"Could not load the paper library: {ex}")

    if not store.get(limit=1)["ids"]:
        st.warning("Your library is empty. Run: `uv run python -m paper_pilot/ingest`")
        return

    retriever = store.as_retriever(search_kwargs={"k": 4})

    rag_chain = build_rag_chain(model, retriever, fallback=fallback)

    question = st.text_input(
        "Ask a question about your prompt",
        placeholder="What optimizer did the Attention paper used",
    )

    if not st.button("Ask", disabled=not question.strip()):
        return
    
    if len(question.strip()) < 15:
        st.warning("Please ask a more descriptive question (at least 15 characters).")
        return

    with st.spinner("Searching your papers..."):
        try:
            rag_output = rag_chain.invoke(question)
        except Exception as ex:
            st.error(f"Could not answer: {ex}")
            return
    rag_answer = rag_output["answer"]
    if not isinstance(rag_answer, RagAnswer):
        st.error("The model returned an unexpected format. Please try again")
        return

    st.markdown(rag_answer.answer)
    if rag_answer.sources:
        st.caption("Sources: " + " .".join(f"{s.paper},p.{s.page}" for s in rag_answer.sources))

    # show what the retriever actually found
    with st.expander("Retrieved chunks"):
        for doc in rag_output["docs"]:
            st.markdown(f"**{doc.metadata['paper']},p.{doc.metadata['page']}**")
            st.text(doc.page_content[:500])


with tab_summary:
    render_summary_tab()

with tab_ask:
    render_ask_tab()

# ---History panel to show previous summaries
with st.sidebar:
    st.subheader("Summary History")
    if not st.session_state.summary_history:
        st.caption("No summaries yet. Click 'Summarise' to generate a summary.")

    # newest first
    for item in reversed(st.session_state.summary_history):
        with st.expander(item["title"]):
            st.caption("Keywords : " + ", ".join(item["keywords"]))
            st.write(item["key_ideas"])
            st.caption(f"Difficulty Level: {item['difficulty_level']}/5")

    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.session_state.summary_history = []
        st.rerun()
