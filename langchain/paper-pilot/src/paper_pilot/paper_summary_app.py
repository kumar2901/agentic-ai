import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from paper_pilot.chains import build_summary_and_keyword_chain
from paper_pilot.models import get_model
from paper_pilot.schemas import Keywords, PaperSummary

st.header("Research Paper Summary Tool")

try:
    model = get_model()
except ValueError as e:
    st.error(f"Error initializing model: {e}")
    st.stop()


# get chain
chain = build_summary_and_keyword_chain(model)


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
    "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"],
)

# Initialize session history as momory of the model
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize summary history (what user sees) as memory of the model
if "summary_history" not in st.session_state:
    st.session_state.summary_history = []

if st.button("Summarise"):
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

    result = summary  # mypy now knows this a paper summary object

    st.subheader(result.title)
    st.caption("Keywords : " + ", ".join(keywords.keywords))

    st.markdown("### Key Ideas")
    st.write(result.key_ideas)

    if result.equations:
        st.markdown("### Equations")
        for eq in result.equations:
            st.latex(eq)

    st.info(f"**Analogy**: {result.analogy}")
    st.progress(result.difficulty_level / 5, text=f"Difficulty Level: {result.difficulty_level}/5")

    # raw data, handy while learning how to use the model and debug
    with st.expander("Show Raw Summary Data"):
        st.json(result.model_dump())

    # save history as text so the next prompt has context of the previous summary
    st.session_state.chat_history.append(
        HumanMessage(content=f"Summarise paper input {paper_input}.")
    )
    st.session_state.chat_history.append(AIMessage(content=result.model_dump_json()))

    # save the tidy copy for the history panel
    st.session_state.summary_history.append({**result.model_dump(), "keywords": keywords.keywords})


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
