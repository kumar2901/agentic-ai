import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load variables from the .env file into the system environment
load_dotenv()

model = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),  # Placeholder key required by Ollama client
    model="mistral:latest"
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "latest_response" not in st.session_state:
    st.session_state.latest_response = None

# Header & History Popover
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.header("Research Tool")
with col2:
    with st.popover("📜 History"):
        st.write("### Past Interactions")
        if not st.session_state.chat_history:
            st.info("No saved history.")
        else:
            for role, text in st.session_state.chat_history:
                st.markdown(f"**{role.capitalize()}:** {text}")
                st.divider()

# Inputs
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Prompt template setup
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that provides clear and concise summaries of research papers, "
        "including mathematical details and analogies when applicable. If certain information is "
        "not available in the paper, respond with: 'Insufficient information available' instead of guessing."
    ),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    (
        "user",
        """Please summarize the research paper titled "{paper_input}" with the following specifications:
            Explanation Style: {style_input}
            Explanation Length: {length_input}
            1. Mathematical Details:
              - Include relevant mathematical equations if present in the paper.
              - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
            2. Analogies:
              - Use relatable analogies to simplify complex ideas.
            If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
            Ensure the summary is clear, accurate, and aligned with the provided style and length."""
    ),
])

# Submit handling
if st.button("Submit"):
    with st.spinner("Thinking..."):
        try:
            chain = prompt | model
            inputs = {
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": length_input,
                "chat_history": st.session_state.chat_history
            }
            response = chain.invoke(inputs)
            
            # Store the output in session_state before rerunning
            st.session_state.latest_response = response.content
            
            # Store standard role tuples
            st.session_state.chat_history.append(("user", f"Paper: {paper_input}, Style: {style_input}, Length: {length_input}"))
            st.session_state.chat_history.append(("ai", response.content))
            
            st.rerun()   
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display persistent output across reruns
if st.session_state.latest_response:
    st.subheader("Summary Result")
    st.write(st.session_state.latest_response)