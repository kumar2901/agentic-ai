from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# Change ChatMessagePromptTemplate -> ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load variables from the .env file into the system environment
load_dotenv()

model = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key=os.getenv("OLLAMA_API_KEY"),  # A placeholder key is required but ignored by Ollama
    model="gemma4:latest"
)



st.title("LangChain Prompts UI")
st.header("Research Tool")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] )

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


#prompt template for the model
prompt_template = PromptTemplate(
    template="""
      Please summarize the research paper titled "{paper_input}" with the following specifications:
      Explanation Style: {style_input}
      Explanation Length: {length_input}
      1. Mathematical Details:
        - Include relevant mathematical equations if present in the paper.
        - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
      2. Analogies:
        - Use relatable analogies to simplify complex ideas.
     If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
     Ensure the summary is clear, accurate, and aligned with the provided style and length
    """,
    input_variables=['paper_input', 'style_input', 'length_input'],
    validate_template=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides clear and concise summaries of research papers, including mathematical details and analogies when applicable. If certain information is not available in the paper, respond with: 'Insufficient information available' instead of guessing."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user",
        """Please summarize the research paper titled "{paper_input}" with the following specifications:
            Explanation Style: {style_input}
            Explanation Length: {length_input}
            1. Mathematical Details:
              - Include relevant mathematical equations if present in the paper.
              - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
            2. Analogies:
              - Use relatable analogies to simplify complex ideas.
            If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
            Ensure the summary is clear, accurate, and aligned with the provided style and length."""),
])


#fill the prompt template with user inputs
#user_input=prompt_template.invoke({"paper_input": paper_input, "style_input": style_input, "length_input": length_input})

#
#if st.button("Submit"):
    
 #   if user_input:
 #       chain=prompt_template | model
 #       response = chain.invoke(user_input)
 #       st.write(response.content)
 #  else:
 #       st.write("Please enter a prompt before submitting.")


 # create session to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


#using langhain to fill the prompt template with user inputs and get the response from the model
if st.button("Submit"):

    with st.spinner("Thinking..."):
        try:
            # Create a chain by combining the prompt template and the model
            chain = prompt | model
            # Invoke the chain with user inputs
            inputs = {"paper_input": paper_input,
                       "style_input": style_input,
                         "length_input": length_input,
                         "chat_history": st.session_state.chat_history}
            response = chain.invoke(inputs)
            # Display the response content
            st.write(response.content)
            # Update chat history
            st.session_state.chat_history.append(("User", f"Paper: {paper_input}, Style: {style_input}, Length: {length_input}"))
            st.session_state.chat_history.append(("ai", response.content))   
        except Exception as e:
            st.error(f"An error occurred: {e}")
