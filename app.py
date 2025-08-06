import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os
import re

load_dotenv()

def remove_think_block(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

st.set_page_config(page_title="Text to Math Problem Solver", page_icon=":triangular_ruler:")
st.title("Text to Math Problem Solver")

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=groq_api_key, model="deepseek-r1-distill-llama-70b")

# Wikipedia Tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet and solving the math problem"
)

# Math Tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Answering math related problems. Only input mathematical expression"
)

# Reasoning Tool
prompt = """
You are an agent tasked for solving the user's mathematical questions and logically arrive at the solution while providing a detailed explanation and display it point-wise.
Question: {question}
"""
prompt_template = PromptTemplate(
    input_variables=['question'],
    template=prompt
)
chain = LLMChain(llm=llm, prompt=prompt_template)
reasoning = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logic and reasoning based questions"
)

# Agent Setup
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_error=True
)

# Chat UI
if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "I am a math chatbot who can answer all your math related problems."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.text_area("Enter your question")

if st.button("Solve"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(question, callbacks=[st_cb])
            response = remove_think_block(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.success(response)
    else:
        st.warning("Please enter your question.")