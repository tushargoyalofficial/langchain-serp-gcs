import os
import constants

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
import streamlit as st

os.environ["SERPAPI_API_KEY"] = constants.serp_api_key
os.environ['OPENAI_API_KEY'] = constants.openai_api_key

prompt_initials = "Enter your creative prompt"

# App Framework 
st.title("Langchain Google CSE Tool")
prompt = st.text_input(prompt_initials)

llm = OpenAI(temperature=0)

tools = load_tools(["serpapi"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

if prompt:
    result = agent.run(prompt)

    st.write(result)
