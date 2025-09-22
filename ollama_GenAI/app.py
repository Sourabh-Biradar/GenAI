# GenAI with ollama (gemma:2b)

# ChatBot



import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# loading env variables
load_dotenv()

langsmith_api = os.getenv("LANGSMITH_API_KEY")
endpoint = os.getenv("LANGSMITH_ENDPOINT")
project = os.getenv("LANGSMITH_PROJECT")

os.environ['LANGSMITH_TRACING'] = 'true'

# creating prompt template
prompt = ChatPromptTemplate([
    ("system","You are AI assistant , respond to questions asked"),
    ('user',"Question:{question}")
])

# setting up llm 
llm = Ollama(model='gemma:2b')

# setting up output parser
output_parser = StrOutputParser()

# creating chain
chain=prompt | llm | output_parser

# streamlit
st.title("Ollama-Gemma:2b AI assistant")

query = st.text_input('Hello! How can I be of assistance today?')

if query:
    st.write(chain.invoke({'question':query}))