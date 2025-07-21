import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

load_dotenv()

## To use Open-source LLM
# Download and install Ollama
# Run command in terminal 'ollama run <open-source_model_name>'
# open-source_model_name = deepseek-r1, llama2, etc
llm = Ollama(model="deepseek-r1")

chatTemplate = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user questions."),
    ("human", "Question: {question}")
])


chain = chatTemplate | llm | StrOutputParser()


st.title("Chat Bot 🤖")
input_text = st.text_input("Ask your question here...")

if input_text:
    st.write(chain.invoke({"question": input_text}))