import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

# Initialize model directly
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Prompt template
chatTemplate = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user questions."),
    ("human", "Question: {question}")
])

# Build the pipeline
chain = chatTemplate | llm | StrOutputParser()

# Invoke
#response = chain.invoke({"question": "Who are you?"})

#print(response)

# streamlit framework
st.title("Chat Bot 🤖")
input_text = st.text_input("Ask your question here...")

if input_text:
    st.write(chain.invoke({"question": input_text}))