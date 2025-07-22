import requests
import streamlit as st

def get_gemini_response(input_text):
    response=requests.post("http://localhost:8000/gemini/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']['content']

def get_deepseek_response(input_text):
    response=requests.post(
    "http://localhost:8000/deepseek/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']

    ## streamlit framework

st.title('Langchain Demo With Gemini and Deepseek API')
input_text=st.text_input("Write an essay on - using Gemini")
input_text1=st.text_input("Write a poem on - using Deepseek")

if input_text:
    st.write(get_gemini_response(input_text))

if input_text1:
    st.write(get_deepseek_response(input_text1))