from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
import uvicorn
import os
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()
llm1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm2 = OllamaLLM(model="deepseek-r1")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server",
)

prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child with 100 words")

add_routes(
    app,
    prompt1 | llm1,
    path="/gemini"
)

add_routes(
    app,
    prompt2 | llm2,
    path="/deepseek"
)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)

