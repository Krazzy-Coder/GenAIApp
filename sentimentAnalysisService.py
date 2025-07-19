import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

def analyze_journals(journal_text: str) -> str:
    messages = [
        SystemMessage(content=(
            "You are a professional sentiment analyzer. "
            "Analyze the following journal entries separated by commas and write a 1-liner emotional summary of how the person felt overall this week."
        )),
        HumanMessage(content=journal_text)
    ]

    response = model.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)
