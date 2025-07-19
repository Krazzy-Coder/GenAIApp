import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sentimentAnalysisService import analyze_journals
from starlette.status import HTTP_403_FORBIDDEN

app = FastAPI()

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "x-api-key"

class JournalRequest(BaseModel):
    journals: str

@app.post("/analyze-journals")
def analyze(request: JournalRequest, req: Request):
    client_key = req.headers.get(API_KEY_NAME)
    if client_key != API_KEY:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid or missing API key")

    try:
        summary = analyze_journals(request.journals)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
