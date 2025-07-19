from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentimentAnalysisService import analyze_journals

app = FastAPI()

class JournalRequest(BaseModel):
    journals: str  # Combined comma separated string of journal entries

@app.post("/analyze-journals")
def analyze(request: JournalRequest):
    try:
        summary = analyze_journals(request.journals)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
