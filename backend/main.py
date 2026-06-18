from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="Lifeline Relay API")


@app.get("/")
def read_root():
    return {
        "message": "Hello from Lifeline Relay — running on Alibaba Cloud!",
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
