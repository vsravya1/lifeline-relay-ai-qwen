from fastapi import FastAPI
from datetime import datetime

from app.models.schemas import WeatherSignal
from app.agents.watcher_agent import assess_zone_risk
from app.memory.store import disaster_memory

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


@app.post("/watcher/assess")
async def watcher_assess(signal: WeatherSignal):
    """
    Phase 1 entry point. Feed in a mock weather signal, get back
    WatcherAgent's risk assessment for that zone. Also visible
    afterward via /timeline and /zones.
    """
    assessment = await assess_zone_risk(signal)
    return assessment


@app.get("/timeline")
def get_timeline():
    """The Decision Timeline — every agent action logged in plain language."""
    return disaster_memory.get_timeline()


@app.get("/zones")
def get_zones():
    """Current state of all zones the system has seen so far."""
    return disaster_memory.get_all_zones()
