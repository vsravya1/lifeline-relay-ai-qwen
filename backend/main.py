from fastapi import FastAPI
from fastapi.responses import FileResponse
from datetime import datetime
import os

from app.models.schemas import WeatherSignal, SOSMessage
from app.agents.watcher_agent import assess_zone_risk
from app.agents.responder_agent import triage_sos
from app.memory.store import disaster_memory

app = FastAPI(title="Lifeline Relay API")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "app", "static")


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


@app.get("/simulator")
def simulator_page():
    return FileResponse(os.path.join(STATIC_DIR, "simulator.html"))


@app.get("/dashboard")
def dashboard_page():
    return FileResponse(os.path.join(STATIC_DIR, "dashboard.html"))


@app.post("/watcher/assess")
async def watcher_assess(signal: WeatherSignal):
    """
    Phase 1 entry point. Feed in a mock weather signal, get back
    WatcherAgent's risk assessment for that zone. Also visible
    afterward via /timeline and /zones.
    """
    assessment = await assess_zone_risk(signal)
    return assessment


@app.post("/responder/triage")
async def responder_triage(sos: SOSMessage):
    """
    Phase 2 entry point. Feed in a citizen SOS message, get back
    ResponderAgent's urgency assessment — including vulnerability
    priority and zone risk boost mechanics.
    """
    assessment = await triage_sos(sos)
    return assessment


@app.get("/timeline")
def get_timeline():
    """The Decision Timeline — every agent action logged in plain language."""
    return disaster_memory.get_timeline()


@app.get("/zones")
def get_zones():
    """Current state of all zones the system has seen so far."""
    return disaster_memory.get_all_zones()
