from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file in the same directory, if present

from app.models.schemas import WeatherSignal, SOSMessage
from app.agents.watcher_agent import assess_zone_risk
from app.agents.responder_agent import triage_sos
from app.agents.coordinator_agent import check_for_conflict, approve_conflict
from app.agents.recovery_agent import assess_damage, assess_damage_from_image, approve_damage_report, calculate_relief_allocation, recommend_resources, approve_resource_allocation
from app.memory.store import disaster_memory

app = FastAPI(title="Lifeline Relay API")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "app", "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")

# Serve the zone photos as static files so the dashboard can display
# them directly (e.g. /static/images/zone-b.jpg)
app.mount("/static/images", StaticFiles(directory=IMAGES_DIR), name="images")


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

    After triage, automatically checks whether this new evidence
    creates a conflict with WatcherAgent's earlier risk assessment
    for the same zone (CoordinatorAgent's job).
    """
    assessment = await triage_sos(sos)
    conflict = await check_for_conflict(sos.zone_id)
    return {
        "assessment": assessment,
        "conflict_detected": conflict is not None,
        "conflict": conflict
    }


@app.post("/coordinator/approve")
async def coordinator_approve(conflict_id: str, approved: bool):
    """
    Human-in-the-loop endpoint. A human reviewer approves or rejects
    CoordinatorAgent's conflict resolution before it's considered final.
    """
    conflict = await approve_conflict(conflict_id, approved)
    if not conflict:
        return {"error": "Conflict not found"}
    return conflict


@app.post("/recovery/assess")
async def recovery_assess(zone_id: str, citizen_id: str, image_description: str, days_since_disaster: int = None):
    """
    Phase 3 entry point (text path). Feed in a damage description
    (manual reference field on the simulator), get back RecoveryAgent's
    severity score AND the memory-weighted final priority score.
    """
    report = await assess_damage(zone_id, citizen_id, image_description, days_since_disaster)
    return report


@app.post("/recovery/assess-image")
async def recovery_assess_image(zone_id: str, citizen_id: str, days_since_disaster: int = None):
    """
    Phase 3 entry point (real vision path). Sends the actual preset
    photo for this zone to Qwen-VL for genuine multimodal analysis.
    """
    image_path = os.path.join(IMAGES_DIR, f"{zone_id}.jpg")
    if not os.path.exists(image_path):
        return {"error": f"No preset image found for zone {zone_id}"}
    report = await assess_damage_from_image(zone_id, citizen_id, image_path, days_since_disaster)
    return report


@app.get("/relief-allocation")
def get_relief_allocation():
    """
    Ranked relief-unit allocation across zones, based on APPROVED damage
    reports only. Powers the 'where do relief crews go' section on the
    dashboard — a Phase 3 (recovery) decision, distinct from Phase 2's
    rescue-urgency triage.
    """
    reports = disaster_memory.get_damage_reports()
    allocation = calculate_relief_allocation(reports)
    return {
        "total_units": 10,
        "allocated_units": sum(a["units"] for a in allocation),
        "allocations": allocation,
        "pending_zones": [r.zone_id for r in reports if r.human_approved is None],
    }


@app.get("/damage-reports")
def get_damage_reports():
    """Latest damage report per zone, powers the photo gallery on the dashboard."""
    return disaster_memory.get_damage_reports()


@app.post("/recovery/approve")
async def recovery_approve(report_id: str, approved: bool, edited_severity_score: float = None, edited_image_description: str = None):
    """
    Human-in-the-loop endpoint for Phase 3. Approve Qwen-VL's damage
    finding as-is, or supply edited_severity_score / edited_image_description
    to correct it first. Rejecting excludes the report from relief priority.

    Once approved, automatically triggers a resource dispatch
    recommendation (vehicle + materials) for that zone — this only
    happens on approval, never on a pending or rejected report, since
    recommending real-world resources off an unverified AI judgment
    would break the human-in-the-loop pattern.
    """
    report = await approve_damage_report(report_id, approved, edited_severity_score, edited_image_description)
    if not report:
        return {"error": "Damage report not found"}

    if approved:
        await recommend_resources(report)

    return report


@app.post("/resource/approve")
async def resource_approve(allocation_id: str, approved: bool, edited_vehicle: str = None, edited_materials: str = None):
    """
    Human-in-the-loop endpoint for resource dispatch. Approve as-is, or
    pass edited_vehicle and/or edited_materials (as a JSON string, e.g.
    '{"food": 2, "medicine": 1}') to correct the recommendation first.
    Only approval actually deducts from the shared inventory.
    """
    import json
    parsed_materials = json.loads(edited_materials) if edited_materials else None
    allocation = await approve_resource_allocation(allocation_id, approved, edited_vehicle, parsed_materials)
    if not allocation:
        return {"error": "Resource allocation not found"}
    return allocation


@app.get("/resource-allocations")
def get_resource_allocations():
    """Latest resource dispatch recommendation per zone, powers the resource dispatch panel on the dashboard."""
    return disaster_memory.get_resource_allocations()


@app.get("/inventory")
def get_inventory():
    """Current vehicle and material inventory levels."""
    return disaster_memory.get_inventory()


@app.get("/conflicts")
def get_conflicts():
    """All conflicts detected so far, including pending human approval."""
    return list(disaster_memory.conflicts.values())


@app.get("/agent-status")
def get_agent_status():
    """Current status of each agent (idle / processing / conflict_found) — powers the agent panel on the dashboard."""
    return disaster_memory.get_agent_status()


@app.get("/timeline")
def get_timeline():
    """The Decision Timeline — every agent action logged in plain language."""
    return disaster_memory.get_timeline()


@app.get("/zones")
def get_zones():
    """Current state of all zones the system has seen so far."""
    return disaster_memory.get_all_zones()


# ---------------------------------------------------------------------------
# Demo sequence endpoints — power the 5 buttons on the dashboard.
# Each one bundles several agent calls into a single click, so the live
# demo can move through the disaster timeline beat by beat.
# ---------------------------------------------------------------------------

@app.post("/demo/phase1/step1")
async def demo_phase1_step1():
    """9:00 AM — initial calm-to-moderate weather signals across all 3 zones."""
    results = []
    for zone_id, severity, wind, rain in [
        ("zone-a", "calm conditions, light cloud cover", 8, 5),
        ("zone-b", "moderate rainfall beginning", 20, 40),
        ("zone-c", "calm conditions, no rainfall detected", 5, 0),
    ]:
        signal = WeatherSignal(event_type="flood", zone_id=zone_id, severity_raw=severity, wind_speed_mph=wind, rainfall_mm=rain)
        results.append(await assess_zone_risk(signal))
    return {"step": "phase1_step1", "results": results}


@app.post("/demo/phase1/step2")
async def demo_phase1_step2():
    """10:15 AM — conditions worsen sharply in Zone B."""
    signal = WeatherSignal(event_type="flood", zone_id="zone-b", severity_raw="heavy rainfall and rapidly rising water levels", wind_speed_mph=48, rainfall_mm=190)
    result = await assess_zone_risk(signal)
    return {"step": "phase1_step2", "result": result}


@app.post("/demo/phase2/step1")
async def demo_phase2_step1():
    """10:20 AM — first SOS wave from Zone B, including a vulnerability case."""
    results = []
    for citizen_id, msg in [
        ("user-101", "water rising fast in the street, need help"),
        ("user-042", "Water rising fast, 3 kids with me, 2nd floor, please hurry"),
    ]:
        sos = SOSMessage(citizen_id=citizen_id, zone_id="zone-b", message=msg)
        assessment = await triage_sos(sos)
        conflict = await check_for_conflict("zone-b")
        results.append({"assessment": assessment, "conflict": conflict})
    return {"step": "phase2_step1", "results": results}


@app.post("/demo/phase2/step2")
async def demo_phase2_step2():
    """10:45 AM — contradicting reports from Zone C, which sensors still call calm. Triggers the conflict."""
    results = []
    messages = [
        ("demo-c-1", "water rising in my street, need help"),
        ("demo-c-2", "flooding got worse here, trapped on roof"),
        ("demo-c-3", "critical, drowning risk, send boat"),
        ("demo-c-4", "basement fully flooded, elderly neighbor stuck inside"),
        ("demo-c-5", "street impassable, multiple families stranded"),
    ]
    conflict_result = None
    for citizen_id, msg in messages:
        sos = SOSMessage(citizen_id=citizen_id, zone_id="zone-c", message=msg)
        await triage_sos(sos)
        conflict_result = await check_for_conflict("zone-c")
        if conflict_result:
            break
    return {"step": "phase2_step2", "conflict_detected": conflict_result is not None, "conflict": conflict_result}


@app.post("/demo/phase3/step1")
async def demo_phase3_step1():
    """11:30 AM — damage reports across all 3 zones, using REAL Qwen-VL
    analysis of the preset photos, scored with memory-weighted priority."""
    results = []
    for zone_id in ["zone-a", "zone-b", "zone-c"]:
        image_path = os.path.join(IMAGES_DIR, f"{zone_id}.jpg")
        report = await assess_damage_from_image(zone_id, f"demo-photo-{zone_id}", image_path, days_since_disaster=2)
        results.append(report)
    return {"step": "phase3_step1", "results": results}
