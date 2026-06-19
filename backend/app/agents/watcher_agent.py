"""
WatcherAgent — Phase 1 (Before)

Reads incoming weather/risk signals and scores each zone's disaster
risk using Qwen. Writes the result into Disaster Memory, where it
becomes the baseline that ResponderAgent and CoordinatorAgent read
from in later phases.
"""

import uuid
from app.models.schemas import WeatherSignal, ZoneRiskAssessment, RiskLevel, SourceType, TimelineEntry, AgentName
from app.services.qwen_client import qwen_client
from app.memory.store import disaster_memory

SYSTEM_PROMPT = """You are WatcherAgent, part of a disaster response system called Lifeline Relay.
Your job is to assess disaster risk for a specific zone based on a weather/risk signal.
Respond ONLY with JSON in this exact shape, no other text:
{"risk_level": "low" | "medium" | "high" | "critical", "reasoning": "<one sentence explanation>", "confidence": <float 0-1>}
"""


async def assess_zone_risk(signal: WeatherSignal) -> ZoneRiskAssessment:
    disaster_memory.set_agent_status("WatcherAgent", "processing")

    user_prompt = (
        f"Weather signal for zone {signal.zone_id}: event_type={signal.event_type}, "
        f"severity_raw={signal.severity_raw}, wind_speed_mph={signal.wind_speed_mph}, "
        f"rainfall_mm={signal.rainfall_mm}. What is the risk level for this zone?"
    )

    result = await qwen_client.ask(SYSTEM_PROMPT, user_prompt)

    risk_level = RiskLevel(result.get("risk_level", "low"))
    reasoning = result.get("reasoning", "No reasoning provided.")
    confidence = float(result.get("confidence", 0.7))

    assessment = ZoneRiskAssessment(
        zone_id=signal.zone_id,
        risk_level=risk_level,
        reasoning=reasoning,
        source=SourceType.WEATHER_SENSOR,
        confidence=confidence,
    )

    # Write into shared memory — this is what makes Phase 2 "know" about
    # Phase 1's findings instead of operating blind.
    disaster_memory.update_zone_risk(signal.zone_id, risk_level)

    disaster_memory.log(TimelineEntry(
        entry_id=str(uuid.uuid4()),
        agent=AgentName.WATCHER,
        zone_id=signal.zone_id,
        headline=f"Zone {signal.zone_id} risk assessed: {risk_level.value.upper()}",
        detail=reasoning,
    ))

    disaster_memory.set_agent_status("WatcherAgent", "idle")

    return assessment
