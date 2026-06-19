"""
ResponderAgent — Phase 2 (During)

Triages incoming citizen SOS messages. Two specific mechanics make
this more than first-come-first-served:

1. Vulnerability priority: messages mentioning children, elderly,
   disabled individuals get flagged and bumped in urgency — inspired
   by the real-world cases where the most vulnerable people waited
   longest for help.

2. Zone risk boost: if WatcherAgent already flagged this zone as
   high/critical risk (read from Disaster Memory), an SOS from that
   zone gets an automatic urgency boost — the system already expected
   trouble here, so it reacts faster.
"""

import uuid
from app.models.schemas import SOSMessage, SOSAssessment, UrgencyLevel, RiskLevel, TimelineEntry, AgentName
from app.services.qwen_client import qwen_client
from app.memory.store import disaster_memory

SYSTEM_PROMPT = """You are ResponderAgent, part of a disaster response system called Lifeline Relay.
Your job is to triage a citizen SOS message during an active disaster.
Detect: (1) urgency level, (2) whether the message mentions a vulnerable person
(child, elderly, disabled, infant, wheelchair user).
Respond ONLY with JSON in this exact shape, no other text:
{"urgency": "low" | "medium" | "high" | "critical", "vulnerability_flag": true/false, "vulnerability_reason": "<string or null>", "reasoning": "<one sentence explanation>"}
"""

# Urgency levels in order, used to "bump up" urgency without downgrading it.
URGENCY_ORDER = [UrgencyLevel.LOW, UrgencyLevel.MEDIUM, UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]


def _bump_urgency(current: UrgencyLevel, steps: int = 1) -> UrgencyLevel:
    """Move urgency up by `steps` levels, capped at CRITICAL."""
    idx = URGENCY_ORDER.index(current)
    new_idx = min(idx + steps, len(URGENCY_ORDER) - 1)
    return URGENCY_ORDER[new_idx]


async def triage_sos(sos: SOSMessage) -> SOSAssessment:
    disaster_memory.set_agent_status("ResponderAgent", "processing")

    user_prompt = f"SOS message from zone {sos.zone_id}: \"{sos.message}\""

    result = await qwen_client.ask(SYSTEM_PROMPT, user_prompt)

    base_urgency = UrgencyLevel(result.get("urgency", "medium"))
    vulnerability_flag = bool(result.get("vulnerability_flag", False))
    vulnerability_reason = result.get("vulnerability_reason")
    reasoning = result.get("reasoning", "No reasoning provided.")

    # --- Mechanic 1: vulnerability priority ---
    final_urgency = base_urgency
    bump_reasons = []
    if vulnerability_flag:
        final_urgency = _bump_urgency(final_urgency, steps=1)
        bump_reasons.append("escalated due to vulnerability priority")

    # --- Mechanic 2: zone risk boost, read from Disaster Memory ---
    zone_memory = disaster_memory.get_zone(sos.zone_id)
    zone_risk_boost_applied = zone_memory.current_risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
    if zone_risk_boost_applied:
        final_urgency = _bump_urgency(final_urgency, steps=1)
        bump_reasons.append(f"escalated further because zone {sos.zone_id} was already flagged {zone_memory.current_risk_level.value} risk")

    # Build a reasoning string that always reflects the FINAL decision,
    # not the pre-bump base classification — explainability has to match
    # the actual outcome shown to the user.
    if bump_reasons:
        full_reasoning = f"{reasoning} Urgency {', '.join(bump_reasons)}, resulting in final urgency: {final_urgency.value.upper()}."
    else:
        full_reasoning = reasoning

    assessment = SOSAssessment(
        sos_id=str(uuid.uuid4()),
        citizen_id=sos.citizen_id,
        zone_id=sos.zone_id,
        original_message=sos.message,
        urgency=final_urgency,
        vulnerability_flag=vulnerability_flag,
        vulnerability_reason=vulnerability_reason,
        zone_risk_boost_applied=zone_risk_boost_applied,
        reasoning=full_reasoning,
    )

    # Write back into shared memory — CoordinatorAgent and RecoveryAgent
    # will read these counts later.
    disaster_memory.record_sos(
        zone_id=sos.zone_id,
        is_critical=(final_urgency == UrgencyLevel.CRITICAL),
        is_vulnerable=vulnerability_flag,
    )

    headline = f"SOS from {sos.citizen_id} in {sos.zone_id}: {final_urgency.value.upper()}"
    if vulnerability_flag:
        headline += " (vulnerability priority applied)"
    if zone_risk_boost_applied:
        headline += " (zone risk boost applied)"

    disaster_memory.log(TimelineEntry(
        entry_id=str(uuid.uuid4()),
        agent=AgentName.RESPONDER,
        zone_id=sos.zone_id,
        headline=headline,
        detail=full_reasoning,
    ))

    disaster_memory.set_agent_status("ResponderAgent", "idle")

    return assessment
