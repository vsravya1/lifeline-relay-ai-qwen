"""
CoordinatorAgent — the centerpiece feature.

This agent does NOT run on every event. It specifically watches for
DISAGREEMENT between WatcherAgent's zone risk assessment and what's
actually happening on the ground per ResponderAgent's SOS data.

The concrete conflict pattern we detect:
  WatcherAgent says a zone is LOW/MEDIUM risk (sensor-based)
  but the zone has accumulated a high volume of SOS messages,
  especially CRITICAL ones (ground-truth, citizen-based)

When this disagreement is detected, CoordinatorAgent:
  1. Uses Qwen to reason about which source to trust and why
  2. Applies a source-trust hierarchy: citizen reports (volume-backed)
     generally override stale/periodic sensor data
  3. Escalates the zone's risk level if the arbitration favors citizens
  4. Flags the decision as requiring human approval before further
     automated action proceeds — explainable AND human-in-the-loop
  5. Logs everything to the Decision Timeline in plain language

This is called after each SOS triage, checking if the zone has crossed
a conflict threshold.
"""

import uuid
from app.models.schemas import ConflictEvent, SourceType, RiskLevel, TimelineEntry, AgentName
from app.services.qwen_client import qwen_client
from app.memory.store import disaster_memory

SYSTEM_PROMPT = """You are CoordinatorAgent, part of a disaster response system called Lifeline Relay.
Two sources disagree about the risk level of a zone during an active disaster:
one is a weather/sensor-based risk assessment, the other is the volume and
severity of real citizen SOS reports from the ground.
Your job is to arbitrate: decide which source should be trusted for THIS
specific situation, and explain your reasoning clearly.
Respond ONLY with JSON in this exact shape, no other text:
{"winning_source": "citizen_report" | "weather_sensor", "resolution": "<one sentence decision>", "reasoning": "<one or two sentence explanation of WHY>"}
"""

# Conflict detection thresholds — tunable, but kept simple and explainable
# on purpose so judges can immediately understand the trigger condition.
SOS_VOLUME_CONFLICT_THRESHOLD = 3       # total SOS messages from the zone
CRITICAL_SOS_CONFLICT_THRESHOLD = 2     # critical-urgency SOS messages from the zone


def _conflict_should_trigger(zone_risk: RiskLevel, sos_count: int, critical_sos_count: int) -> bool:
    """
    The actual disagreement condition: WatcherAgent rated the zone as
    calm (LOW or MEDIUM), but ground reports tell a different story.
    """
    sensor_says_calm = zone_risk in (RiskLevel.LOW, RiskLevel.MEDIUM)
    ground_says_otherwise = (
        sos_count >= SOS_VOLUME_CONFLICT_THRESHOLD
        or critical_sos_count >= CRITICAL_SOS_CONFLICT_THRESHOLD
    )
    return sensor_says_calm and ground_says_otherwise


async def check_for_conflict(zone_id: str) -> ConflictEvent | None:
    """
    Call this after each SOS triage. Returns a ConflictEvent if a
    disagreement is detected and arbitrated, or None if the zone's
    sensor and ground-truth data are currently in agreement.
    """
    zone = disaster_memory.get_zone(zone_id)

    # Don't re-trigger if this zone already has an unresolved conflict
    if zone.active_conflict:
        return None

    if not _conflict_should_trigger(zone.current_risk_level, zone.sos_count, zone.critical_sos_count):
        return None

    disaster_memory.set_agent_status("CoordinatorAgent", "processing")

    # Capture the ORIGINAL sensor reading before any mutation happens —
    # zone is a reference to the live memory object, so if we read
    # zone.current_risk_level again after update_zone_risk() below,
    # we'd see the already-escalated value, not what the sensor
    # actually said at the time the conflict was detected.
    original_risk_level = zone.current_risk_level

    claim_a = f"WatcherAgent (weather sensor) rated zone {zone_id} as {original_risk_level.value.upper()} risk."
    claim_b = f"ResponderAgent reports {zone.sos_count} SOS messages from zone {zone_id}, including {zone.critical_sos_count} critical."

    user_prompt = (
        f"Conflict detected in zone {zone_id}.\n"
        f"Source A (weather_sensor): {claim_a}\n"
        f"Source B (citizen_report): {claim_b}\n"
        f"Which source should be trusted, and why?"
    )

    result = await qwen_client.ask(SYSTEM_PROMPT, user_prompt)

    winning_source = SourceType(result.get("winning_source", "citizen_report"))
    resolution = result.get("resolution", "Citizen reports override sensor data given volume.")
    reasoning = result.get("reasoning", "No reasoning provided.")

    conflict = ConflictEvent(
        conflict_id=str(uuid.uuid4()),
        zone_id=zone_id,
        source_a=SourceType.WEATHER_SENSOR,
        claim_a=claim_a,
        source_b=SourceType.CITIZEN_REPORT,
        claim_b=claim_b,
        resolution=resolution,
        winning_source=winning_source,
        reasoning=reasoning,
        original_risk_level=original_risk_level,
        requires_human_approval=True,
        human_approved=None,  # awaiting human review
    )

    disaster_memory.add_conflict(conflict)

    # If citizen reports win, escalate the zone's risk level immediately —
    # this is the actual consequence of the arbitration, not just a label.
    if winning_source == SourceType.CITIZEN_REPORT:
        disaster_memory.update_zone_risk(zone_id, RiskLevel.CRITICAL)

    disaster_memory.log(TimelineEntry(
        entry_id=str(uuid.uuid4()),
        agent=AgentName.COORDINATOR,
        zone_id=zone_id,
        headline=f"⚠ Conflict detected in {zone_id}: sensor said {original_risk_level.value.upper()}, "
                  f"but {zone.sos_count} SOS reports say otherwise. Resolved: {winning_source.value} wins.",
        detail=reasoning,
    ))

    # Stays "conflict_found" (not idle) until a human approves/rejects —
    # this persistent visual cue is the point: an unresolved conflict
    # should keep demanding attention on the dashboard.
    disaster_memory.set_agent_status("CoordinatorAgent", "conflict_found")

    return conflict


async def approve_conflict(conflict_id: str, approved: bool) -> ConflictEvent | None:
    """
    Human-in-the-loop step. A human reviewer (on the dashboard) approves
    or rejects the Coordinator's arbitration before it's considered final.
    """
    conflict = disaster_memory.conflicts.get(conflict_id)
    if not conflict:
        return None

    conflict.human_approved = approved

    if approved:
        # Human agrees with Coordinator's escalation — keep the
        # CRITICAL risk level that was already applied when the
        # conflict was first detected. Nothing to change here.
        disaster_memory.set_active_conflict(conflict.zone_id, None)
        headline = f"✓ Human approved Coordinator's resolution for {conflict.zone_id} — escalation stands"
    else:
        # Human overrides the Coordinator — revert the zone back to
        # what the original sensor (source_a) said, undoing the
        # escalation. This is what gives human-in-the-loop real teeth:
        # rejecting actually changes system state, not just dismisses
        # a notification.
        disaster_memory.set_active_conflict(conflict.zone_id, None)
        disaster_memory.update_zone_risk(conflict.zone_id, conflict.original_risk_level)
        headline = (
            f"✗ Human REJECTED Coordinator's resolution for {conflict.zone_id} — "
            f"reverted to {conflict.original_risk_level.value.upper()} per original sensor reading"
        )

    disaster_memory.log(TimelineEntry(
        entry_id=str(uuid.uuid4()),
        agent=AgentName.COORDINATOR,
        zone_id=conflict.zone_id,
        headline=headline,
        detail=f"Human review on conflict {conflict_id[:8]}.",
    ))

    # Conflict resolved either way — agent returns to idle
    disaster_memory.set_agent_status("CoordinatorAgent", "idle")

    return conflict
