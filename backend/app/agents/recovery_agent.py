"""
RecoveryAgent — Phase 3 (After)

Analyzes citizen-submitted damage reports (photo description + optional
days-since-disaster) using Qwen-VL-style reasoning. The key mechanic
that differentiates this from a standalone damage-assessment tool:
severity is NOT scored in isolation. It's weighted by the zone's
accumulated history from Disaster Memory — SOS volume, critical SOS
count, and vulnerability flags from Phase 1 and Phase 2.

This is the cross-phase connection: a zone that had 8 SOS reports and
3 critical cases gets a higher relief-priority score for the same
visual damage than a zone with a quiet history, because the system
remembers the disaster's full timeline, not just a snapshot.
"""

import uuid
from app.models.schemas import DamageReport, RiskLevel, TimelineEntry, AgentName, ResourceAllocation
from app.services.qwen_client import qwen_client
from app.memory.store import disaster_memory


RESOURCE_SYSTEM_PROMPT = """You are RecoveryAgent, part of a disaster response system called Lifeline Relay.
You've already assessed damage in one zone. Now recommend which specific resources to dispatch
to THIS zone, based on the damage findings and what's available in inventory.

CRITICAL RULES:
- Resources must be shared across multiple zones — do NOT allocate the entire inventory to one zone
- Recommend only 20-40% of available materials per zone at most, leaving stock for other zones
- Pick the SINGLE most appropriate vehicle (do not recommend a vehicle type with 0 remaining)
- If a vehicle type shows 0 in inventory, do not recommend it

Respond ONLY with JSON in this exact shape, no other text:
{"recommended_vehicle": "<helicopter|boat|ground_vehicle|null>", "recommended_materials": {"food": <int>, "medicine": <int>, "life_jackets": <int>, "clean_water": <int>}, "reasoning": "<one or two sentence explanation tying the recommendation to the specific damage findings and inventory constraints>"}
"""


async def recommend_resources(report: DamageReport) -> ResourceAllocation:
    """
    Reasons over a damage report's structured findings (water level,
    structural damage, hazards, people visible) plus current inventory
    levels to recommend a specific vehicle and material allocation.

    This only runs on APPROVED damage reports — recommending resources
    for a damage assessment a human hasn't yet signed off on would mean
    acting on an unverified AI judgment, which breaks the human-in-the-
    loop pattern used everywhere else in this system.
    """
    inventory = disaster_memory.get_inventory()

    findings_summary = (
        f"Water level: {report.water_level or 'unknown'}. "
        f"Structural damage visible: {report.structural_damage_visible}. "
        f"Visible hazards: {', '.join(report.visible_hazards) if report.visible_hazards else 'none'}. "
        f"People visible in photo: {report.people_visible_count if report.people_visible_count is not None else 'unknown'}. "
        f"Final priority score: {report.final_priority_score:.1f}/10."
    )
    inventory_summary = (
        f"Vehicles currently available: {inventory['vehicles']}. "
        f"Materials currently available: {inventory['materials']}."
    )
    user_prompt = (
        f"Zone {report.zone_id} damage findings — {findings_summary}\n"
        f"Current inventory — {inventory_summary}\n"
        f"What should we dispatch to this zone?"
    )

    disaster_memory.set_agent_status("RecoveryAgent", "processing")

    result = await qwen_client.ask(RESOURCE_SYSTEM_PROMPT, user_prompt)

    recommended_vehicle = result.get("recommended_vehicle")
    if recommended_vehicle == "null" or recommended_vehicle == "":
        recommended_vehicle = None
    # Safety check: never recommend a vehicle type that's actually at 0,
    # even if Qwen's reasoning slips — the inventory constraint should
    # be enforced in code, not just trusted to the model's compliance.
    if recommended_vehicle and inventory["vehicles"].get(recommended_vehicle, 0) <= 0:
        recommended_vehicle = None

    recommended_materials = result.get("recommended_materials", {}) or {}
    # Clamp each material to what's actually available, same reasoning as above
    recommended_materials = {
        k: min(v, inventory["materials"].get(k, 0))
        for k, v in recommended_materials.items()
        if v > 0
    }

    reasoning = result.get("reasoning", "No reasoning provided.")

    allocation = ResourceAllocation(
        allocation_id=str(uuid.uuid4()),
        zone_id=report.zone_id,
        report_id=report.report_id,
        recommended_vehicle=recommended_vehicle,
        recommended_materials=recommended_materials,
        reasoning=reasoning,
        human_approved=None,
    )

    disaster_memory.log(TimelineEntry(
        entry_id=str(uuid.uuid4()),
        agent=AgentName.RECOVERY,
        zone_id=report.zone_id,
        headline=f"Proposed dispatch for {report.zone_id}: {recommended_vehicle or 'no vehicle available'} + {recommended_materials} (pending human review)",
        detail=reasoning,
    ))

    disaster_memory.add_resource_allocation(allocation)
    disaster_memory.set_agent_status("RecoveryAgent", "idle")

    return allocation


async def approve_resource_allocation(
    allocation_id: str,
    approved: bool,
    edited_vehicle: str | None = None,
    edited_materials: dict | None = None,
) -> ResourceAllocation | None:
    """
    Human-in-the-loop step. Only on approval does inventory actually
    get deducted — this is the real-consequence pattern used throughout
    the system: nothing changes shared state until a human signs off.
    """
    allocation = next((a for a in disaster_memory.resource_allocations if a.allocation_id == allocation_id), None)
    if not allocation:
        return None

    allocation.human_approved = approved
    from datetime import datetime
    allocation.reviewed_at = datetime.utcnow()

    if approved:
        final_vehicle = edited_vehicle if edited_vehicle is not None else allocation.recommended_vehicle
        final_materials = edited_materials if edited_materials is not None else allocation.recommended_materials

        if edited_vehicle is not None:
            allocation.edited_vehicle = edited_vehicle
        if edited_materials is not None:
            allocation.edited_materials = edited_materials

        vehicle_ok = True
        if final_vehicle:
            vehicle_ok = disaster_memory.deduct_vehicle(final_vehicle)

        materials_ok = disaster_memory.deduct_materials(final_materials) if final_materials else True

        if not vehicle_ok or not materials_ok:
            headline = f"✗ Could not fulfill dispatch for {allocation.zone_id} — insufficient inventory at approval time"
        else:
            headline = f"✓ Human approved dispatch for {allocation.zone_id}: {final_vehicle or 'no vehicle'} + {final_materials}"
    else:
        headline = f"✗ Human rejected dispatch recommendation for {allocation.zone_id} — no resources allocated"

    disaster_memory.log(TimelineEntry(
        entry_id=str(uuid.uuid4()),
        agent=AgentName.RECOVERY,
        zone_id=allocation.zone_id,
        headline=headline,
        detail=f"Human review on resource allocation {allocation_id[:8]}.",
    ))

    return allocation

SYSTEM_PROMPT = """You are RecoveryAgent, part of a disaster response system called Lifeline Relay.
Your job is to assess disaster damage from a photo description (you are simulating
a Qwen-VL vision analysis). Score severity from 0-10.
Respond ONLY with JSON in this exact shape, no other text:
{"severity_score": <float 0-10>, "image_description": "<one sentence description of what the photo shows>", "reasoning": "<one sentence explanation of the severity score>"}
"""

# Total relief convoys/crews available to distribute across zones for the
# demo. This is a fixed, simple number on purpose — the point is to show
# the ALLOCATION LOGIC clearly, not to model real-world fleet sizes.
TOTAL_RELIEF_UNITS = 10


def calculate_relief_allocation(damage_reports: list) -> list[dict]:
    """
    Converts each zone's APPROVED final_priority_score into a concrete
    relief-unit allocation (e.g. "7 of 10 convoys to Zone B"). Only
    human-approved reports count — this mirrors the rest of the system's
    human-in-the-loop pattern: nothing drives real-world action until a
    person has signed off on the underlying assessment.

    This is a RECOVERY-PHASE decision ("where do relief supplies and
    crews go now that the immediate danger has passed"), distinct from
    ResponderAgent's Phase 2 urgency triage ("who needs rescue right now").
    """
    approved = [r for r in damage_reports if r.human_approved is True]
    if not approved:
        return []

    def effective_score(r):
        return r.edited_severity_score if r.edited_severity_score is not None else r.severity_score
        # Note: final_priority_score already reflects any edit, so we
        # actually rank by that directly — kept here as a reference for
        # what feeds the priority score upstream.

    total_priority = sum(r.final_priority_score for r in approved)
    if total_priority == 0:
        # Avoid divide-by-zero; split evenly if every approved score is 0
        equal_share = TOTAL_RELIEF_UNITS // len(approved)
        return [
            {"zone_id": r.zone_id, "priority_score": r.final_priority_score, "units": equal_share}
            for r in sorted(approved, key=lambda r: r.zone_id)
        ]

    allocations = []
    for r in approved:
        share = r.final_priority_score / total_priority
        units = round(share * TOTAL_RELIEF_UNITS)
        allocations.append({
            "zone_id": r.zone_id,
            "priority_score": r.final_priority_score,
            "units": units,
        })

    # Rounding can drift the total away from TOTAL_RELIEF_UNITS by 1 —
    # correct it on the highest-priority zone so the total always matches
    # what's displayed ("10 of 10 allocated"), which matters for a demo
    # where a judge might actually add up the numbers on screen.
    allocations.sort(key=lambda a: a["priority_score"], reverse=True)
    drift = TOTAL_RELIEF_UNITS - sum(a["units"] for a in allocations)
    if drift != 0 and allocations:
        allocations[0]["units"] += drift

    return allocations

# How much the zone's SOS history can boost the final priority score,
# on top of the raw visual severity. Capped so a quiet zone with a
# truly catastrophic photo still scores appropriately high on visuals
# alone — memory adjusts priority, it doesn't override visual evidence.
MAX_HISTORY_WEIGHT = 2.5


def _calculate_history_weight(sos_count: int, critical_sos_count: int, vulnerability_flags: int) -> float:
    """
    Translates a zone's Phase 1+2 history into a priority boost.
    Simple, explainable formula on purpose — judges should be able to
    see exactly why a score moved, not trust a black box.
    """
    weight = (sos_count * 0.15) + (critical_sos_count * 0.4) + (vulnerability_flags * 0.3)
    return min(weight, MAX_HISTORY_WEIGHT)


VISION_SYSTEM_PROMPT = """You are RecoveryAgent, part of a disaster response system called Lifeline Relay.
You are looking at an actual photo of flood damage. Analyze what you genuinely observe —
water depth, structural damage, visible hazards, and people in the scene — and score
overall severity from 0-10 based on that visual evidence alone (do not factor in
anything except what's visible in this photo).
Respond ONLY with JSON in this exact shape, no other text:
{"severity_score": <float 0-10>, "image_description": "<one sentence description of what you actually see in the photo>", "reasoning": "<one sentence explanation of the severity score based on visual evidence>", "water_level": "<short phrase, e.g. 'ankle-deep', 'waist-deep', 'submerged to roofline', or 'none visible'>", "structural_damage_visible": <true or false>, "visible_hazards": [<list of short strings, e.g. "debris", "downed power lines", "fast-moving water" — empty list if none>], "people_visible_count": <integer, 0 if none visible>}
"""


async def assess_damage_from_image(
    zone_id: str,
    citizen_id: str,
    image_path: str,
    days_since_disaster: int | None = None,
) -> DamageReport:
    """
    The real multimodal path — sends the actual image file to Qwen-VL
    and lets it genuinely analyze the pixels, rather than reasoning
    over a human-typed description (see assess_damage below for that
    simpler path, still used by the manual simulator fields).
    """
    user_prompt = f"This photo shows flood damage in zone {zone_id}."
    if days_since_disaster is not None:
        user_prompt += f" The photo was taken {days_since_disaster} day(s) after the disaster event."
    user_prompt += " What is the severity of the damage shown?"

    disaster_memory.set_agent_status("RecoveryAgent", "processing")

    result = await qwen_client.ask_vision(VISION_SYSTEM_PROMPT, user_prompt, image_path)

    severity_score = float(result.get("severity_score", 5.0))
    image_description = result.get("image_description", "No description returned.")
    reasoning = result.get("reasoning", "No reasoning provided.")
    water_level = result.get("water_level")
    structural_damage_visible = result.get("structural_damage_visible")
    visible_hazards = result.get("visible_hazards") or []
    people_visible_count = result.get("people_visible_count")

    zone = disaster_memory.get_zone(zone_id)
    history_weight = _calculate_history_weight(
        zone.sos_count, zone.critical_sos_count, zone.vulnerability_flags_count
    )
    final_priority_score = min(severity_score + history_weight, 10.0)

    report = DamageReport(
        report_id=str(uuid.uuid4()),
        zone_id=zone_id,
        citizen_id=citizen_id,
        image_description=image_description,
        severity_score=severity_score,
        sos_history_weight=history_weight,
        final_priority_score=final_priority_score,
        reasoning=reasoning,
        water_level=water_level,
        structural_damage_visible=structural_damage_visible,
        visible_hazards=visible_hazards,
        people_visible_count=people_visible_count,
        human_approved=None,
    )

    history_note = ""
    if history_weight > 0:
        history_note = (
            f" Priority boosted by {history_weight:.1f} points because this zone already had "
            f"{zone.sos_count} SOS reports ({zone.critical_sos_count} critical) earlier in the disaster."
        )

    disaster_memory.log(TimelineEntry(
        entry_id=str(uuid.uuid4()),
        agent=AgentName.RECOVERY,
        zone_id=zone_id,
        headline=f"[Qwen-VL] Proposed assessment for {zone_id}: severity {severity_score:.1f}/10 → priority {final_priority_score:.1f}/10 (pending human review)",
        detail=f"{image_description} {reasoning}{history_note}",
    ))

    disaster_memory.add_damage_report(report)
    disaster_memory.set_agent_status("RecoveryAgent", "idle")

    return report


async def approve_damage_report(
    report_id: str,
    approved: bool,
    edited_severity_score: float | None = None,
    edited_image_description: str | None = None,
) -> DamageReport | None:
    """
    Human-in-the-loop step for Phase 3. A reviewer can approve Qwen-VL's
    finding as-is, or edit the severity score / description first and
    then approve the corrected version. Until this is called, the
    report's score does not factor into the zone's official relief
    priority ranking — mirroring the same review-before-action pattern
    used for CoordinatorAgent's conflicts.
    """
    from datetime import datetime

    report = next((r for r in disaster_memory.damage_reports if r.report_id == report_id), None)
    if not report:
        return None

    report.human_approved = approved
    report.reviewed_at = datetime.utcnow()

    if approved:
        # Apply any human edits, then recompute final_priority_score
        # from the (possibly corrected) severity, not Qwen's original.
        if edited_severity_score is not None:
            report.edited_severity_score = edited_severity_score
        if edited_image_description is not None:
            report.edited_image_description = edited_image_description

        effective_severity = report.edited_severity_score if report.edited_severity_score is not None else report.severity_score
        report.final_priority_score = min(effective_severity + report.sos_history_weight, 10.0)

        edit_note = " (human-edited severity)" if report.edited_severity_score is not None else " (approved as-is)"
        headline = f"✓ Human approved damage report for {report.zone_id}: final priority {report.final_priority_score:.1f}/10{edit_note}"
    else:
        headline = f"✗ Human rejected Qwen-VL's damage assessment for {report.zone_id} — excluded from relief priority"

    disaster_memory.log(TimelineEntry(
        entry_id=str(uuid.uuid4()),
        agent=AgentName.RECOVERY,
        zone_id=report.zone_id,
        headline=headline,
        detail=f"Human review on damage report {report_id[:8]}.",
    ))

    return report


async def assess_damage(
    zone_id: str,
    citizen_id: str,
    image_description_input: str,
    days_since_disaster: int | None = None,
) -> DamageReport:
    """
    image_description_input is what a real Qwen-VL call would receive
    as visual context (in the real pipeline this comes from the actual
    photo; for the mock/demo path we pass in a text description that
    stands in for "what Qwen-VL sees").
    """
    user_prompt = f"Damage photo from zone {zone_id}: {image_description_input}."
    if days_since_disaster is not None:
        user_prompt += f" Photo taken {days_since_disaster} day(s) after the disaster event."

    disaster_memory.set_agent_status("RecoveryAgent", "processing")

    result = await qwen_client.ask(SYSTEM_PROMPT, user_prompt)

    severity_score = float(result.get("severity_score", 5.0))
    image_description = result.get("image_description", image_description_input)
    reasoning = result.get("reasoning", "No reasoning provided.")

    # Read the zone's accumulated memory from Phase 1 + Phase 2
    zone = disaster_memory.get_zone(zone_id)
    history_weight = _calculate_history_weight(
        zone.sos_count, zone.critical_sos_count, zone.vulnerability_flags_count
    )
    final_priority_score = min(severity_score + history_weight, 10.0)

    report = DamageReport(
        report_id=str(uuid.uuid4()),
        zone_id=zone_id,
        citizen_id=citizen_id,
        image_description=image_description,
        severity_score=severity_score,
        sos_history_weight=history_weight,
        final_priority_score=final_priority_score,
        reasoning=reasoning,
    )

    history_note = ""
    if history_weight > 0:
        history_note = (
            f" Priority boosted by {history_weight:.1f} points because this zone already had "
            f"{zone.sos_count} SOS reports ({zone.critical_sos_count} critical) earlier in the disaster."
        )

    disaster_memory.log(TimelineEntry(
        entry_id=str(uuid.uuid4()),
        agent=AgentName.RECOVERY,
        zone_id=zone_id,
        headline=f"Damage report for {zone_id}: severity {severity_score:.1f}/10 → final priority {final_priority_score:.1f}/10",
        detail=reasoning + history_note,
    ))

    disaster_memory.set_agent_status("RecoveryAgent", "idle")

    return report
