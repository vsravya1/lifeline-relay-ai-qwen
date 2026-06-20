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
from app.models.schemas import DamageReport, RiskLevel, TimelineEntry, AgentName
from app.services.qwen_client import qwen_client
from app.memory.store import disaster_memory

SYSTEM_PROMPT = """You are RecoveryAgent, part of a disaster response system called Lifeline Relay.
Your job is to assess disaster damage from a photo description (you are simulating
a Qwen-VL vision analysis). Score severity from 0-10.
Respond ONLY with JSON in this exact shape, no other text:
{"severity_score": <float 0-10>, "image_description": "<one sentence description of what the photo shows>", "reasoning": "<one sentence explanation of the severity score>"}
"""

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
You are looking at an actual photo of flood damage. Score severity from 0-10 based on
what you genuinely observe in the image — water depth, structural damage, debris, and
visible danger to people or property.
Respond ONLY with JSON in this exact shape, no other text:
{"severity_score": <float 0-10>, "image_description": "<one sentence description of what you actually see in the photo>", "reasoning": "<one sentence explanation of the severity score based on visual evidence>"}
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
