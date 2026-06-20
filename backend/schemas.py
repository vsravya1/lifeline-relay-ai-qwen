"""
Lifeline Relay — Core Data Models
==================================

These are the shapes that flow between agents and power the dashboard.
Everything else in the system (WatcherAgent, ResponderAgent, CoordinatorAgent,
the Decision Timeline, the dashboard) reads and writes these shapes.

Design principle: each phase WRITES its findings into shared state (the
"Disaster Memory"), and later phases READ that memory to make smarter
decisions. This is what makes the 3 phases connected instead of 3 separate
demos glued together.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums — shared vocabulary across all agents
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SourceType(str, Enum):
    """Used by CoordinatorAgent to weigh conflicting evidence."""
    WEATHER_SENSOR = "weather_sensor"
    SATELLITE = "satellite"
    CITIZEN_REPORT = "citizen_report"
    HUMAN_OPERATOR = "human_operator"


class AgentName(str, Enum):
    WATCHER = "WatcherAgent"
    RESPONDER = "ResponderAgent"
    COORDINATOR = "CoordinatorAgent"
    RECOVERY = "RecoveryAgent"


# ---------------------------------------------------------------------------
# PHASE 1 (Before) — WatcherAgent output
# ---------------------------------------------------------------------------

class WeatherSignal(BaseModel):
    """Raw mock input — what WatcherAgent reads from a weather/risk feed."""
    event_type: str            # "flood", "earthquake", "storm", etc.
    zone_id: str
    severity_raw: str          # raw description from the "sensor"
    wind_speed_mph: Optional[float] = None
    rainfall_mm: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ZoneRiskAssessment(BaseModel):
    """
    WatcherAgent's output. This becomes part of Disaster Memory and is
    read by ResponderAgent (to boost SOS urgency) and CoordinatorAgent
    (to detect conflicts against ground-truth SOS volume).
    """
    zone_id: str
    risk_level: RiskLevel
    reasoning: str              # Qwen's explanation, shown in Decision Timeline
    source: SourceType = SourceType.WEATHER_SENSOR
    confidence: float = 0.8     # 0-1, used in conflict arbitration
    assessed_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# PHASE 2 (During) — ResponderAgent output
# ---------------------------------------------------------------------------

class SOSMessage(BaseModel):
    """Raw mock input — a citizen's incoming SOS message."""
    citizen_id: str
    zone_id: str
    message: str
    received_at: datetime = Field(default_factory=datetime.utcnow)


class SOSAssessment(BaseModel):
    """
    ResponderAgent's output for a single SOS message. Note `vulnerability_flag`
    and `zone_risk_boost` — these are the two specific mechanics that
    differentiate this from a plain first-come-first-served queue.
    """
    sos_id: str
    citizen_id: str
    zone_id: str
    original_message: str
    urgency: UrgencyLevel
    vulnerability_flag: bool        # True if children/elderly/disabled detected
    vulnerability_reason: Optional[str] = None
    zone_risk_boost_applied: bool   # True if WatcherAgent's zone score raised this
    reasoning: str                  # Qwen's explanation
    assessed_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# PHASE 2.5 — CoordinatorAgent: conflict detection + arbitration
# ---------------------------------------------------------------------------

class ConflictEvent(BaseModel):
    """
    The centerpiece feature. Fired when two sources disagree about the
    same zone — e.g. WatcherAgent says low risk, but SOS volume says
    otherwise. CoordinatorAgent arbitrates using source trust ranking:
    HUMAN_OPERATOR > CITIZEN_REPORT > SATELLITE > WEATHER_SENSOR.
    """
    conflict_id: str
    zone_id: str
    source_a: SourceType
    claim_a: str
    source_b: SourceType
    claim_b: str
    resolution: str                 # what the Coordinator decided
    winning_source: SourceType
    reasoning: str                  # Qwen's explanation, shown prominently in UI
    original_risk_level: RiskLevel  # the sensor's reading BEFORE escalation — needed to revert correctly if a human rejects
    requires_human_approval: bool = True
    human_approved: Optional[bool] = None
    detected_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# PHASE 3 (After) — RecoveryAgent output
# ---------------------------------------------------------------------------

class DamageReport(BaseModel):
    """
    A citizen-submitted photo + Qwen-VL's analysis. Severity score is
    weighted by the zone's SOS history from Phase 2 (Disaster Memory),
    not assessed in isolation — this is the cross-phase connection.

    Human-in-the-loop: Qwen-VL's finding starts as a PROPOSAL. It does
    not count toward relief priority until a human reviews it — and
    may edit the severity score or description first. This mirrors the
    same human-checkpoint pattern used for CoordinatorAgent's conflicts,
    so every AI judgment in the system has a review step before it
    drives action.
    """
    report_id: str
    zone_id: str
    citizen_id: str
    image_description: str          # Qwen-VL's ORIGINAL description (preserved for audit, even if edited)
    severity_score: float           # Qwen-VL's ORIGINAL severity score (preserved for audit, even if edited)
    sos_history_weight: float       # boost applied because zone had high SOS volume
    final_priority_score: float     # severity_score + sos_history_weight, capped at 10 — based on CURRENT (possibly edited) severity
    reasoning: str
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

    # Human-in-the-loop fields
    human_approved: Optional[bool] = None       # None = pending review
    edited_severity_score: Optional[float] = None    # set if a human overrides Qwen's score
    edited_image_description: Optional[str] = None   # set if a human overrides Qwen's description
    reviewed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Decision Timeline — the explainability layer judges specifically respond to
# ---------------------------------------------------------------------------

class TimelineEntry(BaseModel):
    """
    A single line in the live Decision Timeline shown on the dashboard.
    Every agent action gets logged here in plain language, so judges can
    see "what happened, why, and which agent decided" without reading code.
    """
    entry_id: str
    agent: AgentName
    zone_id: Optional[str] = None
    headline: str                   # short summary, e.g. "Zone B escalated to Critical"
    detail: str                     # one-sentence reasoning
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Disaster Memory — the shared state that makes phases connected, not siloed
# ---------------------------------------------------------------------------

class ZoneMemory(BaseModel):
    """
    Per-zone rolling memory. This is what RecoveryAgent reads in Phase 3
    to weight damage reports by what happened earlier in the lifecycle.
    """
    zone_id: str
    current_risk_level: RiskLevel = RiskLevel.LOW
    sos_count: int = 0
    critical_sos_count: int = 0
    vulnerability_flags_count: int = 0
    active_conflict: Optional[str] = None  # conflict_id if unresolved
    last_updated: datetime = Field(default_factory=datetime.utcnow)
