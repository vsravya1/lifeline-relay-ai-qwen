# Lifeline Relay

**A multi-agent disaster intelligence system with conflict-aware decision making and human-in-the-loop governance — built on Qwen Cloud and deployed on Alibaba Cloud.**

Built for the Global AI Hackathon with Qwen Cloud (Track 3: Agent Society).

---

## Why this exists

A year ago, several children died at a camp near Austin during a flash flood — a tragedy rooted not in a lack of warning data, but in a lack of coordination. Around the same time, earthquakes and floods across Asia overwhelmed emergency response systems where sensor data and ground reports told conflicting stories, and no one had a clear answer for which to trust.

Lifeline Relay is built to close that specific gap: a disaster doesn't happen in three disconnected acts (warning, response, recovery) — it's one continuous event, and the system responding to it should be too.

---

## What it does

Lifeline Relay runs three connected phases, where each phase's findings become memory the next phase reads from:

**Before** — `WatcherAgent` continuously scores disaster risk per zone from live weather/risk signals.

**During** — `ResponderAgent` triages incoming citizen SOS messages, automatically boosting urgency for messages mentioning vulnerable people (children, elderly, disabled) and for zones already flagged high-risk.

**Conflict resolution** — `CoordinatorAgent` is the centerpiece: when sources disagree (e.g. a weather sensor says a zone is low-risk, but a flood of citizen SOS messages says otherwise), it arbitrates using a source-trust hierarchy, explains its reasoning, and escalates to a human for approval before action is taken.

**After** — `RecoveryAgent` analyzes citizen-submitted damage photos with Qwen-VL, weighting relief priority not just by the photo's severity score but by what the zone already experienced earlier in the pipeline (SOS volume, vulnerability flags).

Every decision — across all phases — is logged in plain language to a live **Decision Timeline**, so the reasoning behind each action is visible and explainable, not a black box.

---

## Architecture

```
WeatherSignal ──▶ WatcherAgent ──▶ ZoneRiskAssessment ─┐
                                                         ├──▶ CoordinatorAgent ──▶ ConflictEvent / Decision
SOSMessage ──────▶ ResponderAgent ──▶ SOSAssessment ────┘                              │
                                                                                          ▼
                                                                                  Disaster Memory
                                                                                          │
DamagePhoto ─────▶ RecoveryAgent (Qwen-VL) ──▶ DamageReport (weighted by Disaster Memory)
```

Built on:
- **Qwen Cloud** — LLM reasoning (risk scoring, SOS triage, conflict arbitration) and Qwen-VL (damage photo analysis)
- **Alibaba Cloud ECS** — backend hosting
- **FastAPI** — backend orchestration
- **Next.js** — live dashboard (zones, SOS feed, conflict panel, decision timeline)

---

## Project status

This project was built during the active hackathon submission window. See `/backend` for the API and agent logic, and `/frontend` for the dashboard.

## License

MIT — see [LICENSE](LICENSE).
