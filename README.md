# Lifeline Relay

A multi-agent disaster intelligence system with conflict-aware decision making and human-in-the-loop governance — built on Qwen Cloud and deployed on Alibaba Cloud.

Built for the Global AI Hackathon with Qwen Cloud (Track 3: Agent Society).

![Lifeline Relay](wordcloud.png)

## Architecture

![Architecture diagram](architecture.png)

## Why this exists

A year ago, several children died at a camp near Austin during a flash flood — not because the warning data didn't exist, but because nobody was coordinating it with what was actually happening on the ground. Around the same time, earthquakes and floods across parts of Asia were overwhelming emergency response teams in a similar way: sensors said one thing, citizen reports said another, and there was no clear process for deciding which one to trust in the moment.

That gap is what this project tries to close. A disaster isn't three disconnected events — a warning, a response, a recovery — it's one continuous thing, and most of the tools built to help with it treat each phase like it happened to a different incident. Lifeline Relay tries to remember.

## What it actually does

The system runs through three phases, and each one writes to a shared memory the next phase reads from — that's the part I cared most about getting right.

**Before.** `WatcherAgent` scores flood/disaster risk per zone from incoming weather signals, using Qwen to reason about the actual conditions rather than just checking thresholds.

**During.** `ResponderAgent` triages citizen SOS messages as they come in. Two things happen automatically here: messages mentioning vulnerable people (kids, elderly, anyone with a disability) get bumped up in urgency, and so do messages from zones WatcherAgent already flagged as high-risk. The first one is a direct response to what went wrong in Austin — a generic triage system wouldn't have caught that two kids and a wheelchair user in a basement during an earthquake is a fundamentally different situation than the same words in a calmer location.

**The part I'm most proud of.** `CoordinatorAgent` watches for moments when the sensors and the citizens disagree — say, a weather sensor reports a zone as low-risk, but a wave of SOS messages says otherwise. When that happens, it doesn't just pick a winner silently. It uses Qwen to reason through which source should be trusted and why, and then it stops and waits for a human to approve or reject that call before anything actually changes. Reject it, and the zone reverts to what the sensor originally said — the override isn't cosmetic, it has a real effect on the system's state.

**After.** `RecoveryAgent` looks at real citizen-submitted damage photos using Qwen-VL — actual vision analysis on actual pixels, not a text description of a photo. The severity score it produces isn't used in isolation, either: it gets weighted by what that zone already went through earlier in the disaster, so the same flood photo means something different in a zone that already had five critical SOS reports than in a quiet one. From there, approved assessments feed into a relief allocation view — a concrete, ranked breakdown of where relief crews and supplies should go first, not just an abstract severity number.

> **On the vision piece specifically:** every damage assessment in the demo is a real `qwen-vl-plus` call against an actual JPEG, base64-encoded and sent over the wire — not a human-written description fed to a text model and labeled as "vision." If you read the description Qwen returns for a photo, it's describing what's actually in that specific image (water level, debris, structural damage), because it actually looked at it.

Every one of these decisions, across all three phases, gets logged in plain language to a live Decision Timeline on the dashboard. The point is that you can watch the system reason, not just trust that it did.

The shared Disaster Memory is the actual mechanism that connects the three phases — without it, this would just be three separate demos that happen to share a UI.

## Built with

- **Qwen Cloud (text)** — `qwen-plus` for risk scoring, SOS triage, and conflict arbitration reasoning
- **Qwen Cloud (vision)** — `qwen-vl-plus` for real image analysis of citizen-submitted damage photos — genuine multimodal input, not a text proxy
- **Alibaba Cloud ECS** — backend hosting
- **FastAPI** — backend orchestration and the agent logic itself
- **Plain HTML/JS dashboard**, served directly by FastAPI (`backend/app/static/`) — no separate frontend framework or build step, by design, since the priority was keeping the system simple enough to actually finish and demo reliably within the hackathon window

## Project layout

Everything lives under `/backend`:
- `app/agents/` — the four agents (`watcher_agent.py`, `responder_agent.py`, `coordinator_agent.py`, `recovery_agent.py`)
- `app/memory/store.py` — the shared Disaster Memory
- `app/models/schemas.py` — the data models every agent reads/writes
- `app/services/qwen_client.py` — the Qwen Cloud API wrapper (text + vision)
- `app/static/` — the live dashboard (`dashboard.html`) and a manual test-data simulator (`simulator.html`)
- `main.py` — the FastAPI app and all endpoints, including the 5-step guided demo sequence

## Project status

Built during the active hackathon submission window. It's a working system, not a mockup — every agent call in the demo hits the real Qwen Cloud API.

## License

MIT — see [LICENSE](https://github.com/vsravya1/lifeline-relay-ai/blob/main/LICENSE).
