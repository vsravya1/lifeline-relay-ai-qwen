"""
Qwen Client — single entry point for all LLM calls in Lifeline Relay.

Why this exists as its own module: every agent (Watcher, Responder,
Coordinator, Recovery) needs to call Qwen. Centralizing it here means:
  1. One place to swap mock -> real API once your key is ready
  2. One place to handle errors/retries consistently
  3. Agents stay clean — they just call `qwen_client.ask(...)`,
     they don't know or care whether it's mocked or real

Set USE_MOCK_QWEN=false in your environment once your Qwen Cloud API
key is active. Until then, this returns realistic, deterministic mock
responses so you can build and demo the full pipeline immediately.
"""

import os
import json
import httpx
from typing import Optional

USE_MOCK_QWEN = os.getenv("USE_MOCK_QWEN", "true").lower() == "true"
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")


class QwenClient:
    """Thin wrapper so agents don't care whether they're hitting the
    real Qwen API or a local mock. Same method signature either way."""

    async def ask(self, system_prompt: str, user_prompt: str, json_mode: bool = True) -> dict:
        """
        Send a prompt to Qwen and get back a parsed dict.
        json_mode=True instructs Qwen to respond ONLY with JSON (no
        preamble), which we then parse directly.
        """
        if USE_MOCK_QWEN:
            return self._mock_response(system_prompt, user_prompt)

        return await self._real_call(system_prompt, user_prompt)

    async def _real_call(self, system_prompt: str, user_prompt: str) -> dict:
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": QWEN_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(QWEN_BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        raw_text = data["choices"][0]["message"]["content"]
        # Strip markdown code fences if Qwen wraps JSON in ```json ... ```
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # If Qwen didn't return clean JSON, surface the raw text
            # so the calling agent can decide how to handle it rather
            # than silently failing.
            return {"_raw_text": raw_text, "_parse_error": True}

    def _mock_response(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Deterministic mock logic so the pipeline is fully demoable
        before a real Qwen key is wired in. Keyword-matches on the
        user_prompt to decide which agent is calling and returns a
        plausible structured response for that agent's expected shape.
        """
        prompt_lower = user_prompt.lower()

        # --- WatcherAgent: zone risk assessment ---
        if "weather signal" in prompt_lower or "risk level" in prompt_lower:
            if "calm" in prompt_lower or "no rainfall" in prompt_lower or "low" in prompt_lower:
                return {
                    "risk_level": "low",
                    "reasoning": "Weather signal shows calm conditions with minimal rainfall and wind speed below warning thresholds.",
                    "confidence": 0.75
                }
            return {
                "risk_level": "high",
                "reasoning": "Weather signal shows elevated rainfall and wind speed consistent with flood conditions for this zone.",
                "confidence": 0.85
            }

        # --- ResponderAgent: SOS triage ---
        if "sos message" in prompt_lower or "citizen message" in prompt_lower:
            vulnerable_terms = ["child", "kid", "elderly", "wheelchair", "baby", "disabled", "infant"]
            is_vulnerable = any(term in prompt_lower for term in vulnerable_terms)
            critical_terms = ["trapped", "rising", "can't walk", "no water", "gas leak", "drowning"]
            is_critical = any(term in prompt_lower for term in critical_terms)

            urgency = "critical" if (is_vulnerable and is_critical) else ("high" if is_critical or is_vulnerable else "medium")
            return {
                "urgency": urgency,
                "vulnerability_flag": is_vulnerable,
                "vulnerability_reason": "Message references a vulnerable individual requiring priority assistance." if is_vulnerable else None,
                "reasoning": f"Classified as {urgency} based on message content and vulnerability indicators."
            }

        # --- CoordinatorAgent: conflict arbitration ---
        if "conflict" in prompt_lower or "disagreement" in prompt_lower:
            return {
                "resolution": "Citizen ground reports override stale sensor data given volume and consistency of reports.",
                "winning_source": "citizen_report",
                "reasoning": "Multiple independent citizen reports carry higher real-time reliability than periodic sensor readings, especially when sensor data has not refreshed recently.",
                "requires_human_approval": True
            }

        # --- RecoveryAgent: damage assessment ---
        if "damage" in prompt_lower or "photo" in prompt_lower:
            return {
                "severity_score": 7.5,
                "image_description": "Significant flooding visible with partially submerged structures and debris.",
                "reasoning": "Severity scored high due to visible structural impact and water depth indicators."
            }

        # Fallback for anything unmatched
        return {
            "reasoning": "Mock response — no specific pattern matched.",
            "_mock_fallback": True
        }


qwen_client = QwenClient()
