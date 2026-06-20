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
import base64
import httpx
from typing import Optional

USE_MOCK_QWEN = os.getenv("USE_MOCK_QWEN", "true").lower() == "true"
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-plus")


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

    async def ask_vision(self, system_prompt: str, user_prompt: str, image_path: str) -> dict:
        """
        Vision variant — sends an actual image (read from disk, base64-encoded)
        to Qwen-VL alongside a text prompt. This is the real multimodal call:
        Qwen genuinely sees the pixels, not a text description of them.
        """
        if USE_MOCK_QWEN:
            return self._mock_vision_response(image_path)

        return await self._real_vision_call(system_prompt, user_prompt, image_path)

    async def _real_vision_call(self, system_prompt: str, user_prompt: str, image_path: str) -> dict:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Infer MIME type from file extension — jpg/jpeg/png cover our use case
        ext = image_path.lower().rsplit(".", 1)[-1]
        mime = "image/png" if ext == "png" else "image/jpeg"

        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": QWEN_VL_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_image}"}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
        }
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(QWEN_BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        raw_text = data["choices"][0]["message"]["content"]
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"_raw_text": raw_text, "_parse_error": True}

    def _mock_vision_response(self, image_path: str) -> dict:
        """Mock fallback — keyed off filename so each zone's preset photo gets a distinct, plausible mock result."""
        filename = image_path.lower()
        if "zone-a" in filename:
            return {
                "severity_score": 2.5,
                "image_description": "Minor surface water pooling near a building entrance, people walking through shallow water. No structural damage visible.",
                "reasoning": "Water depth is shallow and people are mobile, indicating low immediate risk.",
                "water_level": "ankle-deep",
                "structural_damage_visible": False,
                "visible_hazards": [],
                "people_visible_count": 2,
            }
        if "zone-b" in filename:
            return {
                "severity_score": 9.0,
                "image_description": "A house almost completely submerged, with only the roofline and chimneys visible above the waterline.",
                "reasoning": "Near-total submersion of the structure indicates catastrophic flooding and severe risk to anyone still inside.",
                "water_level": "submerged to roofline",
                "structural_damage_visible": True,
                "visible_hazards": ["standing floodwater", "potential structural collapse"],
                "people_visible_count": 0,
            }
        if "zone-c" in filename:
            return {
                "severity_score": 5.8,
                "image_description": "A person wading through knee-deep water with floating debris, including what appears to be a mattress, in a residential street.",
                "reasoning": "Significant flooding with debris poses real hazards, though the scene shows people still mobile rather than trapped.",
                "water_level": "knee-deep",
                "structural_damage_visible": False,
                "visible_hazards": ["floating debris"],
                "people_visible_count": 1,
            }
        return {
            "severity_score": 5.0,
            "image_description": "Flood damage visible in the image.",
            "reasoning": "Mock vision fallback — no zone-specific match found.",
            "water_level": "unknown",
            "structural_damage_visible": None,
            "visible_hazards": [],
            "people_visible_count": 0,
        }

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

        # --- CoordinatorAgent: conflict arbitration ---
        # Checked BEFORE the SOS pattern below, since conflict prompts
        # also reference SOS counts and would otherwise be misrouted
        # to the ResponderAgent mock logic.
        if "conflict detected" in prompt_lower or "which source should be trusted" in prompt_lower:
            return {
                "winning_source": "citizen_report",
                "resolution": "Citizen ground reports override sensor data given the volume and consistency of reports.",
                "reasoning": "Multiple independent citizen reports carry higher real-time reliability than periodic sensor readings, especially when sensor data has not refreshed recently."
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

        # --- RecoveryAgent: damage assessment ---
        if "damage" in prompt_lower or "photo" in prompt_lower:
            if "minor" in prompt_lower or "pooling" in prompt_lower or "no structural damage" in prompt_lower:
                return {
                    "severity_score": 2.5,
                    "image_description": "Minor surface water with no visible structural damage.",
                    "reasoning": "Severity scored low — water pooling only, no signs of structural impact."
                }
            if "severe" in prompt_lower or "first-floor windows" in prompt_lower or "structural damage" in prompt_lower:
                return {
                    "severity_score": 8.8,
                    "image_description": "Severe flooding with water reaching first-floor windows and visible structural damage.",
                    "reasoning": "Severity scored high due to water depth and clear structural impact to the building."
                }
            return {
                "severity_score": 5.5,
                "image_description": "Moderate flooding with partially submerged vehicles and street debris.",
                "reasoning": "Severity scored medium — visible flooding and debris, but no structural collapse evident."
            }

        # Fallback for anything unmatched
        return {
            "reasoning": "Mock response — no specific pattern matched.",
            "_mock_fallback": True
        }


qwen_client = QwenClient()
