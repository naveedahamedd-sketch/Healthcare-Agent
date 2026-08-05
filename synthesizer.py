from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from datetime import datetime
from typing import Any

from .config import settings
from .state import AgentState

_ollama_unavailable_until = 0.0


def _truncate_context(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=True)
    if len(text) <= settings.llm_max_context_chars:
        return text
    return text[: settings.llm_max_context_chars] + "...[truncated]"


def get_llm_health() -> dict[str, Any]:
    missing = []
    provider = settings.llm_provider.strip().lower()
    if not settings.llm_enabled:
        missing.append("HEALTHCARE_LLM_ENABLED")
    if provider in {"deepseek", "openai", "openai-compatible"} and not settings.llm_api_key:
        missing.append("DEEPSEEK_API_KEY")
    if not settings.llm_base_url:
        missing.append("HEALTHCARE_LLM_BASE_URL")

    return {
        "provider": settings.llm_provider,
        "primary_model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "status": "configured" if not missing else "fallback",
        "missing": missing,
        "fallback": "local_synthesizer",
        "requires_api_key": provider in {"deepseek", "openai", "openai-compatible"},
    }


def _format_datetime(value: str | None) -> str:
    if not value:
        return "the requested time"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return parsed.strftime("%A, %B %d, %Y at %I:%M %p").replace(" 0", " ")


def _local_synthesize_response(state: AgentState) -> str:
    intent = state.get("intent", "rag")
    output = state.get("specialist_output", {})

    if intent == "alert":
        return (
            "I am concerned this could be urgent. Please call local emergency services now "
            "or have someone nearby call for you. I have also started the caregiver alert workflow."
        )

    if intent == "pharmacy":
        medications = output.get("medications", [])
        if not medications:
            return output.get("answer", "Please share the medication name so I can look it up.")

        sections = []
        for medication in medications:
            side_effects = medication.get("side_effects", [])
            warnings = medication.get("warnings", [])
            risk_notes = medication.get("risk_notes", [])
            section = [
                f"Here is general information about {medication.get('name', 'this medicine')}:",
                medication.get("common_use", ""),
                medication.get("general_guidance", ""),
            ]
            if side_effects:
                section.append("Common side effects can include " + ", ".join(side_effects) + ".")
            if risk_notes:
                section.append("Extra caution based on what you shared: " + " ".join(risk_notes[:3]))
            if warnings:
                section.append("Important cautions: " + " ".join(warnings))
            source = medication.get("source") or {}
            if source.get("name"):
                section.append("Source: " + source["name"] + ".")
            sections.append(" ".join(part for part in section if part))
        return "\n\n".join(sections)

    if intent == "scheduling":
        appointment: dict[str, Any] | None = output.get("appointment")
        if not appointment:
            missing = output.get("missing_information", [])
            if missing:
                return "Before I create that appointment request, I still need: " + "; ".join(missing[:3]) + "."
            return output.get("answer", "I could not find an appointment for you yet.")

        when = _format_datetime(appointment.get("when"))
        status = appointment.get("status", "scheduled")
        clinician = appointment.get("clinician", "your care team")
        if status == "requested":
            return f"I created an appointment request with {clinician} for {when}. Your care team should confirm it."
        return f"Your next appointment is with {clinician} on {when}."

    answer = output.get("answer")
    if answer:
        if state.get("needs_clarification") or output.get("needs_clarification"):
            missing = state.get("missing_information", []) or output.get("missing_information", [])
            if missing:
                return (
                    "I can help, but I need a little more detail to route this safely. "
                    "Please share: " + "; ".join(missing[:3]) + "."
                    f"\n\nClosest local match: {answer}"
                )
        sections = [str(answer)]
        self_care = [item for item in output.get("self_care", []) if item]
        monitor = [item for item in output.get("monitor", []) if item]
        red_flags = [item for item in output.get("red_flags", []) if item]
        if self_care:
            sections.append("Helpful next steps: " + "; ".join(self_care[:3]) + ".")
        if monitor:
            sections.append("Monitor: " + "; ".join(monitor[:3]) + ".")
        if red_flags:
            sections.append("Get urgent help for: " + "; ".join(red_flags[:3]) + ".")
        return "\n\n".join(sections)
    return "I can share general health information, but I may need a little more detail to help."


def _llm_synthesize_response(state: AgentState) -> str | None:
    if not settings.llm_enabled:
        return None

    provider = settings.llm_provider.strip().lower()
    if provider == "ollama":
        return _ollama_synthesize_response(state)
    if not settings.llm_api_key:
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    prompt = (
        "You are DeepSeek-R1 acting as the patient-facing response synthesizer for a healthcare agent. "
        "Reason internally over the provided care plan and specialist output, but do not reveal chain-of-thought. "
        "Return only the final patient-facing message. Use simple, empathetic language. "
        "Do not diagnose, prescribe, calculate medication doses, or invent facts. "
        "Keep emergency advice direct and prioritize clinician or emergency care when indicated.\n\n"
        f"Intent: {state.get('intent')}\n"
        f"Patient message: {state.get('user_input')}\n"
        f"Care plan: {_truncate_context(state.get('care_plan', {}))}\n"
        f"Specialist output: {_truncate_context(state.get('specialist_output', {}))}"
    )
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.2,
        timeout=settings.llm_timeout_seconds,
    )
    try:
        result = llm.invoke(prompt)
    except Exception:
        return None
    return str(result.content)


def _ollama_synthesize_response(state: AgentState) -> str | None:
    global _ollama_unavailable_until

    if time.monotonic() < _ollama_unavailable_until:
        return None

    prompt = (
        "You are a local open-source DeepSeek-R1 healthcare response synthesizer. "
        "Use reasoning internally, but do not reveal chain-of-thought. "
        "Return only the final patient-facing message. Use simple, empathetic language. "
        "Do not diagnose, prescribe, calculate medication doses, or invent facts. "
        "Keep emergency advice direct and prioritize clinician or emergency care when indicated.\n\n"
        f"Intent: {state.get('intent')}\n"
        f"Patient message: {state.get('user_input')}\n"
        f"Care plan: {_truncate_context(state.get('care_plan', {}))}\n"
        f"Specialist output: {_truncate_context(state.get('specialist_output', {}))}"
    )
    base_url = settings.llm_base_url.rstrip("/")
    url = f"{base_url}/api/generate"
    payload = {
        "model": settings.llm_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
        },
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.llm_timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, OSError):
        _ollama_unavailable_until = time.monotonic() + 60
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    text = str(data.get("response") or "").strip()
    if not text:
        return None
    return _strip_reasoning_tags(text)


def _strip_reasoning_tags(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", "").strip()


def _append_followup_prompts(response: str, state: AgentState) -> str:
    if state.get("intent") == "alert":
        return response
    if state.get("needs_clarification"):
        return response
    if state.get("specialist_output", {}).get("needs_clarification"):
        return response

    missing = [item.strip() for item in state.get("missing_information", []) if item.strip()]
    followups = [item.strip() for item in state.get("suggested_followups", []) if item.strip()]
    prompts = missing[:3] or followups[:3]
    if not prompts:
        return response

    label = "Still needed" if missing else "Helpful details to share next"
    prompt = label + ": " + "; ".join(prompts)
    if prompt in response:
        return response
    return f"{response}\n\n{prompt}"


def _append_secondary_route_note(response: str, state: AgentState) -> str:
    if state.get("intent") == "alert" or state.get("urgency_level") == "emergency":
        return response

    secondary_outputs = [
        output
        for output in state.get("secondary_outputs", [])
        if isinstance(output, dict) and output.get("intent") != state.get("intent")
    ]
    if secondary_outputs:
        labels = {
            "rag": "health information",
            "pharmacy": "medication question",
            "scheduling": "appointment need",
            "alert": "urgent safety concern",
        }
        notes = []
        for output in secondary_outputs[:2]:
            intent = str(output.get("intent") or "")
            summary = str(output.get("summary") or "Preview completed.").strip()
            notes.append(f"{labels.get(intent, intent or 'secondary need')}: {summary}")
        note = "I also previewed the other care need: " + " ".join(notes)
        if note in response:
            return response
        return f"{response}\n\n{note}"

    secondary = [item for item in state.get("secondary_intents", []) if item != state.get("intent")]
    if not secondary:
        return response

    labels = {
        "rag": "general health information",
        "pharmacy": "a medication question",
        "scheduling": "an appointment need",
        "alert": "an urgent safety concern",
    }
    readable = ", ".join(labels.get(item, item) for item in secondary[:2])
    note = f"I also noticed this may involve {readable}; we can handle that after this step."
    if note in response:
        return response
    return f"{response}\n\n{note}"


def synthesize_response(state: AgentState) -> str:
    response = _llm_synthesize_response(state) or _local_synthesize_response(state)
    response = _append_secondary_route_note(response, state)
    return _append_followup_prompts(response, state)
