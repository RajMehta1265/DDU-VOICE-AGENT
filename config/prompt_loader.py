"""
Modular Prompt Loader with Dynamic Domain Triggering & Hot-Reloading.
Loads prompt templates, persona rules, and domain directives from config/prompts.yaml.
Automatically refreshes when the YAML file is modified without restarting the server.
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import yaml

# Path to the prompts YAML configuration
PROMPTS_FILE = Path(__file__).resolve().parent / "prompts.yaml"

_cached_prompts: dict[str, Any] = {}
_last_mtime: float = 0.0


def _load_prompts_yaml() -> dict[str, Any]:
    """Loads and caches the YAML file, refreshing if modified on disk."""
    global _cached_prompts, _last_mtime
    try:
        if not PROMPTS_FILE.exists():
            return _cached_prompts

        current_mtime = PROMPTS_FILE.stat().st_mtime
        if current_mtime != _last_mtime or not _cached_prompts:
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            _cached_prompts = data
            _last_mtime = current_mtime
    except Exception as e:
        print(f"[prompt_loader] Notice reading {PROMPTS_FILE.name}: {e}")

    return _cached_prompts


def load_prompts() -> dict[str, Any]:
    """Public helper to get the latest loaded prompt configuration dict."""
    return _load_prompts_yaml()


def get_welcome_message() -> str:
    """Returns the initial call greeting message."""
    cfg = _load_prompts_yaml()
    return cfg.get(
        "welcome_message",
        "નમસ્તે, હું ડીડીયુ આઈટી ડિપાર્ટમેન્ટમાંથી પ્રિયા વાત કરું છું. હું તમારી શું મદદ કરી શકું? તમે ગુજરાતી, હિન્દી અથવા અંગ્રેજીમાં વાત કરી શકો છો.",
    )


def get_call_hangup_prompt() -> str:
    """Returns the supervisory prompt for classifying call hangup / completion."""
    cfg = _load_prompts_yaml()
    return cfg.get(
        "call_hangup_prompt",
        """\
You are an intelligent call supervisor analyzing a voice call between a caller and a university voice assistant.
Determine whether the caller or the conversation has concluded and the telephone call should be hung up.
Return {"call_hangup": true} or {"call_hangup": false}.
""",
    )


def get_post_call_prompt() -> str:
    """Returns the CRM intelligence analysis prompt for finished calls."""
    cfg = _load_prompts_yaml()
    return cfg.get(
        "post_call_prompt",
        """\
You are an expert College Admissions & Student Affairs CRM Auditor analyzing a concluded voice call transcript.
Return ONLY a raw, valid JSON object with keys: intent, lead_status, sentiment, summary.
""",
    )


def get_fallback(key: str, default: str = "") -> str:
    """Retrieves a fallback phrase from the configuration."""
    cfg = _load_prompts_yaml()
    fallbacks = cfg.get("fallbacks", {})
    return fallbacks.get(key, default)


def detect_active_domains(user_text: str, retrieved_docs: Optional[list[dict[str, Any]]] = None) -> list[str]:
    """Retained for backward-compatibility; returns empty list as knowledge is self-contained."""
    return []


def get_system_prompt(active_domains: Optional[list[str]] = None) -> str:
    """
    Constructs the system prompt:
    - Base Persona & Description
    - Hardcoded Step-by-Step Call Flow
    - Multilingual Policy (Gujarati primary, Hindi/English secondary)
    - Strict Spoken Voice Rules (No markdown, no tools, no hold phrases)
    - Admissions & Eligibility Knowledge (10th/12th, GUJCET result & rank, fees)
    - Complete Curriculum Knowledge (Semesters 1 to 8 subjects & electives)
    - GATE Subjects Knowledge & Important Topics
    - Placements Knowledge (6 LPA avg, 24 LPA highest off-campus, 200 placed, recruiters)
    - Hangup and Silence Handling
    - Guardrails & Emotional Intelligence
    """
    cfg = _load_prompts_yaml()
    persona = cfg.get("base_persona", {})
    p_name = persona.get("name", "Priya")
    p_role = persona.get("role", "Official AI Voice Assistant for DDU IT Department")
    p_inst = persona.get("institution", "Dharamsinh Desai University (DDU) IT Department in Nadiad, Gujarat")
    p_desc = persona.get("description", "You are on an active live telephone call. Everything you output is spoken aloud. You know all information directly. There is NO database and NO document search / RAG tool.")

    # Persona header
    parts = [
        f"You are {p_name}, the {p_role} for {p_inst}.",
        p_desc,
    ]

    # Call Flow
    call_flow = cfg.get("call_flow", "").strip()
    if call_flow:
        parts.append("")
        parts.append("HARDCODED CONVERSATION FLOW (FOLLOW IN EXACT SEQUENCE):")
        parts.append(call_flow)

    # Multilingual Priority & Sticky Language
    lang_policy = cfg.get("language_policy", {})
    if lang_policy:
        parts.append("")
        parts.append("MULTILINGUAL POLICY (PRIMARY: GUJARATI, SECONDARY: HINDI & ENGLISH):")
        parts.append(f"- Primary Language: {lang_policy.get('primary', 'Gujarati (ગુજરાતી)')}")
        secondaries = lang_policy.get('secondary', ['Hindi (हिंदी)', 'English'])
        parts.append(f"- Secondary Languages: {', '.join(secondaries)}")
        for r in lang_policy.get("rules", []):
            parts.append(f"- {r}")

    # Voice rules
    rules = cfg.get("voice_rules", [])
    if rules:
        parts.append("")
        parts.append("STRICT SPOKEN VOICE RULES:")
        for rule in rules:
            parts.append(rule)

    # Admissions Knowledge
    adm = cfg.get("admissions_knowledge", {})
    if adm:
        parts.append("")
        parts.append("ADMISSION & ELIGIBILITY DETAILS:")
        elig = adm.get("eligibility", {})
        parts.append(f"- 10th and 12th Board Results: {elig.get('board_requirements', '')}")
        parts.append(f"- GUJCET Result: {elig.get('entrance_exams', '')}")
        parts.append(f"- GUJCET Merit Rank: {elig.get('merit_counseling', '')}")
        fees = adm.get("fees_structure", {})
        parts.append(f"- Fees Structure: 1st Year (Admission): {fees.get('first_year_fee', '')}; Subsequent Years (2nd, 3rd, 4th): {fees.get('subsequent_years_fee', '')}.")
        parts.append(f"- Course Duration: {fees.get('course_duration', '4 years (8 semesters)')}")

    # Complete Curriculum Knowledge (Semesters 1 to 8)
    curr = cfg.get("curriculum_knowledge", {})
    if curr:
        parts.append("")
        parts.append("OFFICIAL B.TECH IT CURRICULUM (SEMESTERS 1 TO 8):")
        sem_labels = {
            "sem_1": "Semester 1",
            "sem_2": "Semester 2",
            "sem_3": "Semester 3",
            "sem_4": "Semester 4",
            "sem_5": "Semester 5",
            "sem_6": "Semester 6",
            "sem_7": "Semester 7",
            "sem_8": "Semester 8",
        }
        for sem_key, sem_title in sem_labels.items():
            subs = curr.get(sem_key, [])
            if subs:
                subs_list = ", ".join(subs)
                parts.append(f"- {sem_title}: {subs_list}")

        pec = curr.get("professional_electives", {})
        if pec:
            pec_strs = [f"{k.upper()}: {', '.join(v)}" for k, v in pec.items()]
            parts.append(f"- Professional Electives (PEC): {'; '.join(pec_strs)}")

        oe = curr.get("open_electives", {})
        if oe:
            oe_strs = [f"{v}" for k, v in oe.items()]
            parts.append(f"- Open Electives (OE): {'; '.join(oe_strs)}")

    # GATE Subjects Knowledge
    gate_data = cfg.get("gate_subjects_knowledge", {})
    if gate_data:
        parts.append("")
        parts.append("GATE SUBJECTS COVERED IN DDU CURRICULUM & IMPORTANT TOPICS:")
        parts.append(f"Overview: {gate_data.get('overview', '')}")
        for item in gate_data.get("mapping", []):
            subj = item.get("subject", "")
            topics = item.get("topics", "")
            covered = item.get("covered_in_ddu", "")
            parts.append(f"- {subj}: Important Topics: {topics} (Covered in: {covered})")

    # Placements Knowledge
    place = cfg.get("placements_knowledge", {})
    if place:
        parts.append("")
        parts.append("PLACEMENT DETAILS:")
        parts.append(f"- Average Package: {place.get('average_package', '6 LPA')}")
        parts.append(f"- Highest Package: {place.get('highest_package', '24 LPA (off-campus)')}")
        parts.append(f"- Total Students Placed: {place.get('total_placed_students', '200 placed students')}")
        parts.append(f"- Top Recruiters: {place.get('top_recruiters', 'TCS, Infosys, Crest Data Systems, TatvaSoft, Zeus Learning, eInfochips')}")

    # Hangup and Silence Handling (Global Overrides)
    hangup_silence = cfg.get("hangup_and_silence_handling", "").strip()
    if hangup_silence:
        parts.append("")
        parts.append("HANGUP COMMAND AND SILENCE HANDLING (GLOBAL OVERRIDES):")
        parts.append(hangup_silence)

    # Guardrails
    guardrails = cfg.get("guardrails", {})
    if guardrails:
        parts.append("")
        parts.append("SAFETY & INTEGRITY GUARDRAILS:")
        for g_cat, g_rules in guardrails.items():
            if isinstance(g_rules, list):
                for gr in g_rules:
                    parts.append(f"- {gr}")

    # Emotional Intelligence & Empathy
    eq_cfg = cfg.get("emotional_intelligence", {})
    if eq_cfg:
        parts.append("")
        parts.append("EMPATHY & ADAPTIVE TONE:")
        for eq_key, eq_val in eq_cfg.items():
            if isinstance(eq_val, str) and eq_val.strip():
                parts.append(f"- [{eq_key.replace('_', ' ').title()}]: {eq_val.strip()}")

    return "\n".join(parts)
