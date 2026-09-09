"""Explainable risk detection using local legal-knowledge RAG + Ollama."""

from __future__ import annotations

import json
import re
from typing import Callable

from ..legal_knowledge.retriever import retrieve_legal_guidance
from ..legal_knowledge.retriever import map_clause_category
from ..rag.generator import call_ollama

EMPTY_SOURCE = {"name": None, "url": None, "title": None}
ProgressCallback = Callable[[dict], None]

# MVP indicators are intentionally small and correspond to the documented
# source-backed indicators in backend/data/legal_knowledge/risk_indicators.md.
RISK_INDICATOR_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "Limitation of Liability": [
        ("uncapped or unlimited liability wording", r"\b(?:uncapped|unlimited|without limitation)\b"),
        ("broad damages exclusion", r"\b(?:indirect|consequential|incidental|special|punitive)\s+damages\b"),
        ("unclear liability cap or carve-out structure", r"\b(?:cap|liability shall not exceed|aggregate liability|fees paid)\b.{0,160}\b(?:except|excluding|carve[- ]?out|not apply|shall not apply|indemnif|confidential|data|fraud|willful|statutory)\b"),
        ("carve-out language", r"\b(?:except|excluding|carve[- ]?out|not apply|shall not apply)\b"),
        ("indemnity coordination reference", r"\b(?:indemnify|indemnification|defend|hold harmless)\b"),
    ],
    "Indemnification": [
        ("defend, indemnify, or hold harmless obligation", r"\b(?:defend|indemnif(?:y|ies|ication)|hold harmless)\b"),
        ("third-party claim scope", r"\bthird[- ]party\s+claims?\b"),
        ("settlement control or approval", r"\b(?:settle|settlement|consent|approval)\b"),
        ("notice procedure", r"\bnotice\b"),
        ("broad loss language", r"\b(?:all|any)\s+(?:losses|claims|damages|liabilities|expenses|costs)\b"),
    ],
    "Termination": [
        ("immediate termination", r"\bterminat(?:e|ion)\b.{0,80}\b(?:immediately|without notice)\b"),
        ("termination without cause or stated process", r"\bterminat(?:e|ion)\b.{0,120}\b(?:without cause|at any time)\b"),
        ("cure period", r"\b(?:cure period|opportunity to cure|\d+\s+days?\s+(?:to\s+)?cure)\b"),
        ("notice period", r"\b\d+\s+days?\s+(?:prior\s+)?(?:written\s+)?notice\b"),
        ("renewal window", r"\b(?:renewal|auto[- ]?renew|non[- ]?renewal)\b"),
        ("post-termination obligations", r"\b(?:transition|survival|return|retention|accrued|payment|data)\b"),
    ],
    "Intellectual Property": [
        ("ownership allocation wording", r"\b(?:own|owns|owned|ownership|belong|belongs|vest|vests)\b.{0,160}\b(?:intellectual property|ip|rights?|technology|inventions?|work product|developed|created|results?|improvements?)\b|\b(?:intellectual property|ip|rights?|technology|inventions?|work product|developed|created|results?|improvements?)\b.{0,160}\b(?:own|owns|owned|ownership|belong|belongs|vest|vests)\b"),
        ("assignment of IP rights", r"\bassign(?:s|ed|ment)?\b.{0,160}\b(?:all\s+)?(?:right,\s*title,\s+and\s+interest|intellectual property|ip|patents?|copyrights?|trademarks?|technology|inventions?|know[- ]?how|trade secrets?)\b"),
        ("broad license scope", r"\blicen[cs](?:e|es|ed|ing)?\b.{0,260}\b(?:worldwide|perpetual|irrevocable|exclusive|sublicen[cs]e|transferable|any purpose|modify|reproduce|distribute|commerciali[sz]e|sell|make|have made)\b"),
        ("unclear license rights", r"\blicen[cs](?:e|es|ed|ing)?\b"),
        ("software or source-code licensing", r"\b(?:software|source code|object code|technology|technical information)\b.{0,160}\blicen[cs](?:e|es|ed|ing)?\b|\blicen[cs](?:e|es|ed|ing)?\b.{0,160}\b(?:software|source code|object code|technology|technical information)\b"),
        ("confidentiality in technology transfer", r"\b(?:confidential|non[- ]?disclosure|nda|trade secret|know[- ]?how)\b.{0,180}\b(?:technology|intellectual property|ip|license|transfer|background)\b"),
    ],
    "Confidentiality": [
        ("unclear confidential information definition", r"\bconfidential(?: information)?\b.{0,180}\b(?:all information|anything|any information)\b"),
        ("unclear access restrictions", r"\bconfidential\b.{0,220}\b(?:disclose|access|use|third part(?:y|ies)|representative)s?\b"),
        ("unclear confidentiality duration", r"\bconfidential\b.{0,260}\b(?:forever|perpetual|indefinite|duration|years?|months?)\b"),
    ],
    "Payment": [
        ("unclear payment terms", r"\b(?:pay(?:ment)?|invoice|fees?|charges?)\b.{0,180}\b(?:promptly|reasonable time|as agreed|upon receipt)\b"),
        ("unclear late payment consequences", r"\b(?:late|overdue|delinquent)\b.{0,120}\b(?:interest|suspend|terminate|fee|charge|remedy)\b"),
    ],
    "Force Majeure": [
        ("broad force majeure definition", r"\bforce majeure\b.{0,220}\b(?:any event|all events|anything|without limitation)\b"),
        ("unclear force majeure notice requirement", r"\bforce majeure\b.{0,220}\b(?:notice|notify|inform)\b"),
        ("unclear mitigation obligation", r"\bforce majeure\b.{0,260}\b(?:mitigat|overcome|reasonable efforts|alternative)\w*\b"),
    ],
    "Dispute Resolution": [
        ("unclear arbitration or ADR mechanics", r"\b(?:arbitrat(?:e|ion)|mediat(?:e|ion)|adr)\b.{0,260}\b(?:place|seat|rules?|number|appoint|notice|confidential|costs?)\b"),
    ],
    "Insurance": [
        ("unclear insurance coverage", r"\binsurance\b.{0,240}\b(?:coverage|limit|policy|certificate|additional insured|maintain)\b"),
    ],
    "Warranties": [
        ("unclear warranty scope or disclaimer", r"\bwarrant(?:y|ies)\b.{0,220}\b(?:as is|disclaim|excluded|limited|remedy|duration)\b"),
    ],
    "Assignment": [
        ("unclear assignment or change-of-control scope", r"\bassign(?:ment|ed|s)?\b.{0,220}\b(?:affiliate|control|consent|without consent|all rights)\b"),
    ],
}


def _source_from_guidance(guidance: list[dict]) -> dict:
    if not guidance:
        return EMPTY_SOURCE.copy()
    first = guidance[0]
    return {
        "name": first.get("source_name"),
        "url": first.get("source_url"),
        "title": first.get("title"),
    }


def _legal_knowledge_from_guidance(guidance: list[dict]) -> list[dict]:
    return [
        {
            "source_name": item.get("source_name"),
            "source_url": item.get("source_url"),
            "title": item.get("title"),
            "clause_category": item.get("clause_category"),
            "relevance_score": item.get("relevance_score"),
            "retrieved_text": item.get("retrieved_text", "")[:700],
        }
        for item in guidance
    ]


def _format_guidance(guidance: list[dict]) -> str:
    if not guidance:
        return "None"
    blocks = []
    for index, item in enumerate(guidance, start=1):
        blocks.append(
            f"{index}. Source: {item.get('source_name')}\n"
            f"   Title: {item.get('title')}\n"
            f"   URL: {item.get('source_url')}\n"
            f"   Category: {item.get('clause_category')}\n"
            f"   Guidance: {item.get('retrieved_text')}"
        )
    return "\n\n".join(blocks)


def detect_legal_indicators(clause_text: str, clause_type: str) -> dict[str, list[str]]:
    category = map_clause_category(clause_type)
    if not category:
        return {
            "clause_category": [],
            "matched_indicators": [],
            "source_documentation": ["backend/data/legal_knowledge/risk_indicators.md"],
        }

    text = str(clause_text or "")
    matches = [
        label
        for label, pattern in RISK_INDICATOR_PATTERNS.get(category, [])
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    ]
    return {
        "clause_category": [category],
        "matched_indicators": matches,
        "source_documentation": ["backend/data/legal_knowledge/risk_indicators.md"],
    }


def build_risk_prompt(
    clause_text: str,
    clause_type: str,
    legal_indicators: dict[str, list[str]],
    guidance: list[dict],
) -> str:
    indicators_json = json.dumps(legal_indicators, indent=2, ensure_ascii=False)
    return f"""You are a contract-review support assistant. This is not legal advice.

CUAD classification identifies clause type only. It does not decide whether a clause is risky.
Retrieved legal guidance is supporting evidence only. Do not invent legal sources.
Analyze the actual clause text. Do not assume every unusual clause is risky.
Do not classify a clause as risky based only on semantic similarity to guidance.
Detected indicators are review cues only. Do not classify a clause as risky based only on an indicator label.
A liability cap with a clear monetary limit is not a potential uncapped-liability risk by itself.
Do not infer risk from missing terms unless the clause itself makes that omission operationally important.
Do not mark a simple capped-liability clause as risky merely because it does not list carve-outs.
For Intellectual Property clauses, use documented indicators for ownership, assignment, license scope, license rights, software licensing, and technology-transfer confidentiality.
For Intellectual Property clauses, do not treat every license, assignment, ownership, or confidentiality clause as risky.
For Intellectual Property clauses, a license that is limited by territory, duration, non-exclusivity, and internal or specific use is usually not a broad license-scope indicator by itself.
For Intellectual Property assignments, a reference to specifically described IP, a schedule, exhibit, named technology, patent, copyright, trademark, or defined asset can be sufficient evidence of clearer assignment scope.
Set risk to true only when the clause contains exact language that creates a potential review issue.
Use "Potential Risk" or "Risk Indicator" wording, not definitive legal violation wording.
Include exact evidence copied from the contract clause.
Explain why the clause may require human review.
Return JSON only.

Clause Type:
{clause_type}

Clause:
{clause_text}

Detected Legal Indicators:
{indicators_json}

Retrieved Legal Guidance:
{_format_guidance(guidance)}

Return exactly this JSON shape:
{{
  "risk": true,
  "risk_type": "Potential Risk Indicator",
  "reason": "Why this may require review, grounded in the clause and guidance.",
  "evidence": "Exact words from the contract clause.",
  "source": {{
    "name": "source name from retrieved guidance or null",
    "url": "source URL from retrieved guidance or null",
    "title": "source title from retrieved guidance or null"
  }}
}}

For no detected issue, set risk to false and risk_type to null.
"""


def parse_risk_response(raw_response: str, fallback_source: dict) -> dict:
    parsed = None
    try:
        parsed = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", str(raw_response), re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, dict):
        return {
            "risk": False,
            "risk_type": None,
            "reason": "Risk detector returned an invalid response, so no potential risk was recorded.",
            "evidence": "",
            "source": fallback_source,
        }

    risk = bool(parsed.get("risk", False))
    risk_type = parsed.get("risk_type") if risk else None
    source = parsed.get("source") if isinstance(parsed.get("source"), dict) else {}
    return {
        "risk": risk,
        "risk_type": str(risk_type).strip() if risk_type else None,
        "reason": str(parsed.get("reason") or "").strip(),
        "evidence": str(parsed.get("evidence") or "").strip(),
        "source": {
            "name": source.get("name") or fallback_source.get("name"),
            "url": source.get("url") or fallback_source.get("url"),
            "title": source.get("title") or fallback_source.get("title"),
        },
    }


def no_risk_result(
    reason: str,
    source: dict | None = None,
    legal_knowledge: list[dict] | None = None,
) -> dict:
    return {
        "risk": False,
        "risk_type": None,
        "reason": reason,
        "evidence": "",
        "source": source or EMPTY_SOURCE.copy(),
        "legal_knowledge": legal_knowledge or [],
    }


def _specific_risk_type(result: dict, indicators: dict[str, list[str]], clause_type: str) -> dict:
    if not result.get("risk"):
        return result
    if str(result.get("risk_type") or "").strip().lower() not in {"", "potential risk indicator", "risk indicator", "potential risk"}:
        return result
    category = map_clause_category(clause_type)
    labels = set(indicators.get("matched_indicators", []))
    by_category = {
        "Limitation of Liability": {
            "unclear liability cap or carve-out structure": "Unclear Liability Cap",
            "uncapped or unlimited liability wording": "Potential Uncapped Liability",
            "broad damages exclusion": "Broad Liability Exclusion",
        },
        "Indemnification": {
            "defend, indemnify, or hold harmless obligation": "Broad Indemnification",
            "broad loss language": "Unclear Indemnity Scope",
            "settlement control or approval": "Unclear Defense Obligation",
        },
        "Termination": {
            "termination without cause or stated process": "Unilateral Termination",
            "immediate termination": "Immediate Termination Right",
            "cure period": "Missing Cure Period",
            "notice period": "Unclear Termination Mechanics",
        },
        "Intellectual Property": {
            "ownership allocation wording": "Unclear IP Ownership",
            "broad license scope": "Broad IP License Scope",
            "assignment of IP rights": "Unclear Assignment Scope",
            "unclear license rights": "Unclear License Rights",
        },
        "Confidentiality": {
            "unclear confidential information definition": "Unclear Confidential Information Definition",
            "unclear access restrictions": "Unclear Access Restrictions",
            "unclear confidentiality duration": "Unclear Confidentiality Duration",
        },
        "Force Majeure": {
            "broad force majeure definition": "Broad Force Majeure Definition",
            "unclear force majeure notice requirement": "Unclear Notice Requirement",
            "unclear mitigation obligation": "Unclear Mitigation Obligation",
        },
        "Payment": {
            "unclear payment terms": "Unclear Payment Terms",
            "unclear late payment consequences": "Unclear Late Payment Consequences",
        },
        "Dispute Resolution": {"unclear arbitration or ADR mechanics": "Unclear ADR Mechanics"},
        "Insurance": {"unclear insurance coverage": "Unclear Insurance Coverage"},
        "Warranties": {"unclear warranty scope or disclaimer": "Unclear Warranty Scope"},
        "Assignment": {"unclear assignment or change-of-control scope": "Unclear Assignment Scope"},
    }
    for label, risk_type in by_category.get(category or "", {}).items():
        if label in labels:
            return {**result, "risk_type": risk_type}
    return result


def _is_clear_liability_cap_without_indicator(
    clause_text: str,
    clause_type: str,
    indicators: dict[str, list[str]],
) -> bool:
    if map_clause_category(clause_type) != "Limitation of Liability":
        return False
    if indicators.get("matched_indicators"):
        return False

    text = str(clause_text or "")
    has_cap = re.search(
        r"\b(?:total\s+)?(?:aggregate\s+)?liability\b.{0,80}\b(?:shall\s+not\s+exceed|will\s+not\s+exceed|is\s+limited\s+to)\b.{0,80}(?:[$]\s?\d|usd\s?\d|\d[\d,]*(?:\.\d+)?\s?usd)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    has_uncapped_language = re.search(
        r"\b(?:uncapped|unlimited|without limitation)\b",
        text,
        re.IGNORECASE,
    )
    return bool(has_cap and not has_uncapped_language)


def _is_clear_limited_ip_license_without_indicator(clause_text: str, clause_type: str) -> bool:
    if map_clause_category(clause_type) != "Intellectual Property":
        return False
    text = str(clause_text or "")
    has_license = re.search(r"\blicen[cs](?:e|es|ed|ing)?\b", text, re.IGNORECASE)
    has_clear_limits = [
        re.search(r"\bnon[- ]?exclusive\b", text, re.IGNORECASE),
        re.search(r"\b(?:limited to|solely for|only for|internal use|field of use|territory|united kingdom|uk)\b", text, re.IGNORECASE),
        re.search(r"\b(?:\d+\s+years?|for\s+three\s+years?|valid for)\b", text, re.IGNORECASE),
    ]
    has_broad_bundle = re.search(
        r"\b(?:irrevocable|worldwide|perpetual|(?<!non-)exclusive|sublicen[cs]e|any purpose|transferable)\b",
        text,
        re.IGNORECASE,
    )
    return bool(has_license and all(has_clear_limits) and not has_broad_bundle)


def _is_clear_ip_assignment_without_indicator(clause_text: str, clause_type: str) -> bool:
    if map_clause_category(clause_type) != "Intellectual Property":
        return False
    text = str(clause_text or "")
    has_assignment = re.search(r"\bassign(?:s|ed|ment)?\b", text, re.IGNORECASE)
    has_subject_matter = re.search(
        r"\b(?:described in|set forth in|listed in|identified in|schedule|exhibit|patent|copyright|trademark|technology|work product|invention|source code|software)\b",
        text,
        re.IGNORECASE,
    )
    return bool(has_assignment and has_subject_matter)


def _is_clear_ip_ownership_without_indicator(clause_text: str, clause_type: str) -> bool:
    if map_clause_category(clause_type) != "Intellectual Property":
        return False
    text = str(clause_text or "")
    has_ownership = re.search(
        r"\b(?:intellectual property|ip|work product|inventions?|technology|results?)\b.{0,180}\b(?:created|developed|made|generated)\b.{0,180}\b(?:belong|belongs|owned by|vest|vests|shall be owned)\b|\b(?:belong|belongs|owned by|vest|vests|shall be owned)\b.{0,180}\b(?:customer|supplier|licensee|licensor|party)\b",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    has_ambiguous_owner = re.search(
        r"\b(?:as agreed|to be agreed|mutually agreed|jointly owned|shared ownership|respective rights|as applicable)\b",
        text,
        re.IGNORECASE,
    )
    return bool(has_ownership and not has_ambiguous_owner)


def _strong_ip_indicator_result(
    clause_text: str,
    clause_type: str,
    indicators: dict[str, list[str]],
    source: dict,
    parsed_result: dict,
    legal_knowledge: list[dict] | None = None,
) -> dict | None:
    if map_clause_category(clause_type) != "Intellectual Property":
        return None
    if parsed_result.get("risk"):
        return None

    text = str(clause_text or "")
    matches = indicators.get("matched_indicators", [])
    has_broad_license = (
        "broad license scope" in matches
        and re.search(r"\blicen[cs](?:e|es|ed|ing)?\b", text, re.IGNORECASE)
        and len(
            re.findall(
                r"\b(?:irrevocable|worldwide|perpetual|exclusive|sublicen[cs]e|any purpose|modify|reproduce|distribute)\b",
                text,
                re.IGNORECASE,
            )
        )
        >= 4
    )
    if has_broad_license:
        return {
            "risk": True,
            "risk_type": "Broad IP License Scope",
            "reason": "Potential Risk Indicator: the clause grants a broad bundle of IP license rights across duration, territory, exclusivity, permitted acts, sublicensing, or purpose, which may require review against the documented IP licensing guidance.",
            "evidence": _sanitize_evidence(clause_text, clause_text),
            "source": source,
            "legal_knowledge": legal_knowledge or [],
        }
    return None


def _normalize_ip_risk_type(result: dict, indicators: dict[str, list[str]], clause_type: str) -> dict:
    if not result.get("risk") or map_clause_category(clause_type) != "Intellectual Property":
        return result

    matches = indicators.get("matched_indicators", [])
    risk_type = str(result.get("risk_type") or "").strip().lower()
    if risk_type and risk_type != "potential risk indicator":
        return result

    for label, normalized in (
        ("broad license scope", "Broad IP License Scope"),
        ("assignment of IP rights", "Unclear Assignment Scope"),
        ("ownership allocation wording", "Unclear IP Ownership"),
        ("unclear license rights", "Unclear License Rights"),
    ):
        if label in matches:
            return {**result, "risk_type": normalized}
    return result


def _sanitize_evidence(evidence: str, clause_text: str) -> str:
    evidence_text = str(evidence or "").strip()
    clause = str(clause_text or "").strip()
    if not evidence_text or not clause:
        return ""
    if evidence_text in clause:
        return evidence_text

    quoted = re.findall(r"['\"]([^'\"]+)['\"]", evidence_text)
    for candidate in quoted:
        candidate = candidate.strip()
        if candidate and candidate in clause:
            return candidate

    first_line = evidence_text.splitlines()[0].strip()
    if first_line in clause:
        return first_line
    return clause


def detect_clause_risk(
    clause_text: str,
    clause_type: str,
    *,
    cohere_client=None,
    qdrant_client=None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    if progress_callback:
        progress_callback({
            "type": "progress",
            "stage": "risk",
            "message": "Retrieving legal guidance for risk analysis",
        })
    guidance = retrieve_legal_guidance(
        clause_text,
        clause_type,
        cohere_client=cohere_client,
        qdrant_client=qdrant_client,
        progress_callback=progress_callback,
    )
    source = _source_from_guidance(guidance)
    if not guidance:
        return no_risk_result(
            "No documented MVP risk indicator is currently configured for this clause category.",
            source,
        )

    indicators = detect_legal_indicators(clause_text, clause_type)
    prompt = build_risk_prompt(
        clause_text=clause_text,
        clause_type=clause_type,
        legal_indicators=indicators,
        guidance=guidance,
    )
    if progress_callback:
        progress_callback({
            "type": "progress",
            "stage": "risk",
            "message": "Running the local risk analysis model",
        })
    raw_response = call_ollama(prompt)
    result = parse_risk_response(raw_response, source)
    if progress_callback:
        progress_callback({
            "type": "progress",
            "stage": "risk",
            "message": "Risk analysis complete",
            "risk": bool(result.get("risk", False)),
        })
    result["legal_knowledge"] = _legal_knowledge_from_guidance(guidance)
    allowed_sources = {
        (item.get("source_name"), item.get("source_url"), item.get("title"))
        for item in guidance
    }
    if (
        result.get("source", {}).get("name"),
        result.get("source", {}).get("url"),
        result.get("source", {}).get("title"),
    ) not in allowed_sources:
        result["source"] = source
    result["evidence"] = _sanitize_evidence(result.get("evidence", ""), clause_text)
    if result.get("risk") and not indicators.get("matched_indicators"):
        result = no_risk_result(
            "No documented MVP risk indicator was identified in the actual clause text.",
            source,
            result.get("legal_knowledge"),
        )
    result = _specific_risk_type(result, indicators, clause_type)
    result = _normalize_ip_risk_type(result, indicators, clause_type)
    strong_ip_result = _strong_ip_indicator_result(
        clause_text,
        clause_type,
        indicators,
        source,
        result,
        result.get("legal_knowledge"),
    )
    if strong_ip_result:
        return strong_ip_result
    if result["risk"] and _is_clear_liability_cap_without_indicator(
        clause_text,
        clause_type,
        indicators,
    ):
        return no_risk_result(
            "The clause states a clear monetary liability cap and no documented MVP risk indicator was detected in the clause text.",
            source,
            result.get("legal_knowledge"),
        )
    if result["risk"] and _is_clear_limited_ip_license_without_indicator(
        clause_text,
        clause_type,
    ):
        return no_risk_result(
            "The IP license is limited by non-exclusivity, territory, duration, and internal or specific use, so no documented broad license-scope indicator was detected.",
            source,
            result.get("legal_knowledge"),
        )
    if result["risk"] and _is_clear_ip_assignment_without_indicator(
        clause_text,
        clause_type,
    ):
        return no_risk_result(
            "The IP assignment identifies assigned rights by reference to described subject matter, so no documented unclear-assignment-scope indicator was detected.",
            source,
            result.get("legal_knowledge"),
        )
    if result["risk"] and _is_clear_ip_ownership_without_indicator(
        clause_text,
        clause_type,
    ):
        return no_risk_result(
            "The clause identifies developed intellectual property and states which party owns it, so no documented unclear-ownership indicator was detected.",
            source,
            result.get("legal_knowledge"),
        )
    return result
