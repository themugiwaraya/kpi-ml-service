"""Canonicalization helpers for departments and roles."""

from __future__ import annotations

from .role_block_config import normalize_role_name

CANONICAL_DEPARTMENTS = (
    "School of Software Engineering",
    "School of Digital Public Administration",
    "School of Creative Industries",
    "School of Intelligent Systems",
    "School of Cybersecurity",
    "School of Artificial Intelligence and Data Science",
)

CANONICAL_ROLES = (
    "PROFESSOR",
    "ASSOCIATE_PROFESSOR",
    "ASSISTANT_PROFESSOR",
    "SENIOR_LECTURER",
    "LECTURER",
    "ACM_HEAD_COACH",
    "ACM_COACH",
    "PE_TEACHER",
    "PE_SENIOR_TEACHER",
)

_CANONICAL_DEPARTMENT_KEYS = {
    " ".join(name.upper().split()): name
    for name in CANONICAL_DEPARTMENTS
}

# Add aliases here if raw data uses non-canonical department names.
_DEPARTMENT_ALIASES = {
    "ENGINEERING": "School of Software Engineering",
}


def _normalize_department_key(value: str) -> str:
    return " ".join(str(value or "").strip().upper().split())


def canonicalize_department(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    key = _normalize_department_key(raw)
    if key in _DEPARTMENT_ALIASES:
        return _DEPARTMENT_ALIASES[key]
    if key in _CANONICAL_DEPARTMENT_KEYS:
        return _CANONICAL_DEPARTMENT_KEYS[key]
    return raw


def department_candidates(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    canonical = canonicalize_department(raw)
    candidates = {raw, canonical}
    key = _normalize_department_key(raw)
    if key in _DEPARTMENT_ALIASES:
        candidates.add(key)
    for alias_key, canon in _DEPARTMENT_ALIASES.items():
        if canon == canonical:
            candidates.add(alias_key)
    return [c for c in candidates if c]


def canonicalize_role(value: str) -> str:
    normalized = normalize_role_name(value)
    if normalized in CANONICAL_ROLES:
        return normalized
    return normalized or str(value or "").strip()


def role_candidates(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    normalized = normalize_role_name(raw)
    candidates = {raw}
    if normalized:
        candidates.update({
            normalized,
            normalized.replace("_", " "),
            normalized.replace("_", "-"),
            normalized.replace("_", " ").title(),
        })
    return [c for c in candidates if c]
