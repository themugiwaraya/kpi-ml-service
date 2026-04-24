"""Role-to-KPI-block configuration for ML normalization."""

from __future__ import annotations

BLOCK_FEATURES = ("block_1", "block_2", "block_3", "block_4")

# Which KPI blocks are applicable per role.
ROLE_BLOCK_MAP: dict[str, set[str]] = {
    "PROFESSOR": {"block_1", "block_2", "block_3"},
    "ASSOCIATE_PROFESSOR": {"block_1", "block_2", "block_3"},
    "ASSISTANT_PROFESSOR": {"block_1", "block_2", "block_3"},
    "SENIOR_LECTURER": {"block_1", "block_2", "block_3"},
    "LECTURER": {"block_1", "block_2"},
    "PE_TEACHER": {"block_1", "block_2"},
    "PE_SENIOR_TEACHER": {"block_1", "block_2"},
    "SENIOR_PE_TEACHER": {"block_1", "block_2"},
    "ACM_HEAD_COACH": {"block_1", "block_2", "block_3", "block_4"},
    "ACM_COACH": {"block_1", "block_2", "block_3", "block_4"},
}


def normalize_role_name(role: str) -> str:
    return str(role or "").strip().upper().replace("-", "_").replace(" ", "_")


def applicable_blocks_for_role(role: str) -> set[str]:
    # Keep all blocks for unknown roles to avoid accidental data loss.
    return ROLE_BLOCK_MAP.get(normalize_role_name(role), set(BLOCK_FEATURES))
