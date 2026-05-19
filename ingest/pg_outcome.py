"""
Personal grievance outcome extraction for NZ ERA / NZEmpC decisions.

Extracts structured fields to power a PG Tracker comparable to Westlaw NZ.
Stored under payload key 'pg'.

Fields extracted:
  grievance_types         - list of strings: unjustified_dismissal, constructive_dismissal,
                            disadvantage, harassment, discrimination, unjustified_action
  reinstatement_ordered   - bool (True = ordered, False = declined/refused)
  contributory_conduct_pct - float (0-100), reduction applied for employee's own conduct
  has_contributory_conduct - bool (contributory conduct discussed but % not parsed)
  has_data                - bool
"""

import re
from typing import Any

# ---------------------------------------------------------------------------
# Grievance type detection
# ---------------------------------------------------------------------------

_GRIEVANCE_TYPES: list[tuple[str, re.Pattern]] = [
    (
        "unjustified_dismissal",
        re.compile(r"\bunjustif(?:ied|iable)\s+dismissal\b", re.IGNORECASE),
    ),
    (
        "constructive_dismissal",
        re.compile(r"\bconstructive\s+dismissal\b", re.IGNORECASE),
    ),
    (
        "disadvantage",
        re.compile(
            r"\bdisadvantage\s+grievance\b"
            r"|\bunjustified\s+disadvantage\b"
            r"|\bbreach\s+of\s+(?:the\s+)?duty\s+of\s+good\s+faith\b",
            re.IGNORECASE,
        ),
    ),
    (
        "harassment",
        re.compile(
            r"\b(?:sexual\s+harassment|bullying\s+and\s+harassment|"
            r"workplace\s+harassment|harassment\s+grievance)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "discrimination",
        re.compile(
            r"\bdiscrimination\s+(?:claim|grievance|complaint)\b"
            r"|\bdiscriminat(?:ory|ed)\s+(?:on\s+)?(?:the\s+)?ground",
            re.IGNORECASE,
        ),
    ),
    (
        "unjustified_action",
        re.compile(
            r"\bunjustified\s+(?:action|disadvantage)\s+grievance\b",
            re.IGNORECASE,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Reinstatement
# ---------------------------------------------------------------------------

_REINSTATEMENT_YES = re.compile(
    r"\border(?:ed|ing)?\s+(?:the\s+)?(?:applicant(?:'s)?\s+)?reinstatement\b"
    r"|\breinstatement\s+(?:is\s+|was\s+)?(?:ordered|granted|allowed|appropriate)\b"
    r"|\bapplicant\s+(?:be\s+|is\s+)?reinstated\b"
    r"|\bgrant(?:ed)?\s+(?:an?\s+order\s+of\s+)?reinstatement\b",
    re.IGNORECASE,
)

_REINSTATEMENT_NO = re.compile(
    r"\bdeclin(?:ed?|ing)\s+(?:an?\s+order\s+(?:of|for)\s+)?reinstatement\b"
    r"|\brefus(?:ed?|ing)\s+(?:to\s+order\s+)?reinstatement\b"
    r"|\breinstatement\s+(?:is\s+(?:not\s+)?(?:ordered|appropriate|practicable|reasonable)|"
    r"declined|refused|not\s+ordered|impracticable)\b"
    r"|\bnot\s+(?:practicable|appropriate|reasonable)\s+to\s+(?:order\s+)?reinstate",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Contributory conduct
# ---------------------------------------------------------------------------

_CONTRIB_PCT = re.compile(
    r"(\d{1,2})\s*(?:per\s*cent|%)\s*(?:reduction\s+)?(?:for\s+)?contributory\s+conduct"
    r"|contributory\s+conduct[^.]{0,80}?(\d{1,2})\s*(?:per\s*cent|%)"
    r"|reduce[ds]?\s+by\s+(\d{1,2})\s*(?:per\s*cent|%)\s+(?:for\s+)?contributory",
    re.IGNORECASE,
)

_CONTRIB_PRESENT = re.compile(
    r"\bcontributory\s+conduct\b|\bemployee(?:'s)?\s+contribution\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pg_outcome(text: str) -> dict[str, Any]:
    """
    Extract personal grievance outcome from an ERA/NZEmpC decision chunk.
    Returns dict with has_data=True if meaningful outcome data found.
    """
    result: dict[str, Any] = {"has_data": False}

    # Grievance types (collect all that match)
    types_found = [name for name, pat in _GRIEVANCE_TYPES if pat.search(text)]
    if types_found:
        result["grievance_types"] = types_found

    # Reinstatement determination
    yes = bool(_REINSTATEMENT_YES.search(text))
    no = bool(_REINSTATEMENT_NO.search(text))
    if yes and not no:
        result["reinstatement_ordered"] = True
    elif no:
        result["reinstatement_ordered"] = False

    # Contributory conduct %
    m = _CONTRIB_PCT.search(text)
    if m:
        pct_str = m.group(1) or m.group(2) or m.group(3)
        if pct_str:
            pct = int(pct_str)
            if 0 < pct <= 100:
                result["contributory_conduct_pct"] = float(pct)
    elif _CONTRIB_PRESENT.search(text):
        result["has_contributory_conduct"] = True

    if types_found or "reinstatement_ordered" in result or "contributory_conduct_pct" in result:
        result["has_data"] = True

    return result
