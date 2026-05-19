"""
Counsel/appearances extraction for NZ court decisions.

Parses the standard appearances block near the top of every NZ court decision
and returns structured counsel data suitable for Qdrant payload storage.

Payload stored under 'counsel':
  {
    "has_data": bool,
    "raw": "A Jones for Crown\nB Smith for defendant",
    "entries": [
      {"names": ["A Jones"], "role": "crown"},
      {"names": ["B Smith"], "role": "defendant"},
    ],
    "all_names":    ["A Jones", "B Smith"],   # keyword array - any counsel
    "all_surnames": ["Jones", "Smith"],        # keyword array - surname search
    "crown":        ["A Jones"],               # keyword array - prosecution
    "defence":      ["B Smith"],               # keyword array - defence
  }
"""

import re

# ---------------------------------------------------------------------------
# Block detection
# ---------------------------------------------------------------------------

# Standard format: header keyword on its own line, content on lines below.
# Handles Appearances: / Counsel: / COUNSEL (all courts except NZERA).
_STD_HEADER = re.compile(
    r"(?:^|\n)(?:Appearances?|Counsel|COUNSEL)\s*[:\t]?\s*\n+",
    re.MULTILINE,
)

# NZERA inline format: "Representatives:\tName, for the Applicant\n..."
# Content starts on the same line as the header (tab-separated).
_ERA_HEADER = re.compile(
    r"(?:^|\n)Representatives?\s*:\s*\t(.+?)(?:\n\n|Investigation Meeting:|Determination:|$)",
    re.MULTILINE | re.DOTALL,
)

# Block terminator - blank line or known section transition.
_BLOCK_END = re.compile(
    r"\n\n"
    r"|\n(?=Judgment\b|Hearing\b|Determination\b|Decision\b|Further\b|Order\b|"
    r"Date of|\[1\]|\[\d|\d{4}\s)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Role normalization
# ---------------------------------------------------------------------------

_ROLE_MAP: dict[str, str] = {
    "crown":               "crown",
    "prosecution":         "crown",
    "solicitor general":   "crown",
    "solicitor-general":   "crown",
    "appellant":           "appellant",
    "appellants":          "appellant",
    "respondent":          "respondent",
    "respondents":         "respondent",
    "first respondent":    "respondent",
    "second respondent":   "respondent",
    "applicant":           "applicant",
    "applicants":          "applicant",
    "plaintiff":           "plaintiff",
    "plaintiffs":          "plaintiff",
    "defendant":           "defendant",
    "defendants":          "defendant",
    "accused":             "defendant",
    "first defendant":     "defendant",
    "second defendant":    "defendant",
    "third defendant":     "defendant",
    "standards committee": "standards_committee",
    "practitioner":        "practitioner",
    "child":               "child",
    "complainant":         "complainant",
}

_DEFENCE_ROLES = {"defendant", "appellant"}

# ---------------------------------------------------------------------------
# Entry-level patterns
# ---------------------------------------------------------------------------

# Strips professional titles prepended to names ("Ms", "Mr", "Dr", etc.)
_TITLE_PFX  = re.compile(r"^(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+", re.IGNORECASE)
# Strips bar designations appended to names ("QC", "KC", "SC")
_BAR_SUFFIX = re.compile(r"\b(?:QC|KC|SC|JP)\b\.?\s*", re.IGNORECASE)

# Finds a role marker "for [the] Role" or "as Lawyer for [the] Role".
# Lookahead terminates at: next uppercase run, end-of-string, bracket, paren.
# Re.IGNORECASE so "for applicant" (lowercase) is matched.
_ROLE_MARKER = re.compile(
    r"(?:,\s*(?:counsel|advocate|solicitor|barrister|lawyer)\s*)?"
    r"(?:for|as\s+Lawyer\s+for)\s+(?:the\s+)?"
    r"((?:(?:First|Second|Third|Standards?\s+Committee)\s+)?[A-Za-z][a-zA-Z\s\-']{1,35}?)"
    r"(?=\s+[A-Z\[\(]|\s*[,\n]|\s*$)",
    re.IGNORECASE,
)


def _normalize_role(raw: str) -> str:
    key = raw.strip().lower()
    if key in _ROLE_MAP:
        return _ROLE_MAP[key]
    for pat, norm in _ROLE_MAP.items():
        if key.startswith(pat):
            return norm
    return key.replace(" ", "_")[:30]


def _clean_name(raw: str) -> str:
    name = _BAR_SUFFIX.sub("", raw)
    name = _TITLE_PFX.sub("", name)
    name = re.sub(r"[\s,]+$", "", name.strip())
    return re.sub(r"\s{2,}", " ", name).strip()


def _split_names(raw: str) -> list[str]:
    """
    Split 'A Jones and B Smith' or 'A Jones, B Smith' into individual names,
    cleaning each component and rejecting non-name tokens.
    """
    # Strip leading garbage: "[Butler]", "(via telephone)", "in person" prefix
    raw = re.sub(r"^(?:\[[^\]]{0,40}\]|\([^\)]{0,40}\))\s*", "", raw.strip())
    raw = re.sub(r"^.*?\bin\s+person\b\s*", "", raw, flags=re.IGNORECASE)
    raw = raw.strip()

    parts = re.split(r"\s+(?:and|with)\s+|,\s+(?=[A-Z])", raw, flags=re.IGNORECASE)
    result = []
    for p in parts:
        n = _clean_name(p)
        if (
            n and len(n) >= 2
            and re.search(r"[A-Za-z]", n)
            # Reject role words, generic terms and obvious non-names
            and not re.match(
                r"^(?:No\b|Plaintiff|Defendant|Applicant|Respondent|Appellant|Crown|"
                r"Standards|Practitioner|Counsel|Senior\s+Counsel|Junior\s+Counsel|"
                r"agent\b|advocate\b|solicitor\b|via\b|\[)",
                n, re.IGNORECASE,
            )
        ):
            result.append(n)
    return result


def _surname(name: str) -> str:
    clean = _BAR_SUFFIX.sub("", name)
    clean = _TITLE_PFX.sub("", clean)
    words = clean.split()
    return words[-1] if words else ""


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------

def _extract_block(text: str) -> str | None:
    """Return the raw appearances block, or None if the section is absent."""
    # Standard format: content on lines below the header
    m = _STD_HEADER.search(text)
    if m:
        start = m.end()
        rest  = text[start:]
        end_m = _BLOCK_END.search(rest)
        block = rest[: end_m.start()].strip() if end_m else rest[:600].strip()
        return block or None

    # NZERA inline format: "Representatives:\tName, for the Applicant\n..."
    m = _ERA_HEADER.search(text)
    if m:
        return m.group(1).strip() or None

    return None


# ---------------------------------------------------------------------------
# Entry parsing
# ---------------------------------------------------------------------------

def _parse_entries(block: str) -> list[dict]:
    """
    Parse name-role pairs from an appearances block.

    Strategy: scan for each 'for [Role]' marker; the text between the
    previous marker's end and the current marker's start is the names.
    This handles multiple entries on the same line naturally.
    """
    # Normalize whitespace: tabs -> spaces, lone newlines -> spaces
    flat = re.sub(r"\t", " ", block)
    flat = re.sub(r"\n", " ", flat)
    flat = re.sub(r"\s{2,}", " ", flat).strip()

    entries: list[dict] = []
    seen: set[str] = set()
    prev_end = 0

    for m in _ROLE_MARKER.finditer(flat):
        raw_names = flat[prev_end : m.start()].strip()
        raw_role  = m.group(1).strip()

        prev_end = m.end()

        # Skip "No appearance for ..."
        if re.match(r"no\s+appearance", raw_names, re.IGNORECASE):
            continue
        # Skip social worker, psychologist etc. (not legal counsel)
        if re.match(r"social\s+worker|psychologist|interpreter", raw_role, re.IGNORECASE):
            continue

        names = _split_names(raw_names)
        if not names:
            continue

        role = _normalize_role(raw_role)

        # Deduplicate
        key = f"{role}:{names[0]}"
        if key in seen:
            continue
        seen.add(key)

        entries.append({"names": names, "role": role})

    return entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_counsel(text: str) -> dict:
    """
    Extract counsel data from a chunk.
    Returns a dict for the Qdrant 'counsel' payload field.
    Always returns a dict; check has_data to see if extraction succeeded.
    """
    result: dict = {"has_data": False}

    block = _extract_block(text)
    if not block:
        return result

    result["raw"] = block[:500]

    entries = _parse_entries(block)
    if not entries:
        return result

    result["has_data"] = True
    result["entries"]  = entries

    all_names:    list[str] = []
    all_surnames: list[str] = []
    crown:        list[str] = []
    defence:      list[str] = []

    for e in entries:
        for n in e["names"]:
            if n not in all_names:
                all_names.append(n)
                s = _surname(n)
                if s and s not in all_surnames:
                    all_surnames.append(s)
        if e["role"] == "crown":
            crown.extend(n for n in e["names"] if n not in crown)
        elif e["role"] in _DEFENCE_ROLES:
            defence.extend(n for n in e["names"] if n not in defence)

    if all_names:    result["all_names"]    = all_names
    if all_surnames: result["all_surnames"] = all_surnames
    if crown:        result["crown"]        = crown
    if defence:      result["defence"]      = defence

    return result
