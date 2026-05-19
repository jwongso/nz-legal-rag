"""
Penalty/outcome extraction for NZ legal decisions.

Extracts prosecution asks and judicial outcomes from chunk text,
computes Outcome Severity Index (OSI) for criminal cases and
recovery rate for civil financial cases.

All values stored in a 'penalty' dict in the Qdrant payload.
has_data=False means extraction failed for that chunk (not all
chunks in a case contain sentencing/outcome language).
"""

import re

# --- Court type classification ---

_CRIMINAL_CONTEXT = {"NZHC", "NZCA", "NZSC"}   # also handle civil
_CIVIL_FINANCIAL  = {"NZTT", "NZERA", "NZEmpC", "NZACC"}
_CIVIL_MIXED      = {"NZFC"}
_CIVIL_ENV        = {"NZEnvC"}
_CORONAL          = {"NZCorC"}
_DISCIPLINARY     = {"NZLCDT", "NZREADT", "NZHRRT"}
_LEGISLATION      = {"NZLEG"}

_CRIMINAL_SIGNALS = re.compile(
    r"\bconvicted\b|\bsentenced to\b|\bimprisonment\b|\bguilty plea\b"
    r"|\bthe Crown\b|\bprosecution\b|\bcharged with\b|\boffend(?:er|ing)\b"
    r"|\bSolicitor.General\b|\bacquit",
    re.IGNORECASE,
)

def detect_court_type(court: str, text: str) -> str:
    if court in _LEGISLATION:
        return "legislation"
    if court in _CORONAL:
        return "coronal"
    if court in _CIVIL_ENV:
        return "civil_nonfinancial"
    if court in _CIVIL_MIXED:
        return "civil_mixed"
    if court in _CIVIL_FINANCIAL:
        return "civil_financial"
    if court in _DISCIPLINARY:
        return "civil_disciplinary"
    # NZHC/NZCA/NZSC: detect from text
    if _CRIMINAL_SIGNALS.search(text):
        return "criminal"
    return "civil_financial"


# --- Criminal OSI scale ---

def _months_to_osi(months: int) -> float:
    if months < 6:   return 0.35
    if months < 12:  return 0.42
    if months < 24:  return 0.50
    if months < 48:  return 0.58
    if months < 84:  return 0.65
    if months < 120: return 0.73
    if months < 168: return 0.81
    return 0.88


_LIFE_NO_PAROLE = re.compile(r"life\s+imprisonment\s+without\s+parole", re.IGNORECASE)
_LIFE           = re.compile(r"\blife\s+imprisonment\b", re.IGNORECASE)
_PREV_DET       = re.compile(r"\bpreventive\s+detention\b", re.IGNORECASE)
_HOME_DET       = re.compile(r"home\s+detention\s+(?:for\s+)?(\d+)\s*(months?|years?)", re.IGNORECASE)
_COMMUNITY_WORK = re.compile(r"(\d+)\s*hours?\s*(?:of\s+)?community\s+work", re.IGNORECASE)
_SUPERVISION    = re.compile(r"\bintensive\s+supervision\b|\bsupervision\s+(?:order|for)\b", re.IGNORECASE)
_COMM_DET       = re.compile(r"\bcommunity\s+detention\b", re.IGNORECASE)
_DISQUALIFY     = re.compile(r"disqualif\w+\s+from\s+(?:driving|holding)", re.IGNORECASE)
_DISCHARGE_NO   = re.compile(r"discharged?\s+without\s+conviction", re.IGNORECASE)
_CONV_DISCHARGE = re.compile(r"convicted?\s+and\s+discharged?", re.IGNORECASE)
_FINE           = re.compile(r"fined?\s+\$?([\d,]+)", re.IGNORECASE)

# Primary imprisonment pattern: "X years Y months" or "X months"
_IMPRISON_YM = re.compile(
    r"(?:sentenced?\s+to|sentence\s+of|imprisonment\s+(?:for|of)|"
    r"term\s+of\s+imprisonment\s+of)\s+"
    r"(\d+)\s*years?\s*(?:and\s*)?(\d+)?\s*months?",
    re.IGNORECASE,
)
_IMPRISON_M = re.compile(
    r"(?:sentenced?\s+to|sentence\s+of|imprisonment\s+(?:for|of))\s+"
    r"(\d+)\s*months?\s*(?:imprisonment|'s imprisonment)",
    re.IGNORECASE,
)
_IMPRISON_Y = re.compile(
    r"(?:sentenced?\s+to|sentence\s+of|imprisonment\s+(?:for|of))\s+"
    r"(\d+)\s*years?\s*(?:imprisonment|'s imprisonment|\b)",
    re.IGNORECASE,
)

# Crown recommendation patterns
_CROWN_YM = re.compile(
    r"(?:Crown\s+(?:submits?|seeks?|recommends?|asks?\s+for|suggests?)|"
    r"starting\s+point\s+of)\s+"
    r"(\d+)\s*years?\s*(?:and\s*)?(\d+)?\s*months?",
    re.IGNORECASE,
)
_CROWN_M = re.compile(
    r"(?:Crown\s+(?:submits?|seeks?|recommends?|asks?\s+for)|"
    r"starting\s+point\s+of)\s+"
    r"(\d+)\s*months?",
    re.IGNORECASE,
)
_CROWN_Y = re.compile(
    r"(?:Crown\s+(?:submits?|seeks?|recommends?|asks?\s+for)|"
    r"starting\s+point\s+of)\s+"
    r"(\d+)\s*years?",
    re.IGNORECASE,
)
_CROWN_LIFE = re.compile(
    r"Crown\s+(?:submits?|seeks?|recommends?).{0,40}life\s+imprisonment",
    re.IGNORECASE,
)


def _extract_imprisonment_months(text: str) -> int | None:
    m = _IMPRISON_YM.search(text)
    if m:
        years = int(m.group(1))
        months = int(m.group(2)) if m.group(2) else 0
        return years * 12 + months
    m = _IMPRISON_M.search(text)
    if m:
        return int(m.group(1))
    m = _IMPRISON_Y.search(text)
    if m:
        return int(m.group(1)) * 12
    return None


def _extract_crown_months(text: str) -> int | None:
    if _CROWN_LIFE.search(text):
        return 9999
    m = _CROWN_YM.search(text)
    if m:
        years = int(m.group(1))
        months = int(m.group(2)) if m.group(2) else 0
        return years * 12 + months
    m = _CROWN_M.search(text)
    if m:
        return int(m.group(1))
    m = _CROWN_Y.search(text)
    if m:
        return int(m.group(1)) * 12
    return None


def _extract_criminal(text: str) -> dict | None:
    result: dict = {"court_type": "criminal", "has_data": False}

    # Outcome
    outcome_osi: float | None = None
    outcome_label: str | None = None

    if _LIFE_NO_PAROLE.search(text):
        outcome_osi = 1.00
        outcome_label = "Life imprisonment without parole"
    elif _LIFE.search(text):
        outcome_osi = 0.95
        outcome_label = "Life imprisonment"
    elif _PREV_DET.search(text):
        outcome_osi = 0.92
        outcome_label = "Preventive detention"
    else:
        months = _extract_imprisonment_months(text)
        if months is not None:
            outcome_osi = _months_to_osi(months)
            y, m = divmod(months, 12)
            outcome_label = (
                f"{y} year{'s' if y != 1 else ''}"
                + (f" {m} month{'s' if m != 1 else ''}" if m else "")
                + " imprisonment"
            )
        else:
            m_hd = _HOME_DET.search(text)
            if m_hd:
                n = int(m_hd.group(1))
                unit = m_hd.group(2).lower()
                months_hd = n * 12 if "year" in unit else n
                outcome_osi = 0.28
                outcome_label = f"Home detention {n} {unit}"
            elif _COMMUNITY_WORK.search(text):
                m_cw = _COMMUNITY_WORK.search(text)
                hrs = int(m_cw.group(1))
                outcome_osi = 0.12
                outcome_label = f"{hrs} hours community work"
            elif _COMM_DET.search(text):
                outcome_osi = 0.22
                outcome_label = "Community detention"
            elif _SUPERVISION.search(text):
                outcome_osi = 0.15
                outcome_label = "Supervision"
            elif _DISCHARGE_NO.search(text):
                outcome_osi = 0.00
                outcome_label = "Discharged without conviction"
            elif _CONV_DISCHARGE.search(text):
                outcome_osi = 0.02
                outcome_label = "Convicted and discharged"
            elif _FINE.search(text):
                m_fine = _FINE.search(text)
                amt = m_fine.group(1).replace(",", "")
                outcome_osi = 0.08
                outcome_label = f"Fine ${amt}"
            elif _DISQUALIFY.search(text):
                outcome_osi = 0.10
                outcome_label = "Driving disqualification"

    if outcome_osi is None:
        return result  # has_data stays False

    result["outcome_osi"] = round(outcome_osi, 3)
    result["outcome_label"] = outcome_label
    result["has_data"] = True

    # Crown ask (optional)
    crown_months = _extract_crown_months(text)
    if crown_months is not None:
        if crown_months == 9999:
            prosecution_osi = 1.00
            prosecution_label = "Life imprisonment (Crown)"
        else:
            prosecution_osi = _months_to_osi(crown_months)
            y, m = divmod(crown_months, 12)
            prosecution_label = (
                f"{y} year{'s' if y != 1 else ''}"
                + (f" {m} month{'s' if m != 1 else ''}" if m else "")
                + " (Crown)"
            )
        result["prosecution_osi"] = round(prosecution_osi, 3)
        result["prosecution_label"] = prosecution_label
        gap = round(prosecution_osi - outcome_osi, 3)
        result["gap"] = gap
        if gap > 0.05:
            result["gap_class"] = "lighter"
        elif gap < -0.05:
            result["gap_class"] = "heavier"
        else:
            result["gap_class"] = "agreed"

    return result


# --- Civil financial extraction ---

_DOLLAR = re.compile(r"\$\s*([\d,]+(?:\.\d{2})?)", re.IGNORECASE)

_AWARD_PATTERNS = [
    re.compile(r"(?:I\s+award|awarded?|ordered?\s+to\s+pay|entitled\s+to|"
               r"I\s+order.{0,20}pay)\s+\$\s*([\d,]+)", re.IGNORECASE),
    re.compile(r"(?:compensation|damages?|refund)\s+of\s+\$\s*([\d,]+)", re.IGNORECASE),
    re.compile(r"\$\s*([\d,]+)\s+(?:is|are)\s+awarded", re.IGNORECASE),
]
_CLAIM_PATTERNS = [
    re.compile(r"(?:claims?\s+(?:the\s+sum\s+of\s+)?\$|"
               r"seeks?\s+\$|seeking\s+\$|claimed\s+\$)\s*([\d,]+)", re.IGNORECASE),
    re.compile(r"amount\s+claimed\D{0,10}\$\s*([\d,]+)", re.IGNORECASE),
    re.compile(r"application\s+(?:is\s+)?for\s+\$\s*([\d,]+)", re.IGNORECASE),
]

_REINSTATEMENT = re.compile(r"\breinstate(?:ment|d)?\b", re.IGNORECASE)
_EVICTION      = re.compile(r"\bterminat\w+\s+the\s+tenancy\b|\beviction\s+order\b|\bpossession\s+order\b", re.IGNORECASE)
_WORK_ORDER    = re.compile(r"\bwork\s+order\b|\blandlord\s+is\s+ordered\s+to\b", re.IGNORECASE)
_INJUNCTION    = re.compile(r"\binjunction\b|\brestrained\s+from\b", re.IGNORECASE)

_RECOVERY_CLASSES = [
    (0.10,  "minimal"),
    (0.30,  "low"),
    (0.60,  "partial"),
    (0.85,  "substantial"),
    (1.00,  "near_full"),
    (1.001, "full"),
]


def _dollar_from_patterns(text: str, patterns: list) -> float | None:
    for p in patterns:
        m = p.search(text)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except (ValueError, IndexError):
                pass
    return None


def _recovery_class(rate: float) -> str:
    if rate > 1.001:
        return "exceeded"
    for threshold, label in _RECOVERY_CLASSES:
        if rate <= threshold:
            return label
    return "full"


def _extract_civil_financial(text: str, court: str) -> dict | None:
    result: dict = {"court_type": "civil_financial", "has_data": False}

    awarded = _dollar_from_patterns(text, _AWARD_PATTERNS)
    claimed = _dollar_from_patterns(text, _CLAIM_PATTERNS)

    # Non-monetary remedies
    remedies = []
    if _REINSTATEMENT.search(text):
        remedies.append("reinstatement")
    if _EVICTION.search(text):
        remedies.append("eviction")
    if _WORK_ORDER.search(text):
        remedies.append("work_order")
    if _INJUNCTION.search(text):
        remedies.append("injunction")

    if awarded is None and not remedies:
        return result

    result["has_data"] = True
    if awarded is not None:
        result["awarded_amount"] = awarded
    if claimed is not None:
        result["claimed_amount"] = claimed
    if awarded is not None and claimed is not None and claimed > 0:
        rate = round(awarded / claimed, 3)
        result["recovery_rate"] = rate
        result["recovery_class"] = _recovery_class(rate)
    if remedies:
        result["remedies"] = remedies

    return result


def _extract_civil_env(text: str) -> dict | None:
    result: dict = {"court_type": "civil_nonfinancial", "has_data": False}
    granted = re.search(r"(?:consent|application|appeal)\s+(?:is\s+)?granted", text, re.IGNORECASE)
    declined = re.search(r"(?:consent|application|appeal)\s+(?:is\s+)?(?:declined|refused|dismissed)", text, re.IGNORECASE)
    if granted:
        result["has_data"] = True
        result["outcome_class"] = "granted"
    elif declined:
        result["has_data"] = True
        result["outcome_class"] = "declined"
    return result


def _extract_coronal(text: str) -> dict | None:
    result: dict = {"court_type": "coronal", "has_data": False}
    if re.search(r"cause\s+of\s+death|finding\s+(?:is|was|:)", text, re.IGNORECASE):
        result["has_data"] = True
        result["outcome_class"] = "finding"
    return result


# --- Main entry point ---

def extract_penalty(court: str, text: str) -> dict:
    """
    Extract penalty/outcome data from a chunk.
    Returns a dict suitable for storing in Qdrant payload under 'penalty'.
    Always returns a dict; check has_data to see if extraction succeeded.
    """
    court_type = detect_court_type(court, text)

    if court_type == "legislation":
        return {"court_type": "legislation", "has_data": False}
    if court_type == "coronal":
        return _extract_coronal(text) or {"court_type": "coronal", "has_data": False}
    if court_type == "civil_nonfinancial":
        return _extract_civil_env(text) or {"court_type": "civil_nonfinancial", "has_data": False}
    if court_type == "criminal":
        return _extract_criminal(text) or {"court_type": "criminal", "has_data": False}

    # civil_financial, civil_mixed, civil_disciplinary
    result = _extract_civil_financial(text, court)
    if result:
        return result
    # Fallback: try criminal patterns (NZFC sometimes has criminal-adjacent matters)
    if court_type == "civil_mixed":
        cr = _extract_criminal(text)
        if cr and cr.get("has_data"):
            return cr
    return {"court_type": court_type, "has_data": False}
