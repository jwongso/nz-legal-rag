"""
Sentencing factor extraction for NZ criminal decisions.

Extracts structured fields to power a Sentencing Tracker comparable to Westlaw NZ.
Stored under payload key 'sentencing'.

Fields extracted:
  starting_point_months   - judicial starting point before discounts (float)
  final_sentence_months   - imprisonment term imposed (float)
  home_detention_months   - home detention term (float)
  community_work_hours    - community work hours (int)
  reparation_amount       - reparation ordered ($) (float)
  fine_amount             - fine imposed ($) (float)
  guilty_plea_discount_pct - discount applied for guilty plea (float, 0-50)
  sentence_type           - imprisonment | home_detention | community_work | fine | supervision
  has_guilty_plea         - bool
  has_remorse             - bool
  has_previous_convictions - bool (True only when positively stated, not just mentioned)
  has_data                - bool (True if any key field extracted)
"""

import re
from typing import Any

# ---------------------------------------------------------------------------
# Number helpers
# ---------------------------------------------------------------------------

_WORD_NUMS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

# Numeric word pattern (up to 20)
_N = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
)

# Months-only numbers (up to 36 - covers most NZ sentences in months)
_MO = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|twenty-six|"
    r"twenty-seven|twenty-eight|twenty-nine|thirty|thirty-six)"
)

# Duration: "X years [and Y months]" OR "X months"
_DUR_PAT = re.compile(
    rf"({_N})\s+years?(?:\s+(?:and\s+)?({_MO})\s+months?)?"
    rf"|({_MO})\s+months?",
    re.IGNORECASE,
)


def _to_num(s: str | None) -> int | None:
    if not s:
        return None
    s = s.strip().lower().replace(",", "")
    if s.isdigit():
        return int(s)
    return _WORD_NUMS.get(s)


def _first_dur(text: str) -> float | None:
    """Return the first duration found in text as months, or None."""
    m = _DUR_PAT.search(text)
    if not m:
        return None
    if m.group(1) is not None:
        y = _to_num(m.group(1))
        mo = _to_num(m.group(2)) if m.group(2) else 0
        if y is None:
            return None
        return float(y * 12 + (mo or 0))
    if m.group(3) is not None:
        mo = _to_num(m.group(3))
        return float(mo) if mo is not None else None
    return None


# ---------------------------------------------------------------------------
# Trigger patterns - find the phrase, parse duration from the following window
# ---------------------------------------------------------------------------

_TRIG_START = re.compile(
    r"(?:fix(?:ing)?|take|taking|adopt(?:ing)?|set(?:ting)?)\s+a\s+starting\s+point\s+of\s*"
    r"|starting\s+point\s*(?:is\s+|of\s+)?(?:imprisonment\s+of\s+)?"
    r"|start\s+(?:from|at)\s+a\s+(?:point|sentence)\s+of\s+",
    re.IGNORECASE,
)

_TRIG_SENTENCE = re.compile(
    r"sentenced?\s+to\s+(?:imprisonment\s+(?:for\s+a\s+term\s+of\s+)?)?(?:a\s+term\s+of\s+)?"
    r"|imprisoned\s+for\s+(?:a\s+(?:period|term)\s+of\s+)?"
    r"|term\s+of\s+imprisonment\s+(?:of\s+|is\s+(?:imposed|set|confirmed|substituted|varied)\s+(?:at\s+|to\s+)?)",
    re.IGNORECASE,
)

# Post-keyword: "home detention for 6 months" / "home detention of 12 months"
_TRIG_HOME_POST = re.compile(
    r"sentenced?\s+to\s+home\s+detention\s+(?:for\s+(?:a\s+(?:period|term)\s+of\s+)?)?"
    r"|home\s+detention\s+(?:of\s+|for\s+(?:a\s+(?:period|term)\s+of\s+)?)",
    re.IGNORECASE,
)

# Pre-keyword: "12 months' home detention" or "two years home detention"
_TRIG_HOME_PRE = re.compile(
    rf"({_MO})\s+months?'?\s*home\s+detention"
    rf"|({_N})\s+years?'?\s*home\s+detention",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Other patterns
# ---------------------------------------------------------------------------

_COMM_WORK = re.compile(
    r"(\d{2,3})\s+hours?\s+(?:of\s+)?community\s+work"
    r"|community\s+work\s+(?:of\s+|order\s+of\s+)?(\d{2,3})\s+hours?",
    re.IGNORECASE,
)

_REPARATION = re.compile(
    r"(?:pay(?:ment\s+of)?\s+)?reparation\s+(?:of\s+)?(?:the\s+sum\s+of\s+)?\$\s*([\d,]+)"
    r"|reparation\s+(?:is\s+)?(?:ordered|imposed|set|fixed)\s+(?:at\s+|in\s+the\s+sum\s+of\s+)?\$\s*([\d,]+)"
    r"|reparation\s+order\s+of\s+\$\s*([\d,]+)",
    re.IGNORECASE,
)

_FINE_PAT = re.compile(
    r"fined?\s+\$\s*([\d,]+)"
    r"|fine\s+(?:is\s+)?(?:of\s+|imposed\s+(?:at\s+)?|set\s+(?:at\s+)?|fixed\s+(?:at\s+)?)\$\s*([\d,]+)",
    re.IGNORECASE,
)

# Guilty plea discount as percentage
_GP_PCT = re.compile(
    r"(?:discount|reduction|deduction)\s+(?:of\s+)?(\d{1,2})\s*(?:per\s*cent|%)"
    r"(?:[^.]{0,60}(?:guilty\s+plea|plea\s+of\s+guilty))?"
    r"|(\d{1,2})\s*(?:per\s*cent|%)\s*(?:discount|reduction|deduction)"
    r"(?:[^.]{0,60}(?:guilty\s+plea|plea\s+of\s+guilty))?",
    re.IGNORECASE,
)

# Fractional discounts: "one-third discount for the guilty plea"
_GP_FRAC = re.compile(
    r"(?:one|a)\s+(quarter|third|fifth|half)\s+(?:discount|deduction|reduction)"
    r"(?:[^.]{0,60}guilty\s+plea)?",
    re.IGNORECASE,
)
_FRAC_PCT: dict[str, float] = {"quarter": 25.0, "third": 33.0, "fifth": 20.0, "half": 50.0}

_GP_PRESENT = re.compile(
    r"\b(?:pleaded?\s+guilty|early\s+guilty\s+plea|guilty\s+plea|plea\s+of\s+guilty)\b",
    re.IGNORECASE,
)

_SUPERVISION = re.compile(
    r"\bsentenced?\s+to\s+supervision\b|\bsupervision\s+order\b",
    re.IGNORECASE,
)

_PREV_CONV = re.compile(
    r"\b(?:previous|prior)\s+(?:criminal\s+)?convictions?\b"
    r"|\brecidivis[mt]\b|\brepeat\s+offend",
    re.IGNORECASE,
)

_NO_PREV_CONV = re.compile(
    r"\bno\s+(?:previous|prior)\s+(?:criminal\s+)?convictions?\b"
    r"|\bfirst[-\s]time\s+offend|\bno\s+criminal\s+history\b",
    re.IGNORECASE,
)

_REMORSE = re.compile(
    r"\b(?:remorse|remorseful|expressed?\s+(?:genuine\s+)?(?:remorse|sorry)|"
    r"genuinely\s+sorry|apologised?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_sentencing(text: str) -> dict[str, Any]:
    """
    Extract sentencing factors from a text chunk.
    Returns dict with has_data=True if any key structural field was found.
    """
    result: dict[str, Any] = {"has_data": False}

    # Starting point
    m = _TRIG_START.search(text)
    if m:
        val = _first_dur(text[m.end():m.end() + 60])
        if val and 1.0 <= val <= 360.0:
            result["starting_point_months"] = val

    # Final imprisonment sentence
    m = _TRIG_SENTENCE.search(text)
    if m:
        val = _first_dur(text[m.end():m.end() + 60])
        if val and 1.0 <= val <= 360.0:
            result["final_sentence_months"] = val

    # Home detention - post-keyword form
    m = _TRIG_HOME_POST.search(text)
    if m:
        val = _first_dur(text[m.end():m.end() + 30])
        if val and 1.0 <= val <= 24.0:
            result["home_detention_months"] = val

    # Home detention - pre-keyword form: "12 months' home detention"
    if "home_detention_months" not in result:
        m = _TRIG_HOME_PRE.search(text)
        if m:
            val = _first_dur(m.group(0))
            if val and 1.0 <= val <= 24.0:
                result["home_detention_months"] = val

    # Community work (40-400 hours is the NZ range)
    m = _COMM_WORK.search(text)
    if m:
        raw = m.group(1) or m.group(2)
        if raw and raw.isdigit():
            h = int(raw)
            if 40 <= h <= 400:
                result["community_work_hours"] = h

    # Reparation
    m = _REPARATION.search(text)
    if m:
        raw = (m.group(1) or m.group(2) or m.group(3) or "").replace(",", "")
        if raw.isdigit():
            result["reparation_amount"] = float(raw)

    # Fine
    m = _FINE_PAT.search(text)
    if m:
        raw = (m.group(1) or m.group(2) or "").replace(",", "")
        if raw.isdigit():
            amt = float(raw)
            if amt >= 100.0:
                result["fine_amount"] = amt

    # Guilty plea discount percentage
    discount: float | None = None
    for m in _GP_PCT.finditer(text):
        pct_str = m.group(1) or m.group(2)
        if pct_str:
            pct = int(pct_str)
            if 5 <= pct <= 50:
                discount = float(pct)
                break
    if discount is None:
        m = _GP_FRAC.search(text)
        if m:
            frac = m.group(1).lower()
            discount = _FRAC_PCT.get(frac)

    if discount is not None:
        result["guilty_plea_discount_pct"] = discount

    # Boolean indicators
    result["has_guilty_plea"] = bool(_GP_PRESENT.search(text))
    result["has_remorse"] = bool(_REMORSE.search(text))
    has_prev = bool(_PREV_CONV.search(text))
    no_prev = bool(_NO_PREV_CONV.search(text))
    result["has_previous_convictions"] = has_prev and not no_prev

    # Sentence type
    if "final_sentence_months" in result:
        result["sentence_type"] = "imprisonment"
    elif "home_detention_months" in result:
        result["sentence_type"] = "home_detention"
    elif "community_work_hours" in result:
        result["sentence_type"] = "community_work"
    elif "fine_amount" in result:
        result["sentence_type"] = "fine"
    elif _SUPERVISION.search(text):
        result["sentence_type"] = "supervision"

    if any(k in result for k in (
        "starting_point_months", "final_sentence_months", "home_detention_months",
        "community_work_hours", "guilty_plea_discount_pct",
    )):
        result["has_data"] = True

    return result
