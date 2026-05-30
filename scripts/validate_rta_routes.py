#!/usr/bin/env python3
"""Validate all forced_sections in ROUTES against Qdrant.

Usage:
  python scripts/validate_rta_routes.py               # warn on failure
  TENANCY_STRICT_ROUTE_VALIDATION=1 python scripts/validate_rta_routes.py  # fail on any error

Exit code 0 = all sections valid.
Exit code 1 = one or more sections missing or have wrong content (strict mode only).

Add this to your pre-deploy checklist:
  python scripts/validate_rta_routes.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import VectorStore
from tenancy.rta_routes import ROUTES

SECTION_EXPECTATIONS: dict[str, dict] = {
    "NZLEG/RTA/s13A": {
        "must_contain_any": ["tenancy agreement", "agreement in writing", "written", "contents"],
        "must_not_contain_any": ["smoke alarm", "infringement fee", "Schedule 1A"],
    },
    "NZLEG/RTA/s13B": {
        "must_contain_any": ["agreement", "written", "copy", "landlord"],
        "must_not_contain_any": ["smoke alarm", "infringement fee"],
    },
    "NZLEG/RTA/s18": {
        "must_contain_any": ["bond", "receipt", "weeks", "landlord"],
        "must_not_contain_any": ["smoke alarm", "infringement fee", "19(2)"],
    },
    "NZLEG/RTA/s28": {
        "must_contain_any": ["rent", "increase", "notice"],
        "must_not_contain_any": ["infringement fee", "Schedule 1A"],
    },
    "NZLEG/RTA/s28A": {
        "must_contain_any": ["rent", "increase", "order"],
        "must_not_contain_any": ["infringement fee", "Schedule 1A"],
    },
    "NZLEG/RTA/s40": {
        "must_contain_any": ["tenant", "clean", "land", "garden", "responsibilities"],
        "must_not_contain_any": ["40(3A)(a)", "infringement fee", "1,800", "4,000"],
    },
    "NZLEG/RTA/s42A": {
        "must_contain_any": ["fixture", "improvement", "consent"],
        "must_not_contain_any": ["42A(7)", "infringement fee", "1,500"],
    },
    "NZLEG/RTA/s42B": {
        "must_contain_any": ["minor", "change"],
        "must_not_contain_any": ["42B(3)", "42B(6)", "infringement fee", "1,500"],
    },
    "NZLEG/RTA/s45": {
        "must_contain_any": ["repair", "maintain", "landlord"],
        "must_not_contain_any": ["infringement fee", "Schedule 1A"],
    },
    "NZLEG/RTA/s48": {
        "must_contain_any": ["entry", "enter", "landlord"],
        "must_not_contain_any": ["48(5)", "infringement fee"],
    },
    "NZLEG/RTA/s49A": {
        "must_contain_any": ["wear", "tear", "tenant"],
        "must_not_contain_any": ["infringement fee", "Schedule 1A"],
    },
    "NZLEG/RTA/s49B": {
        "must_contain_any": ["damage", "tenant", "landlord"],
        "must_not_contain_any": ["infringement fee", "Schedule 1A"],
    },
    "NZLEG/RTA/s51": {
        "must_contain_any": ["terminate", "notice", "periodic"],
        "must_not_contain_any": ["infringement fee", "Schedule 1A"],
    },
}

STRICT = os.getenv("TENANCY_STRICT_ROUTE_VALIDATION", "") == "1"


def _collect_forced_sections() -> set[str]:
    sections: set[str] = set()
    for route in ROUTES:
        sections.update(route.forced_sections)
    return sections


def validate() -> bool:
    try:
        store = VectorStore()
    except Exception as exc:
        print(f"[SKIP] Qdrant not available: {exc}")
        return True

    forced = _collect_forced_sections()
    failures: list[str] = []

    missing_exp = forced - set(SECTION_EXPECTATIONS)
    if missing_exp:
        for sid in sorted(missing_exp):
            msg = f"[WARN] {sid}: no expectation entry in SECTION_EXPECTATIONS"
            print(msg)
            failures.append(msg)

    print(f"Checking {len(forced)} forced section(s) against Qdrant...")
    for section_id in sorted(forced):
        results = store.get_by_case_id(section_id)
        if not results:
            msg = (
                f"[FAIL] {section_id}: not found in Qdrant. "
                "Verify the ID with the Qdrant scroll API - the corpus may use a different name."
            )
            print(msg)
            failures.append(msg)
            continue

        exp = SECTION_EXPECTATIONS.get(section_id)
        if not exp:
            print(f"[ OK ] {section_id}: exists (no content check defined)")
            continue

        combined = " ".join(r.text.lower() for r in results)

        must_any = exp.get("must_contain_any", [])
        if must_any and not any(term.lower() in combined for term in must_any):
            msg = (
                f"[FAIL] {section_id}: text does not contain any of {must_any!r}. "
                f"Snippet: {combined[:200]!r}"
            )
            print(msg)
            failures.append(msg)
            continue

        bad_term = next(
            (t for t in exp.get("must_not_contain_any", []) if t.lower() in combined),
            None,
        )
        if bad_term:
            msg = (
                f"[FAIL] {section_id}: contains forbidden term {bad_term!r}. "
                "This ID may resolve to regulation text or a penalty table row."
            )
            print(msg)
            failures.append(msg)
            continue

        print(f"[ OK ] {section_id}")

    if failures:
        print(f"\n{len(failures)} issue(s) found.")
        return False
    print(f"\nAll {len(forced)} section(s) valid.")
    return True


if __name__ == "__main__":
    ok = validate()
    if not ok and STRICT:
        sys.exit(1)
