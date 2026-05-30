"""Statute routing lexicon for the NZ tenancy RAG tool.

Maps tenancy question types to the RTA sections that embeddings frequently
miss. Each route is matched against the combined original + rewritten query,
then the forced sections are prepended to the leg_store vector results.

Keep ROUTES ordered: more specific routes before generic ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RouteIntent(str, Enum):
    AGREEMENT_FORM = "agreement_form"
    BOND = "bond"
    RENT_PAYMENT = "rent_payment"
    PROPERTY_CHANGE = "property_change"
    WEAR_AND_TEAR = "wear_and_tear"
    LANDLORD_ENTRY = "landlord_entry"
    TERMINATION_NOTICE = "termination_notice"
    REPAIRS_MAINTENANCE = "repairs_maintenance"


@dataclass(frozen=True)
class StatuteRoute:
    intent: RouteIntent
    include_any: tuple[str, ...]
    forced_sections: tuple[str, ...]
    synthetic_query: str               # embedded to locate forced_sections in leg_store
    include_all: tuple[str, ...] = ()
    exclude_any: tuple[str, ...] = ()
    notes: str = ""


ROUTES: list[StatuteRoute] = [
    StatuteRoute(
        intent=RouteIntent.WEAR_AND_TEAR,
        include_any=(
            "fair wear and tear", "wear and tear", "normal wear",
            "tenant damage", "damage claim", "repair cost",
            "landlord charge", "liable for damage", "damage to the",
            "s49a", "s49b",
        ),
        forced_sections=("NZLEG/RTA/s49A", "NZLEG/RTA/s49B", "NZLEG/RTA/s40"),
        synthetic_query=(
            "tenant not liable fair wear tear exception section 49A damage "
            "landlord cannot charge deterioration reasonable use natural forces "
            "residential tenancies act"
        ),
        notes="Tenant damage liability and fair wear and tear exception.",
    ),
    StatuteRoute(
        intent=RouteIntent.PROPERTY_CHANGE,
        include_any=(
            "plant", "planted", "tree", "trees", "shrub", "hedge",
            "garden", "backyard", "back yard", "lawn",
            "fixture", "alteration", "alter", "altered",
            "install", "installed", "renovate", "renovation",
            "minor change", "improvement", "fence",
        ),
        forced_sections=("NZLEG/RTA/s40", "NZLEG/RTA/s42A", "NZLEG/RTA/s42B"),
        synthetic_query=(
            "tenant obligations alter improve add fixtures to land garden "
            "written consent landlord section 40 42A 42B residential tenancies act"
        ),
        notes="Tenant changes to premises, garden, land, fixtures.",
    ),
    StatuteRoute(
        intent=RouteIntent.REPAIRS_MAINTENANCE,
        include_any=(
            "not working", "broken", "won't fix", "wont fix",
            "hasn't fixed", "hasnt fixed", "not fixed", "not repaired",
            "repair request", "maintenance",
            "hot water", "no hot water", "heating", "no heating",
            "mould", "mold", "damp", "dampness",
            "leak", "leaking", "dripping",
            "weathertight", "habitable", "uninhabitable",
            "appliance", "oven", "stove", "fridge",
            "landlord obligation", "landlord's obligation",
            "s45",
        ),
        exclude_any=(
            "fair wear and tear", "wear and tear",
        ),
        forced_sections=("NZLEG/RTA/s45",),
        synthetic_query=(
            "landlord responsibility maintain premises reasonable state repair "
            "section 45 habitable condition heating hot water weathertight "
            "residential tenancies act tenant remedies maintenance obligations"
        ),
        notes="Landlord maintenance and repair obligations (s45).",
    ),
    StatuteRoute(
        intent=RouteIntent.AGREEMENT_FORM,
        include_any=(
            "tenancy agreement", "written agreement", "copy of agreement",
            "sign agreement", "signing agreement", "before signing",
            "provide agreement", "give the agreement", "before getting the agreement",
            "form of agreement", "written tenancy",
        ),
        forced_sections=("NZLEG/RTA/s13",),
        synthetic_query=(
            "tenancy agreement must be in writing landlord provide signed copy "
            "form of agreement section 13 residential tenancies act"
        ),
        notes="Form, content, and copy obligations for tenancy agreements.",
    ),
    StatuteRoute(
        intent=RouteIntent.BOND,
        include_any=(
            "bond lodgement", "bond lodged", "lodge the bond", "lodge bond",
            "bond receipt", "proof of bond", "bond proof",
            "bond before", "bond form", "bond help",
            "work and income", "winz", "bond guarantee",
            "can pay the bond", "pay the bond",
            "s18", "s19",
        ),
        forced_sections=("NZLEG/RTA/s18", "NZLEG/RTA/s19"),
        synthetic_query=(
            "bond landlord duties receipt lodgement section 18 section 19 "
            "residential tenancies act 23 working days bond centre"
        ),
        notes="Bond receipt and lodgement duties (s18, s19).",
    ),
    StatuteRoute(
        intent=RouteIntent.LANDLORD_ENTRY,
        include_any=(
            "landlord entry", "landlord enter", "right of entry",
            "inspection notice", "24 hour notice", "24 hours notice",
            "landlord came in", "landlord access",
            "notice before entering", "notice to enter",
            "s48",
        ),
        forced_sections=("NZLEG/RTA/s48",),
        synthetic_query=(
            "landlord right of entry inspection notice 24 hours section 48 "
            "residential tenancies act access premises"
        ),
        notes="Landlord entry and inspection rules (s48).",
    ),
    StatuteRoute(
        intent=RouteIntent.TERMINATION_NOTICE,
        include_any=(
            "evict", "eviction",
            "notice to leave", "notice to vacate",
            "end the tenancy", "end my tenancy", "terminate tenancy",
            "90 day notice", "90 days notice", "90-day notice",
            "42 day notice", "42 days notice", "42-day notice",
            "21 day notice", "21 days notice",
            "periodic tenancy end", "asked to leave",
            "termination notice", "s51", "s56",
        ),
        forced_sections=("NZLEG/RTA/s51",),
        synthetic_query=(
            "landlord terminate periodic tenancy notice 90 days 42 days "
            "section 51 residential tenancies act tenant notice 21 days "
            "lawful grounds termination"
        ),
        notes="Termination of periodic tenancy, notice periods (s51).",
    ),
    StatuteRoute(
        intent=RouteIntent.RENT_PAYMENT,
        include_any=(
            "rent increase", "increase the rent", "raise the rent", "raised the rent",
            "rent rise", "rent review", "maximum rent", "rent in advance",
            "weeks rent in advance", "how much rent",
            "s54", "s55",
        ),
        forced_sections=("NZLEG/RTA/s54", "NZLEG/RTA/s55"),
        synthetic_query=(
            "landlord increase rent notice 90 days section 54 55 residential "
            "tenancies act rent review maximum advance weeks"
        ),
        notes="Rent increases, review, and advance rent limits (s54, s55).",
    ),
]

# Sections that vector search frequently returns as false positives.
# Only allowed if the query explicitly contains the listed terms.
LOW_PRIORITY_SECTIONS: dict[str, tuple[str, ...]] = {
    "NZLEG/RTA/s16A": (
        "landlord overseas", "landlord out of new zealand",
        "agent if landlord", "21 consecutive days",
        "out of new zealand", "overseas landlord",
    ),
}


def normalize_query(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def match_routes(original_query: str, rewritten_query: str) -> list[StatuteRoute]:
    """Match both original and rewritten query against the routing table.

    The original often contains colloquial signals (e.g. "work and income");
    the rewrite often contains formal legal terms (e.g. "bond lodgement").
    """
    q = normalize_query(original_query + " " + rewritten_query)
    matches: list[StatuteRoute] = []
    for route in ROUTES:
        if route.exclude_any and any(term in q for term in route.exclude_any):
            continue
        any_ok = any(term in q for term in route.include_any)
        all_ok = (not route.include_all) or all(term in q for term in route.include_all)
        if any_ok and all_ok:
            matches.append(route)
    return matches


def allow_section(case_id: str, combined_query: str) -> bool:
    """Return False to suppress sections that are almost-never relevant."""
    rule = LOW_PRIORITY_SECTIONS.get(case_id)
    if not rule:
        return True
    return any(term in combined_query for term in rule)


def route_debug_info(
    matched: list[StatuteRoute],
    injected_ids: list[str],
    suppressed: list[str],
    original: str,
    rewritten: str,
) -> dict:
    """Return the statute_routing block for context_debug."""
    q = normalize_query(original + " " + rewritten)
    trigger_terms: list[str] = []
    for route in matched:
        for term in route.include_any:
            if term in q and term not in trigger_terms:
                trigger_terms.append(term)
    return {
        "triggered": bool(matched),
        "matched_routes": [r.intent.value for r in matched],
        "trigger_terms": trigger_terms[:8],
        "forced_sections": injected_ids,
        "suppressed_sections": [
            {
                "section": s,
                "reason": "low_priority_section - query does not mention relevant terms",
            }
            for s in suppressed
        ],
    }
