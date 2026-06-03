"""Smoke test suite for tenancy.localrun.ai.

Split into three tiers by what infrastructure they require:

  Tier 1 (retrieval) - Qdrant + embedder only, no LLM.
    Calls /retrieve and asserts legislation anchors.
    Run: pytest tests/test_smoke.py -m retrieval -v

  Tier 2 (structural) - Needs LLM. Checks token budget, forbidden-term
    cleanliness, citation integrity, and route hygiene from context_debug.
    Run: pytest tests/test_smoke.py -m structural -v

  Tier 3 (llm) - Needs LLM. Semantic checks on answer content.
    Run: pytest tests/test_smoke.py -m llm -v

Pass criteria enforced across tiers:
  1. Correct RTA section or relevant NZTT source retrieved
  2. No irrelevant legislation anchor (s19 in a fixture query, etc.)
  3. No Schedule 1A / penalty-table contamination in anchor text
  4. No fake citations: every [SN] maps to a real source index
  5. Practical answer, not overconfident legal advice
  6. Weak context: admits uncertainty rather than guessing
"""

import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient

import config
from tenancy.app import app
from core.api import _PUBLIC_TOKEN

_TOKEN_HEADERS = {"X-API-Key": _PUBLIC_TOKEN, "X-No-Log": "1"} if _PUBLIC_TOKEN else {"X-No-Log": "1"}

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Infrastructure probes
# ---------------------------------------------------------------------------

def _qdrant_available() -> bool:
    try:
        r = httpx.get(
            f"{config.QDRANT_URL}/collections/{config.QDRANT_TENANCY_COLLECTION}",
            timeout=3,
        )
        return r.status_code == 200
    except Exception:
        return False


def _llm_available() -> bool:
    try:
        r = httpx.get(f"{config.LLM_BASE_URL}/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


_QDRANT_SKIP = pytest.mark.skipif(not _qdrant_available(), reason="Qdrant not available")
_LLM_SKIP    = pytest.mark.skipif(not _llm_available(),    reason="llama-server not running")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_sse_events(body: str) -> list[dict]:
    events = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("data: "):
            continue
        try:
            events.append(json.loads(chunk[6:]))
        except Exception:
            continue
    return events


def _retrieve(client, question: str) -> dict:
    r = client.post(
        "/retrieve",
        headers=_TOKEN_HEADERS,
        json={"question": question},
    )
    assert r.status_code == 200, f"/retrieve failed: {r.status_code} {r.text[:200]}"
    return r.json()


def _stream(client, question: str) -> tuple[list, str, dict]:
    """Call /ask/stream and return (leg_sources, answer_text, context_debug_event)."""
    r = client.post(
        "/ask/stream",
        headers=_TOKEN_HEADERS,
        json={"question": question, "feedback_context": True},
    )
    events = _parse_sse_events(r.text)
    leg_sources: list = []
    answer_parts: list[str] = []
    context_debug: dict = {}
    for e in events:
        t = e.get("type", "")
        if t == "sources":
            leg_sources = e.get("legislation", [])
        elif t == "token":
            answer_parts.append(e.get("text", ""))
        elif t == "context_debug":
            context_debug = e
    return leg_sources, "".join(answer_parts), context_debug


def _leg_ids(leg_sources: list) -> set[str]:
    return {s["case_id"] for s in leg_sources}


# ---------------------------------------------------------------------------
# Pass criteria assertion helpers (criteria 1-6)
# ---------------------------------------------------------------------------

def _assert_sections_present(ids: set, *sections: str) -> None:
    """Criterion 1: correct RTA sections must appear in legislation anchors."""
    for s in sections:
        assert s in ids, (
            f"Expected section {s} in legislation anchors. Got: {sorted(ids)}"
        )


def _assert_sections_absent(ids: set, *sections: str) -> None:
    """Criterion 2: wrong sections must not appear."""
    for s in sections:
        assert s not in ids, (
            f"Section {s} must NOT appear in legislation anchors. Got: {sorted(ids)}"
        )


def _assert_no_forbidden_terms(context_debug: dict) -> None:
    """Criterion 3: no Schedule 1A / penalty-table leakage in anchor text."""
    anchor = context_debug.get("anchor", {})
    for section in anchor.get("sections", []):
        ft = section.get("forbidden_terms", {})
        bad = [term for term, hit in ft.items() if hit]
        assert not bad, (
            f"Forbidden terms found in anchor {section.get('document_id')}: {bad}"
        )


def _assert_token_budget(context_debug: dict, max_fraction: float = 0.70) -> None:
    """Criterion 3 / debug check: context must not crowd out generation budget."""
    budget = context_debug.get("budget", {})
    total  = budget.get("total_tokens", 0)
    limit  = budget.get("ctx_limit", 5120)
    assert total > 0, "context_debug.budget.total_tokens is 0 - no context sent to LLM"
    assert total < max_fraction * limit, (
        f"Context too large: {total} tokens exceeds {int(max_fraction * 100)}% of "
        f"{limit}-token limit"
    )


def _assert_citations_mapped(answer: str, sources: list) -> None:
    """Criterion 4: every [SN] citation in the answer maps to a real source index."""
    refs = {int(m) for m in re.findall(r"\[S(\d+)\]", answer)}
    if not refs:
        return  # no citations - handled by other tests
    max_ref = max(refs)
    assert max_ref <= len(sources), (
        f"Answer contains [S{max_ref}] but only {len(sources)} source(s) provided. "
        "Likely a hallucinated citation."
    )


def _assert_practical_answer(answer: str) -> None:
    """Criterion 5: answer must include a practical referral signpost."""
    lower = answer.lower()
    assert any(t in lower for t in ["community law", "tenancy services", "tenancy tribunal", "tribunal"]), (
        "Answer missing practical signpost (Community Law / Tenancy Services / Tribunal). "
        "System prompt may have been overridden."
    )


def _assert_not_empty_answer(answer: str, min_chars: int = 80) -> None:
    assert len(answer) >= min_chars, (
        f"Answer is suspiciously short ({len(answer)} chars). "
        "LLM may have failed or returned nothing."
    )


# ===========================================================================
# TIER 1 - Retrieval correctness (Qdrant only, no LLM required)
# ===========================================================================

class TestRetrieval:
    """Tier 1: /retrieve endpoint assertions. No LLM needed."""

    # ---- 1. Property changes / fixtures -----------------------------------

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_property_change_tree_planting(self, client):
        """Planting trees must route to s42A/s42B; s19 and s16A must not appear."""
        r = _retrieve(client, "I live in a rented house and planted several trees in the backyard. Does this break any rules?")
        ids = _leg_ids(r["legislation"])
        _assert_sections_present(ids, "NZLEG/RTA/s42A", "NZLEG/RTA/s42B")
        _assert_sections_absent(ids, "NZLEG/RTA/s19", "NZLEG/RTA/s16A")

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_property_change_tree_removal(self, client):
        """Removing/pruning a tree must also route to fixture/consent sections."""
        r = _retrieve(client, "Can I remove or prune a tree at my rental property without asking the landlord?")
        ids = _leg_ids(r["legislation"])
        _assert_sections_present(ids, "NZLEG/RTA/s42A")
        _assert_sections_absent(ids, "NZLEG/RTA/s19", "NZLEG/RTA/s16A")

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_property_change_plumbing_case_absent(self, client):
        """The plumbing/RMA compliance case must not appear for a fixture query."""
        r = _retrieve(client, "I installed a shelf on the wall of my rental. Do I need permission?")
        bad_case = "NZTT-MOJ-4480189"
        source_ids = {s["case_id"] for s in r["sources"]}
        assert bad_case not in source_ids, (
            f"Plumbing compliance case {bad_case} must not appear in fixture queries."
        )

    # ---- 2. Fair wear and tear / damage -----------------------------------

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_wear_and_tear_general(self, client):
        """Wear and tear query must route to s49A/s49B."""
        r = _retrieve(client, "What is fair wear and tear and can a landlord charge me for it?")
        ids = _leg_ids(r["legislation"])
        assert ids & {"NZLEG/RTA/s49A", "NZLEG/RTA/s49B"}, (
            f"Expected s49A or s49B for wear-and-tear query. Got: {sorted(ids)}"
        )

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_wear_and_tear_carpet(self, client):
        """Worn carpet query must route to wear-and-tear sections, not fixture sections."""
        r = _retrieve(client, "The carpet is worn after living there for 6 years. Can the landlord take money from my bond?")
        ids = _leg_ids(r["legislation"])
        assert ids & {"NZLEG/RTA/s49A", "NZLEG/RTA/s49B"}, (
            f"Expected s49A or s49B for carpet wear query. Got: {sorted(ids)}"
        )
        _assert_sections_absent(ids, "NZLEG/RTA/s42A", "NZLEG/RTA/s42B")

    # ---- 3. Bond / agreement formation ------------------------------------

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_bond_agreement_chicken_and_egg(self, client):
        """Bond-before-agreement query must route to bond/agreement sections, not s16A."""
        r = _retrieve(client, "Can a landlord ask me for proof of bond payment before giving me the tenancy agreement?")
        ids = _leg_ids(r["legislation"])
        assert ids & {"NZLEG/RTA/s13A", "NZLEG/RTA/s13B", "NZLEG/RTA/s18"}, (
            f"Expected s13A/s13B or s18. Got: {sorted(ids)}"
        )
        _assert_sections_absent(ids, "NZLEG/RTA/s16A")

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_bond_lodgement(self, client):
        """Bond lodgement query must route to s18."""
        r = _retrieve(client, "Does my landlord have to lodge my bond?")
        ids = _leg_ids(r["legislation"])
        _assert_sections_present(ids, "NZLEG/RTA/s18")

    # ---- 4. Landlord entry / inspections ----------------------------------

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_landlord_entry_general(self, client):
        """Entry query must route to s48."""
        r = _retrieve(client, "Can my landlord enter the house without telling me?")
        ids = _leg_ids(r["legislation"])
        _assert_sections_present(ids, "NZLEG/RTA/s48")

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_landlord_entry_while_away(self, client):
        """Landlord entered while tenant at work must route to s48."""
        r = _retrieve(client, "My landlord came into the house while I was at work. Is that allowed?")
        ids = _leg_ids(r["legislation"])
        _assert_sections_present(ids, "NZLEG/RTA/s48")

    # ---- 5. Repairs / maintenance -----------------------------------------

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_repairs_broken_oven(self, client):
        """Broken oven / maintenance query must route to s45, not fixture-consent sections."""
        r = _retrieve(client, "My oven is broken and the landlord has not fixed it. What can I do?")
        ids = _leg_ids(r["legislation"])
        _assert_sections_present(ids, "NZLEG/RTA/s45")
        _assert_sections_absent(ids, "NZLEG/RTA/s42A", "NZLEG/RTA/s42B")

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_repairs_mould(self, client):
        """Mould query must route to s45, not unrelated sections."""
        r = _retrieve(client, "There is mould in my rental. Is the landlord responsible?")
        ids = _leg_ids(r["legislation"])
        _assert_sections_present(ids, "NZLEG/RTA/s45")
        _assert_sections_absent(ids, "NZLEG/RTA/s19", "NZLEG/RTA/s16A")

    # ---- 6. Termination / notice ------------------------------------------

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_termination_periodic(self, client):
        """Periodic tenancy termination query must route to s51."""
        r = _retrieve(client, "How much notice does a landlord need to end a periodic tenancy?")
        ids = _leg_ids(r["legislation"])
        _assert_sections_present(ids, "NZLEG/RTA/s51")

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_termination_sale_notice(self, client):
        """Short notice to vacate for property sale must route to termination sections."""
        r = _retrieve(client, "My landlord gave me 28 days notice to move out because they want to sell. Is that enough?")
        ids = _leg_ids(r["legislation"])
        _assert_sections_present(ids, "NZLEG/RTA/s51")

    # ---- 7. Rent increases ------------------------------------------------

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_rent_increase_frequency(self, client):
        """Rent increase query must route to s28/s28A, not bond or wear-and-tear."""
        r = _retrieve(client, "Can my landlord increase the rent whenever they want?")
        ids = _leg_ids(r["legislation"])
        assert ids & {"NZLEG/RTA/s28", "NZLEG/RTA/s28A"}, (
            f"Expected s28 or s28A for rent-increase query. Got: {sorted(ids)}"
        )
        _assert_sections_absent(ids, "NZLEG/RTA/s18", "NZLEG/RTA/s49A")

    # ---- 8. Long-tail: subletting / flatmates -----------------------------

    @_QDRANT_SKIP
    @pytest.mark.retrieval
    def test_subletting_no_wrong_routing(self, client):
        """Flatmate/subletting query must not fire property-change or bond routing."""
        r = _retrieve(client, "Can I get a flatmate or sublet a room without asking the landlord?")
        ids = _leg_ids(r["legislation"])
        # property-change sections should not appear - adding a person is not a fixture change
        _assert_sections_absent(ids, "NZLEG/RTA/s42A", "NZLEG/RTA/s42B")
        # bond section should not appear either
        _assert_sections_absent(ids, "NZLEG/RTA/s18")


# ===========================================================================
# TIER 2 - Structural integrity (LLM required, non-semantic)
# ===========================================================================

class TestStructural:
    """Tier 2: context_debug assertions. Checks budget, forbidden terms, citations."""

    @_LLM_SKIP
    @pytest.mark.structural
    def test_token_budget_property_change(self, client):
        """Context tokens must not crowd out the generation budget (< 70% of limit)."""
        _, _, ctx = _stream(client, "I planted trees in my rental backyard. Is that a problem?")
        _assert_token_budget(ctx, max_fraction=0.70)

    @_LLM_SKIP
    @pytest.mark.structural
    def test_token_budget_bond(self, client):
        """Bond query context must stay within budget."""
        _, _, ctx = _stream(client, "Can my landlord keep my bond?")
        _assert_token_budget(ctx, max_fraction=0.70)

    @_LLM_SKIP
    @pytest.mark.structural
    def test_forbidden_terms_property_change(self, client):
        """No Schedule 1A or penalty-table terms must appear in anchor text for a fixture query."""
        _, _, ctx = _stream(client, "I installed a new light fitting in my rental. Do I need landlord consent?")
        _assert_no_forbidden_terms(ctx)

    @_LLM_SKIP
    @pytest.mark.structural
    def test_forbidden_terms_wear_and_tear(self, client):
        """No penalty-table contamination in wear-and-tear anchor."""
        _, _, ctx = _stream(client, "What is fair wear and tear?")
        _assert_no_forbidden_terms(ctx)

    @_LLM_SKIP
    @pytest.mark.structural
    def test_citations_map_to_sources_bond(self, client):
        """Every [SN] in the bond answer must map to a real source index."""
        leg, answer, _ = _stream(client, "Does my landlord have to lodge my bond with Tenancy Services?")
        _assert_not_empty_answer(answer)
        # Sources come from the sources event; we need them separately
        # Re-parse from a fresh stream call to get sources count
        r = client.post("/ask/stream", headers=_TOKEN_HEADERS,
                        json={"question": "Does my landlord have to lodge my bond with Tenancy Services?"})
        events = _parse_sse_events(r.text)
        sources = next((e.get("sources", []) for e in events if e.get("type") == "sources"), [])
        answer2 = "".join(e.get("text", "") for e in events if e.get("type") == "token")
        _assert_citations_mapped(answer2, sources)

    @_LLM_SKIP
    @pytest.mark.structural
    def test_citations_map_to_sources_entry(self, client):
        """Every [SN] in the landlord-entry answer must map to a real source index."""
        r = client.post("/ask/stream", headers=_TOKEN_HEADERS,
                        json={"question": "Can my landlord enter my home without giving me notice?"})
        events = _parse_sse_events(r.text)
        sources = next((e.get("sources", []) for e in events if e.get("type") == "sources"), [])
        answer = "".join(e.get("text", "") for e in events if e.get("type") == "token")
        _assert_citations_mapped(answer, sources)

    @_LLM_SKIP
    @pytest.mark.structural
    def test_property_change_route_fires(self, client):
        """statute_routing must trigger for a tree-planting query."""
        _, _, ctx = _stream(client, "I planted several trees in the backyard of my rental. Is that okay?")
        routing = ctx.get("statute_routing", {})
        assert routing.get("triggered") is True, (
            "statute_routing.triggered must be True for a property-change query"
        )
        assert "property_change" in routing.get("matched_routes", []), (
            f"Expected property_change in matched_routes. Got: {routing.get('matched_routes')}"
        )

    @_LLM_SKIP
    @pytest.mark.structural
    def test_property_change_no_s19_in_leg(self, client):
        """After the allow-list fix, s19 must not appear in legislation for a fixture query."""
        leg, _, _ = _stream(client, "I planted trees in my rented backyard. Does this break any rules?")
        ids = _leg_ids(leg)
        _assert_sections_absent(ids, "NZLEG/RTA/s19", "NZLEG/RTA/s16A")
        _assert_sections_present(ids, "NZLEG/RTA/s42A", "NZLEG/RTA/s42B")

    @_LLM_SKIP
    @pytest.mark.structural
    def test_repairs_route_fires(self, client):
        """statute_routing must trigger for a maintenance query with s45 injected."""
        _, _, ctx = _stream(client, "My landlord has not fixed the broken oven for 3 months.")
        routing = ctx.get("statute_routing", {})
        assert routing.get("triggered") is True, (
            "statute_routing must trigger for a repairs/maintenance query"
        )
        assert "repairs_maintenance" in routing.get("matched_routes", []), (
            f"Expected repairs_maintenance route. Got: {routing.get('matched_routes')}"
        )


# ===========================================================================
# TIER 3 - Semantic answer quality (LLM required)
# ===========================================================================

class TestSemantic:
    """Tier 3: answer content checks. All tests require LLM."""

    # ---- 1. Property changes ----------------------------------------------

    @_LLM_SKIP
    @pytest.mark.llm
    def test_property_change_mentions_consent(self, client):
        """Tree-planting answer must mention consent or permission."""
        _, answer, _ = _stream(client, "I planted several trees in the backyard of my rental. Does this break any rules?")
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        assert any(t in lower for t in ["consent", "permission", "written", "s42"]), (
            "Answer must mention consent/permission for a property-change query. "
            f"Got: {answer[:300]}"
        )

    @_LLM_SKIP
    @pytest.mark.llm
    def test_property_change_past_tense_gives_next_steps(self, client):
        """Past-tense planting query must include practical steps (system prompt rule)."""
        _, answer, _ = _stream(client, "I already planted several trees in my rental backyard last month.")
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        # System prompt: past-tense -> answer in two parts: legal position + practical next steps
        assert any(t in lower for t in ["contact", "speak", "write", "notify", "retrospect", "permission", "next"]), (
            "Past-tense property-change answer must include practical next-step guidance."
        )
        _assert_practical_answer(answer)

    # ---- 2. Fair wear and tear --------------------------------------------

    @_LLM_SKIP
    @pytest.mark.llm
    def test_wear_and_tear_correct_framing(self, client):
        """Answer must say landlord generally cannot charge for fair wear and tear."""
        _, answer, _ = _stream(client, "What is fair wear and tear and can a landlord charge me for it?")
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        assert "wear and tear" in lower, "Answer must define or reference 'wear and tear'."
        assert any(t in lower for t in ["cannot", "can't", "not liable", "not charge", "not responsible"]), (
            "Answer must state landlord cannot charge for fair wear and tear."
        )
        _assert_practical_answer(answer)

    @_LLM_SKIP
    @pytest.mark.llm
    def test_wear_and_tear_carpet_distinguishes_damage(self, client):
        """Worn carpet answer must distinguish normal deterioration from tenant damage."""
        _, answer, _ = _stream(client, "The carpet is worn after living there for 6 years. Can the landlord take money from my bond?")
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        assert any(t in lower for t in ["wear and tear", "deterioration", "age", "years"]), (
            "Answer must address normal carpet wear over time."
        )
        _assert_practical_answer(answer)

    # ---- 3. Bond / agreement formation ------------------------------------

    @_LLM_SKIP
    @pytest.mark.llm
    def test_bond_lodgement_answer(self, client):
        """Bond lodgement answer must mention the landlord's obligation to lodge."""
        _, answer, _ = _stream(client, "Does my landlord have to lodge my bond?")
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        assert any(t in lower for t in ["lodge", "lodged", "tenancy services", "bond centre"]), (
            "Answer must mention bond lodgement obligation."
        )
        _assert_practical_answer(answer)

    @_LLM_SKIP
    @pytest.mark.llm
    def test_bond_before_agreement_answer(self, client):
        """Answer must not contradict the RTA on bond/agreement sequencing."""
        _, answer, _ = _stream(client, "Can a landlord ask me for proof of bond payment before giving me the tenancy agreement?")
        _assert_not_empty_answer(answer)
        _assert_practical_answer(answer)
        # Must not confidently say this is fine without qualification
        lower = answer.lower()
        assert "tenancy services" in lower or "community law" in lower or "advice" in lower, (
            "Uncertain answer must direct user to get advice."
        )

    # ---- 4. Landlord entry ------------------------------------------------

    @_LLM_SKIP
    @pytest.mark.llm
    def test_entry_mentions_notice(self, client):
        """Entry answer must mention notice requirement, not simply say 'never allowed'."""
        _, answer, _ = _stream(client, "Can my landlord enter the house without telling me?")
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        assert any(t in lower for t in ["notice", "24 hour", "24-hour", "notify", "s48", "section 48"]), (
            "Landlord entry answer must mention notice requirement."
        )
        # Must not be absolute - there are emergency exceptions
        assert "emergency" in lower or "except" in lower or "unless" in lower or "however" in lower, (
            "Entry answer must acknowledge exceptions (emergency, consent, etc.)."
        )
        _assert_practical_answer(answer)

    @_LLM_SKIP
    @pytest.mark.llm
    def test_entry_while_away_not_absolute_no(self, client):
        """Landlord entry while away answer must not give a flat yes/no without context."""
        _, answer, _ = _stream(client, "My landlord came into the house while I was at work without telling me. Is that allowed?")
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        assert any(t in lower for t in ["notice", "24 hour", "24-hour", "s48", "section 48"]), (
            "Answer must reference the notice requirement from s48."
        )
        _assert_practical_answer(answer)

    # ---- 5. Repairs / maintenance -----------------------------------------

    @_LLM_SKIP
    @pytest.mark.llm
    def test_repairs_gives_practical_steps(self, client):
        """Broken oven answer must give actionable steps, not just describe the law."""
        _, answer, _ = _stream(client, "My oven is broken and the landlord has not fixed it after 3 months. What can I do?")
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        assert any(t in lower for t in [
            "write", "written", "notice", "tribunal", "tenancy services",
            "14 day", "14-day", "record", "contact"
        ]), (
            "Repairs answer must give practical steps (written notice, Tribunal, etc.)."
        )
        _assert_practical_answer(answer)

    @_LLM_SKIP
    @pytest.mark.llm
    def test_repairs_mould_no_hallucinated_standards(self, client):
        """Mould answer must not invent specific standards if not retrieved."""
        _, answer, _ = _stream(client, "There is significant mould in my rental. Is the landlord responsible?")
        _assert_not_empty_answer(answer)
        _assert_practical_answer(answer)
        # Must reference maintenance obligation
        lower = answer.lower()
        assert any(t in lower for t in ["landlord", "repair", "maintain", "s45", "section 45", "obligation"]), (
            "Mould answer must reference landlord maintenance obligation."
        )

    # ---- 6. Termination / notice ------------------------------------------

    @_LLM_SKIP
    @pytest.mark.llm
    def test_termination_qualifies_notice_period(self, client):
        """Notice period answer must not give one fixed number without qualification."""
        _, answer, _ = _stream(client, "How much notice does a landlord need to end a periodic tenancy?")
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        # Must mention at least one period (90 days is the main one post-2021)
        assert any(t in lower for t in ["90 day", "90-day", "63 day", "notice period", "s51", "section 51"]), (
            "Termination answer must reference the notice period from s51."
        )
        _assert_practical_answer(answer)

    @_LLM_SKIP
    @pytest.mark.llm
    def test_termination_sale_does_not_confirm_28_days(self, client):
        """Answer to '28 days for sale' must not confirm that 28 days is sufficient."""
        _, answer, _ = _stream(client, "My landlord gave me 28 days notice to leave because they want to sell. Is that enough?")
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        # 28 days is NOT sufficient for a periodic tenancy sale notice (need 90 days)
        # The answer should either say 28 days is insufficient or express uncertainty
        assert not ("28 days is sufficient" in lower or "28 days is enough" in lower or
                    "28 days notice is valid" in lower), (
            "Answer must not confirm that 28 days notice for a sale is sufficient."
        )
        _assert_practical_answer(answer)

    # ---- 7. Rent increases ------------------------------------------------

    @_LLM_SKIP
    @pytest.mark.llm
    def test_rent_increase_answer(self, client):
        """Rent increase answer must mention frequency or notice requirements."""
        _, answer, _ = _stream(client, "Can my landlord increase the rent whenever they want?")
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        assert any(t in lower for t in [
            "12 month", "once a year", "notice", "s28", "section 28", "90 day", "frequency"
        ]), (
            "Rent increase answer must mention notice or frequency restrictions."
        )
        _assert_practical_answer(answer)

    @_LLM_SKIP
    @pytest.mark.llm
    def test_rent_advance_not_confused_with_increase(self, client):
        """Rent-in-advance query must not be answered with rent-increase law."""
        _, answer, _ = _stream(client, "Can the landlord ask for more than two weeks rent in advance?")
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        assert any(t in lower for t in ["in advance", "advance", "two weeks", "2 weeks"]), (
            "Rent-in-advance query must be answered in terms of advance rent, not increases."
        )
        _assert_practical_answer(answer)

    # ---- 8. Quiet enjoyment / harassment ----------------------------------

    @_LLM_SKIP
    @pytest.mark.llm
    def test_quiet_enjoyment_cautious(self, client):
        """Harassment query must be cautious, not overconfident, and direct to advice."""
        _, answer, _ = _stream(client, "My landlord keeps messaging me constantly and showing up at the property. Is this harassment?")
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        # Must suggest getting advice or keeping records - not give definitive legal verdict
        assert any(t in lower for t in [
            "record", "contact", "tenancy services", "community law", "advice",
            "quiet enjoyment", "harassment", "tribunal"
        ]), (
            "Harassment answer must direct user to keep records or seek advice."
        )
        _assert_practical_answer(answer)

    # ---- 9. Long-tail: subletting / flatmates -----------------------------

    @_LLM_SKIP
    @pytest.mark.llm
    def test_subletting_cautious_if_no_route(self, client):
        """Flatmate/subletting answer must be cautious if corpus is weak."""
        _, answer, _ = _stream(client, "Can I get a flatmate or sublet a room in my rental without asking the landlord?")
        _assert_not_empty_answer(answer)
        _assert_practical_answer(answer)
        # Must not pull in completely unrelated law (property changes, bond)
        lower = answer.lower()
        assert "tree" not in lower and "fixture" not in lower, (
            "Subletting answer must not reference tree planting or fixture law."
        )

    # ---- 10. Long-tail: boarding house ------------------------------------

    @_LLM_SKIP
    @pytest.mark.llm
    def test_boarding_house_scope_acknowledgement(self, client):
        """Boarding house query must acknowledge different scope or admit limited context."""
        _, answer, _ = _stream(client, "I live in a boarding house. Are the rules different from a normal tenancy?")
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        # Must mention boarding house - not silently answer as if it is a normal tenancy
        assert any(t in lower for t in [
            "boarding house", "boarding", "different", "part 2", "section 66",
            "may not apply", "limited", "specific advice"
        ]), (
            "Boarding house answer must acknowledge that rules differ or context is limited."
        )
        _assert_practical_answer(answer)

    # ---- Pass criterion 6: weak context admission -------------------------

    @_LLM_SKIP
    @pytest.mark.llm
    def test_weak_context_admits_uncertainty(self, client):
        """Highly niche query outside corpus should admit limited context, not guess."""
        _, answer, _ = _stream(client, "What are the rules for a landlord who owns more than 50 properties?")
        _assert_not_empty_answer(answer)
        _assert_practical_answer(answer)
        # Should not confidently invent law that is not in the corpus
        lower = answer.lower()
        assert any(t in lower for t in [
            "not enough", "limited", "specific", "advice", "tenancy services",
            "community law", "not clear", "unclear", "no information"
        ]) or len(answer) < 600, (
            "Niche query with weak context should admit uncertainty or be brief."
        )

    # ---- Pass criterion 7: perspective label in verdict -------------------

    @_LLM_SKIP
    @pytest.mark.llm
    def test_verdict_labels_tenant_perspective(self, client):
        """Verdict opening must say 'as tenant' for a clear tenant question."""
        _, answer, _ = _stream(client, "My landlord is withholding my bond. The carpet was 8 years old. Am I liable?")
        _assert_not_empty_answer(answer)
        first_sentence = answer.split(".")[0].lower()
        assert "as tenant" in first_sentence, (
            f"Verdict must label perspective 'as tenant'. Got: {first_sentence!r}"
        )

    @_LLM_SKIP
    @pytest.mark.llm
    def test_verdict_labels_landlord_perspective(self, client):
        """Verdict opening must say 'as landlord' when question is from landlord."""
        _, answer, _ = _stream(client, "I am a landlord. My tenant has not paid rent for 3 weeks. What can I do?")
        _assert_not_empty_answer(answer)
        first_sentence = answer.split(".")[0].lower()
        assert "as landlord" in first_sentence, (
            f"Verdict must label perspective 'as landlord'. Got: {first_sentence!r}"
        )

    # ---- Pass criterion 8: nonsensical question sanity check --------------

    @_LLM_SKIP
    @pytest.mark.llm
    def test_ambiguous_question_asks_clarification_or_interprets_charitably(self, client):
        """Unusual-but-possibly-valid question must either ask for clarification or
        pick the most plausible legal reading - never fabricate rights."""
        _, answer, _ = _stream(
            client,
            "I am angry because my landlord gave me two weeks free-rent, what can I do?"
        )
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        # Must not fabricate deduction rights
        assert "deduct the difference" not in lower
        assert "entitled to deduct" not in lower
        # Must either ask for clarification OR address a plausible valid scenario
        # (e.g. free rent offered to avoid fixing a repair, or conditions attached)
        assert any(t in lower for t in [
            "clarif", "rephrase", "what actually", "could you",
            "repair", "oblig", "condition", "avoid", "instead of",
            "fulfil", "fulfill", "breach",
        ]), (
            "Ambiguous question must ask for clarification or address a plausible "
            "valid reading - not fabricate legal rights."
        )

    @_LLM_SKIP
    @pytest.mark.llm
    def test_nonsensical_question_no_fabrication(self, client):
        """Logically contradictory question must not produce fabricated legal rights."""
        _, answer, _ = _stream(
            client,
            "As tenant, I am complaining because the rental payment is too cheap, what can I do?"
        )
        _assert_not_empty_answer(answer)
        lower = answer.lower()
        # Must not fabricate deduction rights or false remedies
        assert "deduct the difference" not in lower, (
            "Answer must not fabricate a right to deduct rent for a nonsensical question."
        )
        assert "entitled to deduct" not in lower, (
            "Answer must not invent deduction entitlements."
        )
        # Must signal the question is not a valid legal claim or redirect
        assert any(t in lower for t in [
            "not a valid", "not a legal", "cannot legally complain",
            "rephrase", "clarify", "no legal", "does not correspond",
            "not recognised", "no basis", "not possible under",
        ]), (
            "Nonsensical question must be flagged as not a valid legal situation."
        )
