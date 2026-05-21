"""LLM-based structured extraction pipeline.

Enriches employment_cases and sentencing_cases with fields the regex pipelines miss:
reinstatement outcomes, compensation amounts, offence descriptions, aggravating and
mitigating factors, etc.

Only processes documents that already have a row in the target table (i.e. the regex
pipeline already detected relevance). The LLM reads a focused excerpt - document header
plus the final remedy/conclusion chunks - to stay within the 4096-token context window.

Progress is tracked in data/llm_extract_progress.json so runs can be interrupted and
resumed without reprocessing completed documents.

Run:
    python -u -m ingest.llm_extract_pipeline --domain employment
    python -u -m ingest.llm_extract_pipeline --domain criminal --limit 200
    python -u -m ingest.llm_extract_pipeline --domain employment --dry-run
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import httpx
import psycopg2
import psycopg2.extras

import config


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROGRESS_FILE = Path("data/llm_extract_progress.json")

# Text budget: first N chunks (header) + last M chunks (outcome/remedy)
_HEAD_CHUNKS = 1
_TAIL_CHUNKS = 3
_MAX_CHARS_PER_CHUNK = 500   # truncate each chunk to stay inside 4096 ctx

# LLM settings for extraction - zero temperature, enough tokens for verbose JSON
_EXTRACT_MAX_TOKENS = 600
_EXTRACT_TEMPERATURE = 0.0

_SYSTEM_PROMPT = (
    "You are a data extraction tool. "
    "Output only a single valid JSON object. "
    "No thinking, no explanation, no markdown. JSON only."
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_EMPLOYMENT_PROMPT = """\
You are analyzing a New Zealand Employment Relations Authority or Employment Court decision.

Extract the following fields. Output ONLY a single JSON object, no explanation.

Fields:
- "grievance_types": list of zero or more strings from this set: \
["unjustified_dismissal", "constructive_dismissal", "disadvantage", \
"harassment", "discrimination", "unjustified_action"]
- "reinstatement_ordered": true if the court ordered reinstatement, \
false if reinstatement was explicitly considered and declined, null if not discussed
- "compensation_nzd": numeric NZD amount awarded as compensation or lost wages \
(exclude GST, exclude costs), null if none awarded or amount not stated
- "contributory_conduct_pct": integer 0-100 reduction for the employee's \
own contributory conduct, null if not mentioned

Text:
{text}"""

_CRIMINAL_PROMPT = """\
You are analyzing a New Zealand criminal sentencing decision.

Extract the following fields. Output ONLY a single JSON object, no explanation.

Fields:
- "offence": primary offence being sentenced (e.g. "aggravated robbery", \
"manslaughter"), null if not identifiable
- "starting_point_months": numeric months for the judicial starting point \
before adjustments, null if not stated
- "final_sentence_months": numeric months imprisonment actually imposed, \
null if not an imprisonment sentence
- "home_detention_months": numeric months home detention imposed, \
null if not home detention
- "guilty_plea_discount_pct": integer 0-100 discount applied for a guilty plea, \
null if not mentioned
- "aggravating_factors": short comma-separated list of aggravating factors \
(e.g. "weapon use, group offending"), null if none mentioned
- "mitigating_factors": short comma-separated list of mitigating factors \
(e.g. "youth, remorse, no prior convictions"), null if none mentioned
- "appeal_outcome": "allowed", "dismissed", or "varied" if this is an appeal \
decision, null otherwise

Text:
{text}"""

_PROMPTS = {"employment": _EMPLOYMENT_PROMPT, "criminal": _CRIMINAL_PROMPT}


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def _load_progress() -> dict[str, set[int]]:
    if _PROGRESS_FILE.exists():
        raw = json.loads(_PROGRESS_FILE.read_text())
        return {k: set(v) for k, v in raw.items()}
    return {"employment": set(), "criminal": set()}


def _save_progress(progress: dict[str, set[int]]) -> None:
    _PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PROGRESS_FILE.write_text(json.dumps(
        {k: sorted(v) for k, v in progress.items()}, indent=2
    ))


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_candidate_docs(conn, domain: str) -> list[tuple[int, str]]:
    """Return (document_id, citation) for all documents in the target table."""
    table = "employment_cases" if domain == "employment" else "sentencing_cases"
    cur = conn.cursor()
    cur.execute(f"""
        SELECT ec.document_id, d.citation
        FROM {table} ec
        JOIN documents d ON d.id = ec.document_id
        ORDER BY ec.document_id
    """)
    return cur.fetchall()


def _get_chunks(conn, document_id: int) -> list[str]:
    """Fetch text chunks for a document, ordered by chunk_index."""
    cur = conn.cursor()
    cur.execute(
        "SELECT text FROM chunks WHERE document_id = %s ORDER BY chunk_index",
        (document_id,),
    )
    return [row[0] for row in cur.fetchall() if row[0]]


def _select_text(chunks: list[str]) -> str:
    """Take head + tail chunks to stay within context budget, without overlap."""
    if not chunks:
        return ""
    head = chunks[:_HEAD_CHUNKS]
    tail_start = max(_HEAD_CHUNKS, len(chunks) - _TAIL_CHUNKS)
    tail = chunks[tail_start:]
    selected = head + tail
    parts = [c[:_MAX_CHARS_PER_CHUNK] for c in selected]
    return "\n\n---\n\n".join(parts)


def _update_employment(conn, document_id: int, fields: dict) -> None:
    cur = conn.cursor()
    # COALESCE keeps existing regex-extracted values, fills in NULLs with LLM values
    gtypes = fields.get("grievance_types") or []
    cur.execute("""
        UPDATE employment_cases SET
            grievance_types = CASE
                WHEN grievance_types IS NULL OR grievance_types = '{}'
                THEN %s::text[]
                ELSE grievance_types
            END,
            reinstatement = COALESCE(reinstatement, %s),
            compensation  = COALESCE(compensation,  %s),
            contributory_conduct_pct = COALESCE(contributory_conduct_pct, %s)
        WHERE document_id = %s
    """, (
        gtypes if gtypes else None,
        fields.get("reinstatement_ordered"),
        fields.get("compensation_nzd"),
        fields.get("contributory_conduct_pct"),
        document_id,
    ))
    conn.commit()


def _update_sentencing(conn, document_id: int, fields: dict) -> None:
    cur = conn.cursor()
    cur.execute("""
        UPDATE sentencing_cases SET
            offence              = COALESCE(offence,              %s),
            aggravating_factors  = COALESCE(aggravating_factors,  %s),
            mitigating_factors   = COALESCE(mitigating_factors,   %s),
            guilty_plea_discount = COALESCE(guilty_plea_discount, %s),
            appeal_outcome       = COALESCE(appeal_outcome,       %s)
        WHERE document_id = %s
    """, (
        fields.get("offence"),
        fields.get("aggravating_factors"),
        fields.get("mitigating_factors"),
        fields.get("guilty_plea_discount_pct"),
        fields.get("appeal_outcome"),
        document_id,
    ))
    conn.commit()


_UPDATERS = {"employment": _update_employment, "criminal": _update_sentencing}


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Parse JSON from LLM output - handles preamble, postamble, and truncation."""
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find complete {...} block
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Recover truncated JSON - find opening brace and try closing it
    m = re.search(r'\{.*', text, re.DOTALL)
    if m:
        partial = m.group()
        # Strip trailing incomplete key/value (e.g. `"reinst`)
        partial = re.sub(r',?\s*"[^"]*$', '', partial)
        for suffix in ('}', 'null}'):
            try:
                return json.loads(partial + suffix)
            except json.JSONDecodeError:
                pass
    return None


def _validate_employment(fields: dict) -> dict:
    """Coerce and bound-check employment fields."""
    valid_types = {
        "unjustified_dismissal", "constructive_dismissal", "disadvantage",
        "harassment", "discrimination", "unjustified_action",
    }
    gtypes = fields.get("grievance_types")
    if isinstance(gtypes, list):
        fields["grievance_types"] = [g for g in gtypes if g in valid_types]
    else:
        fields["grievance_types"] = []

    pct = fields.get("contributory_conduct_pct")
    if pct is not None:
        try:
            pct = float(pct)
            fields["contributory_conduct_pct"] = max(0.0, min(100.0, pct))
        except (TypeError, ValueError):
            fields["contributory_conduct_pct"] = None

    comp = fields.get("compensation_nzd")
    if comp is not None:
        try:
            fields["compensation_nzd"] = max(0.0, float(comp))
        except (TypeError, ValueError):
            fields["compensation_nzd"] = None

    reinst = fields.get("reinstatement_ordered")
    if not isinstance(reinst, bool):
        fields["reinstatement_ordered"] = None

    return fields


def _validate_sentencing(fields: dict) -> dict:
    """Coerce and bound-check sentencing fields."""
    for col in ("starting_point_months", "final_sentence_months", "home_detention_months"):
        val = fields.get(col)
        if val is not None:
            try:
                v = float(val)
                fields[col] = v if 0 < v < 1200 else None
            except (TypeError, ValueError):
                fields[col] = None

    pct = fields.get("guilty_plea_discount_pct")
    if pct is not None:
        try:
            pct = float(pct)
            fields["guilty_plea_discount_pct"] = max(0.0, min(100.0, pct))
        except (TypeError, ValueError):
            fields["guilty_plea_discount_pct"] = None

    outcome = fields.get("appeal_outcome")
    if outcome not in ("allowed", "dismissed", "varied"):
        fields["appeal_outcome"] = None

    return fields


_VALIDATORS = {"employment": _validate_employment, "criminal": _validate_sentencing}


async def _call_llm(client: httpx.AsyncClient, prompt: str, model: str) -> dict | None:
    """Send extraction prompt to LLM, retry once on parse failure.

    When the base URL looks like an Ollama instance (port 11434), uses Ollama's
    native /api/chat endpoint with think:false to suppress reasoning tokens.
    Otherwise uses the OpenAI-compatible /chat/completions endpoint.
    """
    is_ollama = ":11434" in str(client.base_url)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(2):
        try:
            if is_ollama:
                # Native Ollama API - think:false actually works here
                payload = {
                    "model": model,
                    "messages": messages,
                    "think": False,
                    "stream": False,
                    "options": {"temperature": _EXTRACT_TEMPERATURE, "num_predict": _EXTRACT_MAX_TOKENS},
                }
                base = str(client.base_url).replace("/v1", "").rstrip("/")
                resp = await client.post(f"{base}/api/chat", json=payload, timeout=90)
                resp.raise_for_status()
                content = resp.json()["message"]["content"].strip()
            else:
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": _EXTRACT_MAX_TOKENS,
                    "temperature": _EXTRACT_TEMPERATURE,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                resp = await client.post("/chat/completions", json=payload, timeout=90)
                if resp.status_code == 500 and attempt == 0:
                    await asyncio.sleep(2)
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()

            result = _extract_json(content)
            if result is not None:
                return result
            print(f"  [warn] JSON parse failed (attempt {attempt + 1}), raw: {content[:120]}")
        except Exception as exc:
            print(f"  [warn] LLM call failed (attempt {attempt + 1}): {exc}")
        if attempt == 0:
            await asyncio.sleep(3)
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run(domain: str, limit: int | None, dry_run: bool,
              llm_url: str, llm_model: str) -> None:
    print(f"LLM extraction pipeline: domain={domain} limit={limit} "
          f"model={llm_model} url={llm_url} dry_run={dry_run}")

    progress = _load_progress()
    done_ids = progress.get(domain, set())
    print(f"Already processed: {len(done_ids)} documents")

    conn = psycopg2.connect(dbname="nz_legal")
    candidates = _get_candidate_docs(conn, domain)
    pending = [(doc_id, cit) for doc_id, cit in candidates if doc_id not in done_ids]

    if limit:
        pending = pending[:limit]

    print(f"Pending: {len(pending)} documents")

    if not pending:
        print("Nothing to do.")
        conn.close()
        return

    prompt_template = _PROMPTS[domain]
    validator = _VALIDATORS[domain]
    updater = _UPDATERS[domain]

    async with httpx.AsyncClient(base_url=llm_url) as client:
        for i, (doc_id, citation) in enumerate(pending, 1):
            t0 = time.monotonic()
            chunks = _get_chunks(conn, doc_id)
            text = _select_text(chunks)

            if not text:
                print(f"[{i}/{len(pending)}] {citation}: no text, skipping")
                done_ids.add(doc_id)
                continue

            prompt = prompt_template.format(text=text)

            print(f"[{i}/{len(pending)}] {citation} ({len(chunks)} chunks, {len(text)} chars)...",
                  end=" ", flush=True)

            if dry_run:
                print("(dry-run, skip)")
                continue

            fields = await _call_llm(client, prompt, llm_model)
            elapsed = time.monotonic() - t0

            if fields is None:
                print(f"FAILED ({elapsed:.1f}s)")
            else:
                fields = validator(fields)
                updater(conn, doc_id, fields)
                done_ids.add(doc_id)
                progress[domain] = done_ids
                _save_progress(progress)
                # Brief summary of what was extracted
                summary = {k: v for k, v in fields.items() if v not in (None, [], "")}
                print(f"ok ({elapsed:.1f}s) -> {summary}")

    conn.close()
    print(f"\nDone. Processed {len(done_ids)} total {domain} documents.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-based structured extraction pipeline")
    parser.add_argument("--domain", choices=["employment", "criminal"], required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum documents to process in this run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and show text without calling the LLM or writing to DB")
    parser.add_argument("--llm-url", default=config.LLM_BASE_URL,
                        help="OpenAI-compatible base URL (default: config.LLM_BASE_URL)")
    parser.add_argument("--llm-model", default=config.LLM_MODEL,
                        help="Model name to pass in the API request (default: config.LLM_MODEL)")
    args = parser.parse_args()
    asyncio.run(run(args.domain, args.limit, args.dry_run, args.llm_url, args.llm_model))


if __name__ == "__main__":
    main()
