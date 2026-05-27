"""
Scrape NZ Tenancy Tribunal decisions from the Ministry of Justice public search.

Source: https://forms.justice.govt.nz/search/TT/
Backed by Apache Solr at: https://forms.justice.govt.nz/forms/publicSolrProxy/solr/TTV2/select

The MoJ Solr index is directly accessible with browser-like HTTP headers.
Decision text is embedded in the document_text_abstract field - no PDF fetching needed.
Coverage: approximately 3 years of decisions (~32,000 documents).

Copyright: Crown copyright. Non-commercial reuse with attribution permitted under
tenancy.govt.nz terms and the NZ Government Open Access and Licensing (NZGOAL) framework.
Attribution required: "Source: Ministry of Justice, forms.justice.govt.nz"
"""

import asyncio
import gzip
import json
import re
import time
from typing import AsyncIterator
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ingest.scraper import CaseDocument

_SOLR_URL = "https://forms.justice.govt.nz/forms/publicSolrProxy/solr/TTV2/select"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Referer": "https://forms.justice.govt.nz/search/TT/",
    "Accept": "text/javascript, application/javascript, */*",
    "Accept-Encoding": "gzip, deflate",
}
_PAGE_SIZE = 100
_RATE_LIMIT_S = 0.5


def _fetch_solr(start: int = 0, rows: int = _PAGE_SIZE) -> dict:
    params = {
        "q": "jurisdictionCode_s:TT",
        "fq": "jurisdictionCode_s:TT",
        "rows": str(rows),
        "start": str(start),
        "fl": (
            "id,applicationNumber_s,publishedDate_dt,publishedDate_s,"
            "document_text_abstract,orderDetailJson_s,"
            "casePerOrgApplicant_s,casePerOrgRespondent_s"
        ),
        "sort": "decisionDateIndex_l asc",
        "wt": "json",
        "json.wrf": "cb",
    }
    url = _SOLR_URL + "?" + urlencode(params)
    req = Request(url, headers=_HEADERS)
    resp = urlopen(req, timeout=30)
    raw = resp.read()
    body = gzip.decompress(raw).decode() if raw[:2] == b"\x1f\x8b" else raw.decode()
    start_idx = body.index("(") + 1
    end_idx = body.rindex(")")
    return json.loads(body[start_idx:end_idx])


def _parse_doc(doc: dict) -> CaseDocument | None:
    text = doc.get("document_text_abstract", "").strip()
    if len(text) < 100:
        return None

    app_numbers = doc.get("applicationNumber_s", [])
    app_number = app_numbers[0] if app_numbers else ""
    if not app_number:
        app_number = doc.get("id", "unknown").split("_")[0]

    dt = doc.get("publishedDate_dt", "")
    year = int(dt[:4]) if dt and len(dt) >= 4 else 0

    # DD/MM/YY -> DD/MM/YYYY for consistency
    date_short = doc.get("publishedDate_s", [""])[0]
    if date_short and len(date_short) == 8:
        d, m, y = date_short.split("/")
        date_str = f"{d}/{m}/20{y}"
    else:
        date_str = date_short

    applicants = [a for a in doc.get("casePerOrgApplicant_s", []) if a and a not in ("NONE", "")]
    respondents = [r for r in doc.get("casePerOrgRespondent_s", []) if r and r not in ("NONE", "")]
    parties = applicants + respondents

    if applicants and respondents:
        title = f"{applicants[0]} v {respondents[0]}"
    elif parties:
        title = " v ".join(parties[:2])
    else:
        title = f"[{year}] NZTT {app_number}"

    # Extract any [YEAR] NZTT NNNN citations from the text
    citations = re.findall(r"\[\d{4}\]\s+NZTT\s+\d+", text)

    return CaseDocument(
        case_id=f"NZTT-MOJ-{app_number}",
        court="NZTT",
        court_name="Tenancy Tribunal",
        year=year,
        number=int(app_number) if app_number.isdigit() else 0,
        url=f"https://forms.justice.govt.nz/search/TT/",
        title=title,
        date=date_str,
        parties=parties,
        text=text,
        citations=citations,
    )


def count_total() -> int:
    """Return total number of TT decisions in the MoJ Solr index."""
    data = _fetch_solr(start=0, rows=1)
    return data["response"]["numFound"]


async def scrape_moj(verbose: bool = True) -> AsyncIterator[CaseDocument]:
    """Yield all TT decisions from the MoJ Solr index, paginated."""
    total = count_total()
    if verbose:
        print(f"MoJ TT index: {total} decisions")

    start = 0
    fetched = 0
    while start < total:
        data = _fetch_solr(start=start, rows=_PAGE_SIZE)
        docs = data["response"]["docs"]
        if not docs:
            break

        for doc in docs:
            case = _parse_doc(doc)
            if case:
                fetched += 1
                yield case

        start += len(docs)
        if verbose and start % 500 == 0:
            print(f"  {start}/{total} fetched, {fetched} parsed")

        await asyncio.sleep(_RATE_LIMIT_S)

    if verbose:
        print(f"Done: {fetched} decisions parsed from {start} records")
