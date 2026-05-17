"""
Scrape NZ court decisions from NZLII (nzlii.org).

NZLII is a public legal information repository. All content is freely accessible.
URL pattern: https://www.nzlii.org/nz/cases/{COURT}/{YEAR}/{N}.html
"""

import asyncio
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import httpx
from bs4 import BeautifulSoup

import config

_HEADERS = {
    "User-Agent": "nz-legal-rag/1.0 (legal research; contact: research@example.nz)",
}
_RATE_LIMIT_S = 1.5  # be polite to NZLII


@dataclass
class CaseDocument:
    case_id: str          # e.g. NZTT/2023/42
    court: str            # e.g. NZTT
    court_name: str       # e.g. NZ Tenancy Tribunal
    year: int
    number: int
    url: str
    title: str
    date: str
    parties: list[str]
    text: str
    citations: list[str]  # other cases referenced in the decision


async def list_case_urls(client: httpx.AsyncClient, court: str, year: int) -> list[str]:
    """Return all decision URLs for a given court and year."""
    index_url = f"{config.NZLII_BASE}/nz/cases/{court}/{year}/"
    try:
        resp = await client.get(index_url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return []
        raise

    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.match(r"^\d+\.html$", href.split("/")[-1]):
            full = href if href.startswith("http") else f"{config.NZLII_BASE}{href}"
            urls.append(full)
    return urls


def _extract_parties(title: str) -> list[str]:
    """Split 'Smith v Jones' into ['Smith', 'Jones']."""
    for sep in [" v ", " v. ", " vs ", " vs. "]:
        if sep in title:
            return [p.strip() for p in title.split(sep, 1)]
    return [title.strip()]


def _extract_citations(text: str) -> list[str]:
    """Find case citations like [2023] NZHC 42 or (2021) 3 NZLR 100."""
    pattern = r"\[\d{4}\]\s+[A-Z]+[A-Za-z]*\s+\d+"
    return list(set(re.findall(pattern, text)))


async def fetch_case(client: httpx.AsyncClient, url: str, court: str, year: int) -> CaseDocument | None:
    try:
        resp = await client.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException):
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove nav/header/footer noise
    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    title = re.sub(r"\s*[-|]\s*NZLII.*$", "", title).strip()

    # Extract decision date
    date = ""
    date_match = re.search(
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4})\b",
        resp.text,
    )
    if date_match:
        date = date_match.group(1)

    body = soup.find("body")
    text = body.get_text(separator="\n", strip=True) if body else ""

    # Remove boilerplate NZLII footer text
    text = re.sub(r"NZLII:\s*Copyright Policy.*$", "", text, flags=re.DOTALL).strip()

    number = int(url.rstrip("/").split("/")[-1].replace(".html", ""))
    case_id = f"{court}/{year}/{number}"

    return CaseDocument(
        case_id=case_id,
        court=court,
        court_name=config.COURTS.get(court, court),
        year=year,
        number=number,
        url=url,
        title=title,
        date=date,
        parties=_extract_parties(title),
        text=text,
        citations=_extract_citations(text),
    )


async def scrape_court(
    court: str,
    years: list[int],
    max_per_year: int = 200,
    cache_dir: Path | None = None,
) -> AsyncIterator[CaseDocument]:
    """
    Yield CaseDocument objects for the given court and years.
    Caches raw HTML to cache_dir to avoid re-scraping.
    """
    if cache_dir is None:
        cache_dir = config.DATA_DIR / "raw" / court
    cache_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        for year in years:
            urls = await list_case_urls(client, court, year)
            urls = urls[:max_per_year]
            print(f"  {court}/{year}: {len(urls)} decisions found")

            for url in urls:
                filename = url.rstrip("/").split("/")[-1]
                cache_file = cache_dir / str(year) / filename
                cache_file.parent.mkdir(parents=True, exist_ok=True)

                if cache_file.exists():
                    html = cache_file.read_text(encoding="utf-8", errors="replace")
                    doc = await _parse_cached(html, url, court, year)
                else:
                    await asyncio.sleep(_RATE_LIMIT_S)
                    try:
                        resp = await client.get(url, headers=_HEADERS, timeout=20)
                        resp.raise_for_status()
                        html = resp.text
                        cache_file.write_text(html, encoding="utf-8")
                        doc = await _parse_cached(html, url, court, year)
                    except (httpx.HTTPError, httpx.TimeoutException):
                        doc = None

                if doc and len(doc.text) > 200:
                    yield doc


async def _parse_cached(html: str, url: str, court: str, year: int) -> CaseDocument | None:
    """Parse a CaseDocument from cached HTML without a network call."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    title = re.sub(r"\s*[-|]\s*NZLII.*$", "", title).strip()

    date_match = re.search(
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4})\b",
        html,
    )
    date = date_match.group(1) if date_match else ""

    body = soup.find("body")
    text = body.get_text(separator="\n", strip=True) if body else ""
    text = re.sub(r"NZLII:\s*Copyright Policy.*$", "", text, flags=re.DOTALL).strip()

    number = int(url.rstrip("/").split("/")[-1].replace(".html", ""))
    case_id = f"{court}/{year}/{number}"

    return CaseDocument(
        case_id=case_id,
        court=court,
        court_name=config.COURTS.get(court, court),
        year=year,
        number=number,
        url=url,
        title=title,
        date=date,
        parties=_extract_parties(title),
        text=text,
        citations=_extract_citations(text),
    )
