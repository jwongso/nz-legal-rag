"""
Scrape NZ court decisions from NZLII (nzlii.org).

NZLII is a public legal information repository. All content is freely accessible.
URL pattern: https://www.nzlii.org/nz/cases/{COURT}/{YEAR}/{N}.html

Uses subprocess curl instead of httpx - NZLII is behind Cloudflare which blocks
httpx based on TLS fingerprinting regardless of headers. curl uses the system
libcurl with a browser-compatible TLS fingerprint.
"""

import asyncio
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from pypdf import PdfReader

import config

# Courts whose NZLII pages are metadata-only wrappers - actual text is in PDF
_PDF_COURTS = {"NZTT"}

_CURL_CMD = [
    "curl", "-s", "-L", "--compressed",
    "-A", "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "-H", "Accept-Language: en-NZ,en;q=0.5",
    "-H", "DNT: 1",
    "--max-time", "20",
]
_RATE_LIMIT_S = 1.5  # be polite to NZLII


@dataclass
class CaseDocument:
    case_id: str
    court: str
    court_name: str
    year: int
    number: int
    url: str
    title: str
    date: str
    parties: list[str]
    text: str
    citations: list[str]


async def _curl_get(url: str) -> str | None:
    """Fetch a URL via subprocess curl, bypassing Cloudflare TLS checks."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *_CURL_CMD, url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=25)
        if proc.returncode == 0 and stdout:
            return stdout.decode("utf-8", errors="replace")
        return None
    except (asyncio.TimeoutError, Exception):
        return None


async def _curl_get_bytes(url: str) -> bytes | None:
    """Fetch binary content (e.g. PDF) via subprocess curl."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *_CURL_CMD, url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode == 0 and stdout:
            return stdout
        return None
    except (asyncio.TimeoutError, Exception):
        return None


def _pdf_url_from_html(html: str, base_url: str) -> str | None:
    """Extract the PDF URL embedded as <object data="..."> in NZLII HTML pages."""
    match = re.search(r'<object[^>]+data="([^"]+\.pdf)"', html, re.I)
    if match:
        return urljoin(base_url, match.group(1))
    # Fallback: href link to .pdf
    match = re.search(r'href="([^"]+\.pdf)"', html, re.I)
    if match:
        return urljoin(base_url, match.group(1))
    return None


def _is_pdf_wrapper(html: str) -> bool:
    """Return True if this page embeds a PDF as primary content via <object> tag.

    Old NZLII pages (pre-~2000) store decision text as PDFs and serve only a
    metadata wrapper in HTML. Detecting the <object> embed is more reliable than
    checking court membership, since coverage varies by year.
    """
    return bool(re.search(r'<object[^>]+data="[^"]+\.pdf"', html, re.I))


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using pypdf."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        return ""


async def list_case_urls(court: str, year: int) -> list[str]:
    """Return all decision URLs for a given court and year."""
    index_url = f"{config.NZLII_BASE}/nz/cases/{court}/{year}/"
    html = await _curl_get(index_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.match(r"^\d+\.html$", href.split("/")[-1]):
            urls.append(urljoin(index_url, href))
    return urls


def _extract_parties(title: str) -> list[str]:
    for sep in [" v ", " v. ", " vs ", " vs. "]:
        if sep in title:
            return [p.strip() for p in title.split(sep, 1)]
    return [title.strip()]


def _extract_citations(text: str) -> list[str]:
    pattern = r"\[\d{4}\]\s+[A-Z]+[A-Za-z]*\s+\d+"
    return list(set(re.findall(pattern, text)))


def _parse_html(html: str, url: str, court: str, year: int, pdf_text: str = "") -> CaseDocument | None:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    title = re.sub(r"\s*[-|]\s*NZLII.*$", "", title).strip()
    # NZLII uses "NONE" for suppressed party names (court suppression orders)
    title = re.sub(r"\bNONE\b", "[Suppressed]", title)

    date_match = re.search(
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4})\b",
        html,
    )
    date = date_match.group(1) if date_match else ""

    if pdf_text:
        text = pdf_text
    else:
        body = soup.find("body")
        text = body.get_text(separator="\n", strip=True) if body else ""
        # Strip NZLII breadcrumb navigation lines and copyright footer
        text = re.sub(r"(?m)^NZLII\s*$", "", text)
        text = re.sub(r"(?m)^>>\s*.*$", "", text)
        text = re.sub(r"NZLII:\s*Copyright Policy.*$", "", text, flags=re.DOTALL)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

    try:
        number = int(url.rstrip("/").split("/")[-1].replace(".html", ""))
    except ValueError:
        return None

    return CaseDocument(
        case_id=f"{court}/{year}/{number}",
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
    Caches raw HTML to avoid re-scraping on restart.
    """
    if cache_dir is None:
        cache_dir = config.DATA_DIR / "raw" / court
    cache_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        urls = await list_case_urls(court, year)
        urls = urls[:max_per_year]
        print(f"  {court}/{year}: {len(urls)} decisions found")

        for url in urls:
            filename = url.rstrip("/").split("/")[-1]
            cache_file = cache_dir / str(year) / filename
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            if cache_file.exists():
                html = cache_file.read_text(encoding="utf-8", errors="replace")
            else:
                await asyncio.sleep(_RATE_LIMIT_S)
                html = await _curl_get(url)
                if html:
                    cache_file.write_text(html, encoding="utf-8")

            if not html:
                continue

            # PDF extraction: always attempt if page embeds PDF as primary content.
            # Old NZLII pages (pre-~2000) use <object data="...pdf"> wrappers for
            # any court. _PDF_COURTS is kept for courts that are always PDF-only.
            pdf_text = ""
            if court in _PDF_COURTS or _is_pdf_wrapper(html):
                pdf_cache = cache_file.with_suffix(".pdf.txt")
                if pdf_cache.exists():
                    pdf_text = pdf_cache.read_text(encoding="utf-8")
                else:
                    pdf_url = _pdf_url_from_html(html, url)
                    if pdf_url:
                        await asyncio.sleep(_RATE_LIMIT_S)
                        pdf_bytes = await _curl_get_bytes(pdf_url)
                        if pdf_bytes:
                            pdf_text = _extract_text_from_pdf(pdf_bytes)
                            if pdf_text:
                                pdf_cache.write_text(pdf_text, encoding="utf-8")

            # Skip wrapper pages where PDF extraction failed - they contain only
            # navigation breadcrumbs and are not useful for retrieval.
            is_wrapper = court in _PDF_COURTS or _is_pdf_wrapper(html)
            if is_wrapper and not pdf_text:
                continue

            doc = _parse_html(html, url, court, year, pdf_text=pdf_text)
            if doc and len(doc.text) > 200:
                yield doc
