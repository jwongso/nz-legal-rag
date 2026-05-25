"""
Scrape NZ legal academic journals (OJS-hosted) and download new PDFs to data/inbox/.

Maintains a state file (data/journal_scrape_state.json) so each run only fetches
articles not yet seen. Designed to be called by journal_pipeline.py on a schedule.

Run:
    python -m ingest.scrape_journals               # fetch up to 10 new PDFs
    python -m ingest.scrape_journals --limit 20    # fetch up to 20
    python -m ingest.scrape_journals --dry-run     # show what would be downloaded
"""

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

import config

_STATE_FILE = config.DATA_DIR / "journal_scrape_state.json"
_INBOX      = config.DATA_DIR / "inbox"

_JOURNALS = [
    {
        "name":     "Victoria University of Wellington Law Review",
        "short":    "VUWLR",
        "base_url": "https://ojs.victoria.ac.nz/vuwlr",
    },
    {
        "name":     "New Zealand Universities Law Review",
        "short":    "NZULR",
        "base_url": "https://ojs.victoria.ac.nz/nzulr",
    },
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"
    )
}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text())
    return {"seen_pdf_urls": [], "downloaded": []}


def _save_state(state: dict) -> None:
    _STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# OJS scraping
# ---------------------------------------------------------------------------

def _get(client: httpx.Client, url: str) -> BeautifulSoup | None:
    try:
        r = client.get(url, timeout=20, follow_redirects=True)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"  [warn] GET {url}: {e}")
        return None


def _discover_pdf_urls(client: httpx.Client, base_url: str) -> list[tuple[str, str]]:
    """Return list of (title, pdf_url) for all articles in the journal archive."""
    archive_soup = _get(client, f"{base_url}/issue/archive")
    if not archive_soup:
        return []

    issue_urls = list(dict.fromkeys(
        a["href"] for a in archive_soup.select("a[href*='/issue/view/']")
        if a["href"].strip()
    ))

    results: list[tuple[str, str]] = []
    for issue_url in issue_urls:
        issue_soup = _get(client, issue_url)
        if not issue_soup:
            continue
        time.sleep(0.5)

        # Collect (title, pdf_url) pairs from the issue TOC
        # PDF links appear as siblings of article title links
        for a in issue_soup.select("a[href*='/article/view/']"):
            href = a["href"]
            text = a.get_text(strip=True)
            # Direct PDF links contain an extra path segment (article/view/NNN/MMM)
            if re.search(r"/article/view/\d+/\d+", href) and text == "PDF":
                # Find the article title - look back for the nearest title link
                title = ""
                for sib in a.find_all_previous("a", href=re.compile(r"/article/view/\d+$")):
                    title = sib.get_text(strip=True)
                    break
                if href not in [u for _, u in results]:
                    results.append((title or "untitled", href))

    return results


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _safe_filename(journal_short: str, title: str, pdf_url: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower())[:60].strip("_")
    uid  = hashlib.md5(pdf_url.encode()).hexdigest()[:6]
    return f"{journal_short.lower()}_{slug}_{uid}.pdf"


def fetch_new_pdfs(limit: int = 10, dry_run: bool = False) -> list[dict]:
    """Download up to `limit` new PDFs. Returns list of downloaded file dicts."""
    state    = _load_state()
    seen_set = set(state["seen_pdf_urls"])
    _INBOX.mkdir(parents=True, exist_ok=True)

    downloaded: list[dict] = []

    with httpx.Client(headers=_HEADERS) as client:
        for journal in _JOURNALS:
            if len(downloaded) >= limit:
                break

            print(f"Scanning {journal['short']} ...", flush=True)
            pdf_pairs = _discover_pdf_urls(client, journal["base_url"])
            print(f"  {len(pdf_pairs)} articles found in archive", flush=True)

            for title, pdf_url in pdf_pairs:
                if len(downloaded) >= limit:
                    break
                if pdf_url in seen_set:
                    continue

                seen_set.add(pdf_url)

                if dry_run:
                    print(f"  [dry-run] would download: {title[:70]}")
                    downloaded.append({"url": pdf_url, "title": title, "dry_run": True})
                    continue

                filename = _safe_filename(journal["short"], title, pdf_url)
                dest     = _INBOX / filename

                # Skip if already in inbox (re-run safety)
                if dest.exists():
                    continue

                try:
                    r = client.get(pdf_url, timeout=30, follow_redirects=True)
                    r.raise_for_status()
                    if b"%PDF" not in r.content[:8]:
                        print(f"  [skip] not a PDF: {pdf_url}")
                        continue
                    dest.write_bytes(r.content)
                    entry = {
                        "url":      pdf_url,
                        "title":    title,
                        "journal":  journal["short"],
                        "filename": filename,
                        "at":       datetime.now(timezone.utc).isoformat(),
                    }
                    downloaded.append(entry)
                    state["downloaded"].append(entry)
                    print(f"  + {filename}  ({len(r.content)//1024}KB)  {title[:60]}")
                    time.sleep(0.5)
                except Exception as e:
                    print(f"  [error] {pdf_url}: {e}")

    state["seen_pdf_urls"] = sorted(seen_set)
    if not dry_run:
        _save_state(state)

    return downloaded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape NZ law journals for new PDFs")
    parser.add_argument("--limit",   type=int, default=10, help="Max PDFs to download")
    parser.add_argument("--dry-run", action="store_true",  help="Show without downloading")
    args = parser.parse_args()

    results = fetch_new_pdfs(limit=args.limit, dry_run=args.dry_run)
    print(f"\nDownloaded: {len(results)}")


if __name__ == "__main__":
    main()
