"""
Scrape NZ Acts from legislation.govt.nz.

No Cloudflare protection on this site - standard curl with -L is sufficient.
Each provision (div.prov) in the HTML becomes a LegSection document.

Section URL anchors use the DLM element IDs. In a browser, the redirect
from the legacy URL preserves the fragment, landing the user on the right section.
"""

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

import config

_CURL_CMD = [
    "curl", "-s", "-L", "--compressed",
    "-A", "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "--max-time", "30",
]

ACTS: dict[str, dict] = {
    "RTA": {
        "title": "Residential Tenancies Act 1986",
        "year": 1986,
        "url": "https://www.legislation.govt.nz/act/public/1986/0120/latest/whole.html",
    },
    "ERA2000": {
        "title": "Employment Relations Act 2000",
        "year": 2000,
        "url": "https://www.legislation.govt.nz/act/public/2000/0024/latest/whole.html",
    },
    "PA2020": {
        "title": "Privacy Act 2020",
        "year": 2020,
        "url": "https://www.legislation.govt.nz/act/public/2020/0031/latest/whole.html",
    },
    "CCLA2017": {
        "title": "Contract and Commercial Law Act 2017",
        "year": 2017,
        "url": "https://www.legislation.govt.nz/act/public/2017/0005/latest/whole.html",
    },
    "CA1993": {
        "title": "Companies Act 1993",
        "year": 1993,
        "url": "https://www.legislation.govt.nz/act/public/1993/0105/latest/whole.html",
    },
    "CRA1961": {
        "title": "Crimes Act 1961",
        "year": 1961,
        "url": "https://www.legislation.govt.nz/act/public/1961/0043/latest/whole.html",
    },
    "HHS2019": {
        "title": "Housing (Healthy Homes Standards) Regulations 2019",
        "year": 2019,
        "url": "https://www.legislation.govt.nz/regulation/public/2019/0234/latest/whole.html",
    },
}


@dataclass
class LegSection:
    act_code: str
    act_title: str
    act_year: int
    section_num: str    # "24" or "24A" or "" for unnumbered sections
    section_title: str  # "Frequency of rent increases"
    dlm_id: str         # "DLM94407" - used for URL anchor
    url: str            # full URL with #anchor
    text: str           # plain text body of the section


async def _curl_get(url: str) -> str | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            *_CURL_CMD, url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=35)
        if proc.returncode == 0 and stdout:
            return stdout.decode("utf-8", errors="replace")
        return None
    except (asyncio.TimeoutError, Exception):
        return None


def _parse_act(html: str, act_code: str, base_url: str) -> list[LegSection]:
    """Parse all provisions from a legislation.govt.nz act page."""
    meta = ACTS[act_code]
    soup = BeautifulSoup(html, "html.parser")

    sections: list[LegSection] = []
    for prov in soup.find_all("div", class_="prov"):
        dlm_id = prov.get("id", "")

        # Skip repealed/discontinued sections
        prov_classes = prov.get("class", [])
        if "js-discontinued-info" in prov_classes:
            continue

        heading = prov.find("h5", class_="prov")
        if not heading:
            continue

        # span.label holds the section number; rest is the section title
        label = heading.find("span", class_="label")
        section_num = label.get_text(strip=True) if label else ""
        if label:
            label.extract()
        section_title = heading.get_text(strip=True)

        body = prov.find("div", class_="prov-body")
        if not body:
            continue

        # Remove history/amendment notes - not legal text
        for tag in body.find_all("div", class_="history"):
            tag.decompose()

        text = body.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 30:
            continue

        url = f"{base_url}#{dlm_id}" if dlm_id else base_url
        sections.append(LegSection(
            act_code=act_code,
            act_title=meta["title"],
            act_year=meta["year"],
            section_num=section_num,
            section_title=section_title,
            dlm_id=dlm_id,
            url=url,
            text=text,
        ))

    return sections


async def scrape_act(act_code: str, cache_dir: Path | None = None) -> list[LegSection]:
    """Fetch and parse an act. Caches raw HTML to avoid re-fetching."""
    if act_code not in ACTS:
        raise ValueError(f"Unknown act code: {act_code}. Available: {list(ACTS)}")

    meta = ACTS[act_code]
    if cache_dir is None:
        cache_dir = config.DATA_DIR / "raw" / "NZLEG"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"{act_code}.html"
    if cache_file.exists():
        html = cache_file.read_text(encoding="utf-8", errors="replace")
        print(f"  [{act_code}] loaded from cache ({len(html):,} bytes)")
    else:
        print(f"  [{act_code}] fetching {meta['url']} ...")
        html = await _curl_get(meta["url"])
        if not html:
            print(f"  [{act_code}] fetch failed")
            return []
        cache_file.write_text(html, encoding="utf-8")
        print(f"  [{act_code}] fetched ({len(html):,} bytes)")

    sections = _parse_act(html, act_code, meta["url"])
    print(f"  [{act_code}] {len(sections)} sections parsed")
    return sections
