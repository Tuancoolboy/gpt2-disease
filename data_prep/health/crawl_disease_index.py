#!/usr/bin/env python3
"""Step 1 for the health domain: fetch the A-Z disease index and save detail links."""

import argparse
from loguru import logger
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests

from src.config import HEALTH_AZ_URL, HEALTH_INDEX_HTML, HEALTH_METADATA_CSV
from src.utils import normalize_text

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RETRY_MAX = 3
RETRY_BASE_DELAY = 3.0

SESSION = requests.Session()
SESSION.headers.update({
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Referer": "https://tamanhhospital.vn/",
})


def fetch_html(url: str) -> str:
    """Fetch HTML with a browser-like client and light retry logic."""
    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = SESSION.get(url, timeout=20, impersonate="chrome120")
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            if attempt == RETRY_MAX:
                raise RuntimeError(f"Could not fetch {url}") from exc
            wait = RETRY_BASE_DELAY * attempt
            logger.warning(
                "[Retry {}/{}] {} -> {}. Wait {:.0f}s...",
                attempt,
                RETRY_MAX,
                url,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Could not fetch {url}")


def is_disease_detail_url(url: str, *, expected_host: str) -> bool:
    """Keep only canonical disease detail links under /benh/<slug>/."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc and parsed.netloc != expected_host:
        return False

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        return False
    if parts[0] != "benh":
        return False

    return True


def extract_disease_links(html: str, index_url: str) -> list[dict]:
    """Parse the A-Z page and return unique disease detail links."""
    soup = BeautifulSoup(html, "html.parser")
    host = urlparse(index_url).netloc
    records_by_url: dict[str, dict] = {}

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href:
            continue

        full_url = urljoin(index_url, href)
        if not is_disease_detail_url(full_url, expected_host=host):
            continue

        title = normalize_text(anchor.get_text(" ", strip=True))
        if not title:
            continue

        path = urlparse(full_url).path.rstrip("/")
        slug = path.split("/")[-1]
        existing = records_by_url.get(full_url)
        record = {
            "title": title,
            "url": full_url,
            "slug": slug,
            "path": path,
            "source_index": index_url,
        }

        # The same URL can appear multiple times; keep the richest visible title.
        if existing is None or len(title) > len(existing["title"]):
            records_by_url[full_url] = record

    return sorted(records_by_url.values(), key=lambda item: item["title"].casefold())


def save_html(html: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Saved raw HTML to {}", output_path)


def save_metadata(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved {} disease links to {}", len(df), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the Tam Anh disease A-Z page and save disease detail links.",
    )
    parser.add_argument(
        "--url",
        default=HEALTH_AZ_URL,
        help="Disease index URL to crawl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(HEALTH_METADATA_CSV),
        help="CSV output path for extracted disease links",
    )
    parser.add_argument(
        "--save-html",
        action="store_true",
        help="Also save the raw HTML for debugging selectors",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path(HEALTH_INDEX_HTML),
        help="Raw HTML output path used with --save-html",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Fetching disease index: {}", args.url)
    html = fetch_html(args.url)

    if args.save_html:
        save_html(html, REPO_ROOT / args.html_output)

    records = extract_disease_links(html, args.url)
    if not records:
        raise RuntimeError("No disease detail links were extracted from the index page")

    save_metadata(records, REPO_ROOT / args.output)


if __name__ == "__main__":
    main()
