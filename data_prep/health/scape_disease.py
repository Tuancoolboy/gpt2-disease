#!/usr/bin/env python3
"""
Step 2: read disease URLs from the step-1 CSV, fetch each disease page,
extract the main text, and save JSONL for later continued pretraining.
"""

import argparse
import csv
import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup, Tag

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "raws" / "health_disease_links.csv"
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "data" / "raws" / "health_disease_content.jsonl"
DEFAULT_HTML_DIR = REPO_ROOT / "data" / "raws" / "health_html"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")

# Stop when article body reaches non-content sections.
STOP_MARKERS = {
    "BÀI VIẾT LIÊN QUAN",
    "CÓ THỂ BẠN QUAN TÂM",
    "ĐẶT LỊCH HẸN",
    "XEM THÊM",
}

# Short UI / boilerplate texts that are not useful for training.
SKIP_EXACT = {
    "Mục lục",
    "ĐẶT LỊCH HẸN",
    "XEM HỒ SƠ",
    "Bệnh viện Đa khoa Tâm Anh",
    "Bệnh viện Đa khoa Tâm Anh TP.HCM",
    "Bệnh viện Đa khoa Tâm Anh Hà Nội",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def clean_text(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_html(url: str, retries: int = 3, timeout: int = 20) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
            "Referer": "https://tamanhhospital.vn/",
        },
    )

    for attempt in range(1, retries + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read().decode(charset, errors="ignore")
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Could not fetch {url}") from exc
            wait = attempt * 2
            logger.warning(
                "Retry %s/%s for %s after error: %s. Waiting %ss",
                attempt,
                retries,
                url,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"Could not fetch {url}")


def load_links(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("url") or "").strip()
            if not url:
                continue
            rows.append({
                "title": clean_text(row.get("title", "")),
                "url": url,
                "slug": (row.get("slug") or "").strip(),
                "path": (row.get("path") or "").strip(),
            })
    return rows


def load_done_urls(output_path: Path) -> set[str]:
    done = set()
    if not output_path.exists():
        return done

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = record.get("url")
            if url:
                done.add(url)
    return done


def save_raw_html(html: str, html_dir: Path, slug: str) -> None:
    html_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{slug or 'page'}.html"
    (html_dir / filename).write_text(html, encoding="utf-8")


def extract_published_date(full_text: str) -> str:
    match = DATE_RE.search(full_text)
    return match.group(0) if match else ""


def extract_doctor_review(full_text: str) -> str:
    match = re.search(r"Tư vấn chuyên môn bài viết\s*([^\n]+)", full_text)
    if not match:
        return ""
    return clean_text(match.group(1))


def should_stop(text: str) -> bool:
    upper = text.upper()
    return any(marker in upper for marker in STOP_MARKERS)


def extract_body_lines(soup: BeautifulSoup, title_tag: Tag | None) -> list[str]:
    if title_tag is None:
        return []

    lines: list[str] = []
    in_toc = False

    for tag in title_tag.find_all_next(["p", "h2", "h3", "li"]):
        if not isinstance(tag, Tag):
            continue

        text = clean_text(tag.get_text(" ", strip=True))
        if not text:
            continue

        if text in SKIP_EXACT:
            continue

        if should_stop(text):
            break

        if text.lower() == "mục lục":
            in_toc = True
            continue

        if in_toc:
            if tag.name == "li":
                continue
            if tag.name in {"h2", "h3"}:
                in_toc = False

        if len(text) < 3:
            continue

        if tag.name == "li":
            text = "- " + text

        lines.append(text)

    deduped: list[str] = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    return deduped


def extract_record(html: str, row: dict) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    full_text = normalize_text(soup.get_text("\n", strip=True))

    title_tag = soup.find("h1")
    title = clean_text(title_tag.get_text(" ", strip=True)) if title_tag else row["title"]
    published_date = extract_published_date(full_text)
    doctor_review = extract_doctor_review(full_text)

    body_lines = extract_body_lines(soup, title_tag)
    body_text = "\n\n".join(body_lines).strip()

    # This is the field you will usually use later for continued pretraining.
    training_text_parts = [title]
    if body_text:
        training_text_parts.append(body_text)

    return {
        "title": title,
        "url": row["url"],
        "slug": row["slug"] or urlparse(row["url"]).path.rstrip("/").split("/")[-1],
        "path": row["path"],
        "published_date": published_date,
        "doctor_review": doctor_review,
        "body_text": body_text,
        "text": "\n\n".join(training_text_parts).strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch each disease page from the step-1 CSV and save article text as JSONL.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input CSV from step 1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--save-html",
        action="store_true",
        help="Also save raw HTML for each disease page",
    )
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=DEFAULT_HTML_DIR,
        help="Directory for raw HTML files when --save-html is used",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N records",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between requests",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip URLs already present in the output JSONL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    rows = load_links(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]

    done_urls = set() if args.no_resume else load_done_urls(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loaded %s disease links from %s", len(rows), args.input)
    if done_urls:
        logger.info("Resume mode: %s URLs already done", len(done_urls))

    written = 0
    skipped = 0

    with args.output.open("a", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows, start=1):
            url = row["url"]

            if url in done_urls:
                skipped += 1
                continue

            logger.info("[%s/%s] Fetching %s", idx, len(rows), url)

            try:
                html = fetch_html(url)
                if args.save_html:
                    save_raw_html(html, args.html_dir, row["slug"])

                record = extract_record(html, row)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                written += 1
                done_urls.add(url)

            except Exception as exc:
                logger.error("Failed to process %s: %s", url, exc)

            time.sleep(args.sleep)

    logger.info("Done. Written=%s | Skipped=%s | Output=%s", written, skipped, args.output)


if __name__ == "__main__":
    main()