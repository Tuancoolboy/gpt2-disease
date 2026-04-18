#!/usr/bin/env python3
"""Clean scraped health articles into train-ready JSONL/Parquet."""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_INPUT = REPO_ROOT / "data" / "raws" / "health_disease_content.jsonl"
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "data" / "train" / "health_disease_clean.jsonl"
DEFAULT_OUTPUT_PARQUET = REPO_ROOT / "data" / "train" / "health_disease_clean.parquet"

STOP_MARKERS = [
    "BÀI VIẾT CÙNG CHỦ ĐỀ",
    "ĐỐI TÁC BẢO HIỂM",
    "HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH",
    "Fanpage:",
    "Website:",
    "Web chat",
    "Messager",
    "Để đặt lịch khám",
    "Quý khách vui lòng liên hệ",
    "Bệnh viện còn được trang bị",
    "BVĐK Tâm Anh",
    "Đây cũng là một trong những đơn vị tiên phong",
]

DROP_LINE_PATTERNS = [
    r"https?://\S+",
    r"\bdoi\.org/\S+",
    r"^Hotline:",
    r"^-\s*Hotline:",
    r"^-\s*Fanpage:",
    r"^-\s*Website:",
    r"^-\s*Web chat$",
    r"^-\s*Messager$",
    r"^-\s*Bệnh viện Đa khoa Tâm Anh",
    r"^-\s*Phòng khám Đa khoa Tâm Anh",
    r"^-\s*\d+[A-Z]?\s+.*",
    r"^\(Đ/c cũ:.*\)$",
]

REFERENCE_HINTS = [
    "MedlinePlus",
    "WebMD",
    "Cleveland Clinic",
    "Johns Hopkins",
    "KidsHealth",
    "St. Jude",
    "https://",
    "http://",
    "doi.org",
]

SKIP_EXACT = {
    "Mục lục",
    "Tiêu hóa",
}

INLINE_CITATION_PATTERNS = [
    r"\(\s*\d+\s*\)",
    r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]",
]


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def clean_spaces(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_inline_citations(text: str) -> str:
    text = normalize_text(text)
    for pattern in INLINE_CITATION_PATTERNS:
        text = re.sub(pattern, "", text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    text = clean_spaces(text)
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def contains_stop_marker(text: str) -> bool:
    upper = text.upper()
    return any(marker.upper() in upper for marker in STOP_MARKERS)


def is_reference_paragraph(text: str) -> bool:
    if any(hint.lower() in text.lower() for hint in REFERENCE_HINTS):
        return True
    if re.search(r"\(\s*\d{4}[^)]*\)", text) and ("http" in text or "." in text):
        return True
    return False


def should_drop_line(text: str) -> bool:
    if not text:
        return True
    if text in SKIP_EXACT:
        return True
    for pattern in DROP_LINE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def strip_bullet_prefix(text: str) -> str:
    return clean_spaces(re.sub(r"^-\s*", "", text))


def remove_leading_toc_block(paragraphs: list[str]) -> list[str]:
    later_lookup = {clean_spaces(p).casefold() for p in paragraphs}

    i = 0
    while i < min(len(paragraphs), 12):
        if not paragraphs[i].startswith("- "):
            i += 1
            continue

        j = i
        while j < len(paragraphs) and paragraphs[j].startswith("- "):
            j += 1

        run = paragraphs[i:j]
        if len(run) < 3:
            i = j
            continue

        repeated_headings = 0
        for item in run:
            candidate = strip_bullet_prefix(item)
            if candidate and candidate.casefold() in later_lookup:
                repeated_headings += 1

        if repeated_headings >= 2:
            return paragraphs[:i] + paragraphs[j:]

        i = j

    return paragraphs


def clean_body_text(text: str) -> str:
    paragraphs = split_paragraphs(text)

    cleaned = []
    for p in paragraphs:
        p = clean_spaces(p)

        if contains_stop_marker(p):
            break
        if should_drop_line(p):
            continue
        if is_reference_paragraph(p):
            continue

        p = strip_inline_citations(p)
        if not p:
            continue

        cleaned.append(p)

    cleaned = remove_leading_toc_block(cleaned)

    deduped = []
    seen = set()
    for p in cleaned:
        key = p.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    return "\n\n".join(deduped).strip()


def build_training_text(_title: str, body_text: str) -> str:
    body_text = clean_spaces(body_text)
    return body_text


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skip invalid JSON line {line_no}")
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean scraped health articles for continued pretraining.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--min-chars", type=int, default=300)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    cleaned_records = []

    for rec in records:
        title = clean_spaces(rec.get("title", ""))
        raw_body = rec.get("body_text") or rec.get("text") or ""
        body_text = clean_body_text(raw_body)
        train_text = build_training_text(title, body_text)

        if len(train_text) < args.min_chars:
            continue

        cleaned_records.append({
            "title": title,
            "url": rec.get("url", ""),
            "slug": rec.get("slug", ""),
            "published_date": rec.get("published_date", ""),
            "doctor_review": clean_spaces(rec.get("doctor_review", "")),
            "text": train_text,
        })

    save_jsonl(cleaned_records, args.output_jsonl)

    df = pd.DataFrame(cleaned_records)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_parquet, index=False)

    print(f"Input records: {len(records)}")
    print(f"Cleaned records: {len(cleaned_records)}")
    print(f"Saved JSONL: {args.output_jsonl}")
    print(f"Saved Parquet: {args.output_parquet}")


if __name__ == "__main__":
    main()
