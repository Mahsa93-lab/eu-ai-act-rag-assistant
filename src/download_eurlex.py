"""Download the official HTML of both regulations in German and English from EUR-Lex → data/raw/.

  python -m src.download_eurlex

Creates data/raw/ai_act_de.html, ai_act_en.html, gdpr_de.html, gdpr_en.html (about 5 MB together).
EUR-Lex sometimes answers scripts with a bot check instead of the text. The script detects this and prints
the manual way: open the printed link in the browser → Ctrl+S → "Webseite, nur HTML" → save under the shown name.
Reuse of EUR-Lex content is permitted with source acknowledgement (Commission Decision 2011/833/EU).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

from src.laws import LANGS, LAWS, eurlex_url

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0 Safari/537.36 (portfolio project; contact via GitHub)",
    "Accept-Language": "de,en;q=0.8",
}


def looks_like_law(html: str) -> bool:
    return "ti-art" in html and html.count("ti-art") > 50


def main(raw_dir: Path = Path("data/raw")) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for law in LAWS:
        for lang in LANGS:
            target, url = raw_dir / f"{law}_{lang}.html", eurlex_url(law, lang)
            if target.exists() and looks_like_law(target.read_text(encoding="utf-8", errors="ignore")):
                print(f"✓ {target.name} already there")
                continue
            try:
                resp = requests.get(url, headers=HEADERS, timeout=90)
                resp.encoding = "utf-8"
                html = resp.text
            except requests.RequestException as exc:
                html = ""
                print(f"  {exc.__class__.__name__}: {exc}")
            if looks_like_law(html):
                target.write_text(html, encoding="utf-8")
                print(f"✓ {target.name}  {len(html) / 1e6:.1f} MB")
            else:
                missing.append((target, url))
                print(f"✗ {target.name}  (no legal text received)")
            time.sleep(2)  # be polite to EUR-Lex

    if missing:
        print("\nPlease save these pages manually (browser → Ctrl+S → 'Webseite, nur HTML'):")
        for target, url in missing:
            print(f"  {url}\n    → {target}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
