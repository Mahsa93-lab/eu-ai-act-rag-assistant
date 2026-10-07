"""Split the EUR-Lex HTML of a regulation into one record per article and per annex.

Why per article? People ask about "Artikel 33 DSGVO", and every answer must cite the article. Fixed-size chunks
cut articles in half and make citations unreliable; the retriever later splits long articles into chunks but
always keeps the article as the unit that is cited.

How it works: the HTML is read as ONE stream of text blocks (paragraphs and table rows, in document order).
A small state machine opens a new unit at every article heading ("Artikel 5" / "Article 5") and every annex
heading ("ANHANG III" / "ANNEX III") and closes it at the next heading, a chapter heading, the signature or a
footnote. This works for both EUR-Lex layouts:
  * newer acts (AI Act):  <p class="oj-ti-art">Artikel 5</p> <p class="oj-sti-art">Verbotene Praktiken …</p>
  * older acts (GDPR):    <p class="ti-art">Artikel 33</p>   <p class="sti-art">Meldung von Verletzungen …</p>
Article numbers must follow each other (1, 2, 3 …) – a quoted "Artikel 58a" inside an amendment is body text.

Usage (after src/download_eurlex.py):
  python -m src.parse_articles          → data/articles.jsonl + a check table (AI Act 113 + 13, GDPR 99)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from src.laws import LANGS, LAWS, doc_id, doc_key, eurlex_url, label

ARTICLE_HEAD = re.compile(r"^(?:Artikel|Article)\s+(\d+)$", re.IGNORECASE)
ANNEX_HEAD = re.compile(r"^(?:ANHANG|ANNEX)\s+([IVXLC]+)$")  # upper case only – body text never looks like this
CHAPTER_HEAD = re.compile(r"^(?:KAPITEL|CHAPTER)\s+[IVXLC]+$", re.IGNORECASE)
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def roman_to_int(s: str) -> int:
    total = 0
    for i, ch in enumerate(s):
        value = ROMAN[ch]
        total += -value if i + 1 < len(s) and ROMAN[s[i + 1]] > value else value
    return total


@dataclass
class Unit:
    law: str
    lang: str
    kind: str  # "article" | "annex"
    number: str  # "5" or "III"
    title: str = ""
    chapter: str = ""
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def to_record(self) -> dict:
        anchor = f"art_{self.number}" if self.kind == "article" else f"anx_{self.number}"
        rec = {
            "id": doc_id(self.law, self.lang, self.kind, self.number),
            "key": doc_key(self.law, self.kind, self.number),
            "label": label(self.law, self.lang, self.kind, self.number),
            "url": eurlex_url(self.law, self.lang, anchor),
            **asdict(self),
            "text": self.text,
        }
        del rec["lines"]
        return rec


def _norm_classes(tag: Tag) -> set[str]:
    """'oj-ti-art' and 'ti-art' mean the same – drop the 'oj-' prefix of the newer layout."""
    return {c.removeprefix("oj-") for c in (tag.get("class") or [])}


def _in_signature(tag: Tag) -> bool:
    for node in [tag, *tag.parents]:
        if isinstance(node, Tag) and any("final" in c or "signatory" in c for c in _norm_classes(node)):
            return True
    return False


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
    return re.sub(r" ([,.;:])(?=\s|$)", r"\1", text)  # "ist es , den" (after a removed footnote) → "ist es, den"


def _prepare(soup: BeautifulSoup) -> None:
    """Remove footnote markers and keep powers readable (10^25 instead of 1025)."""
    for a in soup.find_all("a", id=re.compile(r"^ntc")):
        a.decompose()
    for span in soup.find_all("span", class_=re.compile(r"note-tag")):
        span.decompose()
    for el in soup.find_all(["td", "p"]):
        el.append(" ")  # cells and paragraphs inside a cell must not run together ("i)Unterpunkt")
    for el in soup.find_all(["sup", "span"]):
        if el.name == "sup" or any("super" in c for c in el.get("class") or []):
            el.replace_with("^" + el.get_text())


def _blocks(soup: BeautifulSoup):
    """Paragraphs and table rows in document order. A table row becomes one line: 'a) das Inverkehrbringen …'."""
    for el in soup.find_all(["p", "tr"]):
        if el.name == "p" and el.find_parent("table"):
            continue  # covered by its row
        if el.name == "tr":
            if el.find_parent("tr"):
                continue  # nested list – already part of the outer row
            text = " ".join(_clean(td.get_text()) for td in el.find_all(["td", "th"], recursive=False))
        else:
            text = el.get_text()
        yield el, _clean(text)


def parse_html(html: str, law: str, lang: str) -> list[Unit]:
    soup = BeautifulSoup(html, "lxml")
    _prepare(soup)
    units: list[Unit] = []
    current: Unit | None = None
    expect_title = expect_chapter_title = False
    chapter = ""
    last_article = last_annex = 0

    def close() -> None:
        nonlocal current, expect_title
        if current is not None and current.lines:
            units.append(current)
        current, expect_title = None, False

    for el, text in _blocks(soup):
        if not text:
            continue
        cls = _norm_classes(el)
        if "note" in cls or _in_signature(el):
            close()  # footnotes and "Geschehen zu Brüssel …" end the unit; annexes may still follow
            continue
        if "doc-end" in cls:
            close()
            break

        art = ARTICLE_HEAD.match(text)
        if art and int(art.group(1)) == last_article + 1:
            close()
            last_article = int(art.group(1))
            current, expect_title = Unit(law, lang, "article", art.group(1), chapter=chapter), True
            continue
        anx = ANNEX_HEAD.match(text)
        if anx and roman_to_int(anx.group(1)) == last_annex + 1:
            close()
            last_annex = roman_to_int(anx.group(1))
            current, expect_title = Unit(law, lang, "annex", anx.group(1)), True
            continue
        if "ti-section-1" in cls:  # "KAPITEL III" or "ABSCHNITT 2"
            close()
            if CHAPTER_HEAD.match(text):
                chapter, expect_chapter_title = text, True
            continue
        if "ti-section-2" in cls:
            close()
            if expect_chapter_title:
                chapter, expect_chapter_title = f"{chapter} – {text}", False
            continue
        if current is None:
            continue  # title page, recitals, chapter headings …
        if expect_title:
            expect_title = False
            if cls & {"sti-art", "doc-ti", "ti-annex-2"}:
                current.title = text
                continue
        current.lines.append(text)
    close()
    return units


def check_counts(units: list[Unit], law: str) -> list[str]:
    """Problems compared with the official structure (empty list = all good)."""
    counts = Counter(u.kind for u in units)
    expected = {"article": LAWS[law]["articles"], "annex": LAWS[law]["annexes"]}
    problems = [f"{kind}: {counts[kind]} found, {n} expected" for kind, n in expected.items() if counts[kind] != n]
    problems += [f"{u.kind} {u.number} has no title" for u in units if u.kind == "article" and not u.title]
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description="EUR-Lex HTML → data/articles.jsonl (one record per article/annex)")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/articles.jsonl"))
    args = p.parse_args()

    records, ok = [], True
    print(f"{'file':<16}{'articles':>9}{'annexes':>9}{'chars':>11}  check")
    for law in LAWS:
        for lang in LANGS:
            path = args.raw_dir / f"{law}_{lang}.html"
            if not path.exists():
                print(f"{path.name:<16} missing – run: python -m src.download_eurlex")
                ok = False
                continue
            units = parse_html(path.read_text(encoding="utf-8"), law, lang)
            problems = check_counts(units, law)
            ok &= not problems
            n = Counter(u.kind for u in units)
            chars = sum(len(u.text) for u in units)
            print(f"{path.name:<16}{n['article']:>9}{n['annex']:>9}{chars:>11,}  {'OK' if not problems else problems}")
            records += [u.to_record() for u in units]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n{len(records)} records → {args.out}" + ("" if ok else "   ⚠ check the table above"))


if __name__ == "__main__":
    main()
