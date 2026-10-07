from src.parse_articles import check_counts, parse_html, roman_to_int

NEW_LAYOUT = """
<html><body>
<p class="oj-doc-ti">VERORDNUNG (EU) 2024/1689</p>
<div class="eli-subdivision" id="rct_1"><table><tr><td><p class="oj-normal">(1)</p></td>
  <td><p class="oj-normal">Erwägungsgrund – darf nicht in einem Artikel landen.</p></td></tr></table></div>
<div class="eli-subdivision" id="cpt_I">
  <p class="oj-ti-section-1">KAPITEL I</p>
  <div class="eli-title"><p class="oj-ti-section-2">ALLGEMEINE BESTIMMUNGEN</p></div>
  <div class="eli-subdivision" id="art_1">
    <p class="oj-ti-art">Artikel 1</p>
    <div class="eli-title"><p class="oj-sti-art">Gegenstand</p></div>
    <div id="001.001"><p class="oj-normal">(1)   Zweck dieser Verordnung ist es
      <a id="ntc1-L_202401689DE.000101-E0001" href="#ntr1"> (<span class="oj-super oj-note-tag">1</span>)</a>,
      den Binnenmarkt zu verbessern.</p></div>
    <div id="001.002"><p class="oj-normal">(2)   Diese Verordnung enthält</p>
      <table><col width="4%"/><col width="96%"/><tbody>
        <tr><td><p class="oj-normal">a)</p></td><td><p class="oj-normal">harmonisierte Vorschriften;</p>
          <table><tr><td><p class="oj-normal">i)</p></td><td><p class="oj-normal">Unterpunkt</p></td></tr></table>
        </td></tr>
        <tr><td><p class="oj-normal">b)</p></td><td><p class="oj-normal">Verbote;</p></td></tr>
      </tbody></table></div>
  </div>
  <div class="eli-subdivision" id="art_2">
    <p class="oj-ti-art">Artikel 2</p>
    <div class="eli-title"><p class="oj-sti-art">Rechenleistung</p></div>
    <p class="oj-normal">mehr als 10<span class="oj-super">25</span> Gleitkommaoperationen.</p>
    <p class="oj-ti-art">Artikel 7</p>
    <p class="oj-normal">(zitierter Artikel aus einer Änderung – kein neuer Artikel)</p>
  </div>
</div>
<div class="oj-final"><p class="oj-normal">Geschehen zu Brüssel am 13. Juni 2024.</p></div>
<div class="oj-signatory"><p class="oj-signatory">Im Namen des Europäischen Parlaments</p></div>
<p class="oj-note">(1) ABl. C 517 vom 22.12.2021, S. 56.</p>
<div class="eli-container" id="anx_I">
  <p class="oj-doc-ti">ANHANG I</p>
  <p class="oj-doc-ti">Liste der Harmonisierungsrechtsvorschriften der Union</p>
  <p class="oj-normal">1.   Richtlinie 2006/42/EG (Maschinen);</p>
</div>
</body></html>
"""

OLD_LAYOUT = """
<html><body>
<p class="ti-section-1">KAPITEL IV</p><p class="ti-section-2"><span class="bold">Verantwortlicher</span></p>
<p class="ti-art">Artikel 1</p>
<p class="sti-art">Meldung von Verletzungen des Schutzes personenbezogener Daten an die Aufsichtsbehörde</p>
<p class="normal">(1) Im Falle einer Verletzung … möglichst binnen 72 Stunden …</p>
<p class="ti-section-1">KAPITEL V</p><p class="ti-section-2"><span class="bold">Übermittlungen</span></p>
<p class="ti-art">Artikel 2</p>
<p class="sti-art">Benachrichtigung der betroffenen Person</p>
<p class="normal">(1) Hat die Verletzung … voraussichtlich ein hohes Risiko …</p>
<p class="final">Diese Verordnung ist in allen ihren Teilen verbindlich.</p>
<p class="note">(1) ABl. C 229 vom 31.7.2012, S. 90.</p>
<p class="doc-end">Ende</p>
</body></html>
"""


def by_number(units):
    return {(u.kind, u.number): u for u in units}


def test_new_layout_articles_and_titles():
    units = by_number(parse_html(NEW_LAYOUT, "ai_act", "de"))
    assert list(units) == [("article", "1"), ("article", "2"), ("annex", "I")]
    art1 = units[("article", "1")]
    assert art1.title == "Gegenstand"
    assert art1.chapter == "KAPITEL I – ALLGEMEINE BESTIMMUNGEN"
    assert art1.to_record()["id"] == "ai_act|de|art1"
    assert art1.to_record()["label"] == "AI Act Art. 1"
    assert art1.to_record()["url"].endswith("CELEX:32024R1689#art_1")


def test_new_layout_text_is_clean():
    units = by_number(parse_html(NEW_LAYOUT, "ai_act", "de"))
    text = units[("article", "1")].text
    assert "Artikel 1" not in text and "Gegenstand" not in text  # heading and title not repeated
    assert "Erwägungsgrund" not in text  # recitals are not part of an article
    assert "(1)" in text and "Zweck dieser Verordnung ist es , den" not in text  # footnote marker removed
    assert text.count("harmonisierte Vorschriften") == 1  # table cells not duplicated
    assert "a) harmonisierte Vorschriften; i) Unterpunkt" in text  # nested list stays in its row
    assert "b) Verbote;" in text.split("\n")  # one table row = one line


def test_superscript_quoted_article_and_signature():
    units = by_number(parse_html(NEW_LAYOUT, "ai_act", "de"))
    art2 = units[("article", "2")].text
    assert "10^25 Gleitkommaoperationen" in art2
    assert "zitierter Artikel" in art2  # "Artikel 7" out of sequence → body text, not a new article
    assert "Brüssel" not in art2 and "ABl. C 517" not in art2


def test_annex_after_signature():
    annex = by_number(parse_html(NEW_LAYOUT, "ai_act", "de"))[("annex", "I")]
    assert annex.title == "Liste der Harmonisierungsrechtsvorschriften der Union"
    assert annex.text == "1. Richtlinie 2006/42/EG (Maschinen);"
    assert annex.to_record()["id"] == "ai_act|de|annex_i"
    assert annex.to_record()["label"] == "AI Act Anhang I"


def test_old_layout():
    units = parse_html(OLD_LAYOUT, "gdpr", "de")
    assert [u.number for u in units] == ["1", "2"]
    assert units[0].title.startswith("Meldung von Verletzungen")
    assert units[0].chapter == "KAPITEL IV – Verantwortlicher"
    assert "72 Stunden" in units[0].text
    assert "KAPITEL V" not in units[0].text and "hohes Risiko" not in units[0].text
    assert "verbindlich" not in units[1].text and "ABl." not in units[1].text
    assert units[1].to_record()["label"] == "DSGVO Art. 2"


def test_check_counts_reports_missing_articles():
    units = parse_html(OLD_LAYOUT, "gdpr", "de")
    assert check_counts(units, "gdpr") == ["article: 2 found, 99 expected"]


def test_roman_numbers():
    assert [roman_to_int(s) for s in ["I", "IV", "IX", "XIII"]] == [1, 4, 9, 13]
