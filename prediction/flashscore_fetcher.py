"""
Flashscore Maç Veri Çekici
====================================
Flashscore.com üzerinden maç sonuçlarını çeker.
Kullanım: match_score_fetcher.py içinden get_match_result() çağrıldığında 
          web scraping başarısız olursa fallback olarak devreye girer.

Not: Flashscore resmi bir API yayınlamıyor. Bu modül 
      flashscore.com'un HTML/JS rendering'ini parse eder.
      Sitesi JavaScript ile çalıştığı için seleniumbase veya
      headless browser gerekebilir.
"""
import os
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAHMINLER_DIR = os.path.join(BASE_DIR, "tahminler")
MATCH_RESULTS_FILE = os.path.join(TAHMINLER_DIR, "match_results.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.flashscore.com/",
}

# Flashscore'da maç ID formatı: /mac/ID/
MATCH_URL_RE = re.compile(r'/mac/(\d+)/')

# Skor parse patterns
SCORE_PATTERNS = [
    re.compile(r'(\d+)\s*-\s*(\d+)\s*(?:bitti|bitti\.|ft|ms|sonuç|skor)?', re.IGNORECASE),
    re.compile(r'(?:bitti|ft|ms|sonuç|skor)[:\s]*(\d+)\s*-\s*(\d+)', re.IGNORECASE),
]

def normalize_text(text: str) -> str:
    """Türkçe karakterleri temizler, küçük harfe çevirir."""
    if not text:
        return ""
    tr_map = str.maketrans("çğıöşüÇĞİÖŞÜâîûÂÎÛ", "cgiosuCGIOSUaiuAIU")
    return text.translate(tr_map).lower().strip()

def load_local_match_results() -> dict:
    if os.path.exists(MATCH_RESULTS_FILE):
        try:
            with open(MATCH_RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_local_match_result(cid: str, data: dict):
    os.makedirs(TAHMINLER_DIR, exist_ok=True)
    results = load_local_match_results()
    results[cid] = data
    try:
        with open(MATCH_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Flashscore] JSON yazma hatasi: {e}")

def _fetch_flashscore_html(url: str) -> Optional[str]:
    """Flashscore sayfasını indirir. JS-required içerik için yetersiz olabilir."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None

def _parse_score_from_html(html: str, home_name: str, away_name: str) -> Optional[tuple]:
    """
    HTML'den skor bulmaya çalışır. 
    Flashscore'da maç sayfasında genellikle sonuç divs'te bulunur.
    """
    if not html:
        return None

    norm_h = normalize_text(home_name).split()[0]
    norm_a = normalize_text(away_name).split()[0]

    # 1. "sonuc 2-1" veya "MS 2-1" tarzı ifadeler
    for pat in SCORE_PATTERNS:
        m = pat.search(html)
        if m:
            hg, ag = int(m.group(1)), int(m.group(2))
            if 0 <= hg <= 15 and 0 <= ag <= 15:
                return (hg, ag)

    # 2. Takım isimleri arasında skor arar (fenerbahce 2-1 galatasaray)
    if len(norm_h) >= 3 and len(norm_a) >= 3:
        pat = rf'{re.escape(norm_h)}[^\d]*?(\d{{1,2}})\s*-\s*(\d{{1,2}})[^\d]*?{re.escape(norm_a)}'
        m = re.search(pat, html, re.IGNORECASE)
        if m:
            hg, ag = int(m.group(1)), int(m.group(2))
            if 0 <= hg <= 15 and 0 <= ag <= 15:
                return (hg, ag)

    return None

def find_match_on_flashscore(home_name: str, away_name: str, match_date: str = "") -> Optional[dict]:
    """Flashscore lig sayfalarindan (Playwright) kupon macini bulur ve skoru dondurur.

    Flashscore'da eski "ara/?q=" endpoint'i yok (404); gercek veri lig sayfalarinda:
      https://www.flashscore.com.tr/futbol/{ulke}/{lig}/
    Her mac satirinda ($.event__match) mac linki home/away takim slug'ini
    (gaziantep-fk-ITkIKZ7q / goztepe-0MJlDkrA gibi) ve .event__score skoru tasir.

    Ilk once lokasyon (TR) baglantisiyla dogrudan dener; sayfa acilamazsa
    veya takim eslesmesi bulunamazsa, free_proxies havuzundan proxy ile birkac
    lig sayfasini tekrar tarar (IP tabanli blok/429 atlatmak icin).

    Dondurur: make_result stili dict (source "flashscore") ya da None.
    """
    home_name = (home_name or "").strip()
    away_name = (away_name or "").strip()
    if not home_name or not away_name:
        return None

    try:
        from prediction.aiscore_scraper import _name_match, _norm
    except ImportError:
        return None

    rows = _scan_flashscore_rows(proxies=None)
    if not rows:
        proxies = _get_working_proxies_quiet()
        for px in proxies[:6]:
            rows = _scan_flashscore_rows(proxies=[px])
            if rows:
                break

    if not rows:
        return None

    for (home_slug, away_slug, finished, scores) in rows:
        hs_s = _name_match(home_name, home_slug or "")
        as_s = _name_match(away_name, away_slug or "")
        hr_s = _name_match(home_name, away_slug or "")
        ar_s = _name_match(away_name, home_slug or "")
        if not (hs_s >= 45 and as_s >= 45) and not (hr_s >= 45 and ar_s >= 45):
            continue
        # bitmis: DOM class'i 'finished' veya iki gercekci skor digiti (0-20)
        if not finished:
            digs = [int(x) for x in scores if x and x.isdigit()]
            finished = len(digs) >= 2 and all(0 <= d <= 20 for d in digs)
        if not finished:
            continue
        digits = [int(x) for x in scores if x and x.isdigit()]
        if len(digits) < 2:
            continue
        hg, ag = digits[0], digits[1]
        total = hg + ag
        return {
            "home_goals": hg,
            "away_goals": ag,
            "total_goals": total,
            "result": "H" if hg > ag else ("D" if hg == ag else "A"),
            "btts": 1 if hg > 0 and ag > 0 else 0,
            "over25": 1 if total >= 3 else 0,
            "home_corners": None,
            "away_corners": None,
            "ht_home_goals": None,
            "ht_away_goals": None,
            "date": match_date,
            "source": "flashscore",
        }
    return None


# Kupon maçlarinin bulunabilecegi flashscore lig sayfalari (kolay ulasim).
_LIGA_PAGE = [
    ("turkiye", "super-lig"),
    ("turkiye", "1-lig"),
    ("turkiye", "2-lig"),
    ("turkiye", "3-lig"),
    ("almanya", "bundesliga"),
    ("almanya", "2-bundesliga"),
    ("almanya", "dfb-pokal"),
]

_PW_LOCK = None


def _lock():
    global _PW_LOCK
    if _PW_LOCK is None:
        import threading
        _PW_LOCK = threading.RLock()
    return _PW_LOCK


def _get_working_proxies_quiet():
    try:
        from prediction.free_proxies import get_working_proxies
        return get_working_proxies(limit=10)
    except Exception:
        return []


def _slug_of(segment: str) -> str:
    """'gaziantep-fk-ITkIKZ7q' -> 'gaziantep-fk' ('foo-Bar1234' ayiracigaz)."""
    m = re.match(r"^(.*)-([A-Za-z0-9]{5,9})$", segment or "")
    return m.group(1) if m else (segment or "").strip()


def _scan_worker(browser, _page, proxies: Optional[list]) -> list:
    """Worker thread icinde, ortak browser ile calisan tarama (tek greenlet)."""
    rows_out: list = []
    for country, lig in _LIGA_PAGE:
        url = f"https://www.flashscore.com.tr/futbol/{country}/{lig}/"
        ctx = None
        page = None
        try:
            ctx_kwargs = {
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                "locale": "tr-TR",
                "viewport": {"width": 1280, "height": 900},
            }
            if proxies:
                ctx_kwargs["proxy"] = {"server": "http://" + proxies[0]}
            ctx = browser.new_context(**ctx_kwargs)
            page = ctx.new_page()
            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            try:
                page.wait_for_selector(".event__match", timeout=8000)
            except Exception:
                pass
            page.wait_for_timeout(2000)
            data = page.evaluate("""() => {
                const out = [];
                document.querySelectorAll('.event__match').forEach(el => {
                    const a = el.querySelector('a.eventRowLink');
                    const href = (a && a.getAttribute('href')) || '';
                    // /mac/futbol/{home}-{hid}/{away}-{aid}/  -> iki slug segmenti
                    const m = href.match(/\\/mac\\/[^/]+\\/([^/]+)\\/([^/]+)\\//);
                    const sc = Array.from(el.querySelectorAll('.event__score'))
                                   .map(s => s.textContent.trim().replace(/[^\\d-]/g, ''))
                                   .filter(x => x !== '');
                    out.push({
                        h: m ? m[1] : '',
                        a: m ? m[2] : '',
                        f: (el.className || '').indexOf('finished') >= 0,
                        s: sc,
                    });
                });
                return out;
            }""")
            for row in data:
                rows_out.append((_slug_of(row.get("h") or ""), _slug_of(row.get("a") or ""), bool(row.get("f")), row.get("s") or []))
        except Exception:
            continue
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass
            if ctx is not None:
                try:
                    ctx.close()
                except Exception:
                    pass
    return rows_out


def _scan_flashscore_rows(proxies: Optional[list] = None) -> list:
    """Flashscore lig sayfalarini Playwright ile tek worker thread'inde tarar.

    Return: [(home_slug, away_slug, finished: bool, scores: [..]), ...]
    Proxies verilirse o proxy'lerden birini kullanir.

    Thread guvenligi: tum Playwright isi pw_worker'daki tek isci thread'inde
    calisir; bu, 'greenlet.error: Cannot switch to a different thread' hatalarini
    onler (Flask request + arka plan 'Canli kontrol' thread'leri cakismaz).
    """
    try:
        from prediction.pw_worker import run_with_browser as _run_with_browser
    except Exception:
        _run_with_browser = None
    if _run_with_browser is not None:
        try:
            return _run_with_browser(_scan_worker, proxies, timeout=240) or []
        except Exception:
            return []
    return _scan_worker(None, None, proxies) or []

def get_match_result_flashscore(home_id, away_id, home_name, away_name, match_date="") -> Optional[dict]:
    """
    Flashscore'dan maç sonucu çeker.
    
    Önce önbellek kontrolü yapılır. Eğer yoksa flashscore aranır.
    """
    cid = f"{home_id}-{away_id}"
    cache = load_local_match_results()
    
    if cid in cache:
        res = cache[cid]
        if isinstance(res, dict) and "home_goals" in res and res.get("home_goals") is not None:
            return res
    
    if not home_name or not away_name:
        return None
    
    # Gelecek maç ise
    if match_date:
        try:
            md = datetime.fromisoformat(match_date.replace("Z", ""))
            if md.date() > datetime.now().date():
                return None
        except Exception:
            pass
    
    # Flashscore'dan ara
    res = find_match_on_flashscore(home_name, away_name, match_date)
    if res:
        save_local_match_result(cid, res)
        return res
    
    return None

# Test
if __name__ == "__main__":
    print("FlashscoreFetcher test...")
    
    # Test maçı - gerçek bir sonucu olan
    test_cases = [
        # Dünyanın en çok takip edilen maçlarından biri
        ("746", "1000787", "Çaykur Rizespor", "Alanyaspor", "2026-09-03"),
    ]
    
    for hid, aid, hname, aname, mdate in test_cases:
        print(f"\nMaç: {hname} vs {aname} ({mdate})")
        result = get_match_result_flashscore(hid, aid, hname, aname, mdate)
        if result:
            print(f"  Buldum: {result['home_goals']}-{result['away_goals']} ({result['source']})")
        else:
            print(f"  Bulunamadi (flashscore'da aranacak)")
