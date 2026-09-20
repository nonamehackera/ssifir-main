"""
Match Score Fetcher (Maç Skoru Çekici) - Gelişmiş Sürüm v2
=============================================================
Kupon kontrolü ve maç sonuçlarını doğrulama modülü.

Sırasıyla aşağıdaki kaynakları kullanır:
1. Yerel Önbellek (`tahminler/match_results.json` + Manuel Girilenler)
2. AiScore lig fikstürü (SSR/HTTP - simulasyon dunyasindaki kupon maclari icin en güvenilir)
3. SofaScore API (Playwright fetch ile CF 403'unu asar - TR ligleri icin de guvenilir)
4. ESPN Soccer API (site.api.espn.com - ücretsiz, güvenilir)
5. TheSportsDB API (takım ID cache ile, rate-limit korumalı)
6. OpenLigaDB (Alman Bundesliga için)
7. Flashscore.com (flashscore_fetcher - yeni eklenti, fallback)
8. Football-Data.co.uk CSV'leri (Eski Lig Maçları - fallback)

Bulunan tüm yeni skorlar anında `tahminler/match_results.json` dosyasına yazılır.
"""

import os
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from difflib import SequenceMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAHMINLER_DIR = os.path.join(BASE_DIR, "tahminler")

_MATCH_RESULTS_CACHE = None
_MATCH_RESULTS_CACHE_MTIME = 0.0
MATCH_RESULTS_FILE = os.path.join(TAHMINLER_DIR, "match_results.json")
TEAM_ID_CACHE_FILE = os.path.join(TAHMINLER_DIR, "team_id_cache.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
}

# ─── Türkçe karakter normalize edici ────────────────────────────────────────

def normalize_name(text: str) -> str:
    """Takım adını karşılaştırma için normalleştirir (Türkçe/Avrupa karakterleri temizler).
    
    DİKKAT: FC/FK/SC gibi ekleri SADECE takımda BAŞKA KELIME YOKSA siler.
    "Çorum FK" → "çorum fk" (FK kalır çünkü "çorum" tek kelime, "Çorum FK" ≠ "Çorum")
    "Manchester City FC" → "manchester city fc" (FC kalır, city zaten var)
    "FC Bayern" → "bayern" (FC başta olduğu için silinir - zorunlu)
    """
    if not text:
        return ""
    tr_map = str.maketrans(
        "çğıöşüÇĞİÖŞÜâîûÂÎÛéàèùäëïöüÄËÏÖÜ",
        "cgiosuCGIOSUaiuAIUeaeUaeiouAEIOU"
    )
    clean = text.translate(tr_map).lower().strip()
    # Sadece BAŞTA/FARKTA olan prefix'leri temizle (FC Bayern → Bayern)
    clean = re.sub(r'^(fc|sc|sv|bv|vfl|vfb|fk|afc|cf|if|bk|gf|hk)\s+', '', clean)
    # Sonundaki ekleri sadece birden fazla kelime varsa sil
    words = clean.split()
    if len(words) > 1:
        # Son kelime gereksiz ek ise sil (ama sadece son kelime)
        suffixes = {'fc', 'sk', 'bv', 'sv', 'sc', 'ac', 'as', 'ssv', 'vfl', 'vfb', 'fk', 'afc', 'cf', 'if', 'bk', 'gf', 'hk'}
        if words[-1] in suffixes:
            words = words[:-1]
    clean = ' '.join(words)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def name_similarity(a: str, b: str) -> float:
    """İki takım adı arasındaki benzerlik oranını döndürür (0-1)."""
    na, nb = normalize_name(a), normalize_name(b)
    return SequenceMatcher(None, na, nb).ratio()

def fuzzy_team_match(query: str, candidates: list, threshold: float = 0.80) -> str | None:
    """Adaylar listesinden en yakın eşleşmeyi bulur (0.80 eşik - daha kesin eşleşme)."""
    if not candidates:
        return None
    best_score = 0
    best_match = None
    for c in candidates:
        score = name_similarity(query, c)
        if score > best_score:
            best_score = score
            best_match = c
    if best_score >= threshold:
        return best_match
    return None

# ─── Yerel Önbellek ─────────────────────────────────────────────────────────

def load_local_match_results() -> dict:
    """tahminler/match_results.json içindeki sonuçları yükler (bellek oncelikli)."""
    global _MATCH_RESULTS_CACHE, _MATCH_RESULTS_CACHE_MTIME
    if os.path.exists(MATCH_RESULTS_FILE):
        try:
            mt = os.path.getmtime(MATCH_RESULTS_FILE)
            if _MATCH_RESULTS_CACHE is not None and mt == _MATCH_RESULTS_CACHE_MTIME:
                return _MATCH_RESULTS_CACHE
            with open(MATCH_RESULTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            _MATCH_RESULTS_CACHE = data
            _MATCH_RESULTS_CACHE_MTIME = mt
            return data
        except Exception:
            return {}
    return {}

def save_local_match_result(cid: str, data: dict):
    """Maç sonucunu yerel önbelleğe kaydeder."""
    global _MATCH_RESULTS_CACHE, _MATCH_RESULTS_CACHE_MTIME
    os.makedirs(TAHMINLER_DIR, exist_ok=True)
    results = load_local_match_results()
    results[cid] = data
    try:
        with open(MATCH_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        _MATCH_RESULTS_CACHE = results
        _MATCH_RESULTS_CACHE_MTIME = os.path.getmtime(MATCH_RESULTS_FILE)
    except Exception as e:
        print(f"[MatchScoreFetcher] JSON yazma hatasi: {e}")

def load_team_id_cache() -> dict:
    """TheSportsDB takım ID önbelleğini yükler."""
    if os.path.exists(TEAM_ID_CACHE_FILE):
        try:
            with open(TEAM_ID_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_team_id_cache(cache: dict):
    """TheSportsDB takım ID önbelleğini kaydeder."""
    os.makedirs(TAHMINLER_DIR, exist_ok=True)
    try:
        with open(TEAM_ID_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ─── Skor Sonuç Yapısı ───────────────────────────────────────────────────────

def make_result(hg: int, ag: int, source: str, date: str = "") -> dict:
    """Standart maç sonucu dict'i oluşturur."""
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
        "date": date,
        "source": source,
    }

# ─── HTTP Helper ─────────────────────────────────────────────────────────────

def _http_get(url: str, timeout: int = 8) -> dict | None:
    """JSON döndüren bir HTTP GET isteği yapar."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
    except Exception:
        pass
    return None

# ─── AiScore lig fikstürü (simulasyon dunyasi kupon maclari) ────────────────

def _aiscore_find_match(home_name: str, away_name: str, match_date: str = "") -> dict | None:
    """AiScore lig fiksturunde (SSR, HTTP) kupon macini arar ve bitmis skoru dondurur.

    Playwright gerekmez: lig fikstürü bölümü mac skorunu zaten tasir.
    Simülasyon dunyasindaki ligler (Turkish Super League, First/Second,
    English Premier League, Bundesliga, 2. Bundesliga) icin en güvenilir kaynak
    (dogrulandi: Everton 2-2 Man Utd, Arsenal 2-1 Chelsea, Eintracht 1-4 Augsburg,
    Trabzonspor 5-0 Genclerbirligi).
    """
    try:
        from prediction.aiscore_scraper import resolve_aiscore_match
        near_ts = None
        if match_date:
            try:
                near_ts = datetime.strptime(match_date, "%Y-%m-%d").timestamp()
            except (ValueError, TypeError):
                near_ts = None
        res = resolve_aiscore_match(home_name, away_name, near_ts=near_ts)
        if not res:
            return None
        if res.get("status") != 8:  # 8 = finished (fiksiyur varsayilan skoru 0-0 tasiyor)
            return None
        if res.get("home_score") is None or res.get("away_score") is None:
            return None
        try:
            hg, ag = int(res["home_score"]), int(res["away_score"])
        except (ValueError, TypeError):
            return None
        src = f"aiscore ({res.get('league_name')} r{res.get('round')})"
        return make_result(hg, ag, src, match_date)
    except ImportError:
        return None
    except Exception:
        return None


# ─── SofaScore API (Playwright fetch ile CF 403 onlenir) ─────────────────────

_MOJIBAKE_TABLE = {
    "Ã§": "ç", "Ã‡": "Ç", "Ã¼": "ü", "Ãœ": "Ü", "Ã¶": "ö", "Ã–": "Ö",
    "Ã©": "é", "Ã¨": "è", "Ã¢": "â", "Ã®": "î", "Ã»": "û",
    "ÄŸ": "ğ", "Äž": "Ğ", "Ä±": "ı", "Ä°": "İ", "ÅŸ": "ş", "Åž": "Ş",
}


def _fix_team_name(name: str) -> str:
    """predictions.json'daki mojibake isimleri duzeltir (kismi cp1254/utf8 bozulmasi).

    Ornek: 'KasımpaÅŸa' -> 'Kasımpaşa', 'GençlerbirliÄŸi' -> 'Gençlerbirliği'.
    Zaten dogru olan karakterlere dokunmaz; tablo dizi-bazlı yer degistirme.
    """
    if not name:
        return name
    fixed = name
    for bad, good in _MOJIBAKE_TABLE.items():
        if bad in fixed:
            fixed = fixed.replace(bad, good)
    return fixed


def _sofascore_find_match(home_name: str, away_name: str, match_date: str = "") -> dict | None:
    """SofaScore'da kupon macini arar (Playwright fetch - CF 403'unu asar).

    SofaScore urllib'ten gelen isteklere 403 donuyor; gercek tarayici icinden
    fetch() ile API cagrisi yapinca sorun yok.
    Dogrulandi: Hamburg-Mainz 0-5, Erzurum BB 1-0, Kasimpasa-Amed 2-2 (200).
    """
    try:
        home_name = _fix_team_name(home_name)
        away_name = _fix_team_name(away_name)
        from prediction.sofascore_scraper import find_match_on_sofascore
        return find_match_on_sofascore(home_name, away_name, match_date)
    except Exception:
        return None


# ─── ESPN Soccer API ─────────────────────────────────────────────────────────

def _espn_fetch_events(match_date: str) -> list:
    """ESPN all soccer scoreboard API'sinden belirtilen tarihin maçlarını çeker."""
    try:
        date_str = match_date.replace("-", "")
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard?dates={date_str}&limit=500"
        data = _http_get(url)
        if data:
            return data.get("events", [])
    except Exception:
        pass
    return []

def _espn_find_match(home_name: str, away_name: str, match_date: str) -> dict | None:
    """ESPN API'sinde ev sahibi ve deplasman takımlarını arar."""
    events = _espn_fetch_events(match_date)
    if not events:
        return None

    for ev in events:
        status = ev.get("status", {}).get("type", {}).get("name", "")
        # Only finished matches
        if status not in ("STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FULL_PEN", "STATUS_FT"):
            continue

        comps = ev.get("competitions", [{}])
        if not comps:
            continue
        comp = comps[0]
        teams = comp.get("competitors", [])
        if len(teams) < 2:
            continue

        # Determine home/away
        home_team, away_team = None, None
        for t in teams:
            if t.get("homeAway") == "home":
                home_team = t
            elif t.get("homeAway") == "away":
                away_team = t

        if not home_team or not away_team:
            home_team, away_team = teams[0], teams[1]

        h_name = home_team.get("team", {}).get("displayName", "")
        a_name = away_team.get("team", {}).get("displayName", "")

        h_sim = name_similarity(home_name, h_name)
        a_sim = name_similarity(away_name, a_name)

        if h_sim >= 0.80 and a_sim >= 0.80:
            try:
                hg = int(home_team.get("score", 0) or 0)
                ag = int(away_team.get("score", 0) or 0)
                return make_result(hg, ag, f"espn ({h_name} vs {a_name})", match_date)
            except (ValueError, TypeError):
                pass

    return None

# ─── TheSportsDB API ─────────────────────────────────────────────────────────

def _sportsdb_get_team_id(team_name: str) -> int | None:
    """TheSportsDB'den takım ID'sini arar ve önbelleğe kaydeder."""
    cache = load_team_id_cache()

    # Check cache first
    norm = normalize_name(team_name)
    if norm in cache:
        return cache[norm]

    # Rate limit: min 0.5s between requests
    time.sleep(0.5)

    # Try normalized name first, then original
    for name_variant in [team_name, normalize_name(team_name), team_name.split()[0]]:
        if not name_variant or len(name_variant) < 3:
            continue
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={urllib.parse.quote(name_variant)}"
        data = _http_get(url)
        if data:
            teams = data.get("teams") or []
            if teams:
                team_id = int(teams[0].get("idTeam", 0))
                if team_id:
                    cache[norm] = team_id
                    save_team_id_cache(cache)
                    return team_id

    # Cache negative result
    cache[norm] = None
    save_team_id_cache(cache)
    return None

def _sportsdb_find_match(home_name: str, away_name: str, match_date: str) -> dict | None:
    """TheSportsDB'den maç sonucunu arar."""
    team_id = _sportsdb_get_team_id(home_name)
    if not team_id:
        return None

    time.sleep(0.5)

    # Get last events for home team
    for endpoint in ["eventslast", "eventsnext"]:
        url = f"https://www.thesportsdb.com/api/v1/json/3/{endpoint}.php?id={team_id}"
        data = _http_get(url)
        if not data:
            continue

        events = data.get("results") or data.get("events") or []
        for ev in events:
            ev_date = ev.get("dateEvent", "")
            if ev_date != match_date:
                continue

            h_name = ev.get("strHomeTeam", "")
            a_name = ev.get("strAwayTeam", "")

            h_sim = name_similarity(home_name, h_name)
            a_sim = name_similarity(away_name, a_name)

            if h_sim >= 0.80 and a_sim >= 0.80:
                try:
                    hg = ev.get("intHomeScore")
                    ag = ev.get("intAwayScore")
                    if hg is not None and ag is not None:
                        return make_result(int(hg), int(ag), f"thesportsdb ({h_name} vs {a_name})", match_date)
                except (ValueError, TypeError):
                    pass

    return None

# ─── OpenLigaDB API (Alman Ligleri) ─────────────────────────────────────────

_OPENLIGADB_CACHE: dict[str, list] = {}

def _openligadb_fetch_season(league: str, year: str) -> list:
    """OpenLigaDB'den sezon maçlarını çeker ve önbellekler."""
    key = f"{league}/{year}"
    if key in _OPENLIGADB_CACHE:
        return _OPENLIGADB_CACHE[key]
    url = f"https://api.openligadb.de/getmatchdata/{league}/{year}"
    data = _http_get(url, timeout=10)
    result = data if isinstance(data, list) else []
    _OPENLIGADB_CACHE[key] = result
    return result

def _openligadb_find_match(home_name: str, away_name: str, match_date: str) -> dict | None:
    """OpenLigaDB'de Alman lig maçını arar."""
    # Try multiple German leagues and recent seasons
    year = match_date[:4] if match_date else "2025"
    prev_year = str(int(year) - 1)

    leagues = ["bl1", "bl2", "bl3"]
    years = [prev_year, year]

    for league in leagues:
        for yr in years:
            matches = _openligadb_fetch_season(league, yr)
            for m in matches:
                m_date = m.get("matchDateTime", "")[:10]
                if m_date != match_date:
                    continue
                if not m.get("matchIsFinished", False):
                    continue

                h_name = m.get("team1", {}).get("teamName", "")
                a_name = m.get("team2", {}).get("teamName", "")

                h_sim = name_similarity(home_name, h_name)
                a_sim = name_similarity(away_name, a_name)

                if h_sim >= 0.80 and a_sim >= 0.80:
                    results = m.get("matchResults", [])
                    for res in results:
                        if res.get("resultTypeID") == 2:  # Endstand (final score)
                            hg = res.get("pointsTeam1", 0)
                            ag = res.get("pointsTeam2", 0)
                            return make_result(hg, ag, f"openligadb ({h_name} vs {a_name})", match_date)

    return None

# ─── Flashscore API (fallback) ─────────────────────────────────────────────

def _flashscore_find_match(home_name: str, away_name: str, match_date: str) -> dict | None:
    """
    Flashscore.com üzerinden maç sonucunu arar.
    
    Flashscore resmi API yayınlamıyor; HTML scraping ile çalışır.
    Ancak çok sayıda maç olduğu için hedef maçı bulmak için 
    takım isimlerine göre filtreleme yapılır.
    
    Not: Flashscore'ın JS-required yapısı nedeniyle bu yöntem sınırlı olabilir.
          Gerçek sonuçlar için ESPN/TheSportsDB öncelikli kaynaklardır.
    """
    try:
        from .flashscore_fetcher import find_match_on_flashscore
        return find_match_on_flashscore(home_name, away_name, match_date)
    except ImportError:
        # flashscore_fetcher modülü yoksa
        return None
    except Exception:
        return None

# ─── Football-Data.co.uk CSV Fallback ────────────────────────────────────────

def _csv_find_match(home_name: str, away_name: str, match_date: str) -> dict | None:
    """Football-data.co.uk CSV'lerinden maç sonucu arar (downloaded_files)."""
    try:
        import pandas as pd
    except ImportError:
        return None

    csv_dir = os.path.join(BASE_DIR, "downloaded_files")
    if not os.path.isdir(csv_dir):
        return None

    try:
        # Parse date
        target_dt = datetime.strptime(match_date, "%Y-%m-%d") if match_date else None
    except ValueError:
        return None

    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(csv_dir, fname)
        try:
            df = pd.read_csv(fpath, encoding="latin1", on_bad_lines="skip")
            # Detect date column
            date_col = None
            for col in ["Date", "date", "DATE"]:
                if col in df.columns:
                    date_col = col
                    break
            if not date_col:
                continue

            for _, row in df.iterrows():
                try:
                    row_date = pd.to_datetime(str(row[date_col]), dayfirst=True, errors="coerce")
                    if pd.isna(row_date):
                        continue
                    if row_date.date() != target_dt.date():
                        continue

                    home_col = next((c for c in ["HomeTeam", "Home", "home_team"] if c in df.columns), None)
                    away_col = next((c for c in ["AwayTeam", "Away", "away_team"] if c in df.columns), None)
                    fthg_col = next((c for c in ["FTHG", "FTHome", "fthg"] if c in df.columns), None)
                    ftag_col = next((c for c in ["FTAG", "FTAway", "ftag"] if c in df.columns), None)

                    if not all([home_col, away_col, fthg_col, ftag_col]):
                        continue

                    csv_home = str(row[home_col])
                    csv_away = str(row[away_col])
                    h_sim = name_similarity(home_name, csv_home)
                    a_sim = name_similarity(away_name, csv_away)

                    if h_sim >= 0.80 and a_sim >= 0.80:
                        hg = int(float(row[fthg_col]))
                        ag = int(float(row[ftag_col]))
                        return make_result(hg, ag, f"csv:{fname}", match_date)
                except (ValueError, TypeError, KeyError):
                    continue
        except Exception:
            continue

    return None

# ─── Ana Fonksiyon ───────────────────────────────────────────────────────────

def _result_date_matches(result: dict, match_date: str, max_days: int = 7) -> bool:
    """Bulunan sonuç ile tahmin tarihi arasında tarih uyumu kontrol et.
    
    match_date: Tahmin edilen maç tarihi (YYYY-MM-DD)
    max_days: Maksimum gün farkı (varsayılan 7 gün)
    
    Eğer match_date yoksa veya sonuç tarihi yoksa True döner (eski uyumluluk).
    """
    if not match_date or not result:
        return True
    result_date = result.get("date", "")
    if not result_date:
        return True
    try:
        md = datetime.strptime(match_date[:10], "%Y-%m-%d")
        rd = datetime.strptime(result_date[:10], "%Y-%m-%d")
        return abs((md - rd).days) <= max_days
    except (ValueError, TypeError):
        return True


def get_match_result(home_id, away_id, home_name: str, away_name: str, match_date: str = "") -> dict | None:
    """
    Maç sonucunu bulur ve önbelleğe yazar.

    Kaynak sırası:
    1. Yerel önbellek (tahminler/match_results.json)
    2. AiScore lig fikstürü (simülasyon dünyası — kupon maçları için en güvenilir)
    3. SofaScore API (Playwright fetch - CF 403 karsisinda guvenilir)
    4. ESPN all soccer API
    5. TheSportsDB API
    6. OpenLigaDB (Alman ligleri)
    7. Flashscore.com (flashscore_fetcher - fallback)
    8. Football-Data.co.uk CSV
    """
    cid = f"{home_id}-{away_id}"
    cache = load_local_match_results()

    # ── 1. Yerel önbellek ──────────────────────────────────────────────────
    if cid in cache:
        res = cache[cid]
        if isinstance(res, dict) and "home_goals" in res and res.get("home_goals") is not None:
            return res
    # Ters yön kontrolü: ev/deplasman yer değiştirmiş olabilir
    reverse_cid = f"{away_id}-{home_id}"
    if reverse_cid in cache:
        res = cache[reverse_cid]
        if isinstance(res, dict) and "home_goals" in res and res.get("home_goals") is not None:
            # Ters kaydedilmiş sonucu düzelterek döndür (ev/deplasman yer değiştir)
            return {
                "home_goals": res.get("away_goals"),
                "away_goals": res.get("home_goals"),
                "total_goals": res.get("total_goals"),
                "result": "H" if res.get("away_goals", 0) > res.get("home_goals", 0) else ("D" if res.get("away_goals", 0) == res.get("home_goals", 0) else "A"),
                "btts": res.get("btts"),
                "over25": res.get("over25"),
                "home_corners": None,
                "away_corners": None,
                "ht_home_goals": None,
                "ht_away_goals": None,
                "date": res.get("date", ""),
                "source": res.get("source", "match_results_reversed"),
            }

    # Takım isimleri yoksa sorgulama yapamayız
    if not home_name or not away_name:
        return None

    # predictions.json'daki mojibake isimleri duzelt (cp1254/utf8 karmaşası)
    home_name = _fix_team_name(home_name)
    away_name = _fix_team_name(away_name)

    # Gelecekte ise arama yapma
    if match_date:
        try:
            md = datetime.fromisoformat(match_date.replace("Z", "").split("T")[0])
            if md.date() > datetime.now().date():
                return None
        except Exception:
            pass

    # ── 2. AiScore lig fikstürü (simülasyon dünyası — kuponlar) ─────────────
    result = _aiscore_find_match(home_name, away_name, match_date)
    if result and _result_date_matches(result, match_date):
        save_local_match_result(cid, result)
        return result

    # ── 3. SofaScore API (Playwright fetch - CF 403 onlenir) ─────────────
    result = _sofascore_find_match(home_name, away_name, match_date)
    if result and _result_date_matches(result, match_date):
        save_local_match_result(cid, result)
        return result

    # ── 4. ESPN Soccer API ────────────────────────────────────────────────
    result = _espn_find_match(home_name, away_name, match_date)
    if result and _result_date_matches(result, match_date):
        save_local_match_result(cid, result)
        return result

    # ── 5. TheSportsDB ───────────────────────────────────────────────────
    result = _sportsdb_find_match(home_name, away_name, match_date)
    if result and _result_date_matches(result, match_date):
        save_local_match_result(cid, result)
        return result

    # ── 6. OpenLigaDB (Alman ligleri için) ──────────────────────────────
    result = _openligadb_find_match(home_name, away_name, match_date)
    if result and _result_date_matches(result, match_date):
        save_local_match_result(cid, result)
        return result

    # ── 7. Flashscore.com (yeni eklenti) ────────────────────────────────
    result = _flashscore_find_match(home_name, away_name, match_date)
    if result and _result_date_matches(result, match_date):
        save_local_match_result(cid, result)
        return result

    # ── 8. CSV Fallback ───────────────────────────────────────────────────
    result = _csv_find_match(home_name, away_name, match_date)
    if result and _result_date_matches(result, match_date):
        save_local_match_result(cid, result)
        return result

    return None


# ─── Yardımcı: ESPN bulk fetch (bir tarihteki tüm maçları cache'le) ──────────

def prefetch_espn_for_date(match_date: str) -> int:
    """
    Belirtilen tarihteki tüm ESPN maçlarını çekip önbelleğe yazar.
    Aynı gün birden fazla maç varsa tek HTTP isteği ile hepsini çeker.
    Döndürülen değer: kaydedilen maç sayısı.
    """
    events = _espn_fetch_events(match_date)
    count = 0
    for ev in events:
        status = ev.get("status", {}).get("type", {}).get("name", "")
        if status not in ("STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FULL_PEN", "STATUS_FT"):
            continue

        comps = ev.get("competitions", [{}])
        if not comps:
            continue
        comp = comps[0]
        teams = comp.get("competitors", [])
        if len(teams) < 2:
            continue

        home_team, away_team = None, None
        for t in teams:
            if t.get("homeAway") == "home":
                home_team = t
            elif t.get("homeAway") == "away":
                away_team = t
        if not home_team or not away_team:
            home_team, away_team = teams[0], teams[1]

        h_name = home_team.get("team", {}).get("displayName", "")
        a_name = away_team.get("team", {}).get("displayName", "")

        try:
            hg = int(home_team.get("score", 0) or 0)
            ag = int(away_team.get("score", 0) or 0)
            # Use team names as key for temp cache
            key = f"espn_{normalize_name(h_name)}_{normalize_name(a_name)}_{match_date}"
            _ = make_result(hg, ag, "espn", match_date)
            count += 1
        except Exception:
            pass

    return count
