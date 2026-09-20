"""AiScore Scraper - kupon maclari icin skor + istatistik.

Mimari (dogrulanmis kesifler):
  1. Lig SSR sayfasi fiksturu:  https://www.aiscore.com/tournament-{slug}/{comp_id}
     Nuxt SSR icerir: window.__NUXT__.state.football.comp.data
        - matches[] : {id, homeTeam.id, awayTeam.id, matchTime, statusId, scores, roundNum}
        - teams[] / teamObj : id -> {name, slug, logo}
  2. Kupon takim adi -> AiScore lig fikstur eSlemesi (home+away slug/name).
  3. Mac sayfasi Playwright + NUXT state:
        window.__NUXT__.state.football.detail
          - WebMatchData.match  -> skor, status, takimlar
          - stats.items         -> istatistik {
                "2": corner, "3": yellow, "4": red,
                "6": possession(%), "8": ??,
                "21": shots on target, "22": shots off target,
                "23": attacks, "24": dangerous attacks, "25": possession,
                "99": total shots }

Not: AiScore API (api.aiscore.com) protobuf donuyor ve bunlari tasimiyor;
bugun icin bu scraper tek islevsel veri kaynagi (kullanici onayi: Playwright + NUXT state).
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import threading
import time
import unicodedata
import urllib.request
from datetime import datetime, timezone
from typing import Optional

import logging

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tahminler")
_COMP_CACHE_PATH = os.path.join(_CACHE_DIR, "aiscore_comp_cache.json")
_MATCH_CACHE_PATH = os.path.join(_CACHE_DIR, "aiscore_match_cache.json")

# Kusurlu saysal kusurlu bytes onemsiz; temiz parity onemli.
_COMP_SCRAPER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

# Kullaniciya/uygulamaya gore TURK ihtiyac kapak: super + first + second + EPL + Bundesliga.
# (comp_id, slug) protobuf mezzo parslari ile dogrulandi. PCR-7: grenzler.
KNOWN_COMPETITIONS = [
    ("Turkish Super League", "turkish-super-league", "4ndqmliwou5kveg"),
    ("Turkish First League", "turkish-first-league", "2j374oi0lf4qo6d"),
    ("Turkish Second League", "turkish-second-league", "eg6763i5es47ryv"),
    ("English Premier League", "english-premier-league", "mo07dni2vfxknxy"),
    ("Bundesliga", "bundesliga", "1edq09ignayqxgo"),
    ("German 2. Bundesliga", "german-bundesliga-2", "w34kgmixet1ko92"),
]

# items key -> okunur isim
_STAT_LABELS = {
    "2": "Korner",
    "3": "Sari Kart",
    "4": "Kirmizi Kart",
    "6": "Diğer",  # AiScore'da DOM'da karsiligi olmayan ekstra metrik
    "8": "Penalti",
    "21": "Isabetli Sut",
    "22": "Isabetsiz Sut",
    "23": "Atak",
    "24": "Tehlikeli Atak",
    "25": "Topa Sahiplik (%)",
    "99": "Toplam Sut",
}

# ust oncelikli (UI'da once gelsin)
_STAT_ORDER = ["25", "99", "21", "22", "23", "24", "2", "3", "4", "8", "6"]


# ── yardimcilar ──────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Turkce/ascii bagimsiz normalize: 'Çaykur Rizespor' -> 'c aykurrizespor'."""
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", str(name))
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower()
    n = re.sub(r"[^a-z0-9]+", "", n)
    return n


def _fix_mojibake(name: str) -> str:
    """kupon (cp1254 degrade) adlarini dogru UTF-8'e cevirir.

    Ornek: 'BaÅŸakÅŸehir' -> 'Başakşehir', 'GençlerbirliÄŸi' -> 'Gençlerbirliği'.
    Zaten dogru olan adlari bozmaz (encode/decode hata verirse orijinal kalir).
    """
    if not name:
        return name
    try:
        fixed = str(name).encode("cp1254").decode("utf-8", errors="strict")
        return fixed
    except Exception:  # noqa: BLE001
        return name


# yaygin ad kisaltmalari (AiScore slug eksiksiz hali -> kupon adi)
_ALIASES = {
    "manutd": "manunited",
    "manchesterunited": "manunited",
    "manutd.": "manunited",
    "gaziantepfk": "gaziantepspor",
    "gaziantep": "gaziantepspor",
    "basaksehir": "istanbulbasaksehir",
    "basaksehirfk": "istanbulbasaksehir",
    "goztepe": "goztepe",
    "kayserispor": "kayserispor",
}

_ALIAS_TOKEN = ("united", "city", "spor", "fk", "fc", "utd", "bbsk", "sk")


def _name_match(coupon: str, aiscore_name: str) -> int:
    """Kupon takim adi ile AiScore takim adi eslesme skoru.

    0=eslesme yok; buyuk=iyi eslesme. Eslestirme goz goz degil sanki token bazli.
    """
    c = _norm(coupon)
    a = _norm(aiscore_name)
    if not c or not a:
        return 0
    if c == a:
        return 1000
    if _ALIASES.get(c) == a or _ALIASES.get(a) == c:
        return 900
    # token bazli: "man united" vs "manchester united" -> manchester man'a benzer + united tam
    c_toks = [t for t in re.split(r"[^a-z0-9]+", coupon.lower()) if t]
    a_toks = [t for t in re.split(r"[^a-z0-9]+", aiscore_name.lower()) if t]
    score = 0
    for ct in c_toks:
        for at in a_toks:
            if ct == at:
                score += 60
            elif at.startswith(ct) or ct.startswith(at):
                score += 30
            elif len(ct) >= 4 and ct in at:
                score += 20
    return min(score, 200)


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _COMP_SCRAPER_UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _node_eval(js_code: str) -> Optional[str]:
    """window.__NUXT__ ifadesini Node ile degerlendirip JSON dondurur."""
    tmp = os.path.join(os.environ.get("TEMP", _CACHE_DIR), "aiscore_dump_z.js")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(js_code)
    try:
        proc = subprocess.run(
            ["node", tmp],
            capture_output=True, timeout=90, encoding="utf-8", errors="replace",
        )
        out = proc.stdout.strip()
        return out or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[AiScore] node calistiramadi: %s", exc)
        return None


# ── lig fikstur cache + parse ────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def _parse_comp_nuxt(html: str) -> Optional[dict]:
    m = re.search(r"window\.__NUXT__=(.*?);</script>", html, re.S)
    if not m:
        return None
    js = (
        "var window = {}; window.__NUXT__=" + m.group(1) + ";\n"
        "const d = window.__NUXT__.state.football.comp && window.__NUXT__.state.football.comp.data;\n"
        "if (!d) { console.log('NODATA'); }\n"
        "else { console.log(JSON.stringify(d)); }\n"
    )
    out = _node_eval(js)
    if not out or out == "NODATA":
        return None
    try:
        return json.loads(out)
    except Exception:  # noqa: BLE001
        return None


def _comp_fixture(comp_id: str, slug: str) -> Optional[dict]:
    """Lig fiksturunu SSR'dan cekip (tek sefer cache) dondurur.

    Return: {league_name, matches:[{id, home, home_name, away, away_name, time,
            status, round, home_score, away_score}], teams:{slug:name}}
    """
    cache = _load_json(_COMP_CACHE_PATH)
    if comp_id in cache and time.time() - cache[comp_id].get("fetched_at", 0) < 6 * 3600:
        return cache[comp_id]["data"]

    url = f"https://www.aiscore.com/tournament-{slug}/{comp_id}"
    try:
        html = _http_get(url, timeout=40)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[AiScore] lig SSR hata %s: %s", url, exc)
        return None
    data = _parse_comp_nuxt(html)
    if not data:
        return None

    teams = {}
    for t in data.get("teams", []) or []:
        teams.setdefault(t.get("slug"), t)
    team_obj = data.get("teamObj") or {}

    league = (data.get("compDetail", {}).get("competition") or {}).get("name") or slug

    matches = []
    for m in data.get("matches", []) or []:
        h = team_obj.get(m.get("homeTeam", {}).get("id")) or {}
        a = team_obj.get(m.get("awayTeam", {}).get("id")) or {}
        hs = (m.get("homeScores") or [])
        aw = (m.get("awayScores") or [])
        matches.append({
            "id": m.get("id"),
            "home": h.get("slug"),
            "home_name": h.get("name"),
            "away": a.get("slug"),
            "away_name": a.get("name"),
            "time": m.get("matchTime"),
            "status": m.get("statusId"),
            "round": m.get("roundNum"),
            "home_score": hs[0] if hs else None,
            "away_score": aw[0] if aw else None,
        })

    result = {"league_name": league, "matches": matches, "teams": teams}
    cache[comp_id] = {"data": result, "fetched_at": time.time()}
    _save_json(_COMP_CACHE_PATH, cache)
    return result


# ── kupon takim adi -> mac eSlemesi ─────────────────────────────────────────

def resolve_aiscore_match(
    home_name: str, away_name: str,
    near_ts: Optional[float] = None,
    comps: Optional[list] = None,
) -> Optional[dict]:
    """Kupon (home, away) takim adini AiScore liglerinde arar.

    near_ts: kupon olusturma/oynama zamani (epoch) - cift tur turnuvada
             bu zamana en yakin maçi secer.

    Dondurur: {comp_id, league_name, match_id, url, home, away,
               home_name, away_name, time, status, round,
               home_score, away_score}  ya da None.
    """
    if not _norm(home_name) or not _norm(away_name):
        return None

    home_name = _fix_mojibake(home_name)
    away_name = _fix_mojibake(away_name)

    comps = comps or KNOWN_COMPETITIONS
    best = []
    for league_name, slug, comp_id in comps:
        fx = _comp_fixture(comp_id, slug)
        if not fx:
            continue
        for m in fx["matches"]:
            hs = _name_match(home_name, m.get("home_name") or "")
            as_ = _name_match(away_name, m.get("away_name") or "")
            hr = _name_match(home_name, m.get("away_name") or "")
            ar = _name_match(away_name, m.get("home_name") or "")
            if (hs >= 45 and as_ >= 45) or (hr >= 45 and ar >= 45):
                swapped = hr >= 45 and ar >= 45 and (hr + ar) > (hs + as_)
                home_slug = m["away"] if swapped else m["home"]
                away_slug = m["home"] if swapped else m["away"]
                home_dn = m.get("away_name") if swapped else m.get("home_name")
                away_dn = m.get("home_name") if swapped else m.get("away_name")
                best.append({
                    "comp_id": comp_id,
                    "league_name": fx["league_name"],
                    "match_id": m["id"],
                    "url": f"https://www.aiscore.com/match-{home_slug}-{away_slug}/{m['id']}",
                    "home": home_slug,
                    "away": away_slug,
                    "home_name": home_dn,
                    "away_name": away_dn,
                    "time": m["time"],
                    "status": m["status"],
                    "round": m["round"],
                    "home_score": m["home_score"],
                    "away_score": m["away_score"],
                    "_score": (hs + as_) if not swapped else (hr + ar),
                })
    if not best:
        return None
    # once en yuksek eslesme skoru
    best.sort(key=lambda x: -x["_score"])
    top = best[0]["_score"]
    cands = [x for x in best if x["_score"] >= top - 10]
    if len(cands) == 1:
        return cands[0]
    # coklu: near_ts varsa en yakin zaman; yoksa en yakin gelecegi sec
    if near_ts:
        cands.sort(key=lambda x: abs((x["time"] or 0) - near_ts))
        return cands[0]
    cands.sort(key=lambda x: abs((x["time"] or 0) - time.time()))
    return cands[0]


# ── mac sayfasi: Playwright + NUXT ├─┴ detail/stats ─────────────────────────

def _fetch_match_state_worker(browser, _page, match_url: str) -> Optional[dict]:
    """Worker thread icinde, ortak browser ile calisan Playwright isi."""
    ctx = None
    page = None
    try:
        ctx = browser.new_context(
            user_agent=_COMP_SCRAPER_UA,
            locale="tr-TR",
            viewport={"width": 1280, "height": 900},
        )
        page = ctx.new_page()
        page.goto(match_url, timeout=45000, wait_until="domcontentloaded")
        page.wait_for_timeout(7000)
        return page.evaluate("""() => {
            const d = window.__NUXT__?.state?.football?.detail;
            if (!d) { return null; }
            const w = d.WebMatchData || {};
            const m = w.match || {};
            return {
                match: m,
                stats: d.stats || null,
                incidents: d.incidents || null,
                lineups: d.lineups || null,
            };
        }""")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[AiScore] Playwright hata: %s", exc)
        return None
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:  # noqa: BLE001
                pass
        if ctx is not None:
            try:
                ctx.close()
            except Exception:  # noqa: BLE001
                pass


def fetch_match_state(match_url: str) -> Optional[dict]:
    """Playwright ile mac sayfasini acip NUXT detail state'ini dondurur.

    Thread guvenligi: Playwright isi pw_worker'daki tek isci thread'inde calisir;
    bu, Flask request thread'i + arka plan 'Canli kontrol' thread'i ayni anda
    Playwright kullandiginda cikan 'greenlet.error' hatalarini onler.
    """
    try:
        from prediction.pw_worker import run_with_browser as _run_with_browser
    except Exception:
        _run_with_browser = None
    with _LOCK:
        if _run_with_browser is not None:
            try:
                return _run_with_browser(_fetch_match_state_worker, match_url, timeout=120)
            except Exception:
                return None
        return _fetch_match_state_worker(None, None, match_url)


def _stat_dict_to_list(stats: Optional[dict]) -> list:
    if not stats or not isinstance(stats, dict):
        return []
    items = stats.get("items") or {}
    out = []
    for key in _STAT_ORDER:
        row = items.get(key)
        if not row:
            continue
        out.append({
            "key": key,
            "label": _STAT_LABELS.get(key, key),
            "home": row.get("home"),
            "away": row.get("away"),
        })
    return out


def get_aiscore_match_detail(home_name: str, away_name: str, near_ts: Optional[float] = None) -> Optional[dict]:
    """Kupon takim adi -> AiScore mac -> skor + istatistik.

    Dondurur:
      {match_id, url, source: "aiscore", home_name, away_name,
       home_score, away_score, minute, status, is_live, league_name, round,
       stats: [{key,label,home,away}]}
    """
    res = resolve_aiscore_match(home_name, away_name, near_ts=near_ts)
    if not res:
        return None

    state = fetch_match_state(res["url"])
    match = (state or {}).get("match") or {}
    hs = match.get("homeScores") or []
    aw = match.get("awayScores") or []
    status = match.get("statusId")
    is_live = status in (1, 2, 3)  # AiScore live statusler; finished => 8

    kick = match.get("matchTime")
    kick_iso = None
    if kick:
        try:
            kick_iso = datetime.fromtimestamp(kick, tz=timezone.utc).isoformat()
        except Exception:  # noqa: BLE001
            kick_iso = None

    return {
        "match_id": res["match_id"],
        "url": res["url"],
        "source": "aiscore",
        "league_name": res.get("league_name"),
        "league_round": res.get("round"),
        "home_name": res.get("home_name") or home_name,
        "away_name": res.get("away_name") or away_name,
        "home_score": hs[0] if hs else res.get("home_score"),
        "away_score": aw[0] if aw else res.get("away_score"),
        "minute": None,
        "status": "LIVE" if is_live else ("FT" if status == 8 else "SCHEDULED"),
        "status_id": status,
        "is_live": is_live,
        "kickoff": kick_iso,
        "stats": _stat_dict_to_list((state or {}).get("stats")),
    }


if __name__ == "__main__":
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if len(sys.argv) >= 3:
        det = get_aiscore_match_detail(sys.argv[1], sys.argv[2])
        print(json.dumps(det, ensure_ascii=False, indent=2, default=str))
    else:
        print("kullanim: python aiscore_scraper.py <home> <away>")