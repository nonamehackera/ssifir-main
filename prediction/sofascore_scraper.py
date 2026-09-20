"""SofaScore API tuttucu (Playwright fetch ile CF 403'unu asar).

SofaScore, urllib/requests gibi standart TLS parmak izlerinden gelen isteklere
403 (Forbidden) donuyor. Cozum: gercek tarayici (Playwright) acilir, ana sayfaya
gidilir (Cloudflare cookie/token olusur) ve API cagrilari tarayici icinden
fetch() ile yapilir -> ayni origin, 200 doner.

Thread guvenligi: Playwright sync API'si greenlet/thread'e baglidir; Flask
request thread'i + arka plan "Canli kontrol" thread'i ayni anda kullaninca
'greenlet.error: Cannot switch to a different thread' cikar. Tum islemler
prediction.pw_worker icindeki tek worker thread uzerinden gecer.

Kupon kontrolu akisi:
   1. Her iki takimi da search/teams ile arat (isim + slug uzerinden en iyi es).
   2. Ev sahibinin son maclarini cek (team/{id}/events/last/0).
   3. Rakibi ve bitmis status ('finished'/code 100) olan maci bul.
   4. Skoru make_result stili dict olarak dondur.
"""
import threading
import urllib.parse

from prediction.pw_worker import run_on_page

_LOCK = threading.RLock()

_SOFASCORE_HOME = "https://www.sofascore.com/"


def _ensure_origin(page):
    """Worker'in ortak page'i herhangi bir sitede olabilir; SofaScore'a tasinir.

    fetch() relative URL kullandigi icin page'in www.sofascore.com origin'inde
    olmasi sart. Zaten oradaysa yeniden yukleme (CF tokeni korunur).
    """
    cur = ""
    try:
        cur = page.evaluate("() => window.location.origin") or ""
    except Exception:
        cur = ""
    if "sofascore.com" in cur:
        return
    page.goto(_SOFASCORE_HOME, timeout=45000, wait_until="domcontentloaded")
    try:
        page.wait_for_timeout(4000)
    except Exception:
        pass


def _api(page, url, args=None):
    """Tarayici icinden fetch ile SofaScore API cagrisi. Dict/json doner."""
    return page.evaluate(
        """async (u) => {
            const r = await fetch(u, {headers: {'Accept': 'application/json', 'Accept-Language':'en'}});
            const t = await r.text();
            try { return {ok: r.ok, data: JSON.parse(t)}; }
            catch (e) { return {ok: r.ok, data: t}; }
        }""",
        url,
    )


def _search_teams(page, query):
    res = _api(page, "/api/v1/search/teams?q=" + urllib.parse.quote(query or "") + "&page=0")
    out = []
    if not isinstance(res, dict) or not res.get("ok"):
        return out
    for item in (res.get("data") or {}).get("results") or []:
        ent = item.get("entity") or {}
        if not ent or not ent.get("id"):
            continue
        out.append({
            "id": ent.get("id"),
            "name": ent.get("name") or "",
            "slug": ent.get("slug") or "",
            "country": (ent.get("country") or {}).get("name") or "",
        })
    return out


def _team_events(page, team_id, page_no=0):
    kind = "last" if page_no == 0 else "next"
    res = _api(page, f"/api/v1/team/{team_id}/events/{kind}/{page_no}")
    out = []
    if not isinstance(res, dict) or not res.get("ok"):
        return out
    for e in (res.get("data") or {}).get("events") or []:
        hs = e.get("homeScore") or {}
        aw = e.get("awayScore") or {}
        status = e.get("status") or {}
        tour = e.get("tournament") or {}
        home_team = (e.get("homeTeam") or {})
        away_team = (e.get("awayTeam") or {})
        out.append({
            "id": e.get("id"),
            "home_team_id": home_team.get("id"),
            "away_team_id": away_team.get("id"),
            "home_team_name": home_team.get("name"),
            "away_team_name": away_team.get("name"),
            "home_score": hs.get("current"),
            "away_score": aw.get("current"),
            "status_type": status.get("type") or "",
            "status_code": status.get("code"),
            "start_timestamp": e.get("startTimestamp"),
            "tournament": tour.get("name") or "",
        })
    return out


def _score_match(name, cand):
    """Takım adını adaylarla skorlar ve (skor, candidate) listesini sıralı döndürür.

    AiScore _name_match öncelikli; düşükse difflib ratio fallback. Tek aday
    yerine tüm adayları skorlu döndürür ki ana fonksiyon her birini deneyebilsin
    (birebir isimli ama boş takım alt dalları atlanarak gerçek maçın olduğu akraba
    takıma ulaşılır - ornek: kupon 'Erzurum BB' vs Sofa 'Erzurumspor FK').
    """
    try:
        from prediction.aiscore_scraper import _name_match
    except Exception:
        _name_match = None
    from difflib import SequenceMatcher
    scored = []
    for c in cand:
        s = 0.0
        if _name_match is not None:
            s = float(_name_match(name, c["name"]) or _name_match(name, c["slug"] or "") or 0)
        if s < 55:
            na = c["name"] or ""
            s2 = SequenceMatcher(None, (name or "").lower(), na.lower()).ratio() * 100
            s = max(s, s2)
        scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    return scored


def _top_candidates(name, cand, n=6):
    return [c for _, c in _score_match(name, cand)[:n]]


def _is_finished(ev):
    st = (ev.get("status_type") or "").lower()
    code = ev.get("status_code")
    if st in ("finished", "ended", "end"):
        return True
    if code in (100,):
        return True
    return False


_BAD_TOURNAMENT = (
    "u21", "u23", "u19", "u18", "u20", "youth", "reserve",
    "amator", "amateur", "women", "kadın", "ladies", "primavera",
)


def _good_tournament(tour: str) -> bool:
    """U21/rezerv/yarismaci liglerini (kokan yanlis skorlar) eler."""
    tl = (tour or "").lower()
    if tl.startswith("ii "):
        return False
    return not any(b in tl for b in _BAD_TOURNAMENT)


def _to_result(ev, match_date):
    hg, ag = int(ev["home_score"]), int(ev["away_score"])
    src = f"sofascore{(' (' + ev['tournament'] + ')') if ev.get('tournament') else ''}".strip()
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
        "source": src,
    }


def _find_on_page(page, home_name: str, away_name: str, match_date: str) -> dict | None:
    """Worker page'i uzerinde arama yapar (tek thread'te calisir)."""
    _ensure_origin(page)
    homes = _search_teams(page, home_name)
    aways = _search_teams(page, away_name)
    if not homes or not aways:
        return None

    home_cands = _top_candidates(home_name, homes)
    away_cands = _top_candidates(away_name, aways)
    away_ids = {c["id"] for c in away_cands}

    # Ev-sahibi adaylarinin son maclarinda rakip adaylarindan biriyle bitmis eslesme ara
    for hc in home_cands:
        evs = _team_events(page, hc["id"], 0)
        for ev in evs:
            if ev["away_team_id"] not in away_ids:
                continue
            if not _is_finished(ev):
                continue
            if not _good_tournament(ev.get("tournament")):
                continue
            if ev["home_score"] is None or ev["away_score"] is None:
                continue
            return _to_result(ev, match_date)

    # Deplasman adaylarinin maclarinda ev-sahibi adayi arayan ikincil tarama (ters map)
    home_ids = {c["id"] for c in home_cands}
    for ac in away_cands:
        evs = _team_events(page, ac["id"], 0)
        for ev in evs:
            if ev["home_team_id"] not in home_ids:
                continue
            if not _is_finished(ev):
                continue
            if not _good_tournament(ev.get("tournament")):
                continue
            if ev["home_score"] is None or ev["away_score"] is None:
                continue
            return _to_result(ev, match_date)
    return None


def find_match_on_sofascore(home_name: str, away_name: str, match_date: str = "", proxy: str | None = None) -> dict | None:
    """Kupon macini SofaScore'da arar (ev-sahibi adaylari + deplasman adaylari taraniyor).

    Birebir adli takim skor tasimiyorsa sonraki adaya geciyor (Erzurum BB -> Erzurumspor FK gibi).
    """
    home_name = (home_name or "").strip()
    away_name = (away_name or "").strip()
    if not home_name or not away_name:
        return None
    with _LOCK:
        try:
            return run_on_page(_find_on_page, home_name, away_name, match_date, timeout=240)
        except Exception:
            return None


def find_h2h_on_sofascore(home_name: str, away_name: str, match_date: str = "") -> dict | None:
    """Ayni mantigin eski/ozel hali - find_match_on_sofascore zaten cift yonludur."""
    return find_match_on_sofascore(home_name, away_name, match_date)


if __name__ == "__main__":
    import io as _io
    import sys as _sys
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
    for h, a in [("Arsenal", "Chelsea"), ("Hamburg", "Mainz"), ("Göztepe", "Gaziantepspor")]:
        r = find_match_on_sofascore(h, a)
        print(f"{h} vs {a} -> {r}")