"""Flashscore Mobile Scraper (m.flashscore.mobi) - tarih bazli scraping.

URL pattern:
  ?d=-N   -> N gun once (0 = bugun, -1 = dun, 1 = yarin)
  ?s=1    -> All Games (default)
  ?s=2    -> LIVE (sadece canli maclar)
  ?s=3    -> Finished (sadece bitenler)
  ?s=5    -> Odds (oran)

Ornekler:
  https://m.flashscore.mobi/?d=-1&s=3  -> dun biten maclar
  https://m.flashscore.mobi/?d=0&s=2   -> bugun canli maclar
  https://m.flashscore.mobi/?d=2       -> 2 gun sonraki tum maclar

Bu mobile site klasik desktop'tan daha basit ve tarih URL'leri calisiyor.
"""
from __future__ import annotations

import os
import re
import json
import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# Bilinen lig isim haritasi (mobile text formatinda)
_KNOWN_LEAGUES = {
    "premier league": "England Premier League",
    "la liga": "Spain LaLiga",
    "laliga": "Spain LaLiga",
    "serie a": "Italy Serie A",
    "serie b": "Italy Serie B",
    "bundesliga": "Germany Bundesliga",
    "2. bundesliga": "Germany 2. Bundesliga",
    "ligue 1": "France Ligue 1",
    "ligue 2": "France Ligue 2",
    "eredivisie": "Netherlands Eredivisie",
    "primeira liga": "Portugal Primeira Liga",
    "super lig": "Turkey Super Lig",
    "super league": "Switzerland Super League",
    "championship": "England Championship",
    "league one": "England League One",
    "league two": "England League Two",
    "copa del rey": "Spain Copa del Rey",
    "copa italia": "Italy Coppa Italia",
    "fa cup": "England FA Cup",
    "carabao cup": "England EFL Cup",
    "conference league": "UEFA Europa Conference League",
    "europa league": "UEFA Europa League",
    "champions league": "UEFA Champions League",
    "caf champions league": "Africa CAF Champions League",
    "copa libertadores": "South America Copa Libertadores",
    "copa sudamericana": "South America Copa Sudamericana",
    "mls": "USA MLS",
    "liga mx": "Mexico Liga MX",
    "j1 league": "Japan J1 League",
    "k league": "South Korea K League",
    "a-league": "Australia A-League",
    "spl": "Saudi Pro League",
    "botola": "Morocco Botola Pro",
    "torneo federal": "Argentina Torneo Federal",
    "primera b": "Argentina Primera B",
    "primera c": "Argentina Primera C",
    "primera d": "Argentina Primera D",
    "primera nacional": "Argentina Primera Nacional",
    "liga profesional": "Argentina Liga Profesional",
    "brasileirao": "Brazil Serie A",
    "brasileiro women": "Brazil Brasileiro Women",
    "liga de primera": "Chile Liga de Primera",
    "liga de ascenso": "Chile Liga de Ascenso",
    "primera b -": "Colombia Primera B",
    "liga pro": "Ecuador Liga Pro",
    "liga nacional": "Guatemala Liga Nacional",
    "liga de expansion mx": "Mexico Liga de Expansion MX",
    "kate sheppard cup women": "New Zealand Kate Sheppard Cup Women",
    "lpf": "Panama LPF",
    "mls next pro": "USA MLS Next Pro",
    "nwsl women": "USA NWSL Women",
    "amazonsen": "Brazil Amazonense",
    "nwsle women": "USA NWSL Women",
}


def _normalize_team_name(name: str) -> str:
    if not name:
        return ""
    n = str(name).strip().lower()
    return n


def _extract_country(text: str) -> str:
    """Text basindan ulkeyi cikar: 'ENGLAND: Premier League...' -> 'ENGLAND'."""
    if not text:
        return ""
    m = re.match(r'^([A-ZÇĞİÖŞÜ]{3,})\s*:', text)
    return m.group(1) if m else ""


def _extract_league(text: str, country: str = "") -> tuple[str, str]:
    """Lig basligi text'inden (lig adi, round) cikar.

    Text ornekleri:
      'ENGLAND: Premier League Standings' -> ('England Premier League', '')
      'BRAZIL: Serie B Standings'         -> ('Brazil Serie B', '')
      'ARGENTINA: Torneo Federal - Winners stage Standings' -> ('Argentina Torneo Federal', 'Winners stage')
    """
    if not text:
        return ("", "")
    if not country:
        country = _extract_country(text)
    # Ulke kismini at
    raw = re.sub(r'^[A-ZÇĞİÖŞÜ]{3,}\s*:\s*', '', text)
    # 'Standings' / 'Draw' ekini at
    raw = re.sub(r'\s+(Standings?|Draw|Live Standings?)\s*$', '', raw, flags=re.IGNORECASE).strip()
    # Round: ' - Winners stage', ' - Play Offs', vb.
    round_m = re.search(r'\s*-\s*([A-Za-z][A-Za-z0-9\s]+?)$', raw)
    round_name = ""
    if round_m:
        round_name = round_m.group(1).strip()
        raw = raw[:round_m.start()].strip()
    norm = raw.lower().strip()
    # Country-aware mapping: ayni lig ismi farkli ulkelerde olabilir
    country_norm = country.lower().strip()
    # Once ulkeye ozel map'i dene
    country_specific = {
        ("italy", "serie a"): "Italy Serie A",
        ("italy", "serie b"): "Italy Serie B",
        ("italy", "serie c"): "Italy Serie C",
        ("brazil", "serie a"): "Brazil Serie A",
        ("brazil", "serie b"): "Brazil Serie B",
        ("brazil", "brasileiro women"): "Brazil Brasileiro Women",
        ("england", "premier league"): "England Premier League",
        ("england", "championship"): "England Championship",
        ("england", "league one"): "England League One",
        ("england", "league two"): "England League Two",
        ("spain", "la liga"): "Spain LaLiga",
        ("germany", "bundesliga"): "Germany Bundesliga",
        ("germany", "2. bundesliga"): "Germany 2. Bundesliga",
        ("france", "ligue 1"): "France Ligue 1",
        ("france", "ligue 2"): "France Ligue 2",
        ("netherlands", "eredivisie"): "Netherlands Eredivisie",
        ("argentina", "primera b"): "Argentina Primera B",
        ("argentina", "primera c"): "Argentina Primera C",
        ("argentina", "primera d"): "Argentina Primera D",
        ("argentina", "primera nacional"): "Argentina Primera Nacional",
        ("argentina", "liga profesional"): "Argentina Liga Profesional",
        ("argentina", "torneo federal"): "Argentina Torneo Federal",
        ("chile", "liga de primera"): "Chile Liga de Primera",
        ("chile", "liga de ascenso"): "Chile Liga de Ascenso",
        ("colombia", "primera b"): "Colombia Primera B",
        ("ecuador", "liga pro"): "Ecuador Liga Pro",
        ("guatemala", "liga nacional"): "Guatemala Liga Nacional",
        ("honduras", "liga nacional"): "Honduras Liga Nacional",
        ("mexico", "liga de expansion mx"): "Mexico Liga de Expansion MX",
        ("panama", "lpf"): "Panama LPF",
        ("usa", "mls next pro"): "USA MLS Next Pro",
        ("usa", "nwsl women"): "USA NWSL Women",
    }
    league = country_specific.get((country_norm, norm))
    if league is None:
        # Genel fallback
        league = _KNOWN_LEAGUES.get(norm, raw if raw else "Unknown League")
    return (league, round_name)


def _parse_minute(text: str) -> Optional[int]:
    """'70' veya '45+2' gibi dakika bilgisinden sayiyi al."""
    if not text:
        return None
    s = str(text).strip()
    m = re.match(r'^(\d+)(?:\+(\d+))?$', s)
    if m:
        return int(m.group(1))
    return None


def _parse_score(text: str) -> tuple[Optional[int], Optional[int]]:
    """'0-2' veya '2-0' gibi skorlari parse et."""
    if not text:
        return (None, None)
    m = re.match(r'^\s*(\d+)\s*[-:]\s*(\d+)\s*$', str(text).strip())
    if m:
        h, a = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 30 and 0 <= a <= 30:
            return (h, a)
    return (None, None)


_THREAD_LOCAL = threading.local()
_SCRAPER_LOCK = threading.Lock()  # Playwright greenlet conflict'i onler


def _get_browser():
    """Thread-local browser: her Flask thread kendi browser'ini kullanir."""
    if hasattr(_THREAD_LOCAL, "browser") and _THREAD_LOCAL.browser is not None:
        return _THREAD_LOCAL.browser
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("playwright yuklu degil") from exc
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    _THREAD_LOCAL.pw = pw
    _THREAD_LOCAL.browser = browser
    return browser


def _scrape_mobile_day(day_offset: int = 0, status: int = 1, page=None) -> list[dict]:
    """Mobile flashscore'dan tek bir tarih/status icin scrape et.

    Args:
        day_offset: -7..+14 (0 = bugun)
        status: 1=all, 2=live, 3=finished, 5=odds
        page: opsiyonel - mevcut Playwright page (yoksa yeni context acilir)
    """
    own_ctx = None
    if page is None:
        browser = _get_browser()
        own_ctx = browser.new_context(
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            locale="tr-TR",
            viewport={"width": 414, "height": 896},
            timezone_id="Europe/Istanbul",
            is_mobile=True,
            has_touch=True,
        )
        page = own_ctx.new_page()
    try:
        url = f"https://m.flashscore.mobi/?d={day_offset}&s={status}"
        page.goto(url, timeout=30000, wait_until="domcontentloaded")
        try:
            page.wait_for_selector("#score-data", timeout=15000)
        except Exception:
            pass
        time.sleep(2)
        items = page.evaluate("""
            () => {
                const out = [];
                const root = document.querySelector('#score-data');
                if (!root) return out;
                const children = Array.from(root.childNodes);
                let currentLeague = '';
                for (let i = 0; i < children.length; i++) {
                    const ch = children[i];
                    if (ch.nodeType === 1 && ch.tagName === 'H4') {
                        currentLeague = (ch.textContent || '').replace(/Standings?|Draw/i, '').trim();
                        continue;
                    }
                    if (ch.nodeType !== 1) continue;
                    if (ch.tagName !== 'A') continue;
                    const href = ch.getAttribute('href') || '';
                    if (!href.includes('/match/')) continue;
                    const matchId = (href.match(/\\/match\\/([^/?]+)/) || [])[1] || '';
                    const score = (ch.textContent || '').trim();
                    const cls = (ch.className || '');
                    const isFinished = cls.includes('fin');
                    let prevSpan = null;
                    for (let j = i - 1; j >= 0; j--) {
                        const n = children[j];
                        if (n.nodeType === 1 && n.tagName === 'SPAN') { prevSpan = n; break; }
                        if (n.nodeType === 1 && n.tagName === 'A' && (n.getAttribute('href') || '').includes('/match/')) break;
                    }
                    const prevText = prevSpan ? (prevSpan.textContent || '').trim() : '';
                    let minute = null;
                    let statusShort = 'NS';
                    const minMatch = prevText.match(/(\\d+)['′]/);
                    if (minMatch) {
                        minute = parseInt(minMatch[1]);
                        statusShort = minute > 45 ? '2H' : '1H';
                    } else if (/Half Time|^HT$/i.test(prevText)) {
                        statusShort = 'HT';
                    }
                    if (isFinished) statusShort = 'FT';
                    let homeName = '', awayName = '';
                    const prevSibling = ch.previousSibling;
                    if (prevSibling && prevSibling.nodeType === 3) {
                        const text = (prevSibling.textContent || '').trim();
                        const dashIdx = text.lastIndexOf(' - ');
                        if (dashIdx >= 0) {
                            homeName = text.substring(0, dashIdx).trim();
                            awayName = text.substring(dashIdx + 3).trim();
                        } else {
                            const parts = text.split(' - ');
                            if (parts.length === 2) { homeName = parts[0].trim(); awayName = parts[1].trim(); }
                            else { homeName = text; }
                        }
                    }
                    out.push({ matchId, homeName, awayName, score, isFinished, prevText, minute, statusShort, leagueRaw: currentLeague });
                }
                return out;
            }
        """)
        return items
    finally:
        if own_ctx is not None:
            try:
                own_ctx.close()
            except Exception:
                pass


def _normalize_match(raw: dict, day_offset: int, status_kind: str) -> Optional[dict]:
    """Raw mobile scrape sonucunu normalize et."""
    home = (raw.get("homeName", "") or "").strip()
    away = (raw.get("awayName", "") or "").strip()
    if not home or not away or home == away:
        return None
    # Temizlik
    home = re.sub(r'\s+\d+[-:]\d+\s*$', '', home).strip()
    away = re.sub(r'\s+\d+[-:]\d+\s*$', '', away).strip()
    home = re.sub(r'^\d+:\d+\s*', '', home).strip()
    away = re.sub(r'^\d+:\d+\s*', '', away).strip()
    if not home or not away:
        return None
    hg, ag = _parse_score(raw.get("score", ""))
    country = _extract_country(raw.get("leagueRaw", ""))
    league, round_name = _extract_league(raw.get("leagueRaw", ""), country)
    minute = raw.get("minute")
    status_short = raw.get("statusShort", "NS")
    # Status kind override
    if status_kind == "finished" and status_short not in ("FT",):
        if hg is not None:
            status_short = "FT"
    elif status_kind == "live":
        if status_short == "NS":
            status_short = "LIVE"
    elif status_kind == "upcoming":
        status_short = "NS"
    today = datetime.now().date()
    target_date = today + timedelta(days=day_offset)
    kickoff_iso = datetime.combine(target_date, datetime.min.time()).isoformat()
    return {
        "fixture_id": raw.get("matchId", ""),
        "home_name": home,
        "away_name": away,
        "home_score": hg,
        "away_score": ag,
        "status_short": status_short,
        "minute": minute,
        "kickoff": kickoff_iso,
        "is_live": (status_kind == "live"),
        "source": "flashscore",
        "league_name": league,
        "league_country": country,
        "league_round": round_name,
    }


def _get_match_detail_worker(browser, _wpage, match_id: str) -> Optional[dict]:
    """Worker thread icinde, ortak browser ile calisan match detail (tek greenlet)."""
    if not match_id:
        return None
    with _SCRAPER_LOCK:
        ctx = None
        page = None
        try:
            ctx = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                locale="tr-TR",
                viewport={"width": 1280, "height": 900},
            )
            page = ctx.new_page()
            base_url = f"https://www.flashscore.com/match/{match_id}/"
            page.goto(base_url, timeout=30000, wait_until="domcontentloaded")
            try:
                page.wait_for_selector(".duelParticipant", timeout=10000)
            except Exception:
                pass
            time.sleep(2)

            # Ortak bilgileri cek
            data = page.evaluate("""() => {
                const out = {};
                const txt = (el) => (el ? (el.textContent || '').trim() : '');
                const h1 = document.querySelector('h1');
                const h1txt = txt(h1);
                const m = h1txt.match(/(.+?)\\s+[vV]\\s+(.+?)\\s*(?:\\(|$)/);
                if (m) { out.home_team = m[1].trim(); out.away_team = m[2].trim(); }
                const scoreWrap = document.querySelector('.detailScore__wrapper, .fixedScore__wrapper, .detailScore, [class*="detailScore__wrapper"]');
                out.score_raw = txt(scoreWrap);
                const statusEl = document.querySelector('.detailScore__status, .fixedScore__status, [class*="detailScore__status"], [class*="fixedScore__status"]');
                out.status = txt(statusEl);
                const overline = document.querySelectorAll('[class*="wcl-scores-overline-03_Jdp91"], [class*="wcl-scores-overline-03"], [class*="scores-overline-03"]');
                const leagueParts = [];
                overline.forEach(el => leagueParts.push(txt(el)));
                out.league_overlines = leagueParts;
                const allOverlines = document.querySelectorAll('[class*="wcl-scores-overline"]');
                out.all_overlines = [];
                allOverlines.forEach(el => { out.all_overlines.push({cls: el.className, t: txt(el)}); });
                const dateEl = document.querySelector('.duelParticipant__startTime, [class*="startTime"]');
                out.kickoff = txt(dateEl);
                return out;
            }""")

            # Helper: show more tikla
            def click_show_more():
                try:
                    page.evaluate("""() => {
                        document.querySelectorAll('a, button, div, span').forEach(el => {
                            const t = (el.textContent || '').trim().toLowerCase();
                            if (t === 'show more' || t === 'daha fazla goster' || t === 'show all' || t === 'tumunu goster' || t === 'daha fazla bilgi') { el.click(); }
                        });
                    }""")
                    time.sleep(1)
                except Exception:
                    pass

            # Helper: sekmeye tikla
            def click_tab(tab_names):
                for name in tab_names:
                    try:
                        clicked = page.evaluate("""(tabName) => {
                            const tabs = document.querySelectorAll('a, button, div, span, li');
                            for (const el of tabs) {
                                const t = (el.textContent || '').trim().toUpperCase();
                                if (t === tabName.toUpperCase() || t.includes(tabName.toUpperCase())) { el.click(); return true; }
                            }
                            return false;
                        }""", name)
                        if clicked:
                            time.sleep(2)
                            click_show_more()
                            return True
                    except Exception:
                        pass
                return False

            # Helper: full text al
            def get_full_text(limit=15000):
                try:
                    return page.evaluate("(limit) => document.body.innerText.slice(0, limit)", limit)
                except Exception:
                    return ""

            # Helper: DOM'dan structured data cek
            def extract_dom_data():
                return page.evaluate("""() => {
                    const out = {};
                    const txt = (el) => (el ? (el.textContent || '').trim() : '');
                    const isNum = (s) => /^\\d+(\\.\\d+)?%?$/.test((s||'').trim());
                    const isRaw = (s) => /^\\(.+\\)$/.test((s||'').trim());
                    const SECTION_NAMES = ['TOP STATS','SHOTS','SHOTS ON TARGET','ATTACK','PASSES','DEFENSE','DUELS','GOALKEEPER','GOALKEEPING','DISCIPLINE','POSSESSION'];
                    const STAT_NAMES = ['Expected goals (xG)','Ball possession','Total shots','Shots on target','Shots off target','Big chances','Corner kicks','Yellow cards','Red cards','Expected assists (xA)','xG on target (xGOT)','Blocked shots','Shots inside the box','Shots outside the box','Hit the woodwork','Touches in opposition box','Accurate through passes','Offsides','Free kicks','Throw ins','Fouls','Duels won','Clearances','Interceptions','Errors leading to shot','Errors leading to goal','Goalkeeper saves','Goal kicks','xGOT faced','Goals prevented','Passes','Accurate passes','Tackles','Aerial duels won','Ground duels won','Possession','Long passes','Passes in final third','Crosses'];
                    const isStatName = (s) => STAT_NAMES.some(n => n.toLowerCase() === (s||'').trim().toLowerCase());
                    out.stats_sections = [];
                    const allEls = Array.from(document.querySelectorAll('div, section, span'));
                    const headers = [];
                    allEls.forEach(el => {
                        const t = txt(el);
                        if (!t || t.length > 40) return;
                        if (SECTION_NAMES.includes(t.toUpperCase())) {
                            const r = el.getBoundingClientRect();
                            if (r.width > 0 && r.height > 0) headers.push({el, name: t.toUpperCase(), top: r.top});
                        }
                    });
                    const seenH = new Set();
                    const uniqueHeaders = headers.filter(h => { const k = h.name + '|' + Math.round(h.top); if (seenH.has(k)) return false; seenH.add(k); return true; });
                    uniqueHeaders.forEach((h, idx) => {
                        const next = uniqueHeaders[idx+1];
                        const hTop = h.el.getBoundingClientRect().top;
                        const nextTop = next ? next.el.getBoundingClientRect().top : Infinity;
                        const rows = [];
                        allEls.forEach(el => {
                            const r = el.getBoundingClientRect();
                            if (r.top <= hTop + 5 || r.top >= nextTop - 5) return;
                            const t = txt(el);
                            if (!t || t.length > 80) return;
                            if (el.children.length > 0) {
                                const kids = Array.from(el.children);
                                const kidTxts = kids.map(k => txt(k));
                                if (kidTxts.length === 3 && isNum(kidTxts[0]) && isNum(kidTxts[2]) && !isNum(kidTxts[1]) && kidTxts[1].length > 1) {
                                    rows.push({name: kidTxts[1], home: kidTxts[0], away: kidTxts[2]});
                                } else if (kidTxts.length === 5 && isNum(kidTxts[0]) && isRaw(kidTxts[1]) && !isNum(kidTxts[2]) && isNum(kidTxts[3]) && isRaw(kidTxts[4])) {
                                    rows.push({name: kidTxts[2], home: kidTxts[0] + ' ' + kidTxts[1], away: kidTxts[3] + ' ' + kidTxts[4]});
                                } else if (kidTxts.length >= 2) {
                                    const nums = []; let nameLabel = '';
                                    for (const kt of kidTxts) { if (isNum(kt) || isRaw(kt)) nums.push(kt); else if (kt.length > 1 && !isNum(kt)) nameLabel = kt; }
                                    if (nums.length === 2 && nameLabel && (isStatName(nameLabel) || nameLabel.length > 3)) rows.push({name: nameLabel, home: nums[0], away: nums[1]});
                                }
                            }
                        });
                        const seenR = new Set();
                        const uniqRows = rows.filter(r => { if (seenR.has(r.name)) return false; seenR.add(r.name); return true; });
                        if (uniqRows.length > 0) out.stats_sections.push({name: h.name, rows: uniqRows});
                    });
                    // Events
                    out.events = [];
                    try {
                        const sels = ['[class*="smv__Participant"]','[class*="smvHomeParticipant"]','[class*="smvAwayParticipant"]','[class*="incident__"]','[class*="smv__Incident"]','.smvRow','[class*="smvRow"]','[class*="smv"]'];
                        document.querySelectorAll(sels.join(', ')).forEach(inc => {
                            const t = txt(inc);
                            if (!t || t.length < 3 || t.length > 200) return;
                            const m = t.match(/^(\\d{1,3})[\\'\\u2032]?\\s+(.+)/);
                            if (m && m[2].length > 1) out.events.push({minute: m[1], text: m[2].trim(), raw: t});
                        });
                        const seenEv = new Set();
                        out.events = out.events.filter(e => { const k = e.minute + '|' + e.text; if (seenEv.has(k)) return false; seenEv.add(k); return true; });
                    } catch(e) {}
                    // Info
                    out.info = [];
                    try {
                        const iSels = ['[class*="duelParticipant__info"]','[class*="matchInfo"]','[class*="matchInformation"]','[class*="infoSection"]','[class*="wcl-info"]','[class*="match-info"]'];
                        let infoSection = null;
                        for (const sel of iSels) { infoSection = document.querySelector(sel); if (infoSection) break; }
                        if (infoSection) {
                            infoSection.querySelectorAll('[class*="row"], [class*="item"], [class*="info"], [class*="detail"]').forEach(r => {
                                const t = txt(r);
                                if (t && t.includes(':')) {
                                    const parts = t.split(/:\\s*/);
                                    if (parts.length >= 2) {
                                        const lab = parts[0].trim(); const val = parts.slice(1).join(':').trim();
                                        if (lab && val && val.length < 100) out.info.push({label: lab.toUpperCase(), value: val});
                                    }
                                }
                            });
                        }
                        if (out.info.length === 0) {
                            const tLines = document.body.innerText.split('\\n').map(l => l.trim()).filter(l => l);
                            const pat = /^(REFEREE|VENUE|CAPACITY|ATTENDANCE|STADIUM|DATE|TIME|WEATHER|LOCATION|KICK.?OFF|HALFTIME|CITY|STADYUM)\\s*[:\\-]\\s*(.+)/i;
                            for (let i = 0; i < tLines.length; i++) { const m = tLines[i].match(pat); if (m && m[2].trim().length > 1 && m[2].trim().length < 100) out.info.push({label: m[1].toUpperCase(), value: m[2].trim()}); }
                        }
                    } catch(e) {}
                    // Lineups
                    out.lineups = [];
                    try {
                        document.querySelectorAll('[class*="playerName"], [class*="lineup__player"], [class*="smv__ParticipantName"]').forEach(p => { const t = txt(p); if (t && t.length > 1) out.lineups.push(t); });
                    } catch(e) {}
                    return out;
                }""")

            # === 1. SUMMARY SEKMESI (Events/Incidents) ===
            click_tab(['SUMMARY', 'OZET', 'MATCH REPORT', 'MAC RAPORU', 'SUM'])
            summary_dom = extract_dom_data()
            data['events'] = summary_dom.get('events', [])
            data['summary_text'] = get_full_text(5000)

            # === 2. STATS SEKMESI ===
            click_tab(['STATS', 'STATISTICS', 'ISTATISTIKLER', 'ISTATIST', 'STAT'])
            stats_dom = extract_dom_data()
            data['stats_sections'] = stats_dom.get('stats_sections', [])
            data['stats_text'] = get_full_text(10000)

            # === 3. LINEUPS SEKMESI ===
            click_tab(['LINEUPS', 'LINE-UPS', 'KADROLAR', 'ONBIRLER', 'LINEU'])
            lineups_dom = extract_dom_data()
            data['lineups'] = lineups_dom.get('lineups', [])
            data['lineups_text'] = get_full_text(10000)

            # === 4. PLAYER STATS SEKMESI ===
            click_tab(['PLAYER STATS', 'OYUNCU', 'PLAYER', 'PLAYER STATISTICS'])
            pstats_dom = extract_dom_data()
            data['player_stats_text'] = get_full_text(10000)

            # === 5. COMMENTARY SEKMESI ===
            click_tab(['COMMENTARY', 'YORUM', 'COMMENTAR', 'MATCH COMMENTARY'])
            data['commentary_text'] = get_full_text(10000)

            # === 6. MATCH INFORMATION SEKMESI ===
            click_tab(['MATCH INFORMATION', 'MAC BILGISI', 'INFO', 'MATCH INFO'])
            info_dom = extract_dom_data()
            data['info'] = info_dom.get('info', [])
            data['info_text'] = get_full_text(5000)

            # === 7. full_text ===
            data['full_text'] = get_full_text(15000)

            # === POST-PROCESSING ===
            if data.get("score_raw") and not data.get("home_score"):
                ms = re.match(r'(\d+)\s*[\-\u2011\u2013\u2014]\s*(\d+)', data["score_raw"])
                if ms:
                    data["home_score"] = ms.group(1)
                    data["away_score"] = ms.group(2)
            league_parts = data.get("league_overlines") or []
            if league_parts and not data.get("league_name"):
                if len(league_parts) >= 2:
                    data["league_country"] = league_parts[1]
                    if len(league_parts) >= 3:
                        lig_part = league_parts[2]
                        m_round = re.search(r'[\-\u2011\u2013\u2014]?\s*(ROUND|Group|Matchday|Week)\s*[\w\d]+', lig_part, re.IGNORECASE)
                        if m_round:
                            data["league_name"] = lig_part[:m_round.start()].strip()
                            data["round"] = m_round.group(0).strip()
                        else:
                            data["league_name"] = lig_part
                        if len(league_parts) >= 4:
                            data["round"] = league_parts[3]
            if data.get("status"):
                st = data["status"].upper()
                if st in ("FT", "FINISHED"): data["status"] = "FINISHED"
                elif st in ("HT", "HALFTIME", "HALF TIME"): data["status"] = "HT"
                elif st in ("LIVE", "1H", "2H"): data["status"] = st
                else: data["status"] = st
            if not data.get("kickoff"):
                txt = data.get("full_text", "")
                m_dt = re.search(r'(\d{2}\.\d{2}\.\d{4})\s+(\d{1,2}:\d{2})', txt)
                if m_dt: data["kickoff"] = m_dt.group(0)
            if not data.get("home_score") or not data.get("away_score"):
                txt = data.get("full_text", "")
                ms = re.search(r'(\d)\s*\n\s*[\-\u2011\u2013\u2014]\s*\n\s*(\d)\s*\n\s*(\w+)', txt)
                if ms:
                    data["home_score"] = ms.group(1)
                    data["away_score"] = ms.group(2)
                    if not data.get("status"): data["status"] = ms.group(3).upper()
                else:
                    ms2 = re.search(r'(\d)\s*[\-\u2011\u2013\u2014]\s*(\d)', txt)
                    if ms2:
                        data["home_score"] = ms2.group(1)
                        data["away_score"] = ms2.group(2)
            if not data.get("venue"):
                for item in (data.get("info") or []):
                    if item.get("label") in ("VENUE", "STADIUM", "STADYUM", "STADYU"):
                        data["venue"] = item["value"][:80]
                        break
            if not data.get("referee"):
                for item in (data.get("info") or []):
                    if item.get("label") in ("REFEREE", "HAKEM", "REFER"):
                        data["referee"] = item["value"][:80]
                        break
            if not data.get("league_name"):
                txt = data.get("full_text", "")
                m_lg = re.search(r'(?:FOOTBALL|TENNIS|BASKETBALL|HOCKEY|BASEBALL|VOLLEYBALL|HANDBALL|RUGBY\s+UNION|AMERICAN\s+FOOTBALL)\s*\n?\s*([A-ZCEGIOOU]{3,})\s*\n?\s*([A-Z][A-Za-z\s\.\-]+?(?:LEAGUE|CUP|CHAMPIONSHIP|DIVISION|LIGA|SERIE\s*[A-D]?|BUNDESLIGA|LIGUE\s*\d+|EREDIVISIE|PRIMEIRA|SUPER\s*LIG|CLASSIC|OPEN|TROPHY|TOURNAMENT|TOUR|FEDERATION|CONFERENCE|EUROPA|CHAMPIONS|FA\s*CUP|CARABAO|EFL\s*CUP|COPA|DFB\s*POKAL|QUALIFICATION|PLAY[\s\-]?OFFS?|KNOCKOUT|REGULAR\s*SEASON))', txt, re.IGNORECASE)
                if m_lg:
                    data["league_country"] = m_lg.group(1).strip()
                    data["league_name"] = m_lg.group(2).strip()
                    m_rd = re.search(r'[\-\u2011\u2013\u2014:]\s*(ROUND|Group|Matchday|Week)\s*[\w\d]+', txt)
                    if m_rd:
                        data["round"] = m_rd.group(0).strip(' -:')
            return data
        except Exception as exc:
            logger.error("[Flashscore] match detail %s hatasi: %s", match_id, exc)
            return None
        finally:
            try:
                if page is not None:
                    page.close()
            except Exception:
                pass
            try:
                if ctx is not None:
                    ctx.close()
            except Exception:
                pass


def get_match_detail(match_id: str) -> Optional[dict]:
    """Flashscore.com desktop'tan tek macin tum sekmelerinden veri ceker.

    Tum Playwright isi pw_worker'daki tek isci thread'inde calisir; bu, Flask
    request thread'i + arka plan 'Canli kontrol' thread'i ayni anda Playwright
    kullandiginda cikan 'greenlet.error' hatalarini onler.
    """
    if not match_id:
        return None
    from prediction.pw_worker import run_with_browser as _run_with_browser
    try:
        return _run_with_browser(_get_match_detail_worker, match_id, timeout=180)
    except Exception:
        return None

# Cache: kind -> {data, fetched_at}
_CACHE = {"live": {"data": None, "fetched_at": 0},
          "finished": {"data": None, "fetched_at": 0},
          "upcoming": {"data": None, "fetched_at": 0}}
_CACHE_TTL = {"live": 60, "finished": 3600, "upcoming": 3600}

# Disk cache: server restart oldugunde 7dk'lik scrape'i tekrarlatma.
# _CACHE'teki son veriyi diskten geri yukle.
_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tahminler", "fixtures_cache.json")


def _load_disk_cache():
    try:
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                if k in _CACHE and v and v.get("data") is not None:
                    _CACHE[k]["data"] = v["data"]
                    _CACHE[k]["fetched_at"] = v.get("fetched_at", 0)
    except Exception:
        pass


def _save_disk_cache():
    try:
        raw = {k: {"data": v["data"], "fetched_at": v["fetched_at"]} for k, v in _CACHE.items()}
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)
    except Exception:
        pass


# Module yuklenirken diskten geri yukle
_load_disk_cache()


def _fixtures_worker(browser, _page, kind: str) -> list:
    """Worker thread icinde, ortak browser ile calisan scrape (tek greenlet)."""
    out = []
    ctx = None
    page = None
    try:
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            locale="tr-TR",
            viewport={"width": 414, "height": 896},
            timezone_id="Europe/Istanbul",
            is_mobile=True,
            has_touch=True,
        )
        page = ctx.new_page()
        if kind == "live":
            raw = _scrape_mobile_day(day_offset=0, status=2, page=page)
            for r in raw:
                m = _normalize_match(r, day_offset=0, status_kind="live")
                if m:
                    out.append(m)
        elif kind == "finished":
            # Son 30 gunu tara (4 hafta) - bitmis maclarin tamamini almak icin
            for offset in range(-30, 1):
                try:
                    raw = _scrape_mobile_day(day_offset=offset, status=3, page=page)
                    cnt = 0
                    for r in raw:
                        m = _normalize_match(r, day_offset=offset, status_kind="finished")
                        if m:
                            out.append(m)
                            cnt += 1
                    logger.info("[Flashscore] finished d=%d: %d raw -> %d ok", offset, len(raw), cnt)
                except Exception as exc:
                    logger.warning("[Flashscore] finished d=%d hatasi: %s", offset, exc)
        elif kind == "upcoming":
            for offset in range(0, 15):
                raw = _scrape_mobile_day(day_offset=offset, status=1, page=page)
                for r in raw:
                    # Bitmis veya canli maclari hariç tut (sadece NS)
                    if r.get("isFinished"):
                        continue
                    mtxt = r.get("prevText", "")
                    if re.search(r"(\d+)['′]", mtxt) or re.search(r"Half Time", mtxt, re.IGNORECASE):
                        continue  # Canli
                    m = _normalize_match(r, day_offset=offset, status_kind="upcoming")
                    if m:
                        out.append(m)
        # Siralama
        if kind == "finished":
            out.sort(key=lambda x: x.get("kickoff") or "", reverse=True)
        elif kind == "upcoming":
            out.sort(key=lambda x: x.get("kickoff") or "")
        return out
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


def get_fixtures_flashscore(kind: str = "live", limit: int = 50, refresh: bool = False) -> list[dict]:
    """Flashscore mobile'dan canli/bitmis/gelecek maclari cek.

    Tum Playwright isi pw_worker'daki tek isci thread'inde calisir; bu, Flask
    request thread'i + arka plan 'Canli kontrol' thread'i ayni anda Playwright
    kullandiginda cikan 'greenlet.error' hatalarini onler.

    Args:
        kind: 'live' | 'finished' | 'upcoming'
        limit: maksimum dondurulecek mac sayisi
        refresh: True ise cache bypass edilip taze scrape yapilir

    Tarih araligi:
        - bitmis: bugunden 30 gun geriye (s=3, d=-30..0) - tum 1 aylik maclar
        - canli: sadece bugun (d=0, s=2)
        - gelecek: bugunden 14 gun ileri (d=0..14)
    """
    now = time.time()
    cache = _CACHE.get(kind, {"data": None, "fetched_at": 0})
    if not refresh and cache["data"] is not None and (now - cache["fetched_at"]) < _CACHE_TTL.get(kind, 1800):
        return cache["data"][:limit]
    # Lock ile tum scrape islemini seriye al (greenlet conflict onler)
    with _SCRAPER_LOCK:
        try:
            from prediction.pw_worker import run_with_browser as _run_with_browser
            if _run_with_browser is None:
                raise RuntimeError("pw_worker yuklu degil")
            out = _run_with_browser(_fixtures_worker, kind, timeout=900)
        except Exception as exc:
            logger.error("[Flashscore] %s scrape basarisiz: %s", kind, exc)
            return []
        try:
            # Disk'te zaten DOLU finished verisi varken kismi/sifir scrape sonucu
            # (site anlik erisim sorunu vs.) o DOLU veriyi eclmesin. Mevcut verinin
            # en az %60 kadari uretilebilirse guncelle, aksi halde eski veriyi koru.
            if kind == "finished":
                prev = []
                if os.path.exists(_CACHE_FILE):
                    try:
                        with open(_CACHE_FILE, "r", encoding="utf-8") as _f:
                            prev = ((json.load(_f).get("finished") or {}).get("data") or [])
                    except Exception:
                        prev = []
                if prev and len(out) < len(prev) * 0.6:
                    logger.warning("[Flashscore] finished scrape kismi (%d < %d %s) - disk'teki dolu veri korunuyor",
                                   len(out), len(prev), "x0.6")
                    cache["data"] = prev
                    cache["fetched_at"] = now
                    return prev[:limit]
            cache["data"] = out
            cache["fetched_at"] = now
            _save_disk_cache()
        except Exception:
            pass
    return out[:limit]


# CLI test
if __name__ == "__main__":
    import sys
    for k in ["live", "finished", "upcoming"]:
        print(f"\n=== {k.upper()} ===")
        try:
            items = get_fixtures_flashscore(kind=k, limit=10)
            print(f"  Toplam: {len(items)}")
            for m in items[:5]:
                score = f"{m['home_score'] if m['home_score'] is not None else '-'} - {m['away_score'] if m['away_score'] is not None else '-'}"
                ext = m['minute'] or m['kickoff'] or '?'
                lg = m['league_name']
                if m.get('league_round'):
                    lg += f" ({m['league_round']})"
                print(f"  [{m['status_short']:3s}] {m['home_name'][:25].ljust(25)} {score.ljust(8)} {m['away_name'][:25].ljust(25)} | {lg[:50]} ({m['league_country']})")
        except Exception as e:
            print(f"  HATA: {e}")
