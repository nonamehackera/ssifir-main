"""Football Prediction Dashboard - Flask Run."""

import sys
import os
import json
import pickle
import math
import time
import logging
import difflib
from functools import lru_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template, render_template_string, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
)
logger = logging.getLogger(__name__)

_web_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(_web_dir, "templates"), static_folder=os.path.join(_web_dir, "static"))
app.secret_key = os.urandom(24)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

# Tahminler klasoru ayari
TAHMINLER_DIR = os.path.join(os.path.dirname(_web_dir), "tahminler")
os.makedirs(TAHMINLER_DIR, exist_ok=True)
PREDICTIONS_FILE = os.path.join(TAHMINLER_DIR, "predictions.json")
MANUAL_RESULTS_FILE = os.path.join(TAHMINLER_DIR, "manual_results.json")

# ⚠️ ARAYUZ DEGISIKLIKLERI ANINDA GORUNSUN DIYE CACHE TAMAMEN KAPALI.
# Flask varsayilan olarak statik dosyalara (style.css / js) uzun sureli
# Cache-Control header'i ekler; tarayici da eski CSS/JS'i tutar ve
# "arayuzu degistim ama eski gorunum var" sorunu cikar. Bu engeller.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def _no_cache_headers(resp):
    """Her yanita no-cache header'i ekle (template + static)."""
    if resp and resp.headers:
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp

STATE = {
    "pipeline": None, "model_version": None, "loaded_at": None,
    "features": None, "matches": None, "model": None,
    "loading": True, "error": None, "metrics": None,
    "data_info": {},
}

TEAM_INDEX = {}
LEAGUE_TEAMS = {}
TEAM_ID_MAP = {}

_CACHE = {}


def _load_manual_results():
    if os.path.exists(MANUAL_RESULTS_FILE):
        try:
            with open(MANUAL_RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save_manual_result(coupon_id, home_goals, away_goals):
    results = _load_manual_results()
    total = home_goals + away_goals
    res_data = {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "total_goals": total,
        "result": "H" if home_goals > away_goals else ("D" if home_goals == away_goals else "A"),
        "btts": 1 if home_goals > 0 and away_goals > 0 else 0,
        "over25": 1 if total >= 3 else 0,
        "home_corners": None,
        "away_corners": None,
        "ht_home_goals": None,
        "ht_away_goals": None,
        "date": datetime.now().isoformat()[:10],
        "source": "manual"
    }
    results[coupon_id] = res_data
    with open(MANUAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Ayrıca match_score_fetcher önbelleğine yaz
    try:
        import prediction.match_score_fetcher as score_fetcher
        parts = str(coupon_id).split("-")
        if len(parts) >= 2:
            cid = f"{parts[0]}-{parts[1]}"
            score_fetcher.save_local_match_result(cid, res_data)
    except Exception:
        pass


def _load_predictions():
    """Tahminleri yukle. Eski tahminlere match_date backfill yap."""
    if os.path.exists(PREDICTIONS_FILE):
        preds = []
        try:
            with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                preds = json.load(f)
        except Exception:
            try:
                with open(PREDICTIONS_FILE, "r", encoding="cp1254") as f:
                    preds = json.load(f)
            except Exception:
                return []

        # Backfill: match_date yoksa timestamp'ten al
        changed = False
        for p in preds:
            if not p.get("match_date"):
                ts = p.get("timestamp", "")
                if ts:
                    try:
                        t = datetime.fromisoformat(ts.replace("Z", "+00:00").split("+")[0])
                        p["match_date"] = t.strftime("%Y-%m-%d")
                        changed = True
                    except Exception:
                        pass

        if changed:
            try:
                _save_predictions(preds)
            except Exception:
                pass
        return preds
    return []


def _save_predictions(preds):
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)


def _build_prediction_summary(pred):
    """Tahmin sonucundan net ozet uret: kazanan, gol, korner, btts, cift sans."""
    summary = {}
    r = pred.get("result", {})

    winner = r.get("winner", "?")
    win_prob = r.get("winner_prob", 0)
    summary["kazanan"] = f"{winner} (%{win_prob})"

    goals = r.get("total_goals", {})
    o15 = goals.get("over15", 0)
    o25 = goals.get("over25", 0)
    o35 = goals.get("over35", 0)
    summary["toplam_gol_15"] = f"Ust 1.5 (%{o15})" if o15 >= 50 else f"Alt 1.5 (%{100-o15:.0f})"
    summary["toplam_gol_25"] = f"Ust 2.5 (%{o25})" if o25 >= 50 else f"Alt 2.5 (%{100-o25:.0f})"
    summary["toplam_gol_35"] = f"Ust 3.5 (%{o35})" if o35 >= 50 else f"Alt 3.5 (%{100-o35:.0f})"

    corners = r.get("corners", {})
    cor75 = corners.get("over75", 0)
    cor85 = corners.get("over85", 0)
    cor95 = corners.get("over95", 0)
    summary["toplam_korner_75"] = f"Ust 7.5 (%{cor75})" if cor75 >= 50 else f"Alt 7.5 (%{100-cor75:.0f})"
    summary["toplam_korner_85"] = f"Ust 8.5 (%{cor85})" if cor85 >= 50 else f"Alt 8.5 (%{100-cor85:.0f})"
    summary["toplam_korner_95"] = f"Ust 9.5 (%{cor95})" if cor95 >= 50 else f"Alt 9.5 (%{100-cor95:.0f})"

    btts_yes = r.get("btts_yes", 0)
    btts_no = r.get("btts_no", 0)
    if btts_yes >= btts_no:
        summary["btts"] = f"Var (%{btts_yes})"
    else:
        summary["btts"] = f"Yok (%{btts_no})"

    dc = r.get("double_chance", {})
    d1x = dc.get("1X", 0)
    dx2 = dc.get("X2", 0)
    d12 = dc.get("12", 0)
    best_dc = max(d1x, dx2, d12)
    if best_dc == d12:
        summary["cift_sans"] = f"12 (%{d12})"
    elif best_dc == d1x:
        summary["cift_sans"] = f"1X (%{d1x})"
    else:
        summary["cift_sans"] = f"X2 (%{dx2})"

    return summary


def _add_prediction(pred):
    pred["prediction_summary"] = _build_prediction_summary(pred)

    # match_date ekle: features parquet'ten mac tarihini bul
    if not pred.get("match_date"):
        feat = STATE.get("features")
        hid = pred.get("home_id")
        aid = pred.get("away_id")
        if feat is not None and hid and aid:
            try:
                match = feat[
                    ((feat["home_team_id"] == hid) & (feat["away_team_id"] == aid)) |
                    ((feat["home_team_id"] == aid) & (feat["away_team_id"] == hid))
                ]
                if len(match) > 0:
                    last_match = match.iloc[-1]
                    md = last_match.get("date")
                    if md is not None:
                        pred["match_date"] = str(md)[:10] if hasattr(md, "strftime") else str(md)
            except Exception:
                pass

    # Eger hala match_date yoksa, tahmin timestamp'ini kullan
    # (gelecekteki maclar icin - tahmin ne zaman yapildiysa o tarih)
    if not pred.get("match_date"):
        ts = pred.get("timestamp", "")
        if ts:
            try:
                t = datetime.fromisoformat(ts.replace("Z", "+00:00").split("+")[0])
                pred["match_date"] = t.strftime("%Y-%m-%d")
            except Exception:
                pass

    preds = _load_predictions()
    # Ayni mac (home_id + away_id) tekrar tahmin edildiyse eski kaydi kaldir
    hid = pred.get("home_id")
    aid = pred.get("away_id")
    if hid is not None and aid is not None:
        preds = [p for p in preds if not (p.get("home_id") == hid and p.get("away_id") == aid)]
    preds.insert(0, pred)
    if len(preds) > 200:
        preds = preds[:200]
    _save_predictions(preds)


def _delete_prediction(pred_id):
    preds = _load_predictions()
    target = next((p for p in preds if p.get("id") == pred_id), None)
    if target and target.get("home_id") and target.get("away_id"):
        hid, aid = target["home_id"], target["away_id"]
        fixture_key = f"{min(hid, aid)}-{max(hid, aid)}"
        preds = [p for p in preds if not (p.get("home_id") and p.get("away_id") and f"{min(p['home_id'], p['away_id'])}-{max(p['home_id'], p['away_id'])}" == fixture_key)]
    else:
        preds = [p for p in preds if p.get("id") != pred_id]
    _save_predictions(preds)


def _clear_cache():
    _CACHE.clear()


def _memo(fn):
    """Basit sonuc cache'i: veri yenilendiginde _clear_cache ile temizlenir."""
    def wrapper(*args, **kwargs):
        def _hashable(v):
            if isinstance(v, (list, tuple)):
                return tuple(_hashable(x) for x in v)
            if isinstance(v, dict):
                return tuple((_hashable(k), _hashable(val)) for k, val in sorted(v.items()))
            return v
        key = (fn.__name__,) + tuple(_hashable(a) for a in args) + tuple((k, _hashable(v)) for k, v in sorted(kwargs.items()))
        if key not in _CACHE:
            _CACHE[key] = fn(*args, **kwargs)
        return _CACHE[key]
    return wrapper


def _normalize_team_name(name):
    if not name:
        return ""
    # Turkce karakterleri normalize et (kucuk harf + ascii)
    rep = {'ı': 'i', 'i': 'i', 'ö': 'o', 'o': 'o', 'ü': 'u', 'u': 'u',
           'ş': 's', 's': 's', 'ç': 'c', 'c': 'c', 'ğ': 'g', 'g': 'g'}
    n = str(name).lower().strip()
    n = ''.join(rep.get(ch, ch) for ch in n)
    # klup eklerini temizle
    for s in [' fc', ' cf', ' ac', ' sc', ' oss', ' fk', ' sk', ' bk', ' us',
              ' as', ' srl', ' ssd', ' aş', ' k', ' spor', ' belediyespor',
              ' belediye', ' genclik', ' gençlik', ' genç', ' genc',
              ' futbol kulubu', ' futbol kulübü', ' kultur', ' kultur', 'spor kulubu']:
        n = n.replace(s, '')
    n = n.replace('afc ', '').replace(' cf ', ' ').replace(' fc ', ' ')
    n = n.replace('.', '').replace("'", '').replace('-', ' ').replace('&', 'and')
    n = n.replace('  ', ' ')
    n = ' '.join(n.split())
    return n


# Normalize edilmis ismi Turkce karakterli duzgun isme cevir (JSON bozuk oldugu icin)
_TURKISH_FIX = {
    'fenerbahce': 'Fenerbahçe', 'galatasaray': 'Galatasaray', 'besiktas': 'Beşiktaş',
    'trabzonspor': 'Trabzonspor', 'baskansehir': 'Başakşehir', 'basaksehir': 'Başakşehir',
    'istanbul basaksehir': 'İstanbul Başakşehir', 'kasimpasa': 'Kasımpaşa',
    'konyaspor': 'Konyaspor', 'antalyaspor': 'Antalyaspor', 'alanyaspor': 'Alanyaspor',
    'kayserispor': 'Kayserispor', 'sivasspor': 'Sivasspor', 'rizespor': 'Çaykur Rizespor',
    'rizespor': 'Çaykur Rizespor', 'goztep': 'Göztepe', 'goztepe': 'Göztepe',
    'gaziantep': 'Gaziantep FK', 'genclerbirligi': 'Gençlerbirliği',
    'genclerbirligi': 'Gençlerbirliği', 'karagumruk': 'Fatih Karagümrük',
    'karagümrük': 'Fatih Karagümrük', 'hatayspor': 'Hatayspor',
    'samsunspor': 'Samsunspor', 'bursaspor': 'Bursaspor', 'ankaragucu': 'Ankaragücü',
    'malatyaspor': 'Yeni Malatyaspor', 'akhisar belediyespor': 'Akhisar Belediyespor',
    'akhisar belediye': 'Akhisar Belediyespor', 'akhisar': 'Akhisar Belediyespor',
    'eyupspor': 'Eyüpspor', 'eyup': 'Eyüpspor', 'giresunspor': 'Giresunspor',
    'adalet demirspor': 'Adana Demirspor', 'adana demirspor': 'Adana Demirspor',
    'osmanlispor': 'Osmanlıspor', 'karagumruk': 'Fatih Karagümrük',
}


def _title_turkish(s):
    """Turkce karakterlere saygili Title Case."""
    if not s:
        return s
    # ozel kisaltmalar
    upper = {'fk', 'sk', 'bk', 'as', 'us', 'cf', 'ac', 'sc', 'k', 'a', 'b',
             'tv', 'rfc', 'cd', 'fc'}
    out = []
    for i, w in enumerate(str(s).split(' ')):
        wl = w.lower()
        if wl in upper and len(wl) <= 3:
            out.append(wl.upper())
        else:
            out.append(w[:1].upper() + w[1:] if w else w)
    return ' '.join(out)


_TURKISH_DISPLAY = None


def _load_turkish_display():
    """team_id_to_name.json'dan duzgun Turkce isimleri (display) topla."""
    global _TURKISH_DISPLAY
    if _TURKISH_DISPLAY is not None:
        return _TURKISH_DISPLAY
    m = {}
    path = os.path.join(os.path.dirname(__file__), "..", "data", "gold", "team_id_to_name.json")
    try:
        with open(path) as fh:
            raw = json.load(fh)
        for k, v in raw.items():
            m[str(k)] = str(v)
    except Exception:
        pass
    _TURKISH_DISPLAY = m
    return m


def _display_name(tid, fallback):
    """Orjinal Turkce ismi (team_id_to_name'den) dondur, yoksa normalize et."""
    disp = _load_turkish_display()
    name = disp.get(str(tid))
    if name and len(name.strip()) > 1:
        cand = name.strip()
    else:
        cand = fallback
    # Normalize edip Turkce karakterli duzgun isme cevir
    norm = _normalize_team_name(cand)
    if norm in _TURKISH_FIX:
        return _TURKISH_FIX[norm]
    # Turkce karakteri bozuk ama bilinen takim mi?
    if cand and cand.lower() in _TURKISH_FIX:
        return _TURKISH_FIX[cand.lower()]
    return cand


_AVATAR_COLORS = [
    "#22c55e", "#06b6d4", "#3b82f6", "#a855f7", "#f59e0b",
    "#ef4444", "#ec4899", "#14b8a6", "#f97316", "#8b5cf6",
]


def _avatar_for(name, tid):
    """Takim ismi -> bas harfler + deterministik renk."""
    clean = str(name).strip()
    initials = "".join(w[0] for w in clean.split()[:2] if w and w[0].isalnum()).upper()
    if not initials:
        initials = str(tid)[:2]
    initials = initials[:2]
    color = _AVATAR_COLORS[int(tid or 0) % len(_AVATAR_COLORS)]
    return {"initials": initials, "color": color}


# Canonical team names mapping (normalized -> display name)
_CANONICAL_NAMES = {
    'man city': 'Manchester City', 'manchester city': 'Manchester City',
    'man united': 'Manchester United', 'manchester united': 'Manchester United',
    'man utd': 'Manchester United',
    'tottenham': 'Tottenham', 'tottenham hotspur': 'Tottenham',
    'west ham': 'West Ham', 'west ham united': 'West Ham',
    'newcastle': 'Newcastle', 'newcastle united': 'Newcastle',
    'brighton': 'Brighton', 'brighton and hove albion': 'Brighton',
    'wolves': 'Wolverhampton', 'wolverhampton': 'Wolverhampton',
    'wolverhampton wanderers': 'Wolverhampton',
    'nottm forest': 'Nottingham Forest', 'nottingham forest': 'Nottingham Forest',
    'sheffield utd': 'Sheffield United', 'sheffield united': 'Sheffield United',
    'boro': 'Middlesbrough', 'middlesbrough': 'Middlesbrough',
    'brentford': 'Brentford', 'brentford fc': 'Brentford',
    'bournemouth': 'Bournemouth', 'afc bournemouth': 'Bournemouth',
    'luton': 'Luton', 'luton town': 'Luton',
    'burnley': 'Burnley', 'burnley fc': 'Burnley',
    'blackburn': 'Blackburn', 'blackburn rovers': 'Blackburn',
    'bolton': 'Bolton', 'bolton wanderers': 'Bolton',
    'sunderland': 'Sunderland', 'sunderland afc': 'Sunderland',
    'ipswich': 'Ipswich', 'ipswich town': 'Ipswich',
    'norwich': 'Norwich', 'norwich city': 'Norwich',
    'coventry': 'Coventry', 'coventry city': 'Coventry',
    'swansea': 'Swansea', 'swansea city': 'Swansea',
    'cardiff': 'Cardiff', 'cardiff city': 'Cardiff',
    'blackpool': 'Blackpool', 'blackpool fc': 'Blackpool',
    'qpr': 'QPR', "queens park rangers": 'QPR',
    'preston': 'Preston', 'preston north end': 'Preston',
    'hull': 'Hull', 'hull city': 'Hull',
    'stoke': 'Stoke', 'stoke city': 'Stoke',
    'swindon': 'Swindon', 'swindon town': 'Swindon',
    'barcelona': 'Barcelona', 'barca': 'Barcelona',
    'real madrid': 'Real Madrid', 'real madrid cf': 'Real Madrid',
    'atletico madrid': 'Atletico Madrid', 'atletico de madrid': 'Atletico Madrid',
    'bayern munich': 'Bayern Munich', 'bayern munchen': 'Bayern Munich',
    'bayern': 'Bayern Munich',
    'borussia dortmund': 'Borussia Dortmund', 'bvb': 'Borussia Dortmund',
    'psv': 'PSV', 'psv eindhoven': 'PSV',
    'ajax': 'Ajax', 'ajax amsterdam': 'Ajax',
    'juventus': 'Juventus', 'juve': 'Juventus',
    'ac milan': 'AC Milan', 'milan': 'AC Milan',
    'inter milan': 'Inter Milan', 'inter': 'Inter Milan', 'inter cf': 'Inter Milan',
    'galatasaray': 'Galatasaray', 'gs': 'Galatasaray',
    'fenerbahce': 'Fenerbahce', 'fb': 'Fenerbahce',
    'besiktas': 'Besiktas', 'bjk': 'Besiktas',
    'trabzonspor': 'Trabzonspor', 'ts': 'Trabzonspor',
    'basaksehir': 'Istanbul Basaksehir', 'istanbul basaksehir': 'Istanbul Basaksehir',
    'salzburg': 'RB Salzburg', 'fc red bull salzburg': 'RB Salzburg',
    'celtic': 'Celtic', 'celtic fc': 'Celtic',
    'rangers': 'Rangers', 'rangers fc': 'Rangers',
    'porto': 'Porto', 'fc porto': 'Porto',
    'benfica': 'Benfica', 'sl benfica': 'Benfica',
    'sporting': 'Sporting CP', 'sporting cp': 'Sporting CP',
    'olympique lyonnais': 'Lyon', 'lyon': 'Lyon',
    'olympique marseille': 'Marseille', 'marseille': 'Marseille',
    'paris saint-germain': 'PSG', 'psg': 'PSG', 'paris sg': 'PSG',
    'lille': 'Lille', 'osc lille': 'Lille',
    'monaco': 'Monaco', 'as monaco': 'Monaco',
    'leverkusen': 'Bayer Leverkusen', 'bayer leverkusen': 'Bayer Leverkusen',
    'leipzig': 'RB Leipzig', 'rb leipzig': 'RB Leipzig',
    'napoli': 'Napoli', 'ssc napoli': 'Napoli',
    'roma': 'Roma', 'as roma': 'Roma', 'as roma': 'Roma',
    'lazio': 'Lazio', 'ss lazio': 'Lazio',
    'fiorentina': 'Fiorentina', 'acf fiorentina': 'Fiorentina',
    'atalanta': 'Atalanta', 'atalanta bc': 'Atalanta',
    'torino': 'Torino', 'fc torino': 'Torino',
    'genoa': 'Genoa', 'genoa cfc': 'Genoa',
    'real sociedad': 'Real Sociedad', 'real sociedad': 'Real Sociedad',
    'real betis': 'Real Betis', 'real betis balompie': 'Real Betis',
    'villarreal': 'Villarreal', 'villarreal cf': 'Villarreal',
    'athletic bilbao': 'Athletic Bilbao', 'athletic club': 'Athletic Bilbao',
    'sevilla': 'Sevilla', 'sevilla fc': 'Sevilla',
    'valencia': 'Valencia', 'valencia cf': 'Valencia',
    'celta vigo': 'Celta Vigo', 'rc celta': 'Celta Vigo',
    'getafe': 'Getafe', 'getafe cf': 'Getafe',
    'las palmas': 'Las Palmas', 'ud las palmas': 'Las Palmas',
    'alaves': 'Alaves', 'deportivo alaves': 'Alaves',
    'mallorca': 'Mallorca', 'rcd mallorca': 'Mallorca',
    'osasuna': 'Osasuna', 'ca osasuna': 'Osasuna',
    'rayo vallecano': 'Rayo Vallecano', 'rayo': 'Rayo Vallecano',
    'leganes': 'Leganes', 'cd leganes': 'Leganes',
    'espanyol': 'Espanyol', 'rcd espanyol': 'Espanyol',
    'granada': 'Granada', 'granada cf': 'Granada',
    'cadiz': 'Cadiz', 'cadiz cf': 'Cadiz',
    'elche': 'Elche', 'elche cf': 'Elche',
    'almeria': 'Almeria', 'ud almeria': 'Almeria',
    'freiburg': 'Freiburg', 'sc freiburg': 'Freiburg',
    'stuttgart': 'Stuttgart', 'vfb stuttgart': 'Stuttgart',
    'hannover': 'Hannover', 'hannover 96': 'Hannover',
    'hamburg': 'Hamburg', 'hamburger sv': 'Hamburg',
    'koln': 'Koln', '1 fc koln': 'Koln',
    'werder bremen': 'Werder Bremen', 'werder': 'Werder Bremen',
    'eintracht frankfurt': 'Eintracht Frankfurt', 'eintracht': 'Eintracht Frankfurt',
    'gladbach': 'Monchengladbach', 'borussia monchengladbach': 'Monchengladbach',
    'mainz': 'Mainz', 'mainz 05': 'Mainz',
    'augsburg': 'Augsburg', 'fc augsburg': 'Augsburg',
    'hoffenheim': 'Hoffenheim', 'tsg 1899 hoffenheim': 'Hoffenheim',
    'heidenheim': 'Heidenheim', '1 fc heidenheim': 'Heidenheim',
    'darmstadt': 'Darmstadt', 'sv darmstadt 98': 'Darmstadt',
    'nurnberg': 'Nurnberg', '1 fc nurnberg': 'Nurnberg',
    'holstein kiel': 'Holstein Kiel', 'holstein': 'Holstein Kiel',
    'parma': 'Parma', 'parma calcio': 'Parma',
    'lecce': 'Lecce', 'us lecce': 'Lecce',
    'udinese': 'Udinese', 'udinese calcio': 'Udinese',
    'cagliari': 'Cagliari', 'cagliari calcio': 'Cagliari',
    'empoli': 'Empoli', 'empoli fc': 'Empoli',
    'sassuolo': 'Sassuolo', 'us sassuolo': 'Sassuolo',
    'frosinone': 'Frosinone', 'frosinone calcio': 'Frosinone',
    'venezia': 'Venezia', 'venezia fc': 'Venezia',
    'monza': 'Monza', 'ac monza': 'Monza',
    'salernitana': 'Salernitana', 'us salernitana': 'Salernitana',
    'lens': 'RC Lens', 'rc lens': 'RC Lens',
    'nice': 'OGC Nice', 'ogc nice': 'OGC Nice',
    'rennes': 'Rennes', 'stade rennais': 'Rennes',
    'montpellier': 'Montpellier', 'montpellier hsc': 'Montpellier',
    'nantes': 'Nantes', 'fc nantes': 'Nantes',
    'toulouse': 'Toulouse', 'toulouse fc': 'Toulouse',
    'strasbourg': 'Strasbourg', 'rc strasbourg': 'Strasbourg',
    'reims': 'Reims', 'stade de reims': 'Reims',
    'le havre': 'Le Havre', 'le havre ac': 'Le Havre',
    'lorient': 'Lorient', 'fc lorient': 'Lorient',
    'clermont': 'Clermont', 'clermont foot': 'Clermont',
    'metz': 'Metz', 'fc metz': 'Metz',
    'auxerre': 'Auxerre', 'aj auxerre': 'Auxerre',
    'ajaccio': 'Ajaccio', 'ac ajaccio': 'Ajaccio',
    'troyes': 'Troyes', 'estac troyes': 'Troyes',
    'saint-etienne': 'Saint-Etienne', 'saint etienne': 'Saint-Etienne',
    'leyton orient': 'Leyton Orient',
    'chesterfield': 'Chesterfield',
    'notts county': 'Notts County',
    'arbroath': 'Arbroath',
    'bristol rovers': 'Bristol Rovers', 'bristol rvs': 'Bristol Rovers',
    'southend': 'Southend', 'southend united': 'Southend',
    'milton keynes dons': 'MK Dons', 'mk dons': 'MK Dons',
    'wigan': 'Wigan', 'wigan athletic': 'Wigan',
    'barnsley': 'Barnsley',
    'charlton': 'Charlton', 'charlton athletic': 'Charlton',
    'reading': 'Reading',
    'oxford': 'Oxford', 'oxford united': 'Oxford',
    'cambridge': 'Cambridge', 'cambridge united': 'Cambridge',
    'shrewsbury': 'Shrewsbury', 'shrewsbury town': 'Shrewsbury',
    'cheltenham': 'Cheltenham', 'cheltenham town': 'Cheltenham',
    'birmingham': 'Birmingham', 'birmingham city': 'Birmingham',
    'derby': 'Derby', 'derby county': 'Derby',
    'nottingham': 'Nottingham Forest',
    'portsmouth': 'Portsmouth',
    'plymouth': 'Plymouth', 'plymouth argyle': 'Plymouth',
    'wrexham': 'Wrexham',
    'stockport': 'Stockport', 'stockport county': 'Stockport',
    'barnet': 'Barnet',
    'bristol city': 'Bristol City',
    'watford': 'Watford',
    'fulham': 'Fulham',
    'palace': 'Crystal Palace', 'crystal palace': 'Crystal Palace',
    'everton': 'Everton',
    'wolves': 'Wolverhampton',
    'liverpool': 'Liverpool',
    'arsenal': 'Arsenal',
    'chelsea': 'Chelsea',
    'leicester': 'Leicester', 'leicester city': 'Leicester',
    'leeds': 'Leeds', 'leeds united': 'Leeds',
    'sheffield wed': 'Sheffield Wednesday', 'sheffield wednesday': 'Sheffield Wednesday',
    'huddersfield': 'Huddersfield', 'huddersfield town': 'Huddersfield',
}


def _get_canonical_name(name):
    norm = _normalize_team_name(name)
    if norm in _CANONICAL_NAMES:
        return _CANONICAL_NAMES[norm]
    return name.strip()


def _build_canonical_team_index(feat, name_map):
    raw = {}
    id_to_key = {}
    all_tids = pd.unique(feat[["home_team_id", "away_team_id"]].values.ravel())
    all_tids = [t for t in all_tids if pd.notna(t)]
    # En yeni sezon
    all_sn = feat["season"].map(_season_num)
    data_newest_season = int(all_sn.max()) if len(all_sn) else 0
    # Performans icin groupby (O(n) instead of O(n^2))
    try:
        home_g = feat.groupby("home_team_id")
        away_g = feat.groupby("away_team_id")
        use_groupby = True
    except Exception:
        use_groupby = False
    for tid in all_tids:
        if use_groupby:
            try:
                home_s = home_g.get_group(tid) if tid in home_g.groups else pd.DataFrame()
            except Exception:
                home_s = feat[feat["home_team_id"] == tid]
            try:
                away_s = away_g.get_group(tid) if tid in away_g.groups else pd.DataFrame()
            except Exception:
                away_s = feat[feat["away_team_id"] == tid]
        else:
            home_s = feat[feat["home_team_id"] == tid]
            away_s = feat[feat["away_team_id"] == tid]
        s = pd.concat([home_s, away_s]).sort_values("date")
        if len(s) == 0:
            continue
        sn = s["season"].map(_season_num)
        max_sn = int(sn.max())
        # GUNCEL LIG = en yuksek (en yeni) sezonun ligi
        last_season = s[sn == max_sn]
        ls_leagues = last_season["league"].value_counts()
        raw_league = str(ls_leagues.idxmax() if len(ls_leagues) else s.iloc[-1].get("league", "?"))
        last = s.iloc[-1]
        # Form: sadece en yeni sezondan
        recent_season = s[sn == max_sn].sort_values("date").tail(10)
        if len(recent_season) == 0:
            recent_season = s.tail(10)
        recent = recent_season
        raw_name = str(name_map.get(str(tid), str(tid)))
        canonical = _get_canonical_name(raw_name)
        # Orjinal Turkce ismi tercih et (Akhisar Belediyespor vs Akhisar Belediye birlesir)
        canonical = _display_name(tid, canonical)

        def _team_result(row):
            is_home = (str(row["home_team_id"]) == str(tid))
            r = row["result"]
            if r == "D":
                return "D"
            if (is_home and r == "H") or (not is_home and r == "A"):
                return "W"
            return "L"

        form_results = [_team_result(row) for _, row in recent.iterrows()]

        home_recent = home_s.tail(10) if len(home_s) > 0 else pd.DataFrame()
        away_recent = away_s.tail(10) if len(away_s) > 0 else pd.DataFrame()
        avg_goals = 0
        avg_conceded = 0
        if len(home_recent) > 0:
            avg_goals += home_recent["home_goals"].mean()
            avg_conceded += home_recent["away_goals"].mean()
        if len(away_recent) > 0:
            avg_goals += away_recent["away_goals"].mean()
            avg_conceded += away_recent["home_goals"].mean()
        n_sources = (1 if len(home_recent) > 0 else 0) + (1 if len(away_recent) > 0 else 0)
        if n_sources > 0:
            avg_goals = round(avg_goals / n_sources, 2)
            avg_conceded = round(avg_conceded / n_sources, 2)

        is_current = (max_sn >= data_newest_season)
        entry = {
            "id": int(tid),
            "name": canonical,
            "raw_name": raw_name,
            "league": _resolve_league(raw_league),
            "league_code": raw_league,
            "elo": round(float(last.get("home_elo", 1500))),
            "attack_elo": round(float(last.get("home_attack_elo", 1500))),
            "defence_elo": round(float(last.get("home_defence_elo", 1500))),
            "n_matches": int(len(s)),
            "form": "".join(form_results[-5:]) if form_results else "?????",
            "avg_goals_scored": round(float(avg_goals), 2),
            "avg_goals_conceded": round(float(avg_conceded), 2),
            "last_date": str(last.get("date", ""))[:10] if pd.notna(last.get("date")) else "",
            "current_data": bool(is_current),
            "newest_season": int(max_sn),
        }
        entry["avatar"] = _avatar_for(canonical, int(tid))
        entry["norm"] = _normalize_team_name(canonical)
        # Dedupe anahtari: normalize isim + ulke (ayni takim farkli lig kodlarinda
        # tekrar etmesin: takim ayni ligde kalir)
        resolved_league = _resolve_league(raw_league)
        country = resolved_league.split(" ", 1)[0] if resolved_league else "?"
        norm_key = _normalize_team_name(canonical)
        key = (norm_key, country)
        id_to_key[int(tid)] = key
        if key not in raw:
            raw[key] = entry
        else:
            existing = raw[key]
            if entry["n_matches"] > existing["n_matches"]:
                entry["id"] = existing["id"]
                raw[key] = entry
            elif entry["elo"] > existing["elo"]:
                existing["elo"] = entry["elo"]
            existing["n_matches"] = max(existing["n_matches"], entry["n_matches"])

    return raw, id_to_key


def _load_league_names():
    """canonical.py'deki tum lig isimlerini tek sozlukte topla."""
    from ingestion.football_data.canonical import FD_LEAGUES, HF_LEAGUE_NAMES
    m = {}
    m.update(FD_LEAGUES)
    for k, v in HF_LEAGUE_NAMES.items():
        m[f"HF_{k}"] = v
    xg = {
        "E0": "England Premier League", "E1": "England Championship",
        "E2": "England League One", "E3": "England League Two",
        "EC": "England Conference", "SC0": "Scotland Premiership",
        "SC1": "Scotland Championship", "SC2": "Scotland League One",
        "SC3": "Scotland League Two", "SP1": "Spain LaLiga", "SP2": "Spain Segunda Division",
        "I1": "Italy Serie A", "I2": "Italy Serie B", "D1": "Germany Bundesliga",
        "D2": "Germany 2. Bundesliga", "F1": "France Ligue 1", "F2": "France Ligue 2",
        "N1": "Netherlands Eredivisie", "P1": "Portugal Primeira Liga",
        "B1": "Belgium Pro League", "T1": "Turkey Super Lig", "G1": "Greece Super League",
        "ARG": "Argentina Liga Profesional", "BRA": "Brazil Serie A",
        "USA": "USA MLS", "MEX": "Mexico Liga MX",
        "JAP": "Japan J1 League", "CHN": "China Super League",
        "ROM": "Romania SuperLiga", "POL": "Poland Ekstraklasa",
        "SWE": "Sweden Allsvenskan", "NOR": "Norway Eliteserien",
        "RUS": "Russia Premier League", "DEN": "Denmark Superliga",
        "IRL": "Ireland Premier Division", "FIN": "Finland Veikkausliiga",
        "AUT": "Austria Bundesliga", "SUI": "Switzerland Super League",
    }
    for k, v in xg.items():
        m[f"XG_{k}"] = v
    us = {"epl": "England Premier League", "la_liga": "Spain LaLiga",
          "bundesliga": "Germany Bundesliga", "serie_a": "Italy Serie A",
          "ligue_1": "France Ligue 1", "rpl": "Russia Premier League"}
    for k, v in us.items():
        m[f"US_{k}"] = v

    sb = {
        2: "England Premier League", 7: "France Ligue 1", 9: "Germany Bundesliga",
        11: "Spain LaLiga", 12: "Italy Serie A", 16: "Champions League",
        22: "Brazil Serie A", 32: "England Premier League", 35: "Europa League",
        37: "England Women Super League", 43: "FIFA World Cup",
        44: "USA MLS", 49: "USA NWSL", 53: "UEFA Women Euro",
        55: "UEFA Euro", 72: "Women World Cup", 81: "Argentina Liga Profesional",
        87: "Spain Copa del Rey", 116: "USA North American League",
        131: "Italy Serie A Women", 135: "Germany Frauen Bundesliga",
        182: "Spain Liga F", 223: "Copa America", 1238: "India Super League",
        1267: "Africa Cup of Nations",
        3: "England League One", 4: "England League Two", 5: "England National League",
        19: "Netherlands Eredivisie", 26: "Brazil Serie B", 29: "Italy Serie B",
        33: "France Ligue 2", 39: "Champions League Qualification",
        40: "UEFA Europa Conference League", 41: "UEFA Nations League",
        52: "UEFA Super Cup", 73: "FIFA Club World Cup", 75: "England FA Cup",
        76: "England EFL Cup", 88: "Spain Segunda Division", 94: "Germany 2. Bundesliga",
        98: "Japan J1 League", 101: "Mexico Liga MX", 104: "USA MLS",
        108: "Portugal Primeira Liga", 111: "Belgium Pro League", 112: "Scotland Premiership",
        113: "Netherlands Eredivisie", 119: "Turkey Super Lig", 120: "Russia Premier League",
        124: "Austria Bundesliga", 125: "Denmark Superliga", 126: "Sweden Allsvenskan",
        127: "Norway Eliteserien", 128: "Poland Ekstraklasa", 129: "Switzerland Super League",
        130: "Greece Super League", 131: "Italy Serie A Women", 132: "Czech First League",
        133: "Romania Liga I", 134: "Croatia HNL", 136: "Serbia SuperLiga",
        137: "Ukraine Premier League", 138: "Hungary NB I", 139: "Bulgaria First League",
        140: "Slovakia Super Liga", 141: "Finland Veikkausliiga", 142: "Iceland Besta deild",
        143: "Latvia Higher League", 144: "Lithuania A Lyga", 145: "Estonia Meistriliiga",
        146: "Belarus Premier League", 147: "Kazakhstan Premier League",
        148: "Azerbaijan Premier League", 149: "Georgia Erovnuli Liga",
        150: "Armenia Premier League", 151: "Moldova Super Liga",
        152: "Slovenia PrvaLiga", 153: "Bosnia Premier League",
        154: "Northern Ireland Premiership", 155: "Wales Cymru Premier",
        156: "Faroe Islands Premier League", 157: "Luxembourg National Division",
        158: "Malta Premier League", 159: "Gibraltar Premier League",
        160: "San Marino Campionato", 161: "Andorra Primera Divisio",
        162: "Kosovo SuperLiga", 163: "Albania Superliga", 164: "North Macedonia First League",
        165: "Montenegro First League", 166: "Cyprus First Division",
        168: "Israel Premier League", 169: "Qatar Stars League", 170: "UAE Pro League",
        171: "Saudi Pro League", 172: "Bahrain Premier League", 173: "Kuwait Premier League",
        174: "Iraq Premier League", 175: "Jordan Premier League", 176: "Lebanon Premier League",
        177: "Oman Professional League", 178: "Kyrgyzstan Premier League",
        179: "Tajikistan Higher League", 180: "Uzbekistan Super League",
        181: "Turkmenistan Yokary Liga", 183: "Vietnam V-League",
        184: "Thailand Thai League 1", 185: "Malaysia Super League",
        186: "Singapore Premier League", 187: "Indonesia Liga 1", 188: "Philippines PFL",
        189: "China Super League", 190: "Chinese Taipei Premier League",
        191: "Hong Kong Premier League", 192: "Macau Liga de Elite",
        193: "Myanmar National League", 194: "Cambodia Premier League",
        195: "Laos Premier League", 196: "Mongolia Premier League",
        197: "Nepal A Division League", 198: "Bangladesh Premier League",
        199: "India Super League", 200: "Sri Lanka Premier League",
        203: "Australia A-League", 204: "New Zealand National League",
        205: "Fiji Premier League", 206: "Solomon Islands S-League",
        207: "Papua New Guinea National Soccer League", 208: "Vanuatu Premia Divisen",
        209: "Tahiti Ligue 1", 210: "New Caledonia Super Ligue",
        211: "South Africa Premier Division", 212: "Morocco Botola Pro",
        213: "Egypt Premier League", 214: "Tunisia Ligue Professionnelle 1",
        215: "Algeria Ligue Professionnelle 1", 216: "Libya Premier League",
        217: "Sudan Premier League", 218: "Ethiopia Premier League",
        219: "Kenya Premier League", 220: "Uganda Premier League",
        221: "Tanzania Premier League", 222: "Ghana Premier League",
        224: "Nigeria Professional Football League", 225: "Senegal Premier League",
        226: "Mali Premiere Division", 227: "Cameroon Elite One",
        228: "Ivory Coast Ligue 1", 229: "Zambia Super League", 230: "Zimbabwe Premier League",
        231: "Angola Girabola", 232: "DR Congo Linafoot", 233: "Congo Premier League",
        234: "Gabon Championnat National", 235: "Botswana Premier League",
        236: "Burundi Primus League", 237: "Rwanda Premier League",
        238: "Mauritius Premier League", 239: "Namibia Premier League",
        240: "Malawi Super League", 241: "Mozambique Moçambola",
        242: "Benin Premier League", 243: "Togo Premier League", 244: "Guinea Premier League",
        245: "Burkina Faso Premier League", 246: "Sierra Leone Premier League",
        247: "Liberia First Division", 248: "Niger Premier League",
        249: "Chad Premier League", 250: "Mauritania Premier League",
        251: "Gambia Premier League", 252: "Guinea-Bissau Campeonato Nacional",
        253: "Equatorial Guinea Primera Division", 254: "Cape Verde Premier League",
        255: "Comoros Premier League", 256: "Lesotho Premier League",
        257: "Swaziland Premier League", 258: "Seychelles Premier League",
        259: "Madagascar THB Champions League", 260: "South Sudan Premier League",
        261: "Sao Tome Premier League", 262: "Eritrea Premier League",
        263: "Djibouti Premier League", 264: "Somalia Premier League",
        265: "Central African Republic Premier League", 266: "Afghanistan Premier League",
        267: "Pakistan Premier League", 268: "Bhutan Premier League",
        269: "Canada Premier League", 270: "Cuba Primera Division",
        271: "Costa Rica Primera Division", 272: "Panama LPF",
        273: "Honduras Nacional", 274: "Nicaragua Primera Division",
        275: "El Salvador Primera Division", 276: "Guatemala Liga Nacional",
        277: "Belize Premier League", 278: "Jamaica Premier League",
        279: "Trinidad and Tobago Pro League", 280: "Haiti Ligue 1",
        281: "Dominican Republic Liga Mayor", 282: "Puerto Rico Premier League",
        283: "Barbados Premier League", 284: "Bermuda Premier Division",
        285: "Guyana Elite League", 286: "Suriname Hoofdklasse",
        287: "French Guiana Régional", 288: "Martinique Championnat",
        289: "Guadeloupe Championnat", 290: "Curaçao Sekshon Pagá",
        291: "Aruba Division di Honor", 292: "St. Lucia Premier League",
        293: "Antigua and Barbuda Premier League", 294: "St. Kitts Premier League",
        295: "Dominica Premier League", 296: "Grenada Premier League",
        297: "Cayman Islands Premier League", 298: "Bahamas Premier League",
        299: "Turks and Caicos Premier League", 300: "Virgin Islands Premier League",
        301: "Anguilla League", 302: "Montserrat Championship",
        303: "British Virgin Islands Premier League", 304: "St. Vincent Premier League",
        305: "Sint Maarten Premier League", 306: "Bonaire League",
        307: "St. Martin League", 308: "Aruba Division di Honor",
        309: "Congo Brazzaville Premier League", 310: "Chad Premier League",
        311: "Colombia Primera A", 312: "Ecuador Serie A", 313: "Paraguay Primera Division",
        314: "Peru Liga 1", 315: "Venezuela Primera Division", 316: "Chile Primera Division",
        317: "Bolivia Primera Division", 318: "Uruguay Primera Division",
        319: "Argentina Liga Profesional", 320: "Brazil Serie A",
        321: "Paraguay Primera Division", 322: "Bolivia Primera Division",
    }
    for k, v in sb.items():
        m[f"SB_{k}"] = v

    of = {
        "en_1": "England Premier League", "en_2": "England Championship",
        "en_3": "England League One", "en_4": "England League Two",
        "es_1": "Spain LaLiga", "es_2": "Spain LaLiga 2",
        "it_1": "Italy Serie A", "it_2": "Italy Serie B",
        "de_1": "Germany Bundesliga", "de_2": "Germany 2. Bundesliga",
        "fr_1": "France Ligue 1", "fr_2": "France Ligue 2",
        "nl_1": "Netherlands Eredivisie", "pt_1": "Portugal Primeira Liga",
        "tr_1": "Turkey Super Lig", "sco_1": "Scotland Premiership",
        "at_1": "Austria Bundesliga", "be_1": "Belgium Pro League",
        "gr_1": "Greece Super League", "ch_1": "Switzerland Super League",
        "ru_1": "Russia Premier League", "ua_1": "Ukraine Premier League",
        "pl_1": "Poland Ekstraklasa", "cz_1": "Czech First League",
        "ro_1": "Romania Liga I", "rs_1": "Serbia SuperLiga",
        "hr_1": "Croatia HNL", "bg_1": "Bulgaria First League",
        "hu_1": "Hungary NB I", "sk_1": "Slovakia Super Liga",
        "si_1": "Slovenia PrvaLiga", "dk_1": "Denmark Superliga",
        "se_1": "Sweden Allsvenskan", "no_1": "Norway Eliteserien",
        "fi_1": "Finland Veikkausliiga", "is_1": "Iceland Besta deild",
        "ee_1": "Estonia Meistriliiga", "lv_1": "Latvia Higher League",
        "lt_1": "Lithuania A Lyga", "by_1": "Belarus Premier League",
        "ie_1": "Ireland Premier Division", "cy_1": "Cyprus First Division",
        "mk_1": "North Macedonia First League", "al_1": "Albania Superliga",
        "ba_1": "Bosnia Premier League", "me_1": "Montenegro First League",
        "xk_1": "Kosovo SuperLiga", "lu_1": "Luxembourg National Division",
        "mt_1": "Malta Premier League", "ad_1": "Andorra Primera Divisio",
        "sm_1": "San Marino Campionato", "gi_1": "Gibraltar Premier League",
        "fo_1": "Faroe Islands Premier League", "va_1": "Vatican Coppa",
        "ge_1": "Georgia Erovnuli Liga", "am_1": "Armenia Premier League",
        "az_1": "Azerbaijan Premier League", "kz_1": "Kazakhstan Premier League",
        "uz_1": "Uzbekistan Super League", "tj_1": "Tajikistan Higher League",
        "tm_1": "Turkmenistan Yokary Liga", "kg_1": "Kyrgyzstan Premier League",
        "il_1": "Israel Premier League", "jo_1": "Jordan Premier League",
        "lb_1": "Lebanon Premier League", "sy_1": "Syria Premier League",
        "iq_1": "Iraq Premier League", "ir_1": "Iran Pro League",
        "om_1": "Oman Professional League", "qa_1": "Qatar Stars League",
        "ae_1": "UAE Pro League", "sa_1": "Saudi Pro League",
        "bh_1": "Bahrain Premier League", "kw_1": "Kuwait Premier League",
        "ps_1": "Palestine Premier League", "ye_1": "Yemen Premier League",
        "af_1": "Afghanistan Premier League", "pk_1": "Pakistan Premier League",
        "in_1": "India Super League", "lk_1": "Sri Lanka Premier League",
        "np_1": "Nepal A Division League", "bd_1": "Bangladesh Premier League",
        "bt_1": "Bhutan Premier League", "mv_1": "Maldives Premier League",
        "th_1": "Thailand Thai League 1", "vn_1": "Vietnam V-League",
        "id_1": "Indonesia Liga 1", "my_1": "Malaysia Super League",
        "sg_1": "Singapore Premier League", "ph_1": "Philippines PFL",
        "mm_1": "Myanmar National League", "kh_1": "Cambodia Premier League",
        "la_1": "Laos Premier League", "mn_1": "Mongolia Premier League",
        "hk_1": "Hong Kong Premier League", "mo_1": "Macau Liga de Elite",
        "tw_1": "Chinese Taipei Premier League", "kp_1": "North Korea DPR Premier League",
        "jp_1": "Japan J1 League", "kr_1": "South Korea K League 1",
        "au_1": "Australia A-League", "nz_1": "New Zealand National League",
        "us_1": "USA MLS", "ca_1": "Canada Premier League",
        "mx_1": "Mexico Liga MX", "br_1": "Brazil Serie A", "br_2": "Brazil Serie B",
        "ar_1": "Argentina Liga Profesional", "cl_1": "Chile Primera Division",
        "co_1": "Colombia Primera A", "pe_1": "Peru Liga 1",
        "uy_1": "Uruguay Primera Division", "py_1": "Paraguay Primera Division",
        "bo_1": "Bolivia Primera Division", "ec_1": "Ecuador Serie A",
        "ve_1": "Venezuela Primera Division", "cr_1": "Costa Rica Primera Division",
        "pa_1": "Panama LPF", "hn_1": "Honduras Nacional",
        "ni_1": "Nicaragua Primera Division", "sv_1": "El Salvador Primera Division",
        "gt_1": "Guatemala Liga Nacional", "cu_1": "Cuba Primera Division",
        "do_1": "Dominican Republic Liga Mayor", "jm_1": "Jamaica Premier League",
        "tt_1": "Trinidad and Tobago Pro League", "pr_1": "Puerto Rico Premier League",
        "bb_1": "Barbados Premier League", "gy_1": "Guyana Elite League",
        "za_1": "South Africa Premier Division", "ma_1": "Morocco Botola Pro",
        "eg_1": "Egypt Premier League", "tn_1": "Tunisia Ligue Professionnelle 1",
        "dz_1": "Algeria Ligue Professionnelle 1", "ly_1": "Libya Premier League",
        "ng_1": "Nigeria Professional Football League", "gh_1": "Ghana Premier League",
        "cm_1": "Cameroon Elite One", "ci_1": "Ivory Coast Ligue 1",
        "sn_1": "Senegal Premier League", "ml_1": "Mali Premiere Division",
        "zm_1": "Zambia Super League", "zw_1": "Zimbabwe Premier League",
        "ao_1": "Angola Girabola", "cd_1": "DR Congo Linafoot",
        "cg_1": "Congo Premier League", "ga_1": "Gabon Championnat National",
        "bw_1": "Botswana Premier League", "ug_1": "Uganda Premier League",
        "ke_1": "Kenya Premier League", "et_1": "Ethiopia Premier League",
        "sd_1": "Sudan Premier League", "tz_1": "Tanzania Premier League",
        "gn_1": "Guinea Premier League", "gw_1": "Guinea-Bissau Campeonato Nacional",
        "sl_1": "Sierra Leone Premier League", "lr_1": "Liberia First Division",
        "bf_1": "Burkina Faso Premier League", "ne_1": "Niger Premier League",
        "td_1": "Chad Premier League", "mr_1": "Mauritania Premier League",
        "gm_1": "Gambia Premier League", "cv_1": "Cape Verde Premier League",
        "st_1": "Sao Tome Premier League", "gq_1": "Equatorial Guinea Primera Division",
        "mz_1": "Mozambique Moçambola", "mw_1": "Malawi Super League",
        "na_1": "Namibia Premier League", "bi_1": "Burundi Primus League",
        "rw_1": "Rwanda Premier League", "mg_1": "Madagascar THB Champions League",
        "sc_1": "Seychelles Premier League", "ls_1": "Lesotho Premier League",
        "sz_1": "Swaziland Premier League", "km_1": "Comoros Premier League",
        "so_1": "Somalia Premier League", "dj_1": "Djibouti Premier League",
        "er_1": "Eritrea Premier League", "ss_1": "South Sudan Premier League",
        "cf_1": "Central African Republic Premier League", "nir": "Northern Ireland Premiership",
        "wal": "Wales Cymru Premier", "civ": "Ivory Coast Ligue 1",
        "crc": "Costa Rica Primera Division", "tpe": "Chinese Taipei Premier League",
        "sui": "Switzerland Super League", "aut": "Austria Bundesliga",
        "bel": "Belgium Pro League", "smr": "San Marino Campionato",
        "fro": "Faroe Islands Premier League", "kgz": "Kyrgyzstan Premier League",
        "mda": "Moldova Super Liga", "bih": "Bosnia Premier League",
        "gib": "Gibraltar Premier League", "pse": "Palestine Premier League",
        "tls": "East Timor Pertamina", "mac": "Macau Liga de Elite",
        "hkg": "Hong Kong Premier League", "ksa": "Saudi Pro League",
        "kor": "South Korea K League 1", "cze": "Czech First League",
        "svk": "Slovakia Super Liga", "svn": "Slovenia PrvaLiga",
        "hrv": "Croatia HNL", "srb": "Serbia SuperLiga", "mkd": "North Macedonia First League",
        "alb": "Albania Superliga", "geo": "Georgia Erovnuli Liga",
        "arm": "Armenia Premier League", "aze": "Azerbaijan Premier League",
        "kaz": "Kazakhstan Premier League", "uzb": "Uzbekistan Super League",
        "isr": "Israel Premier League", "jor": "Jordan Premier League",
        "lib": "Lebanon Premier League", "irq": "Iraq Premier League",
        "omn": "Oman Professional League", "qat": "Qatar Stars League",
        "uae": "UAE Pro League", "bhr": "Bahrain Premier League",
        "kwt": "Kuwait Premier League", "cyp": "Cyprus First Division",
        "lux": "Luxembourg National Division", "mlt": "Malta Premier League",
        "and": "Andorra Primera Divisio", "kos": "Kosovo SuperLiga",
        "mne": "Montenegro First League", "bul": "Bulgaria First League",
        "hun": "Hungary NB I", "rou": "Romania Liga I", "pol": "Poland Ekstraklasa",
        "ukr": "Ukraine Premier League", "blr": "Belarus Premier League",
        "est": "Estonia Meistriliiga", "lva": "Latvia Higher League",
        "ltu": "Lithuania A Lyga", "fin": "Finland Veikkausliiga",
        "isl": "Iceland Besta deild", "nor": "Norway Eliteserien",
        "swe": "Sweden Allsvenskan", "den": "Denmark Superliga",
        "por": "Portugal Primeira Liga", "che": "Switzerland Super League",
        "fra": "France Ligue 1", "esp": "Spain LaLiga", "ger": "Germany Bundesliga",
        "ita": "Italy Serie A", "ned": "Netherlands Eredivisie",
        "bel": "Belgium Pro League", "eng": "England Premier League",
        "irl": "Ireland Premier Division", "mli": "Mali Premiere Division",
        "sen": "Senegal Premier League", "gha": "Ghana Premier League",
        "nga": "Nigeria Professional Football League", "cmr": "Cameroon Elite One",
        "civ": "Ivory Coast Ligue 1", "zam": "Zambia Super League",
        "zim": "Zimbabwe Premier League", "ang": "Angola Girabola",
        "cod": "DR Congo Linafoot", "cog": "Congo Premier League",
        "gab": "Gabon Championnat National", "bot": "Botswana Premier League",
        "uga": "Uganda Premier League", "ken": "Kenya Premier League",
        "eth": "Ethiopia Premier League", "sdn": "Sudan Premier League",
        "tan": "Tanzania Premier League", "gui": "Guinea Premier League",
        "gin": "Guinea-Bissau Campeonato Nacional", "sle": "Sierra Leone Premier League",
        "lbr": "Liberia First Division", "bfa": "Burkina Faso Premier League",
        "ner": "Niger Premier League", "tcd": "Chad Premier League",
        "mar": "Morocco Botola Pro", "tun": "Tunisia Ligue Professionnelle 1",
        "egy": "Egypt Premier League", "alg": "Algeria Ligue Professionnelle 1",
        "lby": "Libya Premier League", "moz": "Mozambique Moçambola",
        "mwi": "Malawi Super League", "zam": "Zambia Super League",
        "nam": "Namibia Premier League", "bdi": "Burundi Primus League",
        "rwa": "Rwanda Premier League", "mdg": "Madagascar THB Champions League",
        "sey": "Seychelles Premier League", "les": "Lesotho Premier League",
        "swz": "Swaziland Premier League", "cpv": "Cape Verde Premier League",
        "com": "Comoros Premier League", "ssd": "South Sudan Premier League",
        "stp": "Sao Tome Premier League", "eri": "Eritrea Premier League",
        "dji": "Djibouti Premier League", "som": "Somalia Premier League",
        "caf": "Central African Republic Premier League", "afg": "Afghanistan Premier League",
        "pak": "Pakistan Premier League", "btn": "Bhutan Premier League",
        "ind": "India Super League", "lka": "Sri Lanka Premier League",
        "npl": "Nepal A Division League", "bgd": "Bangladesh Premier League",
        "tha": "Thailand Thai League 1", "vnm": "Vietnam V-League",
        "idn": "Indonesia Liga 1", "mys": "Malaysia Super League",
        "sgp": "Singapore Premier League", "phl": "Philippines PFL",
        "mmr": "Myanmar National League", "khm": "Cambodia Premier League",
        "lao": "Laos Premier League", "mng": "Mongolia Premier League",
        "mac": "Macau Liga de Elite", "hkg": "Hong Kong Premier League",
        "twn": "Chinese Taipei Premier League", "jpn": "Japan J1 League",
        "aus": "Australia A-League", "nzl": "New Zealand National League",
    }
    for k, v in of.items():
        m[f"OF_{k}"] = v

    return m


def _resolve_league(code):
    if not code or not isinstance(code, str):
        return str(code) if code else "?"
    if hasattr(_resolve_league, "_map") and code in _resolve_league._map:
        return _resolve_league._map[code]
    return code


def _season_num(s):
    """Sezon kodunu sayisal yapar: 2526 -> 2526, 2023-24 -> 2324."""
    s = str(s)
    if "-" in s:
        y1, y2 = s.split("-")
        return int("20" + y2) if len(y2) == 2 else int(y2)
    return int(s)


def _build_team_index(feat, name_map):
    team_details = {}
    all_tids = set(feat["home_team_id"].unique()) | set(feat["away_team_id"].unique())
    # Tum verinin en yeni sezonu (veri setinin "gunumuz")
    all_seasons = feat["season"].map(_season_num)
    data_newest_season = int(all_seasons.max()) if len(all_seasons) else 0
    for tid in all_tids:
        home_s = feat[feat["home_team_id"] == tid].sort_values("date")
        away_s = feat[feat["away_team_id"] == tid].sort_values("date")
        s = pd.concat([home_s, away_s]).sort_values("date")
        if len(s) == 0:
            continue
        sn = s["season"].map(_season_num)
        max_sn = int(sn.max())
        # GUNCEL LIG = takimin EN YUKSEK (en yeni) sezonundaki ligi (mod yerine son sezon)
        last_season = s[sn == max_sn]
        ls_leagues = last_season["league"].value_counts()
        current_league = ls_leagues.idxmax() if len(ls_leagues) else str(s.iloc[-1].get("league", "?"))
        last = s.iloc[-1]
        # Form ve ortalamalar: sadece EN YENI sezondan (gunumuz takimi icin guncel)
        recent_season = s[sn == max_sn].sort_values("date").tail(10)
        if len(recent_season) == 0:
            recent_season = s.tail(10)
        recent = recent_season
        form_results = recent["result"].map({"H": "W", "D": "D", "A": "L"}).tolist()
        avg_goals = recent["home_goals"].mean() if "home_goals" in recent.columns else 0
        avg_conceded = recent["away_goals"].mean() if "away_goals" in recent.columns else 0
        raw_league = str(current_league)
        # GUNCEL MI? Takimin en yeni sezonu tum verinin en yeni sezonuna esit mi?
        is_current = (max_sn >= data_newest_season)
        team_details[int(tid)] = {
            "id": int(tid),
            "name": str(name_map.get(int(tid), str(tid))),
            "league": _resolve_league(raw_league),
            "league_code": raw_league,
            "elo": round(float(last.get("home_elo", 1500))),
            "attack_elo": round(float(last.get("home_attack_elo", 1500))),
            "defence_elo": round(float(last.get("home_defence_elo", 1500))),
            "n_matches": int(len(s)),
            "form": "".join(form_results[-5:]) if form_results else "?????",
            "avg_goals_scored": round(float(avg_goals), 2),
            "avg_goals_conceded": round(float(avg_conceded), 2),
            "last_match_date": str(last.get("date", "")),
            "current_data": bool(is_current),
            "newest_season": int(max_sn),
        }
    return team_details


def _load_model():
    """Gold veriden takim indeksi + model yukle."""
    global TEAM_INDEX, LEAGUE_TEAMS, TEAM_ID_MAP
    _resolve_league._map = _load_league_names()
    try:
        print("[WEB] Gold veri yukleniyor (features_combined)...", flush=True)
        feat = pd.read_parquet("data/gold/features_combined.parquet")
        feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
        # ID'leri sayisala cevir (TM_xxx gibi gecersizler NaN olur, name_map eslesmez ama crash olmaz)
        for _idc in ["home_team_id", "away_team_id"]:
            if _idc in feat.columns:
                feat[_idc] = pd.to_numeric(feat[_idc], errors="coerce")
        feat = feat.dropna(subset=["home_team_id", "away_team_id"])
        feat["home_team_id"] = feat["home_team_id"].astype(int)
        feat["away_team_id"] = feat["away_team_id"].astype(int)
        # Performans: en yeni 500K mac uzerinden index build
        if len(feat) > 500000:
            feat = feat.sort_values("date").tail(500000).reset_index(drop=True)
        else:
            feat = feat.sort_values("date").reset_index(drop=True)
        for c in ["home_goals", "away_goals"]:
            if c in feat.columns:
                feat[c] = feat[c].fillna(0).astype(int)
        feat["total_goals"] = feat.get("home_goals", 0) + feat.get("away_goals", 0)
        feat["result"] = feat.get("result", pd.Series(["D"] * len(feat))).fillna("D")
        feat["btts"] = ((feat.get("home_goals", 0) > 0) & (feat.get("away_goals", 0) > 0)).astype(int)
        feat["over15"] = (feat["total_goals"] > 1.5).astype(int)
        feat["over25"] = (feat["total_goals"] > 2.5).astype(int)
        feat["over35"] = (feat["total_goals"] > 3.5).astype(int)
        feat["total_corners"] = feat.get("home_corners", pd.Series([0]*len(feat))).fillna(0) + feat.get("away_corners", pd.Series([0]*len(feat))).fillna(0)
        feat["double_1X"] = ((feat["result"] == "H") | (feat["result"] == "D")).astype(int)
        feat["double_X2"] = ((feat["result"] == "D") | (feat["result"] == "A")).astype(int)
        feat["double_12"] = ((feat["result"] == "H") | (feat["result"] == "A")).astype(int)
        feat["corners_over75"] = (feat["total_corners"] >= 8).astype(int)
        feat["corners_over85"] = (feat["total_corners"] >= 9).astype(int)
        feat = feat.dropna(subset=["result"])

        name_map = {}
        name_path = "data/gold/team_id_to_name.json"
        if os.path.exists(name_path):
            with open(name_path) as fh:
                raw_nm = json.load(fh)
                name_map = {str(k): v for k, v in raw_nm.items()}

        n_t, n_l = _build_indices_from_feat(feat, name_map)
        if TEAM_INDEX:
            TEAM_ID_MAP.update(TEAM_INDEX)
        LEAGUE_TEAMS = LEAGUE_TEAMS
        STATE["features"] = feat
        STATE["loading"] = False

        # === VERI BILGISI (gunluklik / guncelluk) ===
        newest_date = feat["date"].max() if "date" in feat.columns else None
        newest_season_num = int(feat["season"].map(_season_num).max())
        n_current = sum(1 for d in TEAM_INDEX.values() if d.get("current_data"))
        n_total = len(TEAM_INDEX)
        STATE["data_info"] = {
            "newest_match_date": str(newest_date) if newest_date is not None else "?",
            "newest_season": newest_season_num,
            "n_teams": n_total,
            "n_current_teams": n_current,
            "n_stale_teams": n_total - n_current,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "data_fresh": bool(n_current > 0),
        }
        print(f"[WEB] {n_t} takim, {n_l} lig yuklendi!", flush=True)
        print(f"[WEB] Veri son mac: {STATE['data_info']['newest_match_date']} | "
              f"Guncel takim: {n_current}/{n_total}", flush=True)

        # === MODEL YUKLE (web_model.pkl - SofaScore LightGBM) ===
        _root = os.path.join(os.path.dirname(__file__), "..")
        web_model_path = os.path.join(_root, "data", "web_model.pkl")
        if not os.path.exists(web_model_path):
            web_model_path = os.path.join(_root, "web_model.pkl")
        if os.path.exists(web_model_path):
            print(f"[WEB] Model yukleniyor ({web_model_path})...", flush=True)
            with open(web_model_path, "rb") as fh:
                wm = pickle.load(fh)
            STATE["web_model"] = wm
            STATE["pipeline"] = None
            STATE["model_version"] = wm.get("version", "unknown")
            STATE["loaded_at"] = datetime.now(timezone.utc).isoformat()
            print(f"[WEB] Model yuklendi! ({wm.get('n_features',0)} feature, v{wm.get('version','?')})", flush=True)
        else:
            print("[WEB] UYARI: web_model.pkl bulunamadi!", flush=True)
            STATE["web_model"] = None

        # === SOFASCORE FULL VERISI (takim form + elo icin) ===
        sf_full_path = os.path.join(os.path.dirname(__file__), "..", "data", "gold", "sofascore_full.parquet")
        if os.path.exists(sf_full_path):
            print("[WEB] SofaScore full verisi yukleniyor...", flush=True)
            sf_full = pd.read_parquet(sf_full_path)
            sf_full["date"] = pd.to_datetime(sf_full["date"], errors="coerce")
            sf_full = sf_full.sort_values("date").reset_index(drop=True)
            STATE["sofa_full"] = sf_full
            print(f"[WEB] SofaScore full: {len(sf_full)} mac yuklendi!", flush=True)
        else:
            print("[WEB] UYARI: sofascore_full.parquet yok!", flush=True)
            STATE["sofa_full"] = None

        # === SOFASCORE ROLLING VERISI (pre-computed rolling features) ===
        sf_roll_path = os.path.join(os.path.dirname(__file__), "..", "data", "gold", "sofascore_rolling.parquet")
        if os.path.exists(sf_roll_path):
            print("[WEB] SofaScore rolling verisi yukleniyor...", flush=True)
            sf_roll = pd.read_parquet(sf_roll_path)
            sf_roll["date"] = pd.to_datetime(sf_roll["date"], errors="coerce")
            sf_roll = sf_roll.sort_values("date").reset_index(drop=True)
            STATE["sofa_rolling"] = sf_roll
            print(f"[WEB] SofaScore rolling: {len(sf_roll)} mac yuklendi!", flush=True)
        else:
            print("[WEB] UYARI: sofascore_rolling.parquet yok!", flush=True)
            STATE["sofa_rolling"] = None

        # === SOFASCORE ISIM ESLESTIRME ===
        sofa_name_path = os.path.join(os.path.dirname(__file__), "..", "data", "gold", "sofa_name_map.pkl")
        if os.path.exists(sofa_name_path):
            with open(sofa_name_path, "rb") as fh:
                STATE["sofa_name_map"] = pickle.load(fh)
            print(f"[WEB] SofaScore isim eslesme: {len(STATE['sofa_name_map'])} takim!", flush=True)
        else:
            STATE["sofa_name_map"] = {}
            print("[WEB] sofa_name_map.pkl yok, isim eslestirme yapilmayacak!", flush=True)

        # === SOFASCORE TARIH VERISI (eski - fallback) ===
        sf_path = os.path.join(os.path.dirname(__file__), "..", "data", "gold", "sofascore_raw.parquet")
        if os.path.exists(sf_path):
            sf_df = pd.read_parquet(sf_path)
            sf_df["date"] = pd.to_datetime(sf_df["date"], errors="coerce")
            sf_df = sf_df.sort_values("date").reset_index(drop=True)
            STATE["sofa_hist"] = sf_df
        else:
            STATE["sofa_hist"] = None

        # === FORM CACHE: Pre-computed team form (predict-time O(1) lookup) ===
        print("[WEB] Form cache olusturuluyor...", flush=True)
        _build_form_cache()
        print(f"[WEB] Form cache: {len(STATE.get('form_cache', {}))} takim!", flush=True)

    except Exception as exc:
        STATE["loading"] = False
        STATE["error"] = str(exc)
        print(f"[WEB] HATA: {exc}", flush=True)


def _build_form_cache():
    """Startup'ta tüm takımların form feature'larını vectorized hesapla.
    features.parquet'taki hazır rolling stats'ları kullanır (iterrows yok)."""
    feat = STATE.get("features")
    if feat is None or len(feat) == 0:
        STATE["form_cache"] = {}
        return

    cache = {}
    all_ids = set(feat["home_team_id"].dropna().astype(int).unique()) | \
              set(feat["away_team_id"].dropna().astype(int).unique())

    feat_sorted = feat.sort_values("date")
    feat_home = feat_sorted[feat_sorted["home_team_id"].notna()].copy()
    feat_home["home_team_id"] = feat_home["home_team_id"].astype(int)
    feat_away = feat_sorted[feat_sorted["away_team_id"].notna()].copy()
    feat_away["away_team_id"] = feat_away["away_team_id"].astype(int)

    home_last = feat_home.groupby("home_team_id").last()
    away_last = feat_away.groupby("away_team_id").last()

    home_cols_map = {
        "home_gf_5": "gf_5", "home_ga_5": "ga_5", "home_pts_5": "pts_5",
        "home_gf_3": "gf_3", "home_ga_3": "ga_3",
        "home_gf_8": "gf_8", "home_ga_8": "ga_8",
        "home_gf_20": "gf_20", "home_ga_20": "ga_20",
        "home_momentum": "momentum", "home_wins_last5": "wins_last5",
        "home_gdiff5": "gdiff5", "home_opp_elo": "opp_elo",
        "home_w_gf": "w_gf", "home_w_ga": "w_ga",
        "home_w_shots": "w_shots", "home_w_sot": "w_sot",
        "home_rest_days": "rest_days",
        "home_hgf_5": "hgf_5", "home_hga_5": "hga_5", "home_hpts_5": "hpts_5",
    }
    away_cols_map = {
        "away_gf_5": "gf_5", "away_ga_5": "ga_5", "away_pts_5": "pts_5",
        "away_gf_3": "gf_3", "away_ga_3": "ga_3",
        "away_gf_8": "gf_8", "away_ga_8": "ga_8",
        "away_gf_20": "gf_20", "away_ga_20": "ga_20",
        "away_momentum": "momentum", "away_wins_last5": "wins_last5",
        "away_gdiff5": "gdiff5", "away_opp_elo": "opp_elo",
        "away_w_gf": "w_gf", "away_w_ga": "w_ga",
        "away_w_shots": "w_shots", "away_w_sot": "w_sot",
        "away_rest_days": "rest_days",
        "away_agf_5": "hgf_5", "away_aga_5": "hga_5", "away_apts_5": "hpts_5",
    }

    for tid in all_ids:
        form = {
            "gf_5": 0, "ga_5": 0, "pts_5": 0,
            "gf_3": 0, "ga_3": 0,
            "gf_8": 0, "ga_8": 0,
            "gf_20": 0, "ga_20": 0,
            "pts_10": 0, "pts_20": 0,
            "rest_days": 5,
            "xg_r5": 0, "sot_r5": 0, "shots_r5": 0,
            "w_gf": 0, "w_ga": 0, "w_shots": 0, "w_sot": 0,
            "momentum": 0, "wins_last5": 0, "gdiff5": 0, "opp_elo": 1500,
            "hgf_5": 0, "hga_5": 0, "hpts_5": 0,
            "agf_5": 0, "aga_5": 0, "apts_5": 0,
        }

        h_form = home_last.loc[tid] if tid in home_last.index else None
        a_form = away_last.loc[tid] if tid in away_last.index else None

        h_date = h_form["date"] if h_form is not None and "date" in h_form.index else pd.Timestamp.min
        a_date = a_form["date"] if a_form is not None and "date" in a_form.index else pd.Timestamp.min
        h_date = h_date if pd.notna(h_date) else pd.Timestamp.min
        a_date = a_date if pd.notna(a_date) else pd.Timestamp.min

        newer = h_form if h_date >= a_date else a_form
        older = a_form if h_date >= a_date else h_form
        ns = "home" if h_date >= a_date else "away"

        if newer is not None:
            for src, dst in (home_cols_map if ns == "home" else away_cols_map).items():
                if src in newer.index:
                    v = newer[src]
                    if pd.notna(v):
                        form[dst] = float(v)

        if older is not None:
            os_map = away_cols_map if ns == "home" else home_cols_map
            for src, dst in os_map.items():
                if src in older.index:
                    v = older[src]
                    if pd.notna(v) and form.get(dst, 0) == 0:
                        form[dst] = float(v)

        if h_form is not None:
            form["hgf_5"] = float(h_form.get("home_hgf_5", 0) or 0)
            form["hga_5"] = float(h_form.get("home_hga_5", 0) or 0)
            form["hpts_5"] = float(h_form.get("home_hpts_5", 0) or 0)
            # xG rolling: use last match xG as estimate
            xg_val = h_form.get("home_xg_real") or h_form.get("home_xg") or h_form.get("sf_home_xg") or 0
            if pd.notna(xg_val) and float(xg_val) > 0:
                form["xg_r5"] = float(xg_val)
        if a_form is not None:
            form["agf_5"] = float(a_form.get("away_agf_5", 0) or 0)
            form["aga_5"] = float(a_form.get("away_aga_5", 0) or 0)
            form["apts_5"] = float(a_form.get("away_apts_5", 0) or 0)
            xg_val = a_form.get("away_xg_real") or a_form.get("away_xg") or a_form.get("sf_away_xg") or 0
            if pd.notna(xg_val) and float(xg_val) > 0:
                form["xg_r5"] = float(xg_val)

        cache[tid] = form

    STATE["form_cache"] = cache


def _check_data_freshness():
    """Veri guncluluk kontrolu: son mac tarihi vs bugün, ve Pazartesi otomatik kontrol.
    Doner: {fresh, newest_match_date, days_old, needs_update, checked_at, note}
    """
    info = STATE.get("data_info") or {}
    newest = info.get("newest_match_date")
    note = ""
    needs_update = False
    days_old = -1
    if newest and newest != "?":
        try:
            d = pd.to_datetime(newest, errors="coerce")
            if d is not None and not pd.isna(d):
                days_old = (datetime.now() - d.tzinfo_and_datetime if d.tzinfo else datetime.now() - d).days if False else (datetime.now() - d).days
                days_old = max(0, days_old)
                # 2026-27 sezonu basladi mi? (Temmuz sonu itibariyle yeni sezon)
                # Veri 60 gunden eskiyse "guncel degil" uyarisi
                if days_old > 60:
                    needs_update = True
                    note = f"Veri {days_old} gun eski. Yeni sezon verisi (2026-27) icin guncelleme gerekli."
                else:
                    note = f"Veri {days_old} gunluk, guncel."
            else:
                note = "Veri tarihi okunamadi."
        except Exception as e:
            note = f"Tarih kontrolu hatasi: {e}"
    else:
        note = "Veri tarihi bilinmiyor."
    # Pazartesi mi? (0=Monday)
    is_monday = datetime.now().weekday() == 0
    return {
        "fresh": info.get("data_fresh", False),
        "newest_match_date": newest or "?",
        "days_old": days_old,
        "needs_update": bool(needs_update),
        "is_monday": bool(is_monday),
        "auto_check_enabled": True,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }


def _build_indices_from_feat(feat, name_map):
    """Gold features'tan takim + lig indexlerini yeniden build et (reload icin).
    Cache: team_index.pkl varsa aninda yukler (build ~5dk suruyor)."""
    global TEAM_INDEX, LEAGUE_TEAMS, TEAM_ID_MAP
    cache_path = os.path.join(os.path.dirname(__file__), "..", "team_index.pkl")
    if os.path.exists(cache_path) and STATE.get("force_rebuild") is not True:
        try:
            with open(cache_path, "rb") as fh:
                cached = pickle.load(fh)
            TEAM_INDEX = cached["team_index"]
            TEAM_ID_MAP = cached["team_id_map"]
            LEAGUE_TEAMS = cached["league_teams"]
            print(f"[WEB] Team index cache'den yuklendi! ({len(TEAM_INDEX)} takim)", flush=True)
            return len(TEAM_INDEX), len(LEAGUE_TEAMS)
        except Exception as e:
            print(f"[WEB] Cache yukleme hatasi, rebuild: {e}", flush=True)
    TEAM_INDEX, id_to_key = _build_canonical_team_index(feat, name_map)
    id_map = {}
    for src_id, key in id_to_key.items():
        if key in TEAM_INDEX:
            id_map[src_id] = TEAM_INDEX[key]
    TEAM_ID_MAP = id_map
    lg_teams = {}
    for (canon, league), det in TEAM_INDEX.items():
        code = det["league_code"]
        if code not in lg_teams:
            lg_teams[code] = []
        lg_teams[code].append(det)
    LEAGUE_TEAMS = lg_teams
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump({"team_index": TEAM_INDEX, "team_id_map": TEAM_ID_MAP,
                         "league_teams": LEAGUE_TEAMS}, fh)
        print(f"[WEB] Team index cache'lendi! ({len(TEAM_INDEX)} takim)", flush=True)
    except Exception:
        pass
    return len(TEAM_INDEX), len(LEAGUE_TEAMS)


def _rebuild_indices():
    """Gold veriden takim/lig indexlerini tazele (modeli yeniden egitmeden)."""
    global STATE
    _clear_cache()
    feat = STATE.get("features")
    if feat is None:
        feat = pd.read_parquet("data/gold/features.parquet")
        feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
        # ID'leri sayisala cevir (TM_xxx gibi gecersizler NaN olur, name_map eslesmez ama crash olmaz)
        for _idc in ["home_team_id", "away_team_id"]:
            if _idc in feat.columns:
                feat[_idc] = pd.to_numeric(feat[_idc], errors="coerce")
        feat = feat.dropna(subset=["home_team_id", "away_team_id"])
        feat["home_team_id"] = feat["home_team_id"].astype(int)
        feat["away_team_id"] = feat["away_team_id"].astype(int)
        feat = feat.sort_values("date").reset_index(drop=True)
        for c in ["home_goals", "away_goals"]:
            if c in feat.columns:
                feat[c] = feat[c].fillna(0).astype(int)
        feat["total_goals"] = feat.get("home_goals", 0) + feat.get("away_goals", 0)
        feat["result"] = feat.get("result", pd.Series(["D"] * len(feat))).fillna("D")
        feat["btts"] = ((feat.get("home_goals", 0) > 0) & (feat.get("away_goals", 0) > 0)).astype(int)
        feat["over25"] = (feat["total_goals"] > 2.5).astype(int)
        feat = feat.dropna(subset=["result"])
        STATE["features"] = feat
    name_map = {}
    name_path = "data/gold/team_id_to_name.json"
    if os.path.exists(name_path):
        with open(name_path) as fh:
            raw_nm = json.load(fh)
            name_map = {str(k): v for k, v in raw_nm.items()}
    n_t, n_l = _build_indices_from_feat(feat, name_map)
    print(f"[WEB] Index yenilendi: {n_t} takim, {n_l} lig", flush=True)
    return {"teams": n_t, "leagues": n_l}


@app.route("/api/reload", methods=["POST"])
def api_reload():
    try:
        res = _rebuild_indices()
        return jsonify({"status": "ok", "teams": res["teams"], "leagues": res["leagues"]})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stats=_get_stats(), loading=STATE["loading"], error=STATE["error"])


@app.route("/predictions")
def predictions():
    return render_template("predictions.html", loading=STATE["loading"])


@app.route("/api/predictions")
def api_predictions():
    preds = _load_predictions()
    seen = {}
    for p in preds:
        hid = p.get("home_id")
        aid = p.get("away_id")
        key = f"{min(hid, aid)}-{max(hid, aid)}" if hid and aid else p.get("id")
        if key not in seen:
            seen[key] = p
    deduped = sorted(seen.values(), key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(deduped)


@app.route("/api/predictions/all")
def api_predictions_all():
    return jsonify(_load_predictions())


@app.route("/api/predictions/delete", methods=["POST"])
def api_predictions_delete():
    data = request.get_json()
    pred_id = data.get("id")
    if not pred_id:
        return jsonify({"error": "ID gerekli"}), 400
    _delete_prediction(pred_id)
    return jsonify({"ok": True})


@app.route("/api/predictions/delete-all", methods=["POST"])
def api_predictions_delete_all():
    """Tum tahminleri sil."""
    _save_predictions([])
    return jsonify({"ok": True})


@app.route("/api/predictions/repredict-all", methods=["POST"])
def api_predictions_repredict_all():
    """Mevcut tum tahminleri yeniden hesapla."""
    try:
        preds = _load_predictions()
        if not preds:
            return jsonify({"error": "Yeniden hesaplanacak tahmin yok"}), 400

        total = len(preds)
        updated = 0
        errors = 0

        for pred in preds:
            home_id = pred.get("home_id")
            away_id = pred.get("away_id")
            if not home_id or not away_id:
                errors += 1
                continue
            try:
                result = _predict_match(int(home_id), int(away_id))
                if result:
                    pred["timestamp"] = datetime.now().isoformat()
                    pred["result"] = {
                        "home_win": result["home_win"],
                        "draw": result["draw"],
                        "away_win": result["away_win"],
                        "winner": result["winner"],
                        "winner_prob": result["winner_prob"],
                    }
                    pred["goals"] = result["goals"]
                    pred["total_goals"] = result["total_goals"]
                    pred["double_chance"] = result["double_chance"]
                    pred["corners"] = result["corners"]
                    pred["btts_yes"] = result["btts_yes"]
                    pred["btts_no"] = result["btts_no"]
                    pred["top_scores"] = result["top_scores"]
                    pred["recommendations"] = result.get("recommendations")
                    pred["v15_markets"] = result.get("v15_markets")
                    pred["prediction_summary"] = _build_prediction_summary(pred)
                    updated += 1
            except Exception as e:
                logger.error(f"[RE_PRED] Hata: {home_id} vs {away_id}: {e}")
                errors += 1

        _save_predictions(preds)
        return jsonify({
            "success": True,
            "total": total,
            "updated": updated,
            "errors": errors,
        })
    except Exception as e:
        logger.error(f"[RE_PRED] Toplu hata: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions/auto-generate", methods=["POST"])
def api_predictions_auto_generate():
    """Yaklaşan maçlar için otomatik tahmin oluştur."""
    try:
        data = request.get_json() or {}
        days = data.get("days", 3)
        
        logger.info(f"[AUTO_PRED] Otomatik tahmin başlatıldı: {days} gün")
        
        # 1. Takım ID haritasını kontrol et
        if not TEAM_ID_MAP:
            return jsonify({"error": "TEAM_ID_MAP yüklenmedi"}), 500
        
        # 2. Yaklaşan maçları Flashscore'dan çek
        from prediction.flashscore_scraper import get_fixtures_flashscore
        from datetime import datetime, timedelta, timezone
        
        all_matches = []
        today = datetime.now(timezone.utc).date()
        
        for day_offset in range(days + 1):
            target_date = today + timedelta(days=day_offset)
            matches = get_fixtures_flashscore(kind="upcoming", limit=500, refresh=True)
            
            if matches:
                for match in matches:
                    kickoff = match.get("kickoff", "")
                    if kickoff:
                        try:
                            match_date = datetime.fromisoformat(kickoff.replace("Z", "")).date()
                            if match_date == target_date:
                                all_matches.append(match)
                        except:
                            pass
        
        logger.info(f"[AUTO_PRED] {len(all_matches)} yaklaşan maç bulundu")
        
        # 3. Her maç için tahmin oluştur
        from prediction.team_matcher import find_team_ids
        
        total = len(all_matches)
        matched = 0
        created = 0
        skipped = 0
        
        for match in all_matches:
            home_name = match.get("home_name", "")
            away_name = match.get("away_name", "")
            
            if not home_name or not away_name:
                continue
            
            # Takımları eşleştir
            home_id, away_id = find_team_ids(home_name, away_name, TEAM_ID_MAP)
            
            if not home_id or not away_id:
                logger.debug(f"[AUTO_PRED] Eşleşmedi: {home_name} vs {away_name}")
                continue
            
            matched += 1
            
            # Zaten var mı kontrol et
            existing = _find_existing_prediction(home_id, away_id)
            if existing:
                logger.debug(f"[AUTO_PRED] Zaten var: {home_name} vs {away_name}")
                skipped += 1
                continue
            
            # Tahmin oluştur
            try:
                result = _predict_match(int(home_id), int(away_id))
                if result:
                    pred_record = {
                        "id": f"{home_id}-{away_id}-{int(datetime.now().timestamp())}",
                        "timestamp": datetime.now().isoformat(),
                        "home_id": home_id,
                        "away_id": away_id,
                        "home_name": result["home"].get("name", str(home_id)),
                        "away_name": result["away"].get("name", str(away_id)),
                        "home_elo": result["home"].get("elo", 0),
                        "away_elo": result["away"].get("elo", 0),
                        "result": {
                            "home_win": result["home_win"],
                            "draw": result["draw"],
                            "away_win": result["away_win"],
                            "winner": result["winner"],
                            "winner_prob": result["winner_prob"],
                        },
                        "goals": result["goals"],
                        "total_goals": result["total_goals"],
                        "double_chance": result["double_chance"],
                        "corners": result["corners"],
                        "btts_yes": result["btts_yes"],
                        "btts_no": result["btts_no"],
                        "top_scores": result["top_scores"],
                        "recommendations": result.get("recommendations"),
                        "v15_markets": result.get("v15_markets"),
                        "match_date": match.get("kickoff"),
                    }
                    _add_prediction(pred_record)
                    created += 1
                    logger.info(f"[AUTO_PRED] ✓ Tahmin oluşturuldu: {home_name} vs {away_name}")
            except Exception as e:
                logger.error(f"[AUTO_PRED] Tahmin hatası: {home_name} vs {away_name}: {e}")
        
        logger.info(f"[AUTO_PRED] Tamamlandı: {created} oluşturuldu, {skipped} atlandı, {matched}/{total} eşleşti")
        
        return jsonify({
            "success": True,
            "total": total,
            "matched": matched,
            "created": created,
            "skipped": skipped,
        })
        
    except Exception as e:
        logger.error(f"[AUTO_PRED] Hata: {e}")
        return jsonify({"error": str(e)}), 500


def _find_existing_prediction(home_id, away_id):
    """Mevcut bir tahmin var mı kontrol et."""
    preds = _load_predictions()
    for pred in preds:
        if pred.get("home_id") == home_id and pred.get("away_id") == away_id:
            # Son 7 gün içinde mi?
            try:
                pred_time = datetime.fromisoformat(pred.get("timestamp", ""))
                if (datetime.now() - pred_time).days < 14:  # 14 gün (2 hafta) içinde
                    return pred
            except:
                pass
    return None


@app.route("/api/predictions/save-json", methods=["POST"])
def api_predictions_save_json():
    """Tahminleri sunucudaki tahminler/ klasorune JSON olarak kaydet."""
    data = request.get_json() or {}
    preds = data.get("predictions") or _load_predictions()
    filename = data.get("filename") or f"tahminler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    export = []
    for p in preds:
        # buildJSON() nested format: home_team, away_team, predictions.1X2, predictions.btts, vb.
        # Eski flat format: home_name, away_name, result, btts_yes, vb.
        # Her iki formati da destekle
        home = p.get("home_team") or p.get("home_name")
        away = p.get("away_team") or p.get("away_name")

        pred = p.get("predictions", {})
        r = p.get("result", {}) or pred.get("1X2", {})
        tg = p.get("total_goals", {}) or pred.get("goals", {}).get("over_under", {})
        cr = p.get("corners", {}) or pred.get("corners", {})
        dc = p.get("double_chance", {}) or pred.get("double_chance", {})
        g = p.get("goals", {}) or pred.get("goals", {}).get("expected", {})
        btts = pred.get("btts", {})
        top_scores_raw = p.get("top_scores") or pred.get("top_scores", [])

        export.append({
            "id": p.get("id"),
            "timestamp": p.get("timestamp"),
            "match": f"{home or '?'} vs {away or '?'}",
            "home_team": home,
            "away_team": away,
            "home_elo": p.get("home_elo", 0),
            "away_elo": p.get("away_elo", 0),
            "predictions": {
                "1X2": {
                    "home": r.get("home") or r.get("home_win"),
                    "draw": r.get("draw"),
                    "away": r.get("away") or r.get("away_win"),
                    "winner": r.get("winner"),
                    "winner_prob": r.get("winner_prob")
                },
                "double_chance": {"1X": dc.get("1X"), "12": dc.get("12"), "X2": dc.get("X2")},
                "btts": {"yes": btts.get("yes") or p.get("btts_yes"), "no": btts.get("no") or p.get("btts_no")},
                "goals": {
                    "expected": {
                        "home_xg": g.get("home_xg") or g.get("home_lambda"),
                        "away_xg": g.get("away_xg") or g.get("away_lambda"),
                        "total_xg": g.get("total_xg") or g.get("expected_total")
                    },
                    "over_under": {
                        "1.5": tg.get("1.5") or tg.get("over15"),
                        "2.5": tg.get("2.5") or tg.get("over25"),
                        "3.5": tg.get("3.5") or tg.get("over35")
                    }
                },
                "corners": {
                    "average": cr.get("average") or cr.get("predicted_total") or cr.get("avg"),
                    "over75": cr.get("over75"),
                    "over85": cr.get("over85")
                },
                "top_scores": [{"score": s.get("score"), "probability": s.get("probability")} for s in top_scores_raw]
            },
            "prediction_summary": p.get("prediction_summary"),
            "recommendations": p.get("recommendations")
        })
    filepath = os.path.join(TAHMINLER_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    return jsonify({"ok": True, "filename": filename, "count": len(export), "path": filepath})


@app.route("/api/predictions/export")
def api_predictions_export():
    preds = _load_predictions()
    export = []
    for p in preds:
        r = p.get("result", {})
        tg = p.get("total_goals", {})
        cr = p.get("corners", {})
        dc = p.get("double_chance", {})
        g = p.get("goals", {})
        export.append({
            "id": p.get("id"),
            "timestamp": p.get("timestamp"),
            "match": f"{p.get('home_name','?')} vs {p.get('away_name','?')}",
            "home_team": p.get("home_name"),
            "away_team": p.get("away_name"),
            "home_elo": p.get("home_elo", 0),
            "away_elo": p.get("away_elo", 0),
            "predictions": {
                "1X2": {"home": r.get("home_win"), "draw": r.get("draw"), "away": r.get("away_win"),
                         "winner": r.get("winner"), "winner_prob": r.get("winner_prob")},
                "double_chance": {"1X": dc.get("1X"), "12": dc.get("12"), "X2": dc.get("X2")},
                "btts": {"yes": p.get("btts_yes"), "no": p.get("btts_no")},
                "goals": {
                    "expected": {"home_xg": g.get("home_lambda"), "away_xg": g.get("away_lambda"), "total_xg": g.get("expected_total")},
                    "over_under": {"1.5": tg.get("over15"), "2.5": tg.get("over25"), "3.5": tg.get("over35")}
                },
                "corners": {"average": cr.get("avg"), "over75": cr.get("over75"), "over85": cr.get("over85")},
                "top_scores": [{"score": s.get("score"), "probability": s.get("probability")} for s in p.get("top_scores", [])]
            },
            "recommendations": p.get("recommendations")
        })
    return jsonify(export)


# ── KUPONLAR (COUPONS) ──────────────────────────────────────────────

def _build_id_lookup():
    """Feature ID ↔ takım adı eşleme sözlüğü oluştur."""
    lookup = {}
    for path in ("data/gold/team_id_to_name.json", "data/gold/team_id_unified.json"):
        fp = os.path.join(os.path.dirname(_web_dir), path)
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                tid = int(k) if str(k).isdigit() else None
                if tid and v:
                    norm = v.strip().lower()
                    if tid not in lookup:
                        lookup[tid] = norm
    return lookup


def _find_match_result(home_id, away_id, home_name=None, away_name=None, match_date=None, use_live=True):
    """Sonuc bulma sirasi (en guvenilir ve HIZLI kaynaklar once):
        1) Manuel + yerel onbellekler (results_cache.json, match_results.json) - disk, aninda
        2) Football-data.co.uk (ana kaynak, hizli, dogru)
        3) features.parquet - sadece match_date +-30 gun icinde
        4) Canli kaynaklar (TheSportsDB / ESPN / Flashscore / SofaScore) -
           YALNIZCA son 10 gun icindeki maclar icin (yeni biten maclari yakala)

    use_live=False ise 4. adim (pahali canli scraping) atlanir.
    Bu mod, kupon LISTESi sayfasinda kullanilir - canli skor kontrolu
    ayri /api/coupons/check_live endpoint'inde yapilir.

    match_date: Tahmin edilen macin tarihi (ISO string ya da None).
    Bu parametre verilirse sadece o tarihe yakin gercek maclar dondurulur.
Eski tarihli gecmis maclar 'sonuc' olarak kabul edilmez.
    """
    from datetime import datetime, timedelta

    if not home_name:
        home_name = (TEAM_INDEX.get(home_id) or {}).get("name") or TEAM_ID_MAP.get(home_id, {}).get("name") or _display_name(home_id, "")
    if not away_name:
        away_name = (TEAM_INDEX.get(away_id) or {}).get("name") or TEAM_ID_MAP.get(away_id, {}).get("name") or _display_name(away_id, "")

    if not home_name or not away_name:
        home_name = str(home_id)
        away_name = str(away_id)

    # 0. Manuel (kullanici girisi) sonuc - en oncelikli
    try:
        manual = _load_manual_results()
        for cid, mr in manual.items():
            prefix = cid.split("-")
            if len(prefix) >= 2 and prefix[0] == str(home_id) and prefix[1] == str(away_id):
                return {
                    "home_goals": int(mr.get("home_goals", 0)),
                    "away_goals": int(mr.get("away_goals", 0)),
                    "total_goals": int(mr.get("home_goals", 0)) + int(mr.get("away_goals", 0)),
                    "result": mr.get("result", "D"),
                    "btts": int(mr.get("btts", 0)),
                    "over25": int(mr.get("over25", 0)),
                    "home_corners": None,
                    "away_corners": None,
                    "ht_home_goals": None,
                    "ht_away_goals": None,
                    "date": mr.get("date", ""),
                    "source": "manual",
                }
    except Exception:
        pass

    # match_date'i datetime objesine cevir
    _match_dt = None
    if match_date:
        try:
            _match_dt = datetime.fromisoformat(str(match_date)[:10])
        except Exception:
            _match_dt = None

    # Eger match_date gelecekte ise (henuz oynanmamis), direkt None don
    if _match_dt and _match_dt.date() > datetime.now().date():
        return None

    def _date_close(source_date, max_days=7):
        """Sonuc tarihi tahmin tarihine yakin mi? (Karisik maci onle)."""
        if not source_date or not _match_dt:
            return True
        try:
            dd = datetime.fromisoformat(str(source_date)[:10])
            return abs((dd - _match_dt).days) <= max_days
        except Exception:
            return True

    # 1. results_cache.json (TheSportsDB onbellegi) - disk, hizli
    try:
        cached = _find_result_from_cache(home_name, away_name, _load_results_cache())
        if cached and _date_close(cached.get("date", "")):
            hg, ag = cached["home_goals"], cached["away_goals"]
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
                "date": cached.get("date", ""),
                "source": cached.get("source", "thesportsdb"),
            }
    except Exception:
        pass

    # 1.5. match_results.json (match_score_fetcher onbellegi) - disk, hizli
    try:
        import prediction.match_score_fetcher as _ms_fetcher
        _mres = _ms_fetcher.load_local_match_results().get(f"{home_id}-{away_id}")
        if _mres and isinstance(_mres, dict) and _mres.get("home_goals") is not None and _date_close(_mres.get("date", "")):
            hg = int(_mres["home_goals"]); ag = int(_mres["away_goals"])
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
                "date": _mres.get("date", ""),
                "source": _mres.get("source", "match_results"),
            }
    except Exception:
        pass

    # 2. Football-data.co.uk (ana kaynak)
    fd = _fetch_result_from_football_data(home_id, away_id, home_name, away_name)
    if fd:
        # FD sonucunun tarihi match_date'e yakin mi kontrol et
        if _match_dt and fd.get("date"):
            try:
                fd_dt = datetime.fromisoformat(str(fd["date"])[:10])
                if abs((fd_dt - _match_dt).days) > 30:
                    fd = None  # Cok eski FD sonucu - gercek mac degil
            except Exception:
                pass
        if fd:
            return fd

    # 3. features.parquet fallback
    feat = STATE.get("features")
    if feat is not None:
        try:
            # Takim ID kümesini olustur (name_map uzerinden tum varyasyonlar)
            name_map = _load_turkish_display()
            name_to_ids = {}
            for k, v in name_map.items():
                norm = v.strip().lower()
                name_to_ids.setdefault(norm, set()).add(str(k))
            
            def get_ids(tid, tname):
                ids = {str(tid)}
                if tname:
                    norm = tname.strip().lower()
                    if norm in name_to_ids:
                        ids.update(name_to_ids[norm])
                    for k, v in name_map.items():
                        if norm in v.lower() or v.lower() in norm:
                            ids.add(str(k))
                return ids
            
            h_ids = get_ids(home_id, home_name)
            a_ids = get_ids(away_id, away_name)
            
            feat_home_str = feat["home_team_id"].astype(str)
            feat_away_str = feat["away_team_id"].astype(str)
            
            h2h_mask = (
                ((feat_home_str.isin(h_ids)) & (feat_away_str.isin(a_ids))) |
                ((feat_home_str.isin(a_ids)) & (feat_away_str.isin(h_ids)))
            )

            if _match_dt and "date" in feat.columns:
                # Sadece match_date +-30 gun icindeki maci kabul et
                mdt = pd.to_datetime(_match_dt)
                date_mask = (
                    (feat["date"] >= mdt - pd.Timedelta(days=30)) &
                    (feat["date"] <= mdt + pd.Timedelta(days=30))
                )
                match = feat[h2h_mask & date_mask]
            else:
                # match_date yoksa son 365 gun, yoksa hic bir sey donme
                cutoff = pd.to_datetime(datetime.now() - timedelta(days=365))
                recent_mask = feat["date"] >= cutoff if "date" in feat.columns else pd.Series(True, index=feat.index)
                match = feat[h2h_mask & recent_mask]
            
            if len(match) > 0:
                if "date" in match.columns:
                    match = match.sort_values("date")
                row = match.iloc[-1]
                hg = int(row["home_goals"])
                ag = int(row["away_goals"])
                hc = int(row["home_corners"]) if "home_corners" in row and pd.notna(row["home_corners"]) else None
                ac = int(row["away_corners"]) if "away_corners" in row and pd.notna(row["away_corners"]) else None
                hth = int(row["ht_home_goals"]) if "ht_home_goals" in row and pd.notna(row["ht_home_goals"]) else None
                hta = int(row["ht_away_goals"]) if "ht_away_goals" in row and pd.notna(row["ht_away_goals"]) else None
                
                is_home = str(row["home_team_id"]) in h_ids
                if not is_home:
                    hg, ag = ag, hg
                    hc, ac = ac, hc
                    hth, hta = hta, hth
                total = hg + ag
                return {
                    "home_goals": hg,
                    "away_goals": ag,
                    "total_goals": total,
                    "result": "H" if hg > ag else ("D" if hg == ag else "A"),
                    "btts": 1 if hg > 0 and ag > 0 else 0,
                    "over25": 1 if total >= 3 else 0,
                    "home_corners": hc,
                    "away_corners": ac,
                    "ht_home_goals": hth,
                    "ht_away_goals": hta,
                    "date": str(row["date"])[:10] if hasattr(row["date"], "strftime") else str(row.get("date", "")),
                    "source": "features_parquet",
                }
        except Exception:
            pass

    # 3.5. Flashscore bitmis mac sonuclari (cache'ler - her zaman kullanilir).
    # Hem LIVE hem FINISHED cache'lerini bitmis skorlari icin birleştir:
    # canli cache'te status_short=="FT" olan maçlar da zaten BITTIR.
    try:
        # Disk'teki dolu finished verisini garantile - bellek cache'i bos olsa bile.
        _ensure_finished_cache()
        finished = []
        for fm in (_FLASHSCORE_FINISHED_CACHE or []):
            hg_v = fm.get("home_score") if fm.get("home_score") is not None else fm.get("home_goals")
            ag_v = fm.get("away_score") if fm.get("away_score") is not None else fm.get("away_goals")
            if hg_v is not None and ag_v is not None:
                finished.append(fm)
        _fix_fin = _FIXTURES_CACHE.get("finished", {}).get("data")
        if _fix_fin:
            for fm in _fix_fin:
                st = (fm.get("status_short") or "").upper()
                if st in ("FT", "AET", "PEN"):
                    hg_v = fm.get("home_score") if fm.get("home_score") is not None else fm.get("home_goals")
                    ag_v = fm.get("away_score") if fm.get("away_score") is not None else fm.get("away_goals")
                    if hg_v is not None and ag_v is not None:
                        finished.append(fm)
        for fm in (_FLASHSCORE_LIVE_CACHE or []):
            st = (fm.get("status_short") or "").upper()
            if st in ("FT", "AET", "PEN"):
                hg_v = fm.get("home_score") if fm.get("home_score") is not None else fm.get("home_goals")
                ag_v = fm.get("away_score") if fm.get("away_score") is not None else fm.get("away_goals")
                if hg_v is not None and ag_v is not None:
                    finished.append(fm)

        if finished:
            best_match = None
            best_score = -1
            for fm in finished:
                fm_home = (fm.get("home_name", "") or "").strip().lower()
                fm_away = (fm.get("away_name", "") or "").strip().lower()
                q_home = (home_name or "").strip().lower()
                q_away = (away_name or "").strip().lower()

                if not q_home or not q_away or not fm_home or not fm_away:
                    continue

                q_home_norm = _norm_name(q_home)
                q_away_norm = _norm_name(q_away)
                fm_home_norm = _norm_name(fm_home)
                fm_away_norm = _norm_name(fm_away)

                q_home_words = set([w for w in q_home_norm.split() if len(w) > 3])
                q_away_words = set([w for w in q_away_norm.split() if len(w) > 3])
                fm_home_words = set([w for w in fm_home_norm.split() if len(w) > 3])
                fm_away_words = set([w for w in fm_away_norm.split() if len(w) > 3])

                # ALT TAKIM FİLTRESİ: Sorgu senior takim ise (U19/U21/rezer/b gibi eklenti
                # yoksa) bulunan mac da senior olmali. Asil takim yerine U19/rezerv mackini
                # yanlis eslestirmeyi engeller (ornek: "Trabzonspor" vs "Trabzonspor U19").
                if _is_senior_team(q_home_norm) and not _is_senior_team(fm_home_norm):
                    continue
                if _is_senior_team(q_away_norm) and not _is_senior_team(fm_away_norm):
                    continue

                home_match = (
                    q_home_norm == fm_home_norm or
                    (q_home_norm in fm_home_norm and len(q_home_norm) > 3) or
                    (fm_home_norm in q_home_norm and len(fm_home_norm) > 3) or
                    (q_home_words and fm_home_words and len(q_home_words & fm_home_words) >= min(2, len(q_home_words))) or
                    (len(q_home_norm) >= 4 and len(fm_home_norm) >= 4 and difflib.SequenceMatcher(None, q_home_norm, fm_home_norm).ratio() >= 0.80)
                )
                away_match = (
                    q_away_norm == fm_away_norm or
                    (q_away_norm in fm_away_norm and len(q_away_norm) > 3) or
                    (fm_away_norm in q_away_norm and len(fm_away_norm) > 3) or
                    (q_away_words and fm_away_words and len(q_away_words & fm_away_words) >= min(2, len(q_away_words))) or
                    (len(q_away_norm) >= 4 and len(fm_away_norm) >= 4 and difflib.SequenceMatcher(None, q_away_norm, fm_away_norm).ratio() >= 0.80)
                )

                if not (home_match and away_match):
                    continue

                # Format destegi: flashscore_scraper home_score, api_fixtures home_goals kullanir
                hg_v = fm.get("home_score") if fm.get("home_score") is not None else fm.get("home_goals")
                ag_v = fm.get("away_score") if fm.get("away_score") is not None else fm.get("away_goals")
                if hg_v is None or ag_v is None:
                    continue
                hg, ag = int(hg_v), int(ag_v)
                kickoff = fm.get("kickoff", "")

                # Tarih yakinligi puani: _match_dt'e en yakin eslesme olsun, maks 7 gun tolerans
                if _match_dt and kickoff:
                    try:
                        fm_dt = datetime.fromisoformat(str(kickoff)[:10])
                        date_diff = abs((fm_dt - _match_dt).days)
                        if date_diff > 7:
                            continue
                        score = -date_diff
                    except Exception:
                        score = 0
                else:
                    score = 0

                if score > best_score:
                    best_score = score
                    best_match = (hg, ag, kickoff)

            if best_match:
                hg, ag, kickoff = best_match
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
                    "date": kickoff[:10] if kickoff else "",
                    "source": "flashscore_finished",
                }
    except Exception as e:
        print(f"[Flashscore] Bitmis mac arama hatasi: {e}")

    # 4. Canli kaynaklar - YALNIZCA guncel maclar (son 10 gun).
    # use_live=False ise bu pahali blok atlanir (canli scraping/API cagrilar).
    if use_live and (_match_dt is None or (datetime.now().date() - _match_dt.date()).days <= 10):
        # 4.1. Akıllı Maç Skoru Çekici (Web Search Scraper + Persistent Cache)
        try:
            import prediction.match_score_fetcher as score_fetcher
            ms_res = score_fetcher.get_match_result(home_id, away_id, home_name, away_name, match_date)
            if ms_res:
                # Tarih doğrulaması: bulunan sonuç tahmin tarihine uygun mu?
                ms_date = ms_res.get("date", "")
                if ms_date and _match_dt:
                    try:
                        ms_dt = datetime.fromisoformat(str(ms_date)[:10])
                        if abs((ms_dt - _match_dt).days) > 7:
                            ms_res = None  # Çok farklı tarih - yanlış maç olabilir
                    except Exception:
                        pass
                if ms_res:
                    return ms_res
        except Exception:
            pass

        # 4.3. SofaScore (yedek, yavas)
        result = _fetch_result_from_sofascore(home_id, away_id)
        if result:
            return result

    return None


# Football-data.co.uk lig eslesmesi (takim adi -> lig kodu)
FD_LEAGUE_MAP = {
    "super lig": "T1", "turkish super lig": "T1", "turkey": "T1", "tuerkiye": "T1",
    "premier league": "E0", "english premier": "E0", "england": "E0", "ingiltere": "E0",
    "championship": "E1", "efl championship": "E1",
    "la liga": "SP1", "primera division": "SP1", "spain": "SP1", "ispanya": "SP1",
    "segunda": "SP2",
    "bundesliga": "D1", "germany": "D1", "almanya": "D1",
    "2. bundesliga": "D2", "2.bundesliga": "D2",
    "serie a": "I1", "italy": "I1", "italya": "I1",
    "serie b": "I2",
    "ligue 1": "F1", "ligue1": "F1", "france": "F1", "fransa": "F1",
    "ligue 2": "F2",
    "eredivisie": "N1", "holland": "N1", "hollanda": "N1",
    "primeira liga": "P1", "portugal": "P1",
}

# Takim id -> lig kodu (team_index.pkl icinden ogreniliyor)
_TEAM_LEAGUE_CACHE = {}

FD_CACHE_FILE = os.path.join(TAHMINLER_DIR, "fd_results_cache.json")
_FD_CACHE_MEMORY = None
_FD_CACHE_MEMORY_MTIME = 0.0


def _resolve_fd_leagues(home_id, away_id, home_name):
    """Takim ID'si ve adindan FD lig kodlarini belirle (oncelik sirasi)."""
    import re
    codes = []
    for tid in (home_id, away_id):
        if tid and tid in TEAM_ID_MAP:
            lg = TEAM_ID_MAP[tid].get("league", "")
            lc = TEAM_ID_MAP[tid].get("league_code", "")
            # league_code T1 gibi temiz ise onu kullan
            if lc and re.match(r"^[A-Z][0-9]$", lc):
                codes.append(lc)
                continue
            # Degilse league adindan cikarim yap
            lk = (lg or "").lower()
            for key, code in FD_LEAGUE_MAP.items():
                if key in lk and code not in codes:
                    codes.append(code)
                    break
    # 2) Takim adindan lig tahmini (geriye donuk uyumluluk)
    hn_lower = (home_name or "").lower()
    for keyword, code in FD_LEAGUE_MAP.items():
        if keyword in hn_lower and code not in codes:
            codes.append(code)
            break
    # 3) Hicbir sey bulunamadiysa tum buyuk 6 ligi dene
    if not codes:
        codes = ["T1", "E0", "SP1", "D1", "D2", "E1", "I1", "F1"]
    return codes


def _fetch_result_from_football_data(home_id, away_id, home_name, away_name):
    """Football-data.co.uk CSV'lerinden mac sonucunu bul (ana kaynak)."""
    import csv
    import io
    import urllib.request
    import re as _re
    from datetime import datetime, timedelta

    def _fd_normalize(name):
        """FD icin takim adini normallestir."""
        if not name: return ""
        tr_map = str.maketrans("çğıöşüâîûê","cgiosuaiue")
        n = name.strip().lower().translate(tr_map)
        return " ".join(n.split())

    FD_ALIASES = {
        "çaykur rizespor": ["rizespor"],
        "göztepe": ["goztep"],
        "gaziantepspor": ["gaziantep", "gaziantep fk"],
        "gaziantep fk": ["gaziantep", "gaziantepspor"],
        "istanbul başakşehir": ["buyuksehyr", "basaksehir"],
        "gençlerbirliği": ["genclerbirligi"],
        "çorum fk": ["corum"],
        "fatih karagümrük": ["karagumruk"],
        "kasımpaşa": ["kasimpasa"],
        "adana demirspor": ["ad. demirspor", "adana demirspor"],
        "ankaragücü": ["ankaragucu"],
        "man united": ["manchester united", "man united"],
        "manchester united": ["man united"],
        "manchester city": ["man city"],
        "wolverhampton": ["wolves"],
        "nottingham forest": ["nott'm forest"],
        "newcastle united": ["newcastle"],
        "leeds united fc": ["leeds"],
        "ipswich town fc": ["ipswich"],
        "hull city afc": ["hull"],
        "1. fc union berlin": ["union berlin"],
        "1. fc köln": ["koeln", "koln"],
        "m'gladbach": ["m-gladbach", "monchengladbach", "m'gladbach"],
    }

    def _fd_match(query, target):
        """Fuzzy takim eslestirme - alias + tam + kelime eslesmesi."""
        q = _fd_normalize(query)
        t = _fd_normalize(target)
        if not q or not t: return False
        if q == t: return True
        if q in t or t in q: return True
        # Aliases
        aliases = FD_ALIASES.get(query.strip().lower(), [])
        for a in aliases:
            na = _fd_normalize(a)
            if na == t or na in t or t in na:
                return True
        q_words = [w for w in q.split() if len(w) >= 4]
        t_words = [w for w in t.split() if len(w) >= 4]
        for qw in q_words:
            for tw in t_words:
                if qw in tw or tw in qw:
                    return True
        return False

    leagues_to_try = _resolve_fd_leagues(home_id, away_id, home_name)

    # Cache yukle (bellek oncelikli: ayni istek icinde tekrar disk okuma)
    global _FD_CACHE_MEMORY, _FD_CACHE_MEMORY_MTIME
    fd_cache = _FD_CACHE_MEMORY
    if fd_cache is None:
        fd_cache = {}
        if os.path.exists(FD_CACHE_FILE):
            try:
                with open(FD_CACHE_FILE, "r", encoding="utf-8") as f:
                    fd_cache = json.load(f)
                _FD_CACHE_MEMORY = fd_cache
                _FD_CACHE_MEMORY_MTIME = os.path.getmtime(FD_CACHE_FILE)
            except Exception:
                fd_cache = {}
    else:
        # Diskti degisiklik varsa yenile
        try:
            if os.path.exists(FD_CACHE_FILE):
                mt = os.path.getmtime(FD_CACHE_FILE)
                if mt != _FD_CACHE_MEMORY_MTIME:
                    with open(FD_CACHE_FILE, "r", encoding="utf-8") as f:
                        fd_cache = json.load(f)
                    _FD_CACHE_MEMORY = fd_cache
                    _FD_CACHE_MEMORY_MTIME = mt
        except Exception:
            pass

    now = datetime.now()
    best_match = None  # (skor, sonuc) tuple

    for lg in leagues_to_try:
        cache_key = f"fd_{lg}"
        rows = None

        # Sayfa/onbellek yolu asla aga takilmasin: cache'te ne varsa onu kullan.
        # Tazelik AUTO-FETCH / gunluk sync arka planda hallediyor.
        if cache_key in fd_cache:
            cached = fd_cache[cache_key]
            try:
                rows = cached.get("rows", [])
            except Exception:
                rows = None

        if rows is None:
            url = f"https://www.football-data.co.uk/mmz4281/2627/{lg}.csv"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    raw = resp.read()
                    # BOM olabilir
                    text = raw.decode("utf-8-sig", errors="replace")
                reader = csv.DictReader(io.StringIO(text))
                rows = [dict(r) for r in reader]
                fd_cache[cache_key] = {"fetched_at": now.isoformat(), "rows": rows}
                with open(FD_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(fd_cache, f, ensure_ascii=False)
                _FD_CACHE_MEMORY = fd_cache
                _FD_CACHE_MEMORY_MTIME = os.path.getmtime(FD_CACHE_FILE)
            except Exception:
                continue

        for row in rows:
            r_home = row.get("HomeTeam", "").strip()
            r_away = row.get("AwayTeam", "").strip()
            if not r_home or not r_away:
                continue

            if not (_fd_match(home_name, r_home) and _fd_match(away_name, r_away)):
                continue

            hg = row.get("FTHG")
            ag = row.get("FTAG")
            if hg is None or ag is None or hg == "" or ag == "":
                continue
            try:
                hg, ag = int(hg), int(ag)
            except (ValueError, TypeError):
                continue

            total = hg + ag
            date_str = row.get("Date", "")
            # mac tarihini ISO formatina cevir ve "yakınlık" puani hesapla
            iso = _parse_fd_date(date_str, lg)
            days_old = 999
            try:
                d = datetime.fromisoformat(iso)
                days_old = abs((now - d).days)
            except Exception:
                pass

            # mac 60 gunden eski ise atla
            if days_old > 60:
                continue

            score = -days_old  # en yeni en iyi
            if best_match is None or score > best_match[0]:
                best_match = (score, {
                    "home_goals": hg,
                    "away_goals": ag,
                    "total_goals": total,
                    "result": "H" if hg > ag else ("D" if hg == ag else "A"),
                    "btts": 1 if hg > 0 and ag > 0 else 0,
                    "over25": 1 if total >= 3 else 0,
                    "home_corners": None,
                    "away_corners": None,
                    "ht_home_goals": _safe_int(row.get("HTHG")),
                    "ht_away_goals": _safe_int(row.get("HTAG")),
                    "date": iso or date_str,
                    "source": "football-data.co.uk",
                })

    if best_match:
        return best_match[1]
    return None


def _safe_int(v):
    """String/int'i guvenli sekilde int'e cevir."""
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _parse_fd_date(date_str, league_code):
    """FD tarihini ISO formatina cevir (dd/mm/yy veya dd/mm/yyyy)."""
    if not date_str:
        return ""
    import re as _re
    m = _re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", date_str.strip())
    if not m:
        return date_str
    d, mo, y = m.groups()
    if len(y) == 2:
        # 26/27 sezonu icin 26 -> 2026, 27 -> 2027 (FD 2627 sezonu)
        yi = int(y)
        y = "20" + y if yi < 50 else "19" + y
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def _fetch_result_from_sofascore(home_id, away_id):
    """SofaScore public API 403 donduruyor (bot korumasi).
    Takim ID'leri de SofaScore'un kendi ID'leri degil - API calismiyor.
    Bos dondur: diger kaynaklar (FD) zaten ana kaynak."""
    return None


def _match_sofascore_events(events, home_id, away_id):
    """SofaScore event listesinde eslesme bul."""
    for ev in events:
        ev_home = ev.get("homeTeam", {})
        ev_away = ev.get("awayTeam", {})
        ev_home_id = ev_home.get("id")
        ev_away_id = ev_away.get("id")
        status = ev.get("status", {})

        # Sadece bitmis macalar
        if status.get("type") != "finished":
            continue

        # Ev sahibi / deplasman eslesmesi
        if (ev_home_id == home_id and ev_away_id == away_id) or \
           (ev_home_id == away_id and ev_away_id == home_id):
            hg = ev.get("homeScore", {}).get("current")
            ag = ev.get("awayScore", {}).get("current")
            if hg is None or ag is None:
                continue

            # Dogru yonde mi?
            is_home = ev_home_id == home_id
            if not is_home:
                hg, ag = ag, hg

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
                "date": ev.get("startTimestamp", ""),
                "source": "sofascore",
            }
    return None


def _evaluate_coupon(pred, actual):
    """Tahmini gerçek sonuçla karşılaştır, her market için tuttu/tutmadı."""
    if not actual:
        return None

    hg = actual.get("home_goals", 0)
    ag = actual.get("away_goals", 0)
    total = actual.get("total_goals", hg + ag)
    actual_result = actual.get("result", "H" if hg > ag else ("D" if hg == ag else "A"))
    actual_btts = actual.get("btts", 1 if hg > 0 and ag > 0 else 0)
    hc = actual.get("home_corners")
    ac = actual.get("away_corners")
    actual_corners = (hc + ac) if (hc is not None and ac is not None) else None

    markets = []

    # 1X2
    r = pred.get("result", {})
    home_w, draw, away_w = r.get("home_win", 0), r.get("draw", 0), r.get("away_win", 0)
    if home_w >= draw and home_w >= away_w:
        pred_1x2 = "1"
        pred_1x2_prob = home_w
    elif draw >= home_w and draw >= away_w:
        pred_1x2 = "X"
        pred_1x2_prob = draw
    else:
        pred_1x2 = "2"
        pred_1x2_prob = away_w

    actual_1x2 = "1" if hg > ag else ("X" if hg == ag else "2")
    markets.append({
        "name": "1X2",
        "prediction": pred_1x2,
        "probability": pred_1x2_prob,
        "actual": actual_1x2,
        "actual_detail": f"{hg}-{ag}",
        "won": pred_1x2 == actual_1x2,
    })

    # Gol Üst/Alt 2.5
    tg = pred.get("total_goals", {})
    over25_pred = tg.get("over25", 50)
    pred_goal_market = "Üst 2.5" if over25_pred >= 50 else "Alt 2.5"
    actual_goal_market = "Üst 2.5" if total >= 3 else "Alt 2.5"
    markets.append({
        "name": "Gol Üst/Alt 2.5",
        "prediction": f"{pred_goal_market} (%{over25_pred:.0f})",
        "probability": over25_pred if over25_pred >= 50 else 100 - over25_pred,
        "actual": actual_goal_market,
        "actual_detail": f"Toplam: {total} gol",
        "won": pred_goal_market == actual_goal_market,
    })

    # BTTS
    btts_yes = pred.get("btts_yes", 50)
    pred_btts = "Var" if btts_yes >= 50 else "Yok"
    actual_btts_str = "Var" if actual_btts else "Yok"
    markets.append({
        "name": "BTTS (Karşılıklı Gol)",
        "prediction": f"{pred_btts} (%{btts_yes:.0f})",
        "probability": btts_yes if btts_yes >= 50 else 100 - btts_yes,
        "actual": actual_btts_str,
        "actual_detail": f"Ev: {hg} Gol, Dep: {ag} Gol",
        "won": pred_btts == actual_btts_str,
    })

    # Korner Üst/Alt 7.5
    cr = pred.get("corners", {})
    cr_over75 = cr.get("over75", 50)
    pred_corner = "Üst 7.5" if cr_over75 >= 50 else "Alt 7.5"
    if actual_corners is not None:
        actual_corner = "Üst 7.5" if actual_corners >= 8 else "Alt 7.5"
        markets.append({
            "name": "Korner Üst/Alt 7.5",
            "prediction": f"{pred_corner} (%{cr_over75:.0f})",
            "probability": cr_over75 if cr_over75 >= 50 else 100 - cr_over75,
            "actual": actual_corner,
            "actual_detail": f"Toplam: {actual_corners} korner",
            "won": pred_corner == actual_corner,
        })

    # Çifte Şans
    dc = pred.get("double_chance", {})
    dc_1x, dc_12, dc_x2 = dc.get("1X", 50), dc.get("12", 50), dc.get("X2", 50)
    best_dc = max(dc_1x, dc_12, dc_x2)
    if best_dc == dc_12:
        pred_dc = "12"
    elif best_dc == dc_1x:
        pred_dc = "1X"
    else:
        pred_dc = "X2"

    actual_res = "1" if hg > ag else ("X" if hg == ag else "2")
    if pred_dc == "1X":
        dc_won = actual_res in ("1", "X")
    elif pred_dc == "12":
        dc_won = actual_res in ("1", "2")
    else:
        dc_won = actual_res in ("X", "2")

    markets.append({
        "name": "Çifte Şans",
        "prediction": pred_dc,
        "probability": best_dc,
        "actual": actual_res,
        "actual_detail": f"{hg}-{ag}",
        "won": dc_won,
    })

    # İlk Yarı Skoru (varsa)
    v15 = pred.get("v15_markets", {})
    ht_home = actual.get("ht_home_goals")
    ht_away = actual.get("ht_away_goals")
    if ht_home is not None and ht_away is not None:
        ht_total = ht_home + ht_away
        iy_over05 = v15.get("iy_over05", 50)
        pred_iy = "Üst 0.5" if iy_over05 >= 50 else "Alt 0.5"
        actual_iy = "Üst 0.5" if ht_total >= 1 else "Alt 0.5"
        markets.append({
            "name": "İlk Yarı Gol Üst/Alt 0.5",
            "prediction": f"{pred_iy} (%{iy_over05:.0f})",
            "probability": iy_over05 if iy_over05 >= 50 else 100 - iy_over05,
            "actual": actual_iy,
            "actual_detail": f"IY: {ht_home}-{ht_away}",
            "won": pred_iy == actual_iy,
        })

    # --- Ek marketler ---
    # Gol Ust/Alt 1.5
    tg = pred.get("total_goals", {})
    over15_pred = tg.get("over15", 50)
    pred_g15 = "Ust 1.5" if over15_pred >= 50 else "Alt 1.5"
    actual_g15 = "Ust 1.5" if total >= 2 else "Alt 1.5"
    markets.append({
        "name": "Gol Ust/Alt 1.5",
        "prediction": f"{pred_g15} (%{over15_pred:.0f})",
        "probability": over15_pred if over15_pred >= 50 else 100 - over15_pred,
        "actual": actual_g15,
        "actual_detail": f"Toplam: {total} gol",
        "won": pred_g15 == actual_g15,
    })

    # Gol Ust/Alt 3.5
    over35_pred = tg.get("over35", 50)
    pred_g35 = "Ust 3.5" if over35_pred >= 50 else "Alt 3.5"
    actual_g35 = "Ust 3.5" if total >= 4 else "Alt 3.5"
    markets.append({
        "name": "Gol Ust/Alt 3.5",
        "prediction": f"{pred_g35} (%{over35_pred:.0f})",
        "probability": over35_pred if over35_pred >= 50 else 100 - over35_pred,
        "actual": actual_g35,
        "actual_detail": f"Toplam: {total} gol",
        "won": pred_g35 == actual_g35,
    })

    # Korner Ust/Alt 8.5
    cr = pred.get("corners", {})
    cr_over85 = cr.get("over85", 50)
    if actual_corners is not None:
        pred_c85 = "Ust 8.5" if cr_over85 >= 50 else "Alt 8.5"
        actual_c85 = "Ust 8.5" if actual_corners >= 9 else "Alt 8.5"
        markets.append({
            "name": "Korner Ust/Alt 8.5",
            "prediction": f"{pred_c85} (%{cr_over85:.0f})",
            "probability": cr_over85 if cr_over85 >= 50 else 100 - cr_over85,
            "actual": actual_c85,
            "actual_detail": f"Toplam: {actual_corners} korner",
            "won": pred_c85 == actual_c85,
        })

    # Korner Ust/Alt 9.5
    cr_over95 = cr.get("over95", 50)
    if actual_corners is not None:
        pred_c95 = "Ust 9.5" if cr_over95 >= 50 else "Alt 9.5"
        actual_c95 = "Ust 9.5" if actual_corners >= 10 else "Alt 9.5"
        markets.append({
            "name": "Korner Ust/Alt 9.5",
            "prediction": f"{pred_c95} (%{cr_over95:.0f})",
            "probability": cr_over95 if cr_over95 >= 50 else 100 - cr_over95,
            "actual": actual_c95,
            "actual_detail": f"Toplam: {actual_corners} korner",
            "won": pred_c95 == actual_c95,
        })

    # CGol (Ev sahibi gol Ust/Alt 1.5)
    goals = pred.get("goals", {})
    home_over15 = goals.get("home_over15", 50)
    if home_over15:
        pred_hg15 = "Ust 1.5" if home_over15 >= 50 else "Alt 1.5"
        actual_hg15 = "Ust 1.5" if hg >= 2 else "Alt 1.5"
        markets.append({
            "name": "Ev Sahibi Gol Ust/Alt 1.5",
            "prediction": f"{pred_hg15} (%{home_over15:.0f})",
            "probability": home_over15 if home_over15 >= 50 else 100 - home_over15,
            "actual": actual_hg15,
            "actual_detail": f"Ev: {hg} gol",
            "won": pred_hg15 == actual_hg15,
        })

    # Dep gol Ust/Alt 1.5
    away_over15 = goals.get("away_over15", 50)
    if away_over15:
        pred_ag15 = "Ust 1.5" if away_over15 >= 50 else "Alt 1.5"
        actual_ag15 = "Ust 1.5" if ag >= 2 else "Alt 1.5"
        markets.append({
            "name": "Dep. Gol Ust/Alt 1.5",
            "prediction": f"{pred_ag15} (%{away_over15:.0f})",
            "probability": away_over15 if away_over15 >= 50 else 100 - away_over15,
            "actual": actual_ag15,
            "actual_detail": f"Dep: {ag} gol",
            "won": pred_ag15 == actual_ag15,
        })

    won_count = sum(1 for m in markets if m["won"])
    return {
        "markets": markets,
        "won_count": won_count,
        "total_markets": len(markets),
        "score": f"{won_count}/{len(markets)}",
    }


@app.route("/coupons")
def coupons():
    return render_template("coupons.html", loading=STATE["loading"])


@app.route("/api/coupons/manual_result", methods=["POST"])
def manual_result():
    data = request.json
    coupon_id = data.get("coupon_id")
    home_goals = data.get("home_goals")
    away_goals = data.get("away_goals")

    if not coupon_id or home_goals is None or away_goals is None:
        return jsonify({"error": "Eksik parametre"}), 400

    try:
        hg = int(home_goals)
        ag = int(away_goals)
        _save_manual_result(coupon_id, hg, ag)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/coupons/sync_scores", methods=["POST"])
def sync_scores():
    """Tüm tahminlerin skorlarını web/search üzerinden senkronize et."""
    try:
        import prediction.match_score_fetcher as score_fetcher
        preds = _load_predictions()
        synced = 0
        for pred in preds:
            hid = pred.get("home_id")
            aid = pred.get("away_id")
            hname = pred.get("home_name")
            aname = pred.get("away_name")
            mdate = pred.get("match_date", "")
            if hid and aid:
                res = score_fetcher.get_match_result(hid, aid, hname, aname, mdate)
                if res:
                    synced += 1
        return jsonify({"success": True, "synced_count": synced, "total_predictions": len(preds)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/coupons/check_live", methods=["POST"])
def api_check_live_coupons():
    """Bekleyen kuponlari canli/bitmis mac sonuclarina gore kontrol et.

    Istege gore filtreleme:
    - only_pending: sadece sonuclanmamis kuponlari kontrol et (default: true)
    - match_ids: belirli mac ID'lerini kontrol et

    Donus:
    - checked: kontrol edilen kupon sayisi
    - found: sonuc bulunan kupon sayisi
    - results: [{id, status, evaluation, actual_score, ...}]
    """
    try:
        import threading
        lock = getattr(app, "_coupons_lock", None)
        if lock is None:
            lock = threading.Lock()
            app._coupons_lock = lock

        req_data = request.get_json(silent=True) or {}
        only_pending = req_data.get("only_pending", True)
        target_ids = req_data.get("match_ids", [])

        # Flashscore cache'ini tek seferde doldur
        _refresh_flashscore_cache()
        
        print(f"[Coupons] Canli kontrol basladi: {len(_FLASHSCORE_LIVE_CACHE)} canli, {len(_FLASHSCORE_FINISHED_CACHE)} bitmis mac cache'de")

        with lock:
            preds = _load_predictions()
            results = []
            checked = 0
            found = 0
            saved = 0

            # Sonuc bulunan maclari kaydetmek icin match_score_fetcher kullan
            import prediction.match_score_fetcher as score_fetcher

            for pred in preds:
                hid = pred.get("home_id")
                aid = pred.get("away_id")
                if not hid or not aid:
                    continue

                # Filtre: sadece bekleyenler
                if only_pending:
                    existing = _find_match_result(
                        hid, aid,
                        pred.get("home_name"), pred.get("away_name"),
                        pred.get("match_date"),
                        use_live=False  # Hizli kontrol, canli aramadan
                    )
                    if existing is not None:
                        continue  # Zaten sonucu var

                # Filtre: belirli mac ID'leri
                if target_ids:
                    pred_id = pred.get("id", "")
                    if pred_id not in target_ids:
                        continue

                checked += 1

                # Canli/bitmis mac sonucu bul (use_live=True ile)
                actual = _find_match_result(
                    hid, aid,
                    pred.get("home_name"), pred.get("away_name"),
                    pred.get("match_date"),
                    use_live=True
                )

                if actual is not None:
                    found += 1
                    evaluation = _evaluate_coupon(pred, actual)
                    
                    # Sonucu kaydet (match_results.json'a)
                    try:
                        cid = f"{hid}-{aid}"
                        score_fetcher.save_local_match_result(cid, {
                            "home_goals": actual.get("home_goals"),
                            "away_goals": actual.get("away_goals"),
                            "total_goals": actual.get("total_goals"),
                            "result": actual.get("result"),
                            "btts": actual.get("btts"),
                            "over25": actual.get("over25"),
                            "date": actual.get("date"),
                            "source": actual.get("source", "live_check")
                        })
                        saved += 1
                        print(f"[Coupons] Sonuc kaydedildi: {pred.get('home_name')} vs {pred.get('away_name')} = {actual['home_goals']}-{actual['away_goals']}")
                    except Exception as e:
                        print(f"[Coupons] Sonuc kaydetme hatasi: {e}")
                    
                    results.append({
                        "id": pred.get("id"),
                        "home_name": pred.get("home_name"),
                        "away_name": pred.get("away_name"),
                        "match_date": pred.get("match_date"),
                        "actual_score": f"{actual['home_goals']}-{actual['away_goals']}",
                        "actual_date": actual.get("date"),
                        "source": actual.get("source"),
                        "evaluation": evaluation,
                        "status": "found",
                    })
                else:
                    # Henuz sonuc yok - canli mi kontrol et
                    match_status = _check_match_liveness(
                        pred.get("home_name"),
                        pred.get("away_name"),
                        pred.get("match_date")
                    )
                    results.append({
                        "id": pred.get("id"),
                        "home_name": pred.get("home_name"),
                        "away_name": pred.get("away_name"),
                        "match_date": pred.get("match_date"),
                        "status": match_status,
                        "evaluation": None,
                    })

            print(f"[Coupons] Canli kontrol tamamlandi: {checked} kontrol, {found} bulundu, {saved} kaydedildi")

            return jsonify({
                "success": True,
                "checked": checked,
                "found": found,
                "saved": saved,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            })

    except Exception as e:
        print("[Coupons] canli kontrol hatasi:", e)
        return jsonify({"error": str(e)}), 500


def _check_match_liveness(home_name, away_name, match_date):
    """Macin canli/bitmis/durumunu kontrol et. Flashscore'dan canli veri cek."""
    if not match_date:
        return "unknown"

    try:
        md = datetime.fromisoformat(str(match_date)[:10])
        days_diff = (datetime.now().date() - md.date()).days

        # Gelecek maclar
        if days_diff < -1:
            return "future"
        elif days_diff == -1:
            return "tomorrow"
        elif days_diff >= 0 and days_diff <= 14:
            # Son 14 gun icindeki maclar - flashscore bitmis cache'i 30 gun kapsar ama 7 gun yeterli
            try:
                from prediction.flashscore_scraper import get_fixtures_flashscore
                _ensure_finished_cache()
                live_matches = _FLASHSCORE_LIVE_CACHE or []
                # Both sources for finished: _FLASHSCORE (coupon cache) + _FIXTURES (canli sekmesi)
                finished_matches = list(_FLASHSCORE_FINISHED_CACHE or [])
                _fix_fin = _FIXTURES_CACHE.get("finished", {}).get("data")
                if _fix_fin:
                    for m in _fix_fin:
                        if (m.get("status_short") or "").upper() in ("FT", "AET", "PEN"):
                            finished_matches.append(m)
                
                # Takim adi normalize
                q_home = (home_name or "").strip().lower()
                q_away = (away_name or "").strip().lower()

                # Once canli maclari kontrol et
                for m in live_matches:
                    m_home = (m.get("home_name", "") or "").strip().lower()
                    m_away = (m.get("away_name", "") or "").strip().lower()

                    # Takim adi eslesmesi - AKILLI kontrol (Türkçe normalize + esneklik)
                    if not q_home or not q_away or not m_home or not m_away:
                        continue
                    
                    # Türkçe karakter normalizasyonu
                    q_home_norm = _norm_name(q_home)
                    q_away_norm = _norm_name(q_away)
                    m_home_norm = _norm_name(m_home)
                    m_away_norm = _norm_name(m_away)
                    
                    # Kelime bazlı eşleştirme (en az 2 ortak kelime)
                    q_home_words = set([w for w in q_home_norm.split() if len(w) > 3])
                    q_away_words = set([w for w in q_away_norm.split() if len(w) > 3])
                    m_home_words = set([w for w in m_home_norm.split() if len(w) > 3])
                    m_away_words = set([w for w in m_away_norm.split() if len(w) > 3])
                    
                    # ALT TAKIM FILTRESI: asil takim yerine U19/rezerv/kadin mackini eslestirme
                    if _is_senior_team(q_home_norm) and not _is_senior_team(m_home_norm):
                        continue
                    if _is_senior_team(q_away_norm) and not _is_senior_team(m_away_norm):
                        continue
                    
                    # Eşleştirme: Tam eşleme VEYA uzun string içerme VEYA 2+ ortak kelime
                    home_match = (
                        q_home_norm == m_home_norm or
                        (q_home_norm in m_home_norm and len(q_home_norm) > 3) or
                        (m_home_norm in q_home_norm and len(m_home_norm) > 3) or
                        (q_home_words and m_home_words and len(q_home_words & m_home_words) >= min(2, len(q_home_words))) or
                        (len(q_home_norm) >= 4 and len(m_home_norm) >= 4 and difflib.SequenceMatcher(None, q_home_norm, m_home_norm).ratio() >= 0.80)
                    )
                    away_match = (
                        q_away_norm == m_away_norm or
                        (q_away_norm in m_away_norm and len(q_away_norm) > 3) or
                        (m_away_norm in q_away_norm and len(m_away_norm) > 3) or
                        (q_away_words and m_away_words and len(q_away_words & m_away_words) >= min(2, len(q_away_words))) or
                        (len(q_away_norm) >= 4 and len(m_away_norm) >= 4 and difflib.SequenceMatcher(None, q_away_norm, m_away_norm).ratio() >= 0.80)
                    )
                    
                    # Sadece HEM ev HEM deplasman eslesiyorsa kabul et
                    if home_match and away_match:
                        status = m.get("status_short", "")
                        minute = m.get("minute")
                        _hs = m.get("home_score") if m.get("home_score") is not None else m.get("home_goals")
                        _as = m.get("away_score") if m.get("away_score") is not None else m.get("away_goals")
                        score = f"{_hs if _hs is not None else '?'}-{_as if _as is not None else '?'}"

                        if status == "FT":
                            return "finished"
                        elif status in ("1H", "2H", "HT", "LIVE"):
                            live_info = f"{status}"
                            if minute:
                                live_info += f" {minute}'"
                            live_info += f" ({score})"
                            return f"live:{live_info}"
                        elif status == "NS":
                            return "scheduled"

                # Canli maclarda bulunamadiysa, bitmis maclari kontrol et
                for m in finished_matches:
                    m_home = (m.get("home_name", "") or "").strip().lower()
                    m_away = (m.get("away_name", "") or "").strip().lower()

                    # Takim adi eslesmesi - AKILLI kontrol (Türkçe normalize + esneklik)
                    if not q_home or not q_away or not m_home or not m_away:
                        continue
                    
                    # Türkçe karakter normalizasyonu
                    q_home_norm = _norm_name(q_home)
                    q_away_norm = _norm_name(q_away)
                    m_home_norm = _norm_name(m_home)
                    m_away_norm = _norm_name(m_away)
                    
                    # Kelime bazlı eşleştirme (en az 2 ortak kelime VEYA tek kelime tam eşleşme)
                    q_home_words = set([w for w in q_home_norm.split() if len(w) > 3])
                    q_away_words = set([w for w in q_away_norm.split() if len(w) > 3])
                    m_home_words = set([w for w in m_home_norm.split() if len(w) > 3])
                    m_away_words = set([w for w in m_away_norm.split() if len(w) > 3])
                    
                    # ALT TAKIM FILTRESI: asil takim yerine U19/rezerv/kadin mackini eslestirme
                    if _is_senior_team(q_home_norm) and not _is_senior_team(m_home_norm):
                        continue
                    if _is_senior_team(q_away_norm) and not _is_senior_team(m_away_norm):
                        continue
                    
                    # Eşleştirme: Tam eşleme VEYA uzun string içerme VEYA 2+ ortak kelime
                    home_match = (
                        q_home_norm == m_home_norm or
                        (q_home_norm in m_home_norm and len(q_home_norm) > 3) or
                        (m_home_norm in q_home_norm and len(m_home_norm) > 3) or
                        (q_home_words and m_home_words and len(q_home_words & m_home_words) >= min(2, len(q_home_words))) or
                        (len(q_home_norm) >= 4 and len(m_home_norm) >= 4 and difflib.SequenceMatcher(None, q_home_norm, m_home_norm).ratio() >= 0.80)
                    )
                    away_match = (
                        q_away_norm == m_away_norm or
                        (q_away_norm in m_away_norm and len(q_away_norm) > 3) or
                        (m_away_norm in q_away_norm and len(m_away_norm) > 3) or
                        (q_away_words and m_away_words and len(q_away_words & m_away_words) >= min(2, len(q_away_words))) or
                        (len(q_away_norm) >= 4 and len(m_away_norm) >= 4 and difflib.SequenceMatcher(None, q_away_norm, m_away_norm).ratio() >= 0.80)
                    )
                    
                    # Sadece HEM ev HEM deplasman eslesiyorsa kabul et
                    if home_match and away_match:
                        # Mac tarihi eslesmesi (mac_date +/- 7 gun toleransi)
                        kickoff = m.get("kickoff", "")
                        if kickoff:
                            try:
                                ko_dt = datetime.fromisoformat(str(kickoff)[:10])
                                if abs((ko_dt.date() - md.date()).days) <= 7:
                                    return "finished"
                            except:
                                pass
                        else:
                            return "finished"

            except Exception as e:
                print("[Coupons] Flashscore canli kontrol hatasi:", e)

            if days_diff == 0:
                return "today"
            elif days_diff == 1:
                return "yesterday"
            elif days_diff <= 7:
                return "recent"
            elif days_diff <= 30:
                return "old"
            else:
                return "old"
        else:
            return "old"
    except Exception:
        return "unknown"


_FLASHSCORE_LIVE_CACHE = []
_FLASHSCORE_FINISHED_CACHE = []
_FLASHSCORE_CACHE_TIME = 0


# TR karakterleri ascii'ye cevirir (eslestirme icin ortak norm)
_TR_MAP = str.maketrans("çğıöşüâîûêéàèùäëïöü", "cgiosuaiueeaeuaeiou")
_ALNUM_KEEP = set("abcdefghijklmnopqrstuvwxyz0123456789 ")

# Flashscore'daki farkli takim isimleri ile tahmin isimlerini esitle
_NAME_ALIASES = {
    "manchester united": "man united",
    "manchester utd": "man united",
    "b monchengladbach": "monchengladbach",
    "borussia monchengladbach": "monchengladbach",
    "monchengladbach": "monchengladbach",
    "mgladbach": "monchengladbach",
    "erzurumspor": "erzurum",
    "istanbul basaksehir": "basaksehir",
    "istanbul b basaksehir": "basaksehir",
    "istanbul buyuksehir belediye": "basaksehir",
}


def _norm_name(s):
    """Takim adlarini eslestirmeye hazir birlestirilmis tekrar duzenleyici.

    TR karakterler ascii'ye cevrilir, her turlu bozuk/yabanci karakter
    (U+FFFD, U+0178 vb.) atilir. Boylece 'Kasımpa?sa' veya 'KasımpaŸşa'
    gibi bozuk kayitlar bile dogru eslesir: 'kasimpasa'.
    Bilinen isim farkliliklari (manchester united -> man united) da burada cozulur.
    """
    if s is None:
        return ""
    t = str(s).strip().lower().translate(_TR_MAP)
    cleaned = "".join(c for c in t if c in _ALNUM_KEEP)
    return _NAME_ALIASES.get(cleaned, cleaned)


def _is_senior_team(name_norm):
    """Asil takim mi? U19/U21/rezer/kadin gibi alt ya da kadin takimlari False.

    Sadece substring degil kelime bazli kontrol yapar; boylece
    'sheffield wednesday' gibi 'w' harfi iceren normal takimlar yanlis engellenmez.
    """
    if not name_norm:
        return False
    if any(t in name_norm for t in ("u19", "u21", "u17", "u18", "u20", "u23", "u16", "u15", "reserves", "rezerv", "youth")):
        return False
    for w in name_norm.split():
        if w in ("w", "women", "kadin"):
            return False
    return True


def _load_finished_from_disk():
    """fixtures_cache.json DOSYASINDAN bitmis maclari güvenle yukle.

    Bellek cache'leri (basarisiz/kismi yazilma) sunucu gotugu icin finished
    verisi HER ZAMAN diskteki dolu dosyadan okunur. Sunucu ne kadar eski
    baslarsa baslasin diskteki guncel 30 gunluk veriyi gorur.
    """
    try:
        from prediction.flashscore_scraper import _CACHE_FILE as fs_cache_file
        if os.path.exists(fs_cache_file):
            with open(fs_cache_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            data = (raw.get("finished") or {}).get("data")
            if data:
                return data
    except Exception:
        pass
    return []


_finished_disk_mtime = 0.0
_finished_disk_cache = None
_FINISHED_DISK_TTL = 60


def _ensure_finished_cache():
    """_FIXTURES_CACHE['finished'] ve _FLASHSCORE_FINISHED_CACHE'i DISK'ten dolu hale getir.

    Bellegin kismi/boş verisi ne olursa olsun diskteki DOLU finished verisi
    (varsa) bellegin üzerine yazilir. Boylece kupon aramalari hep tam veriyle calisir.
    """
    global _FLASHSCORE_FINISHED_CACHE, _finished_disk_mtime, _finished_disk_cache
    from prediction.flashscore_scraper import _CACHE_FILE as fs_cache_file
    reload = False
    try:
        if os.path.exists(fs_cache_file):
            mtime = os.path.getmtime(fs_cache_file)
            if mtime != _finished_disk_mtime or os.path.getsize(fs_cache_file) == 0:
                reload = True
    except Exception:
        reload = True
    if reload:
        _finished_disk_cache = None
        _finished_disk_mtime = 0.0
    disk = _finished_disk_cache
    if disk is None:
        disk = _load_finished_from_disk()
        _finished_disk_cache = disk
        try:
            _finished_disk_mtime = os.path.getmtime(fs_cache_file) if os.path.exists(fs_cache_file) else 0.0
        except Exception:
            _finished_disk_mtime = 0.0
    if disk:
        _FIXTURES_CACHE["finished"]["data"] = disk
        _FIXTURES_CACHE["finished"]["fetched_at"] = time.time()
        _FLASHSCORE_FINISHED_CACHE = disk
    elif _FIXTURES_CACHE.get("finished", {}).get("data"):
        _FLASHSCORE_FINISHED_CACHE = _FIXTURES_CACHE["finished"]["data"]


def _refresh_flashscore_cache():
    """Flashscore canli/bitmis maclarini cache'le (bir kez cek, her yerde kullan).

    Kritik: bitmis maclar (finished) icin 30 gunluk scrape COK pahali (~7dk).
    Isteği bloklamamak icin finished scrape'i ARKA PLANDA bir kez baslat,
    sonuc geldiginde _FLASHSCORE_FINISHED_CACHE'ye yaz. Boylece kuponlar
    canli sekmesindeki BITTMIIS maclari da gorur.

    Canli (live) scrape de kisa olsa worker thread'i kullanir; ilk /api/coupons
    istegini (ve dolayisiyla sayfayi) bu yuzden bloklamamak icin live de arka
    planda tazelenir.
    """
    global _FLASHSCORE_LIVE_CACHE, _FLASHSCORE_FINISHED_CACHE, _FLASHSCORE_CACHE_TIME
    now = time.time()

    # Önce disk'teki dolu finished verisini yukle - kuponlar inmesin diye.
    _ensure_finished_cache()

    if now - _FLASHSCORE_CACHE_TIME < 60:
        return
    _FLASHSCORE_CACHE_TIME = now
    try:
        import threading
        from prediction.flashscore_scraper import get_fixtures_flashscore

        # Finished: önce diskten (kuponlar inmesin), sonra arka planda tazele.
        if not _FIXTURES_CACHE.get("finished", {}).get("data"):
            _ensure_finished_cache()
        if not getattr(app, "_flashscore_finished_fetching", False):
            app._flashscore_finished_fetching = True
            def _bg_finished():
                try:
                    items = get_fixtures_flashscore(kind="finished", limit=100000, refresh=True) or []
                    if items:
                        global _FLASHSCORE_FINISHED_CACHE
                        _FIXTURES_CACHE["finished"]["data"] = items
                        _FIXTURES_CACHE["finished"]["fetched_at"] = time.time()
                        _FLASHSCORE_FINISHED_CACHE = items
                        print(f"[Coupons] Finished scrape tamam: {len(items)} mac")
                except Exception as e:
                    print(f"[Coupons] Finished scrape hatasi: {e}")
                finally:
                    app._flashscore_finished_fetching = False
            threading.Thread(target=_bg_finished, daemon=True).start()

        # Live'i de arka planda tazele - ilk kupon istegini bloklamamak icin.
        if not getattr(app, "_flashscore_live_fetching", False):
            app._flashscore_live_fetching = True
            def _bg_live():
                try:
                    items = get_fixtures_flashscore(kind="live", limit=200, refresh=True) or []
                    if items:
                        global _FLASHSCORE_LIVE_CACHE
                        _FLASHSCORE_LIVE_CACHE = items
                        print(f"[Coupons] Live scrape tamam: {len(items)} mac")
                except Exception as e:
                    print(f"[Coupons] Live scrape hatasi: {e}")
                finally:
                    app._flashscore_live_fetching = False
            threading.Thread(target=_bg_live, daemon=True).start()

        print(f"[Coupons] Flashscore cache: {len(_FLASHSCORE_LIVE_CACHE)} live, {len(_FLASHSCORE_FINISHED_CACHE)} finished")
    except Exception as e:
        print("[Coupons] Flashscore cache hatasi:", e)
        print(f"[Coupons] Flashscore cache: {len(_FLASHSCORE_LIVE_CACHE)} live, {len(_FLASHSCORE_FINISHED_CACHE)} finished")



@app.route("/api/coupons")
def api_coupons():
    """Kupon listesini dondur (gercek sonuclarla karsilastirmali).

    Agir hesaplamayi (her tahmin icin sonuc arama + degerlendirme) tekrar
    tekrar yapmamak icin sonuc imza bazli onbellekliyoruz: predictions.
    json ve sonuc cache dosyalarinin mtime/size imzasi degismedigi surece
    onbellekten aninda dondurur. Boylece kupon sayfasi acilista takilmaz.
    """
    import threading
    # Flashscore disk cache'ini doldur (finished maçlar için - canlı sekmesindeki BITTI'ler).
    # Cache zaten diskte tam dolu ise bunun maliyeti yok; aksi halde tazelenir.
    try:
        _ensure_finished_cache()
    except Exception:
        pass
    # Flashscore refresh sadece arka planda calissin - ilk istegi bloklamasin
    try:
        _refresh_flashscore_cache()
    except Exception:
        pass

    try:
        import time as _time
        from os import stat
        sig_parts = []
        for _fp in (PREDICTIONS_FILE,
                    os.path.join(TAHMINLER_DIR, "results_cache.json"),
                    os.path.join(TAHMINLER_DIR, "fd_results_cache.json"),
                    MANUAL_RESULTS_FILE,
                    os.path.join(TAHMINLER_DIR, "match_results.json"),
                    os.path.join(TAHMINLER_DIR, "fixtures_cache.json")):
            try:
                _st = stat(_fp)
                sig_parts.append(f"{_fp}:{int(_st.st_mtime)}:{_st.st_size}")
            except Exception:
                sig_parts.append(f"{_fp}:missing")
        sig = "|".join(sig_parts)
    except Exception:
        sig = None

    _cache = getattr(app, "_coupons_cache", None)
    if _cache and _cache.get("sig") == sig and _cache.get("data") is not None:
        return jsonify(_cache["data"])

    lock = getattr(app, "_coupons_lock", None)
    if lock is None:
        lock = threading.Lock()
        app._coupons_lock = lock

    with lock:
        # Lock icinde tekrar kontrol et (baska bir istek arada doldurmamis olsun)
        _cache = getattr(app, "_coupons_cache", None)
        if _cache and _cache.get("sig") == sig and _cache.get("data") is not None:
            return jsonify(_cache["data"])

        preds = _load_predictions()
        coupons = []
        new_results_found = False

        # Her tahmin icin sonuc bul ve degerlendir
        for pred in preds:
            hid = pred.get("home_id")
            aid = pred.get("away_id")
            if not hid or not aid:
                continue

            actual = _find_match_result(
                hid, aid,
                pred.get("home_name"), pred.get("away_name"),
                pred.get("match_date"),
                use_live=False
            )
            evaluation = _evaluate_coupon(pred, actual) if actual else None

            # Sonuc bulunduysa match_results.json'a kaydet (sonraki arama hizli olsun)
            if actual and actual.get("source") not in ("match_results", "manual"):
                try:
                    import prediction.match_score_fetcher as _ms
                    cid = f"{hid}-{aid}"
                    _ms.save_local_match_result(cid, {
                        "home_goals": actual.get("home_goals"),
                        "away_goals": actual.get("away_goals"),
                        "total_goals": actual.get("total_goals"),
                        "result": actual.get("result"),
                        "btts": actual.get("btts"),
                        "over25": actual.get("over25"),
                        "date": actual.get("date"),
                        "source": actual.get("source", "api_coupons"),
                    })
                    new_results_found = True
                except Exception:
                    pass

            match_status = "scheduled"
            match_date_str = pred.get("match_date", "")
            if actual is not None:
                match_status = "played"
            elif match_date_str:
                try:
                    md = datetime.fromisoformat(match_date_str)
                    days_since = (datetime.now().date() - md.date()).days
                    if days_since > 0:
                        match_status = "pending"
                except Exception:
                    pass

            coupon = {
                "id": pred.get("id"),
                "home_name": pred.get("home_name", "?"),
                "away_name": pred.get("away_name", "?"),
                "home_id": hid,
                "away_id": aid,
                "home_elo": pred.get("home_elo", 0),
                "away_elo": pred.get("away_elo", 0),
                "timestamp": pred.get("timestamp"),
                "match_date": pred.get("match_date"),
                "match_status": match_status,
                "prediction_summary": pred.get("prediction_summary", {}),
                "has_result": actual is not None,
                "actual_score": f"{actual['home_goals']}-{actual['away_goals']}" if actual else None,
                "actual_date": actual.get("date") if actual else None,
                "source": actual.get("source") if actual else None,
                "evaluation": evaluation,
            }
            coupons.append(coupon)

        # Yeni sonuc kaydedildiyse signature'i guncelle ki cache yenilensin
        if new_results_found:
            app._coupons_cache = {"sig": sig + "_updated", "data": coupons}
        else:
            app._coupons_cache = {"sig": sig, "data": coupons}
        return jsonify(coupons)


# ── SONUÇ CACHE & THE SPORTS DB ──────────────────────────────────────

RESULTS_CACHE_FILE = os.path.join(TAHMINLER_DIR, "results_cache.json")
SOFASCORE_CACHE_FILE = os.path.join(TAHMINLER_DIR, "sofascore_cache.json")

_RESULTS_CACHE_MEMORY = None
_RESULTS_CACHE_MEMORY_MTIME = 0.0


def _load_results_cache():
    """Cache'den sonuclari yukle (bellek oncelikli)."""
    global _RESULTS_CACHE_MEMORY, _RESULTS_CACHE_MEMORY_MTIME
    if os.path.exists(RESULTS_CACHE_FILE):
        try:
            mt = os.path.getmtime(RESULTS_CACHE_FILE)
            if _RESULTS_CACHE_MEMORY is not None and mt == _RESULTS_CACHE_MEMORY_MTIME:
                return _RESULTS_CACHE_MEMORY
            with open(RESULTS_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            _RESULTS_CACHE_MEMORY = data
            _RESULTS_CACHE_MEMORY_MTIME = mt
            return data
        except Exception:
            return {}
    return {}


def _save_results_cache(cache):
    """Cache'i kaydet."""
    global _RESULTS_CACHE_MEMORY, _RESULTS_CACHE_MEMORY_MTIME
    with open(RESULTS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    _RESULTS_CACHE_MEMORY = cache
    _RESULTS_CACHE_MEMORY_MTIME = os.path.getmtime(RESULTS_CACHE_FILE)


def _normalize_name(name):
    """Takim adini normallestir (kucuk harf, bosluk temizleme)."""
    if not name:
        return ""
    import unicodedata
    # Turkce karakterleri ASCII'ye cevir
    tr_map = str.maketrans("çğıöşüâîûê","cgiosuaiue")
    n = name.strip().lower().translate(tr_map)
    # Fazla bosluklari temizle
    return " ".join(n.split())


def _fetch_results_from_sportsdb(date_str):
    """TheSportsDB'den belirli bir tarihteki futbol sonuclarini cek."""
    import urllib.request
    import urllib.error

    url = f"https://www.thesportsdb.com/api/v1/json/3/eventsday.php?d={date_str}&s=Soccer"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FTMS/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[RESULTS] TheSportsDB hatasi ({date_str}): {e}")
        return []

    events = data.get("events") or []
    results = []
    for ev in events:
        if ev.get("strStatus") != "FT":
            continue
        hg = ev.get("intHomeScore")
        ag = ev.get("intAwayScore")
        if hg is None or ag is None:
            continue
        results.append({
            "home_team": ev.get("strHomeTeam", ""),
            "away_team": ev.get("strAwayTeam", ""),
            "home_goals": int(hg),
            "away_goals": int(ag),
            "date": ev.get("dateEvent", date_str),
            "league": ev.get("strLeague", ""),
            "source": "thesportsdb",
        })
    return results


def _fetch_results_for_predictions(preds):
    """Tahmin tarihlerine gore sonuclari cek (agresif cache - 5dk)."""
    cache = _load_results_cache()
    from datetime import datetime, timedelta

    today = datetime.now()

    # Kontrol edilecek tarihleri topla
    dates_to_check = set()

    # Son 14 gunu her zaman kontrol et (yeni biten maclar icin)
    for i in range(14):
        d = today - timedelta(days=i)
        dates_to_check.add(d.strftime("%Y-%m-%d"))

    # Tahminlerin timestamp'lerinden de tarihleri ekle (son 30 gun)
    for pred in preds:
        ts = pred.get("timestamp", "")
        if ts:
            try:
                t = datetime.fromisoformat(ts.replace("Z", "+00:00").split("+")[0])
                if (today - t).days <= 30:
                    dates_to_check.add(t.strftime("%Y-%m-%d"))
            except Exception:
                pass

        # match_date varsa onu da ekle
        md = pred.get("match_date", "")
        if md:
            try:
                t = datetime.fromisoformat(md)
                if (today - t).days <= 30:
                    dates_to_check.add(t.strftime("%Y-%m-%d"))
            except Exception:
                pass

    # Cache'de olmayan veya 15dk'dan eski olan tarihleri cek
    dates_to_fetch = []
    for date_str in dates_to_check:
        cached = cache.get(date_str)
        if not cached:
            dates_to_fetch.append(date_str)
        else:
            fetched_at = cached.get("fetched_at", "")
            if fetched_at:
                try:
                    ft = datetime.fromisoformat(fetched_at)
                    if (today - ft).total_seconds() > 900:  # 15 dk cache
                        dates_to_fetch.append(date_str)
                except Exception:
                    dates_to_fetch.append(date_str)
            else:
                dates_to_fetch.append(date_str)

    fetched = 0
    for date_str in dates_to_fetch:
        results = _fetch_results_from_sportsdb(date_str)
        cache[date_str] = {
            "fetched_at": today.isoformat(),
            "results": results,
        }
        fetched += 1
        if results:
            print(f"[RESULTS] {date_str}: {len(results)} mac sonucu cekildi")

    if fetched > 0:
        _save_results_cache(cache)

    # Ayrica FD cache'ini de tazele ki yeni biten maclar hemen yakalanabilsin
    _refresh_fd_cache_if_stale()

    return cache


def _refresh_fd_cache_if_stale():
    """Football-data cache 30dkdan eski ise otomatik tazele."""
    try:
        if not os.path.exists(FD_CACHE_FILE):
            return
        with open(FD_CACHE_FILE, "r", encoding="utf-8") as f:
            fd_cache = json.load(f)
        # Herhangi bir lig 30dkdan eski ise tazele
        from datetime import datetime
        now = datetime.now()
        stale = False
        for key, val in fd_cache.items():
            try:
                fetched_at = datetime.fromisoformat(val.get("fetched_at", ""))
                if (now - fetched_at).total_seconds() > 900:  # 15 dk cache
                    stale = True
                    break
            except Exception:
                stale = True
                break
        if not stale:
            return
        # Tum ligleri tazele (arka planda, hata durumunda skip)
        import csv, io, urllib.request
        for lg in ["T1", "E0", "E1", "SP1", "SP2", "D1", "D2", "I1", "I2", "F1", "F2", "N1", "P1"]:
            url = f"https://www.football-data.co.uk/mmz4281/2627/{lg}.csv"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    raw = resp.read()
                    text = raw.decode("utf-8-sig", errors="replace")
                reader = csv.DictReader(io.StringIO(text))
                rows = [dict(r) for r in reader]
                fd_cache[f"fd_{lg}"] = {"fetched_at": now.isoformat(), "rows": rows}
            except Exception:
                continue
        with open(FD_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(fd_cache, f, ensure_ascii=False)
        _FD_CACHE_MEMORY = fd_cache
        _FD_CACHE_MEMORY_MTIME = os.path.getmtime(FD_CACHE_FILE)
    except Exception as e:
        print(f"[FD-REFRESH] Hata: {e}")


def _find_result_from_cache(home_name, away_name, cache):
    """Cache'den mac sonucunu bul (akilli fuzzy eslestirme)."""
    h_norm = _normalize_name(home_name)
    a_norm = _normalize_name(away_name)
    if not h_norm or not a_norm:
        return None

    def _cache_team_match(query, target):
        """Fuzzy takim eslesmesi: tam, icinde olma, kelime bazli."""
        if not query or not target:
            return False
        if query == target:
            return True
        if query in target or target in query:
            return True
        q_words = set(w for w in query.split() if len(w) >= 4)
        t_words = set(w for w in target.split() if len(w) >= 4)
        if q_words and t_words and len(q_words & t_words) >= min(2, len(q_words)):
            return True
        return False

    best = None
    best_days = 999

    for date_key, day_data in cache.items():
        for r in day_data.get("results", []):
            r_home = _normalize_name(r.get("home_team", ""))
            r_away = _normalize_name(r.get("away_team", ""))

            home_ok = _cache_team_match(h_norm, r_home)
            away_ok = _cache_team_match(a_norm, r_away)

            if not (home_ok and away_ok):
                continue

            # Tarih yakinligina gore skor hesapla (en yakin olsun)
            try:
                from datetime import datetime
                r_date = r.get("date", "")
                dd = abs((datetime.fromisoformat(str(r_date)[:10]) - datetime.fromisoformat(str(date_key))).days) if r_date else 0
            except Exception:
                dd = 0

            if dd < best_days:
                best_days = dd
                # Ters yon kontrolu
                if h_norm == r_away and a_norm == r_home:
                    best = {
                        "home_team": r.get("away_team"),
                        "away_team": r.get("home_team"),
                        "home_goals": r.get("away_goals"),
                        "away_goals": r.get("home_goals"),
                        "date": r.get("date"),
                        "league": r.get("league"),
                        "source": r.get("source"),
                    }
                else:
                    best = r

    return best


@app.route("/api/coupons/refresh", methods=["POST"])
def api_coupons_refresh():
    """Tum sonuc kaynaklarini zorla tazele (FD + TheSportsDB)."""
    from datetime import datetime
    results = {"ok": True, "actions": []}

    try:
        preds = _load_predictions()
        cache = _fetch_results_for_predictions(preds)
        total_results = sum(len(d.get("results", [])) for d in cache.values())
        results["thesportsdb_results"] = total_results
        results["actions"].append("thesportsdb")
    except Exception as e:
        results["thesportsdb_error"] = str(e)

    # FD cache'i zorla tazele
    try:
        import csv, io, urllib.request
        fd_cache = {}
        if os.path.exists(FD_CACHE_FILE):
            try:
                with open(FD_CACHE_FILE, "r", encoding="utf-8") as f:
                    fd_cache = json.load(f)
            except Exception:
                fd_cache = {}
        now = datetime.now()
        leagues = ["T1", "E0", "E1", "SP1", "SP2", "D1", "D2", "I1", "I2", "F1", "F2", "N1", "P1"]
        for lg in leagues:
            url = f"https://www.football-data.co.uk/mmz4281/2627/{lg}.csv"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    raw = resp.read()
                    text = raw.decode("utf-8-sig", errors="replace")
                reader = csv.DictReader(io.StringIO(text))
                rows = [dict(r) for r in reader]
                fd_cache[f"fd_{lg}"] = {"fetched_at": now.isoformat(), "rows": rows}
            except Exception as e:
                results.setdefault("fd_errors", []).append({"league": lg, "error": str(e)})
                continue
        with open(FD_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(fd_cache, f, ensure_ascii=False)
        _FD_CACHE_MEMORY = fd_cache
        _FD_CACHE_MEMORY_MTIME = os.path.getmtime(FD_CACHE_FILE)
        results["fd_leagues"] = leagues
        results["actions"].append("football-data")
    except Exception as e:
        results["fd_error"] = str(e)

    results["refreshed_at"] = datetime.now().isoformat()
    return jsonify(results)


@app.route("/api/coupons/status")
def api_coupons_status():
    """Cache durumunu dondur (son guncelleme zamanı, toplam sonuc)."""
    cache = _load_results_cache()
    total_results = sum(len(d.get("results", [])) for d in cache.values())
    last_update = max((d.get("fetched_at", "") for d in cache.values()), default="")

    # FD cache bilgisini de ekle (bellekteki oncelikli)
    fd_info = {"leagues": [], "last_update": "", "total_matches": 0}
    fd_cache = _FD_CACHE_MEMORY
    if fd_cache is None and os.path.exists(FD_CACHE_FILE):
        try:
            with open(FD_CACHE_FILE, "r", encoding="utf-8") as f:
                fd_cache = json.load(f)
        except Exception:
            fd_cache = {}
    if fd_cache:
        try:
            for key, val in fd_cache.items():
                if key.startswith("fd_"):
                    rows = val.get("rows", [])
                    fd_info["leagues"].append(key.replace("fd_", ""))
                    fd_info["total_matches"] += len(rows)
                    fa = val.get("fetched_at", "")
                    if fa > fd_info["last_update"]:
                        fd_info["last_update"] = fa
        except Exception:
            pass

    # Tahmin istatistikleri - coupons cache varsa onu kullan (yavas _has_result_for_pred'i atla)
    preds = _load_predictions()
    total_preds = len(preds)
    _coupons_cache = getattr(app, "_coupons_cache", None)
    _cached_data = _coupons_cache.get("data") if _coupons_cache else None
    if _cached_data and isinstance(_cached_data, list):
        pending = sum(1 for c in _cached_data if not c.get("has_result") and c.get("match_status") != "scheduled")
    else:
        from datetime import datetime
        pending = sum(1 for p in preds
                      if p.get("match_date")
                      and not _has_result_for_pred(p))

    return jsonify({
        "cached_dates": len(cache),
        "total_results": total_results,
        "last_update": last_update,
        "fd_cache": fd_info,
        "total_predictions": total_preds,
        "pending_predictions": pending,
    })


def _has_result_for_pred(pred):
    """Tahmin icin sonuc bulunabiliyor mu?"""
    hid = pred.get("home_id")
    aid = pred.get("away_id")
    if not hid or not aid:
        return False
    actual = _find_match_result(hid, aid, pred.get("home_name"), pred.get("away_name"), pred.get("match_date"))
    return actual is not None


# ── OTOMATIK SONUÇ ÇEKME ARKA PLANI ──────────────────────────────────

import threading
import time as _time

_auto_fetch_stop = threading.Event()


def _auto_fetch_loop():
    """Arka planda her 60 saniyede bir sonuclari cek - daha az API yuku."""
    import random
    while not _auto_fetch_stop.is_set():
        try:
            preds = _load_predictions()
            if preds:
                # 1) TheSportsDB cache guncelle
                try:
                    cache = _fetch_results_for_predictions(preds)
                    n_results = sum(len(d.get("results", [])) for d in cache.values())
                    if n_results:
                        print(f"[AUTO-FETCH] TSDB: {n_results} sonuc cache'lendi", flush=True)
                except Exception as e:
                    print(f"[AUTO-FETCH] TSDB hatasi: {e}", flush=True)

                # 2) Tahminlerden sonuc bulunanlari say
                n_results_found = 0
                for pred in preds:
                    hid = pred.get("home_id") or pred.get("home_team_id")
                    aid = pred.get("away_id") or pred.get("away_team_id")
                    if hid and aid:
                        r = _find_match_result(hid, aid, pred.get("home_name"), pred.get("away_name"), pred.get("match_date"))
                        if r:
                            n_results_found += 1
                if n_results_found:
                    print(f"[AUTO-FETCH] {n_results_found}/{len(preds)} tahmin icin sonuc bulundu", flush=True)
        except Exception as e:
            print(f"[AUTO-FETCH] Hata: {e}", flush=True)

        # 30 sn bekle
        _auto_fetch_stop.wait(30)


def _start_auto_fetch():
    """Otomatik sonuc cekme baslat."""
    t = threading.Thread(target=_auto_fetch_loop, daemon=True, name="auto-fetch-results")
    t.start()
    print("[AUTO-FETCH] Otomatik sonuc cekme baslatildi (her 30 sn)")


def _stop_auto_fetch():
    """Otomatik sonuc cekme durdur."""
    _auto_fetch_stop.set()


@app.route("/live")
def live():
    return render_template("live.html", loading=STATE["loading"])


@app.route("/live/predict", methods=["POST"])
def live_predict():
    try:
        data = request.get_json()
        from live.model import live_predict as lp
        return jsonify(lp(
            pre_home_lambda=data.get("pre_home_lambda", 1.5),
            pre_away_lambda=data.get("pre_away_lambda", 1.0),
            minute=data.get("minute", 0),
            xg_home=data.get("xg_home"), xg_away=data.get("xg_away"),
            score_home=data.get("score_home", 0), score_away=data.get("score_away", 0),
        ))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ── FIXTURES (CANLI / BİTMİŞ / GELECEK MAÇLAR) ───────────────────────────────

_FIXTURES_CACHE = {
    "live": {"data": None, "fetched_at": 0, "fetching": False},
    "finished": {"data": None, "fetched_at": 0, "fetching": False},
    "upcoming": {"data": None, "fetched_at": 0, "fetching": False},
}
# TTL: Flashscore her scrape ~3-6sn suruyor (singleton browser).
# Fresh suresi: cache HIT aninda don, arka planda yenile.
# Stale suresi: cache HIT ama stale, yine de don, arka planda yenile.
_FIXTURES_TTL = {"live": 30, "finished": 300, "upcoming": 600}
_FIXTURES_STALE = {"live": 180, "finished": 1800, "upcoming": 3600}


def _preload_fixtures_cache():
    """Sunucu baslarken flashscore disk cache'ini yukle.
    Boylece ilk finished istegi 7dk'lik scrape'i beklemez, aninda doner.
    """
    try:
        from prediction.flashscore_scraper import _CACHE as fs_cache
        now = _time.time()
        for k in ("live", "finished", "upcoming"):
            data = fs_cache.get(k, {}).get("data")
            if data:
                _FIXTURES_CACHE[k]["data"] = data
                _FIXTURES_CACHE[k]["fetched_at"] = now
    except Exception:
        pass


_preload_fixtures_cache()

# Sunucu baslarken finished scraping'i arka planda baslat ki kupon kontrolu
# canli sekmesindeki BITTI maclari hemen/erken gorebilsin (7dk surer, thread'de).
try:
    if not _FIXTURES_CACHE["finished"]["data"]:
        import threading as _th
        def _bg_preload_finished():
            from prediction.flashscore_scraper import get_fixtures_flashscore
            try:
                items = get_fixtures_flashscore(kind="finished", limit=100000, refresh=True) or []
                if items:
                    _FIXTURES_CACHE["finished"]["data"] = items
                    _FIXTURES_CACHE["finished"]["fetched_at"] = _time.time()
                    global _FLASHSCORE_FINISHED_CACHE
                    _FLASHSCORE_FINISHED_CACHE = items
                    print(f"[Coupons] Preload finished scrape: {len(items)} mac")
            except Exception as e:
                print(f"[Coupons] Preload finished scrape hatasi: {e}")
        _th.Thread(target=_bg_preload_finished, daemon=True).start()
except Exception:
    pass


def _normalize_api_fixture(raw, status_kind):
    """API-Football fixture objesini ortak forma çevir."""
    fixture = raw.get("fixture", {}) or {}
    league = raw.get("league", {}) or {}
    teams = raw.get("teams", {}) or {}
    goals = raw.get("goals", {}) or {}
    score = raw.get("score", {}) or {}

    home = teams.get("home", {}) or {}
    away = teams.get("away", {}) or {}

    status_short = (fixture.get("status") or {}).get("short") or ""
    status_long = (fixture.get("status") or {}).get("long") or ""
    minute = (fixture.get("status") or {}).get("elapsed")

    halftime = score.get("halftime") or {}
    fulltime = score.get("fulltime") or {}
    extratime = score.get("extratime") or {}
    penalty = score.get("penalty") or {}

    home_goals = goals.get("home")
    away_goals = goals.get("away")

    return {
        "fixture_id": fixture.get("id"),
        "kickoff": fixture.get("date"),
        "timestamp": fixture.get("timestamp"),
        "status_short": status_short,
        "status_long": status_long,
        "status_kind": status_kind,
        "minute": minute,
        "league_id": league.get("id"),
        "league_name": league.get("name"),
        "league_country": league.get("country"),
        "league_logo": league.get("logo"),
        "league_flag": league.get("flag"),
        "season": league.get("season"),
        "round": league.get("round"),
        "home_id": home.get("id"),
        "home_name": home.get("name"),
        "home_logo": home.get("logo"),
        "away_id": away.get("id"),
        "away_name": away.get("name"),
        "away_logo": away.get("logo"),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "ht_home_goals": halftime.get("home") if isinstance(halftime, dict) else None,
        "ht_away_goals": halftime.get("away") if isinstance(halftime, dict) else None,
        "ft_home_goals": fulltime.get("home") if isinstance(fulltime, dict) else None,
        "ft_away_goals": fulltime.get("away") if isinstance(fulltime, dict) else None,
        "et_home_goals": extratime.get("home") if isinstance(extratime, dict) else None,
        "et_away_goals": extratime.get("away") if isinstance(extratime, dict) else None,
        "pen_home": penalty.get("home") if isinstance(penalty, dict) else None,
        "pen_away": penalty.get("away") if isinstance(penalty, dict) else None,
        "venue": (raw.get("fixture") or {}).get("venue", {}) or {},
        "referee": (raw.get("fixture") or {}).get("referee"),
    }


def _fixtures_from_features(kind, limit=20):
    """API-Football erişilemiyorsa features parquet'tan maç üret (fallback)."""
    feat = STATE.get("features")
    if feat is None or len(feat) == 0:
        return []
    try:
        f = feat.copy()
        if "date" in f.columns:
            f["date"] = pd.to_datetime(f["date"], errors="coerce")
            f = f.dropna(subset=["date"])
        today = pd.Timestamp.now(tz=None).normalize()
        name_map = _load_turkish_display()
        out = []
        if kind == "finished":
            cutoff_start = today - pd.Timedelta(days=7)
            sub = f[(f["date"] >= cutoff_start) & (f["date"] <= today)]
            sub = sub.sort_values("date", ascending=False).head(limit)
        elif kind == "live":
            # Bugün ± 3 gün: canlı görünüm için örnek. Bulunamazsa son 30 günden örnekler.
            mask = (f["date"] >= today - pd.Timedelta(days=3)) & (f["date"] <= today + pd.Timedelta(days=3))
            sub = f[mask]
            if len(sub) == 0:
                sub = f[f["date"] >= today - pd.Timedelta(days=30)].sort_values("date", ascending=False).head(limit)
            else:
                sub = sub.sort_values("date", ascending=False).head(limit)
        else:
            # upcoming: gelecek maç features parquet'ta genelde yok; en yakın tarihli örnekler
            sub = f.sort_values("date", ascending=False).head(limit)
            sub = sub.sort_values("date").head(limit)
        for _, row in sub.iterrows():
            hid = int(row.get("home_team_id", 0)) if pd.notna(row.get("home_team_id")) else 0
            aid = int(row.get("away_team_id", 0)) if pd.notna(row.get("away_team_id")) else 0
            hg = int(row.get("home_goals", 0)) if pd.notna(row.get("home_goals")) else None
            ag = int(row.get("away_goals", 0)) if pd.notna(row.get("away_goals")) else None
            dt = row.get("date")
            if kind == "live":
                status_short = "1H"
                minute = 45
            elif kind == "finished":
                status_short = "FT"
                minute = 90
            else:
                status_short = "NS"
                minute = None
            out.append({
                "fixture_id": f"feat-{hid}-{aid}-{int(dt.timestamp()) if pd.notna(dt) else 0}",
                "kickoff": dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
                "timestamp": int(dt.timestamp()) if hasattr(dt, "timestamp") and pd.notna(dt) else None,
                "status_short": status_short,
                "status_long": status_short,
                "status_kind": kind,
                "minute": minute,
                "league_id": None,
                "league_name": _resolve_league(str(row.get("league", "?"))),
                "league_country": None,
                "league_logo": None,
                "league_flag": None,
                "season": str(row.get("season", "")),
                "round": None,
                "home_id": hid,
                "home_name": name_map.get(str(hid)) or _display_name(hid, f"Team {hid}"),
                "home_logo": None,
                "away_id": aid,
                "away_name": name_map.get(str(aid)) or _display_name(aid, f"Team {aid}"),
                "away_logo": None,
                "home_goals": hg,
                "away_goals": ag,
                "ht_home_goals": int(row["ht_home_goals"]) if "ht_home_goals" in row and pd.notna(row.get("ht_home_goals")) else None,
                "ht_away_goals": int(row["ht_away_goals"]) if "ht_away_goals" in row and pd.notna(row.get("ht_away_goals")) else None,
                "ft_home_goals": hg,
                "ft_away_goals": ag,
                "et_home_goals": None,
                "et_away_goals": None,
                "pen_home": None,
                "pen_away": None,
                "venue": {},
                "referee": None,
            })
        return out
    except Exception:
        return []


def _fixtures_from_api(kind, limit=20):
    """API-Football'dan canlı/bitmiş/gelecek maçları çek."""
    api_key = os.getenv("API_FOOTBALL_KEY", "")
    if not api_key:
        return None
    try:
        from ingestion.api_football.client import ApiFootballClient
        from ingestion.api_football.provider import ApiFootballProvider
        client = ApiFootballClient(api_key=api_key, timeout=15)
        provider = ApiFootballProvider(client=client)
        if kind == "live":
            raw = provider.get_fixtures_live()
        elif kind == "finished":
            raw = provider.get_fixtures_by_status("FT")
        else:
            raw = provider.get_fixtures_by_status("NS")
        items = [_normalize_api_fixture(r, kind) for r in (raw or [])]
        items.sort(key=lambda x: (x.get("timestamp") or 0), reverse=(kind != "upcoming"))
        return items[:limit]
    except Exception as exc:
        print(f"[FIXTURES] API-Football hatasi ({kind}): {exc}", flush=True)
        return None


def _fixtures_from_flashscore(kind, limit=20, refresh=False):
    """Flashscore.com'dan (Playwright ile) canlı/bitmiş/gelecek maçları çeker.
    API key gerektirmez, sadece siteye HTTP isteği yapar.

    Donen her item lig bilgisini (league_name, league_country, league_round)
    ve Flashscore.com DOM yapisina sadik sekilde skor/dakika bilgisi tasir.
    """
    try:
        from prediction.flashscore_scraper import get_fixtures_flashscore
        # Tum kindlar icin limiti yuksek tut (canli/bitmis/gelecek hepsi)
        scrape_limit = max(limit, 5000)
        items = get_fixtures_flashscore(kind=kind, limit=scrape_limit, refresh=refresh)
        # Formatı normalize et (UI'in beklediği sekle)
        out = []
        for m in items:
            kickoff = m.get("kickoff") or ""
            iso_ts = None
            if kickoff:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(kickoff)
                    iso_ts = int(dt.timestamp())
                except Exception:
                    iso_ts = None
            out.append({
                "fixture_id": m.get("fixture_id"),
                "kickoff": kickoff,
                "timestamp": iso_ts,
                "status_short": m.get("status_short"),
                "status_long": m.get("status_short"),
                "status_kind": kind,
                "minute": m.get("minute"),
                "league_id": None,
                "league_name": m.get("league_name") or "",
                "league_country": m.get("league_country") or "",
                "league_round": m.get("league_round") or "",
                "league_logo": None,
                "league_flag": None,
                "season": None,
                "round": m.get("league_round") or None,
                "home_id": None,
                "home_name": m.get("home_name"),
                "home_logo": None,
                "away_id": None,
                "away_name": m.get("away_name"),
                "away_logo": None,
                "home_goals": m.get("home_score"),
                "away_goals": m.get("away_score"),
                "ht_home_goals": None,
                "ht_away_goals": None,
                "ft_home_goals": m.get("home_score") if m.get("status_short") == "FT" else None,
                "ft_away_goals": m.get("away_score") if m.get("status_short") == "FT" else None,
                "et_home_goals": None,
                "et_away_goals": None,
                "pen_home": None,
                "pen_away": None,
                "venue": {},
                "referee": None,
            })
        # Siralama
        if kind == "upcoming":
            out.sort(key=lambda x: (x.get("timestamp") or 0))
        elif kind == "finished":
            out.sort(key=lambda x: (x.get("timestamp") or 0), reverse=True)
        return out
    except Exception as exc:
        print(f"[FIXTURES] Flashscore hatasi ({kind}): {exc}", flush=True)
        return None


@app.route("/api/fixtures")
def api_fixtures():
    """?kind=live|finished|upcoming (default: live) — canlı/bitmiş/gelecek maçlar.

    Stale-while-revalidate: cache varsa anında dön, arka planda yenile.
    Bu sayede ilk istek 6sn yerine anında cevap verir.

    Kaynak siralamasi:
      1. API-Football (.env'de API_FOOTBALL_KEY varsa)
      2. Flashscore (Playwright, API key gerektirmez - BEDAVA, LİMİTSİZ)
      3. Yerel features parquet (fallback)
    """
    kind = request.args.get("kind", "live").lower()
    if kind not in ("live", "finished", "upcoming"):
        return jsonify({"error": "Geçersiz kind. live|finished|upcoming olmali."}), 400
    # Tum kindlar icin yuksek limit (tum maclari getir)
    max_limit = 200000
    limit = max(1, min(int(request.args.get("limit", 20)), max_limit))
    refresh = request.args.get("refresh") == "1"
    now = _time.time()

    cache = _FIXTURES_CACHE[kind]

    def _do_fetch():
        items = _fixtures_from_api(kind, limit=limit)
        source = "api_football"
        if items is None:
            items = _fixtures_from_flashscore(kind, limit=limit, refresh=refresh)
            source = "flashscore"
        if items is None:
            items = _fixtures_from_features(kind, limit=limit)
            source = "features_fallback"
        _FIXTURES_CACHE[kind] = {"data": items, "fetched_at": now, "fetching": False, "source": source}

    # Force refresh: don't block for long scrapes - return current data,
    # kick off background refresh. (finished scrape 30 gun ~7dk suruyor,
    # blocking would hang the request.)
    if refresh:
        items = cache.get("data") if isinstance(cache.get("data"), list) else []
        source = _FIXTURES_CACHE[kind].get("source", "cache")
        if not cache.get("fetching"):
            _FIXTURES_CACHE[kind]["fetching"] = True
            import threading
            def _bg_fetch_force():
                try:
                    _do_fetch()
                except Exception as exc:
                    print(f"[FIXTURES] refresh {kind} hatasi: {exc}", flush=True)
                finally:
                    _FIXTURES_CACHE[kind]["fetching"] = False
            threading.Thread(target=_bg_fetch_force, daemon=True).start()
    # Fresh cache: return immediately
    elif cache["data"] is not None and (now - cache["fetched_at"]) < _FIXTURES_TTL[kind]:
        items = cache["data"]
        source = "cache"
    # Stale cache: return immediately, kick off background refresh
    elif cache["data"] is not None and (now - cache["fetched_at"]) < _FIXTURES_STALE[kind]:
        items = cache["data"]
        source = "cache_stale"
        if not cache.get("fetching"):
            _FIXTURES_CACHE[kind]["fetching"] = True
            import threading
            def _bg_fetch():
                try:
                    _do_fetch()
                except Exception as exc:
                    print(f"[FIXTURES] bg fetch {kind} hatasi: {exc}", flush=True)
                    _FIXTURES_CACHE[kind]["fetching"] = False
            threading.Thread(target=_bg_fetch, daemon=True).start()
    # No cache or too stale: blocking fetch
    else:
        _FIXTURES_CACHE[kind]["fetching"] = True
        try:
            _do_fetch()
        finally:
            _FIXTURES_CACHE[kind]["fetching"] = False
        items = _FIXTURES_CACHE[kind]["data"]
        source = _FIXTURES_CACHE[kind].get("source", "api")

    return jsonify({
        "kind": kind,
        "count": len(items or []),
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "items": items or [],
    })


# ── MAÇ DETAY (Flashscore.com üzerinden) ─────────────────────────────────────

_MATCH_DETAIL_CACHE = {}


def _kupon_detail_from_aiscore(match_id):
    """'home_id-away_id-ts' formatindaki kupon id'sini AiScore'dan ceker.

    Takim adları predictions.json'dan alınır; get_aiscore_match_detail
    (Playwright) skor + istatistik dondurur. match_detail.html kabul eder sekli:
    home_team/away_team/home_score/away_score/status/kickoff/league/round +
    stats_sections[{name, rows:[{home, away, name}]}].

    Kupon id degilse veya AiScore'da bulunamazsa None (fallback Flashscore).
    """
    import re
    if not re.fullmatch(r"\d+-\d+-\d+", match_id or ""):
        return None
    parts = match_id.split("-")
    preds = _load_predictions() or []
    p = next(
        (x for x in preds
         if str(x.get("home_id")) == parts[0] and str(x.get("away_id")) == parts[1]),
        None,
    )
    if not p:
        return None
    home_name = p.get("home_name") or ""
    away_name = p.get("away_name") or ""
    if not home_name or not away_name:
        return None
    try:
        near_ts = float(parts[2])
    except ValueError:
        near_ts = None

    try:
        from prediction.aiscore_scraper import get_aiscore_match_detail
        det = get_aiscore_match_detail(home_name, away_name, near_ts=near_ts)
    except Exception:
        return None
    if not det:
        return None

    rows = [
        {"name": s.get("label", s.get("key", "")), "home": s.get("home"), "away": s.get("away")}
        for s in (det.get("stats") or [])
        if s.get("home") is not None or s.get("away") is not None
    ]
    return {
        "home_team": det.get("home_name") or home_name,
        "away_team": det.get("away_name") or away_name,
        "home_score": det.get("home_score"),
        "away_score": det.get("away_score"),
        "status": det.get("status", "SCHEDULED"),
        "minute": det.get("minute"),
        "kickoff": det.get("kickoff"),
        "league": det.get("league_name"),
        "league_country": "",
        "league_round": det.get("league_round"),
        "round": det.get("league_round"),
        "stats_sections": [{"name": "ANA İSTATİSTİKLER", "rows": rows}] if rows else [],
        "source": "aiscore",
        "url": det.get("url"),
    }


@app.route("/api/fixtures/my_live")
def api_my_live_fixtures():
    """Sadece kullanıcının tahminlerindeki maçları döndür (live/finished/upcoming).

    ?kind=live|finished|upcoming (varsayılan: live)
    Canlı sayfası ve ana sayfa ticker'ı için: dünyadaki TÜM maçları değil,
    sadece tahminlerdeki maçları gösterir.
    """
    try:
        kind = request.args.get("kind", "live").lower()
        if kind not in ("live", "finished", "upcoming"):
            kind = "live"
        refresh = request.args.get("refresh") == "1"
        limit = int(request.args.get("limit", 200000))

        preds = _load_predictions()
        if not preds:
            return jsonify({"items": [], "count": 0, "kind": kind})

        pred_team_pairs = set()
        for p in preds:
            hn = p.get("home_name", "").strip()
            an = p.get("away_name", "").strip()
            if hn and an:
                pred_team_pairs.add((hn.lower(), an.lower()))

        if kind == "live":
            _refresh_flashscore_cache()
            candidates = list(_FLASHSCORE_LIVE_CACHE or [])
        elif kind == "finished":
            _ensure_finished_cache()
            candidates = []
            for m in (_FLASHSCORE_FINISHED_CACHE or []):
                try:
                    ko = m.get("kickoff", "")
                    if ko:
                        ko_dt = datetime.fromisoformat(str(ko)[:10])
                        if (datetime.now() - ko_dt).days <= 7:
                            candidates.append(m)
                except Exception:
                    candidates.append(m)
            _fix_fin = _FIXTURES_CACHE.get("finished", {}).get("data")
            if _fix_fin:
                for m in _fix_fin:
                    st = (m.get("status_short") or "").upper()
                    if st in ("FT", "AET", "PEN"):
                        try:
                            ko = m.get("kickoff", "") or m.get("date", "")
                            if ko:
                                ko_dt = datetime.fromisoformat(str(ko)[:10])
                                if (datetime.now() - ko_dt).days <= 7:
                                    candidates.append(m)
                        except Exception:
                            candidates.append(m)
        else:
            # upcoming: Flashscore'dan çek
            upcoming_items = _fixtures_from_flashscore("upcoming", limit=max(limit, 5000), refresh=refresh)
            candidates = list(upcoming_items or [])

        matched = []
        seen = set()
        for m in candidates:
            m_home = (m.get("home_name", "") or "").strip().lower()
            m_away = (m.get("away_name", "") or "").strip().lower()
            if not m_home or not m_away:
                continue

            for ph, pa in pred_team_pairs:
                ph_norm = _norm_name(ph)
                pa_norm = _norm_name(pa)
                mh_norm = _norm_name(m_home)
                ma_norm = _norm_name(m_away)

                home_ok = (
                    ph_norm == mh_norm or
                    (len(ph_norm) >= 4 and len(mh_norm) >= 4 and difflib.SequenceMatcher(None, ph_norm, mh_norm).ratio() >= 0.80)
                )
                away_ok = (
                    pa_norm == ma_norm or
                    (len(pa_norm) >= 4 and len(ma_norm) >= 4 and difflib.SequenceMatcher(None, pa_norm, ma_norm).ratio() >= 0.80)
                )

                if home_ok and away_ok:
                    key = f"{m_home}|{m_away}"
                    if key not in seen:
                        seen.add(key)
                        matched.append(m)
                    break

        if kind == "upcoming":
            matched.sort(key=lambda x: (x.get("timestamp") or 0))
        elif kind == "finished":
            matched.sort(key=lambda x: (x.get("timestamp") or 0), reverse=True)
        else:
            def _sort_key(m):
                st = (m.get("status_short") or "").upper()
                if st in ("1H", "2H", "HT", "LIVE"):
                    return 0
                return 1
            matched.sort(key=_sort_key)

        return jsonify({
            "items": matched[:limit],
            "count": len(matched),
            "total_candidates": len(candidates),
            "kind": kind,
            "source": "predictions_filtered",
        })
    except Exception as e:
        print(f"[FIXTURES] my_live hatasi: {e}")
        return jsonify({"items": [], "count": 0, "error": str(e)})


@app.route("/api/fixture/<match_id>")
def api_fixture_detail(match_id):
    """Tek bir maçın detaylarını çeker.

    1) Flashscore fixture cache'den hizli bilgi dondur (aninda).
    2) Kupon id'si ("home_id-away_id-ts") ise AiScore'dan (Playwright + istatistik).
    3) Degilse Flashscore detail scrape (yavas, sadece ?force=1 ile).
    ?force=1 ile cache bypass edilir.
    """
    now = _time.time()
    force = request.args.get("force") == "1"

    if not force:
        cached = _MATCH_DETAIL_CACHE.get(match_id)
        if cached:
            st = (cached.get("data", {}).get("status") or "").upper()
            ttl = 60 if st in ("1H", "2H", "HT", "LIVE") else 21600
            if (now - cached.get("fetched_at", 0)) < ttl:
                return jsonify(cached["data"])

    # 1) Flashscore fixture cache'den maçı bul (aninda, Playwright gerektirmez)
    for kind_key in ("live", "finished", "upcoming"):
        fix_cache = _FIXTURES_CACHE.get(kind_key, {})
        fix_data = fix_cache.get("data") or []
        for m in fix_data:
            if m.get("fixture_id") == match_id:
                detail = {
                    "match_id": match_id,
                    "home_team": m.get("home_name") or "?",
                    "away_team": m.get("away_name") or "?",
                    "home_score": m.get("home_goals"),
                    "away_score": m.get("away_goals"),
                    "status": m.get("status_short") or "NS",
                    "status_short": m.get("status_short") or "NS",
                    "minute": m.get("minute"),
                    "kickoff": m.get("kickoff"),
                    "league": m.get("league_name") or "",
                    "league_name": m.get("league_name") or "",
                    "league_country": m.get("league_country") or "",
                    "round": m.get("league_round") or m.get("round") or "",
                    "venue": m.get("venue") or {},
                    "source": "flashscore_cache",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                _MATCH_DETAIL_CACHE[match_id] = {"data": detail, "fetched_at": now}
                return jsonify(detail)

    # 2) Flashscore live/finished cache'de de bul (coupon checker cache)
    for m in (_FLASHSCORE_LIVE_CACHE or []):
        if m.get("fixture_id") == match_id:
            detail = {
                "match_id": match_id,
                "home_team": m.get("home_name") or "?",
                "away_team": m.get("away_name") or "?",
                "home_score": m.get("home_score") or m.get("home_goals"),
                "away_score": m.get("away_score") or m.get("away_goals"),
                "status": m.get("status_short") or "NS",
                "status_short": m.get("status_short") or "NS",
                "minute": m.get("minute"),
                "kickoff": m.get("kickoff"),
                "league": m.get("league_name") or "",
                "league_name": m.get("league_name") or "",
                "league_country": m.get("league_country") or "",
                "round": m.get("league_round") or "",
                "venue": {},
                "source": "flashscore_live_cache",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            _MATCH_DETAIL_CACHE[match_id] = {"data": detail, "fetched_at": now}
            return jsonify(detail)

    for m in (_FLASHSCORE_FINISHED_CACHE or []):
        if m.get("fixture_id") == match_id:
            detail = {
                "match_id": match_id,
                "home_team": m.get("home_name") or "?",
                "away_team": m.get("away_name") or "?",
                "home_score": m.get("home_score") or m.get("home_goals"),
                "away_score": m.get("away_score") or m.get("away_goals"),
                "status": m.get("status_short") or "NS",
                "status_short": m.get("status_short") or "NS",
                "minute": m.get("minute"),
                "kickoff": m.get("kickoff"),
                "league": m.get("league_name") or "",
                "league_name": m.get("league_name") or "",
                "league_country": m.get("league_country") or "",
                "round": m.get("league_round") or "",
                "venue": {},
                "source": "flashscore_finished_cache",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            _MATCH_DETAIL_CACHE[match_id] = {"data": detail, "fetched_at": now}
            return jsonify(detail)

    # 3) Kupon id'si ise AiScore'dan dene
    try:
        data = _kupon_detail_from_aiscore(match_id)
        if data:
            data["fetched_at"] = datetime.now(timezone.utc).isoformat()
            data["match_id"] = match_id
            _MATCH_DETAIL_CACHE[match_id] = {"data": data, "fetched_at": now}
            return jsonify(data)
    except Exception:
        pass

    # 4) Son cares: Playwright scrape (sadece force=1 ile, cok yavas)
    if force:
        try:
            from prediction.flashscore_scraper import get_match_detail
            data = get_match_detail(match_id)
            if data:
                data["fetched_at"] = datetime.now(timezone.utc).isoformat()
                data["match_id"] = match_id
                _MATCH_DETAIL_CACHE[match_id] = {"data": data, "fetched_at": now}
                return jsonify(data)
        except Exception as exc:
            print(f"[FIXTURES] detail scrape hatasi: {exc}", flush=True)

    return jsonify({"error": "Mac detayi bulunamadi"}), 404


@app.route("/match/<match_id>")
def match_detail_page(match_id):
    """Maç detay sayfası."""
    return render_template("match_detail.html", match_id=match_id, loading=STATE["loading"])


@app.route("/performance")
def performance():
    return render_template("performance.html", metrics=STATE.get("metrics") or {}, loading=STATE["loading"])


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip().lower()
    if len(q) < 1:
        return jsonify([])
    qn = _normalize_team_name(q)  # Turkce karakterden bagimsiz arama

    def _score(det):
        name_l = det["name"].lower()
        norm = det.get("norm", _normalize_team_name(det["name"]))
        ini = (det.get("avatar") or {}).get("initials", "").lower()
        s = 0
        if name_l.startswith(q):
            s += 100
        elif norm.startswith(qn):
            s += 90
        elif q in name_l:
            s += 60
        elif qn in norm:
            s += 55
        elif ini.startswith(q):
            s += 40
        elif q in det.get("league", "").lower() or q in det.get("league_code", "").lower():
            s += 30
        else:
            return None
        # populerlik: cok mac oynayan ve yuksek ELO on plana
        s += min(det.get("n_matches", 0), 500) / 20.0
        s += max(min((det.get("elo", 1500) - 1400) / 8.0, 20), 0)
        return s

    scored = []
    for tid, det in TEAM_ID_MAP.items():
        sc = _score(det)
        if sc:
            scored.append((sc, det))
    scored.sort(key=lambda x: (-x[0], -x[1]["elo"]))

    seen_names = {}
    deduped = []
    for sc, r in scored:
        base = _normalize_team_name(r["name"]).strip()
        if base not in seen_names:
            seen_names[base] = True
            deduped.append(r)
    return jsonify(deduped[:30])


@app.route("/api/leagues")
def api_leagues():
    INTERNATIONAL = {"Champions League", "Europa League", "FIFA World Cup", "UEFA Euro",
                     "UEFA Women Euro", "Women World Cup", "Copa America",
                     "Africa Cup of Nations", "England Women Super League",
                     "USA NWSL", "USA MLS", "USA North American League",
                     "Italy Serie A Women", "Germany Frauen Bundesliga"}
    merged = {}
    for code, teams in LEAGUE_TEAMS.items():
        name = _resolve_league(code)
        if name not in merged:
            merged[name] = {"name": name, "codes": [], "teams": {}, "total": 0}
        merged[name]["codes"].append(code)
        merged[name]["total"] += len(teams)
        for t in teams:
            merged[name]["teams"][t["id"]] = t

    countries = {}
    for name, info in merged.items():
        if name in INTERNATIONAL:
            country = "Uluslararasi"
            league_name = name
        else:
            parts = name.split(" ", 1) if name else ["?"]
            country = parts[0] if len(parts) > 1 else "Diger"
            league_name = parts[1] if len(parts) > 1 else name
        if country not in countries:
            countries[country] = {"country": country, "leagues": []}
        best_code = max(info["codes"], key=lambda c: len(LEAGUE_TEAMS.get(c, [])))
        all_teams = sorted(info["teams"].values(), key=lambda x: -x["elo"])
        countries[country]["leagues"].append({
            "name": league_name,
            "code": best_code,
            "codes": info["codes"],
            "count": len(all_teams),
            "teams": all_teams[:10],
        })
    result = sorted(countries.values(), key=lambda c: -sum(l["count"] for l in c["leagues"]))
    for c in result:
        c["leagues"].sort(key=lambda l: -l["count"])
        c["total_teams"] = sum(l["count"] for l in c["leagues"])
    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    home_id = data.get("home_id")
    away_id = data.get("away_id")
    if not home_id or not away_id:
        return jsonify({"error": "Takim ID gerekli"}), 400
    result = _predict_match(int(home_id), int(away_id))
    if result is None:
        return jsonify({"error": "Takim bulunamadi"}), 404

    pred_record = {
        "id": f"{home_id}-{away_id}-{int(datetime.now().timestamp())}",
        "timestamp": datetime.now().isoformat(),
        "home_id": home_id,
        "away_id": away_id,
        "home_name": result["home"].get("name", str(home_id)),
        "away_name": result["away"].get("name", str(away_id)),
        "home_elo": result["home"].get("elo", 0),
        "away_elo": result["away"].get("elo", 0),
        "result": {
            "home_win": result["home_win"],
            "draw": result["draw"],
            "away_win": result["away_win"],
            "winner": result["winner"],
            "winner_prob": result["winner_prob"],
        },
        "goals": result["goals"],
        "total_goals": result["total_goals"],
        "double_chance": result["double_chance"],
        "corners": result["corners"],
        "btts_yes": result["btts_yes"],
        "btts_no": result["btts_no"],
        "top_scores": result["top_scores"],
        "recommendations": result.get("recommendations"),
        "v15_markets": result.get("v15_markets"),
    }
    _add_prediction(pred_record)

    return jsonify(result)


def _build_recommendations(home_prob, draw_prob, away_prob, winner, win_prob,
                           hxg, axg, total_lam,
                           o15, o25, o35,
                           btts_p, corner_avg, cor75, cor85,
                           hd, ad,
                           d1X=None, dX2=None, d12=None,
                           u25=None, btts_no=None,
                           cor95=None, cor105=None,
                           home_cs_p=None, home_over15_p=None,
                           away_over05_p=None, home_over05_p=None,
                           home_win_p=None,
                           iy_draw_p=None, iy_ms_dh_p=None, iy_ms_da_p=None, iy_over05_p=None,
                           two_y_over05_p=None, two_y_over15_p=None, two_y_more_p=None,
                           first_goal_home_p=None,
                           hk1_home_p=None, hk1_away_p=None,
                           corner_most_home_p=None, corner_even_p=None, corner_over95_p=None):
    tips = []

    # MINIMUM CONFIDENCE: %70 — %70 alti hic onerilmez
    MIN_CONF = 0.70

    # 1X2 — sadece %70+ ise oner
    if home_prob >= MIN_CONF:
        tips.append({"market": "Ev Sahibi Galibiyet", "pick": "1", "confidence": round(home_prob * 100, 1), "reason": f"Ev sahibi favori ({round(home_prob*100)}%)"})
    elif away_prob >= MIN_CONF:
        tips.append({"market": "Deplasman Galibiyet", "pick": "2", "confidence": round(away_prob * 100, 1), "reason": f"Deplasman favori ({round(away_prob*100)}%)"})
    elif draw_prob >= MIN_CONF:
        tips.append({"market": "Beraberlik", "pick": "X", "confidence": round(draw_prob * 100, 1), "reason": f"Beraberlik olasligi yuksek ({round(draw_prob*100)}%)"})

    # Double Chance — %70+
    if d1X is not None and d1X >= MIN_CONF:
        tips.append({"market": "Cift Sans 1X", "pick": "1X", "confidence": round(d1X * 100, 1), "reason": "Ev sahibi yenilmez"})
    if dX2 is not None and dX2 >= MIN_CONF:
        tips.append({"market": "Cift Sans X2", "pick": "X2", "confidence": round(dX2 * 100, 1), "reason": "Deplasman yenilmez"})
    if d12 is not None and d12 >= MIN_CONF:
        tips.append({"market": "Cift Sans 12", "pick": "12", "confidence": round(d12 * 100, 1), "reason": "Beraberlik disi"})

    # Gol marketleri — hepsinde %70+ esik
    if o25 >= MIN_CONF:
        tips.append({"market": "Gol 2.5 Ust", "pick": "U2.5", "confidence": round(o25 * 100, 1), "reason": f"Ort. {round(total_lam,1)} gol bekleniyor"})
    elif u25 is not None and u25 >= MIN_CONF:
        tips.append({"market": "Gol 2.5 Alt", "pick": "A2.5", "confidence": round(u25 * 100, 1), "reason": f"Dusuk gol beklentisi ({round(total_lam,1)})"})

    if o15 >= MIN_CONF:
        tips.append({"market": "Gol 1.5 Ust", "pick": "U1.5", "confidence": round(o15 * 100, 1), "reason": "En az 2 gol bekleniyor"})
    elif o15 is not None and (1 - o15) >= MIN_CONF:
        tips.append({"market": "Gol 1.5 Alt", "pick": "A1.5", "confidence": round((1-o15) * 100, 1), "reason": f"Cok dusuk gol beklentisi ({round(total_lam,1)})"})

    if o35 >= MIN_CONF:
        tips.append({"market": "Gol 3.5 Ust", "pick": "U3.5", "confidence": round(o35 * 100, 1), "reason": f"Yuksek gol beklentisi ({round(total_lam,1)})"})
    elif o35 is not None and (1 - o35) >= MIN_CONF:
        tips.append({"market": "Gol 3.5 Alt", "pick": "A3.5", "confidence": round((1-o35) * 100, 1), "reason": f"Dusuk gol beklentisi ({round(total_lam,1)})"})

    # BTTS — %70+
    if btts_p >= MIN_CONF:
        tips.append({"market": "BTTS Evet", "pick": "Var", "confidence": round(btts_p * 100, 1), "reason": "Her iki takim da gol atar"})
    elif btts_no is not None and btts_no >= MIN_CONF:
        tips.append({"market": "BTTS Hayir", "pick": "Yok", "confidence": round(btts_no * 100, 1), "reason": "En az bir takim gol atamaz"})

    # NEW MARKETS (v14) — only calibrated ones
    if home_over05_p is not None and home_over05_p >= MIN_CONF:
        tips.append({"market": "Ev Sahibi Gol Atar", "pick": "Var", "confidence": round(home_over05_p * 100, 1), "reason": "Ev sahibi en az 1 gol atar"})
    if away_over05_p is not None and away_over05_p >= MIN_CONF:
        tips.append({"market": "Deplasman Gol Atar", "pick": "Var", "confidence": round(away_over05_p * 100, 1), "reason": "Deplasman en az 1 gol atar"})

    # v15 NEW MARKETS — First Half
    if iy_draw_p is not None and iy_draw_p >= MIN_CONF:
        tips.append({"market": "IY Beraberlik", "pick": "X", "confidence": round(iy_draw_p * 100, 1), "reason": "Ilk yari beraberlik"})
    if iy_over05_p is not None and iy_over05_p >= MIN_CONF:
        tips.append({"market": "IY Ust 0.5", "pick": "U0.5", "confidence": round(iy_over05_p * 100, 1), "reason": "Ilk yari gol olur"})
    if iy_ms_dh_p is not None and iy_ms_dh_p >= MIN_CONF:
        tips.append({"market": "IY/MS D/H", "pick": "X/1", "confidence": round(iy_ms_dh_p * 100, 1), "reason": "IY beraberlik, MS ev sahibi kazanir"})
    if iy_ms_da_p is not None and iy_ms_da_p >= MIN_CONF:
        tips.append({"market": "IY/MS D/A", "pick": "X/2", "confidence": round(iy_ms_da_p * 100, 1), "reason": "IY beraberlik, MS deplasman kazanir"})

    # v15 — Second Half
    if two_y_over05_p is not None and two_y_over05_p >= MIN_CONF:
        tips.append({"market": "2Y Ust 0.5", "pick": "U0.5", "confidence": round(two_y_over05_p * 100, 1), "reason": "2. yarida gol olur"})
    if two_y_over15_p is not None and two_y_over15_p >= MIN_CONF:
        tips.append({"market": "2Y Ust 1.5", "pick": "U1.5", "confidence": round(two_y_over15_p * 100, 1), "reason": "2. yarida en az 2 gol olur"})
    if two_y_more_p is not None and two_y_more_p >= MIN_CONF:
        tips.append({"market": "2Y Daha Cok Gol", "pick": "2Y", "confidence": round(two_y_more_p * 100, 1), "reason": "2. yarida daha cok gol atilir"})
    if first_goal_home_p is not None and first_goal_home_p >= MIN_CONF:
        tips.append({"market": "Ilk Gol Ev Atar", "pick": "1", "confidence": round(first_goal_home_p * 100, 1), "reason": "Ev sahibi ilk golu atar"})

    # v15 — Handikap
    if hk1_home_p is not None and hk1_home_p >= MIN_CONF:
        tips.append({"market": "HK1:0 Ev Sahibi", "pick": "1", "confidence": round(hk1_home_p * 100, 1), "reason": "Ev sahibi -1 handikapla kazanir"})
    if hk1_away_p is not None and hk1_away_p >= MIN_CONF:
        tips.append({"market": "HK1:0 Depasman", "pick": "2", "confidence": round(hk1_away_p * 100, 1), "reason": "Depasman +1 handikapla kazanir"})

    # v15 — Corners (pre-match features)
    if corner_most_home_p is not None and corner_most_home_p >= MIN_CONF:
        tips.append({"market": "En Cok Korner Ev", "pick": "1", "confidence": round(corner_most_home_p * 100, 1), "reason": "Ev sahibi daha cok korner kullanir"})
    if corner_even_p is not None and corner_even_p >= MIN_CONF:
        tips.append({"market": "Korner Cift", "pick": "Cift", "confidence": round(corner_even_p * 100, 1), "reason": "Toplam korner cift sayi"})
    if corner_over95_p is not None and corner_over95_p >= MIN_CONF:
        tips.append({"market": "Korner 9.5 Ust", "pick": "U9.5", "confidence": round(corner_over95_p * 100, 1), "reason": "10+ korner bekleniyor"})

    # Korner — %70+
    if corner_avg >= 10.5:
        if cor75 >= MIN_CONF:
            tips.append({"market": "Korner 7.5 Ust", "pick": "U7.5", "confidence": round(cor75 * 100, 1), "reason": f"Ort. {round(corner_avg,1)} korner bekleniyor"})
        if cor85 >= MIN_CONF:
            tips.append({"market": "Korner 8.5 Ust", "pick": "U8.5", "confidence": round(cor85 * 100, 1), "reason": f"Ort. {round(corner_avg,1)} korner bekleniyor"})
    if corner_avg >= 11.5 and cor95 is not None and cor95 >= MIN_CONF:
        tips.append({"market": "Korner 9.5 Ust", "pick": "U9.5", "confidence": round(cor95 * 100, 1), "reason": f"Ort. {round(corner_avg,1)} korner bekleniyor"})
    if corner_avg <= 8.5:
        if (1 - cor75) >= MIN_CONF:
            tips.append({"market": "Korner 7.5 Alt", "pick": "A7.5", "confidence": round((1-cor75) * 100, 1), "reason": f"Ort. {round(corner_avg,1)} korner bekleniyor"})
    if corner_avg <= 9.0:
        if cor85 is not None and (1 - cor85) >= MIN_CONF:
            tips.append({"market": "Korner 8.5 Alt", "pick": "A8.5", "confidence": round((1-cor85) * 100, 1), "reason": f"Ort. {round(corner_avg,1)} korner bekleniyor"})

    tips.sort(key=lambda t: -t["confidence"])

    best = tips[0] if tips else {"market": "Tahmin Yok", "pick": "-", "confidence": 0, "reason": "%70 guven esigi altinda — tavsiye edilmez"}

    return {
        "best_bet": best,
        "all_tips": tips[:12],
    }


def _predict_match(home_id, away_id):
    if home_id not in TEAM_ID_MAP or away_id not in TEAM_ID_MAP:
        return None

    hd = TEAM_ID_MAP.get(home_id, {})
    ad = TEAM_ID_MAP.get(away_id, {})

    wm = STATE.get("web_model")
    feat = STATE.get("features")
    sofa_full = STATE.get("sofa_full")
    sofa_roll = STATE.get("sofa_rolling")
    sofa_name_map = STATE.get("sofa_name_map", {})
    if wm is None or feat is None or len(feat) == 0:
        return None

    feat_cols = wm["features"]

    home_matches = feat[feat["home_team_id"] == home_id].sort_values("date")
    away_matches = feat[(feat["home_team_id"] == away_id) | (feat["away_team_id"] == away_id)].sort_values("date")

    h_last = home_matches.iloc[-1] if len(home_matches) > 0 else None
    a_last = away_matches.iloc[-1] if len(away_matches) > 0 else None

    def _safe(row, col, default=0):
        if row is None:
            return default
        v = row.get(col, None)
        if v is None or pd.isna(v):
            return default
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    feat_row = {}

    # --- ELO (from TEAM_INDEX) ---
    feat_row["home_elo"] = hd.get("elo", 1500)
    feat_row["away_elo"] = ad.get("elo", 1500)
    feat_row["elo_diff"] = feat_row["home_elo"] - feat_row["away_elo"]
    feat_row["home_attack_elo"] = hd.get("attack_elo", 1500)
    feat_row["home_defence_elo"] = hd.get("defence_elo", 1500)
    feat_row["away_attack_elo"] = ad.get("attack_elo", 1500)
    feat_row["away_defence_elo"] = ad.get("defence_elo", 1500)
    feat_row["attack_elo_diff"] = feat_row["home_attack_elo"] - feat_row["away_attack_elo"]
    feat_row["defence_elo_diff"] = feat_row["home_defence_elo"] - feat_row["away_defence_elo"]

    # --- LEAGUE (from last match) ---
    for col in ["lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg", "lg_draw_rate",
                "lg_btts_rate", "lg_over25_rate", "lg_corner_avg", "lg_card_avg"]:
        if h_last is not None and col in h_last.index and pd.notna(h_last.get(col)):
            feat_row[col] = float(h_last[col])
        elif a_last is not None and col in a_last.index and pd.notna(a_last.get(col)):
            feat_row[col] = float(a_last[col])
        else:
            feat_row[col] = 0

    # --- H2H ---
    h2h_matches = feat[
        ((feat["home_team_id"] == home_id) & (feat["away_team_id"] == away_id)) |
        ((feat["home_team_id"] == away_id) & (feat["away_team_id"] == home_id))
    ].sort_values("date")

    if len(h2h_matches) > 0:
        h2h_total = len(h2h_matches)
        h2h_home_wins = h2h_draws = h2h_away_wins = h2h_btts_count = 0
        h2h_goals = []
        for _, m in h2h_matches.iterrows():
            hg = int(m.get("home_goals", 0))
            ag = int(m.get("away_goals", 0))
            h2h_goals.append(hg + ag)
            if hg > 0 and ag > 0:
                h2h_btts_count += 1
            if m["home_team_id"] == home_id:
                if hg > ag: h2h_home_wins += 1
                elif hg == ag: h2h_draws += 1
                else: h2h_away_wins += 1
            else:
                if ag > hg: h2h_home_wins += 1
                elif hg == ag: h2h_draws += 1
                else: h2h_away_wins += 1
        feat_row["h2h_home_win"] = h2h_home_wins / max(h2h_total, 1)
        feat_row["h2h_draw"] = h2h_draws / max(h2h_total, 1)
        feat_row["h2h_away_win"] = h2h_away_wins / max(h2h_total, 1)
        feat_row["h2h_goals_avg"] = float(np.mean(h2h_goals)) if h2h_goals else 2.5
        feat_row["h2h_btts"] = h2h_btts_count / max(h2h_total, 1)
    else:
        feat_row["h2h_home_win"] = 0.45
        feat_row["h2h_draw"] = 0.25
        feat_row["h2h_away_win"] = 0.30
        feat_row["h2h_goals_avg"] = 2.5
        feat_row["h2h_btts"] = 0.5

    # --- ROLLING (from sofascore_rolling - pre-computed per match) ---
    ROLL_FEATURES = ["h_xg_r", "a_xg_r", "h_shots_r", "a_shots_r", "h_sot_r", "a_sot_r",
                     "h_big_ch_r", "a_big_ch_r", "h_passes_r", "a_passes_r",
                     "h_fouls_r", "a_fouls_r", "h_tackles_r", "a_tackles_r",
                     "h_yellows_r", "a_yellows_r", "h_dribbles_r", "a_dribbles_r",
                     "h_crosses_r", "a_crosses_r"]

    for col in ROLL_FEATURES:
        feat_row[col] = 0

    if sofa_roll is not None:
        h_name = hd.get("name", "")
        a_name = ad.get("name", "")

        h_team_roll = sofa_roll[(sofa_roll["home_team"] == h_name) | (sofa_roll["away_team"] == h_name)].sort_values("date")
        a_team_roll = sofa_roll[(sofa_roll["home_team"] == a_name) | (sofa_roll["away_team"] == a_name)].sort_values("date")

        def _get_rolling(row_data, is_home_perspective):
            """Extract rolling features from a sofa_rolling row, from this team's perspective."""
            vals = {}
            if is_home_perspective:
                vals["h_xg_r"] = _safe(row_data, "h_xg", 0)
                vals["h_shots_r"] = _safe(row_data, "h_shots", 0)
                vals["h_sot_r"] = _safe(row_data, "h_sot", 0)
                vals["h_big_ch_r"] = _safe(row_data, "h_big_ch", 0)
                vals["h_passes_r"] = _safe(row_data, "h_passes", 0)
                vals["h_fouls_r"] = _safe(row_data, "h_fouls", 0)
                vals["h_tackles_r"] = _safe(row_data, "h_tackles", 0)
                vals["h_yellows_r"] = _safe(row_data, "h_yellows", 0)
                vals["h_dribbles_r"] = _safe(row_data, "h_dribbles", 0)
                vals["h_crosses_r"] = _safe(row_data, "h_crosses", 0)
            else:
                vals["h_xg_r"] = _safe(row_data, "a_xg", 0)
                vals["h_shots_r"] = _safe(row_data, "a_shots", 0)
                vals["h_sot_r"] = _safe(row_data, "a_sot", 0)
                vals["h_big_ch_r"] = _safe(row_data, "a_big_ch", 0)
                vals["h_passes_r"] = _safe(row_data, "a_passes", 0)
                vals["h_fouls_r"] = _safe(row_data, "a_fouls", 0)
                vals["h_tackles_r"] = _safe(row_data, "a_tackles", 0)
                vals["h_yellows_r"] = _safe(row_data, "a_yellows", 0)
                vals["h_dribbles_r"] = _safe(row_data, "a_dribbles", 0)
                vals["h_crosses_r"] = _safe(row_data, "a_crosses", 0)
            return vals

        if len(h_team_roll) > 0:
            last_h = h_team_roll.iloc[-1]
            is_home = (last_h.get("home_team") == h_name)
            feat_row.update(_get_rolling(last_h, is_home))

        if len(a_team_roll) > 0:
            last_a = a_team_roll.iloc[-1]
            is_home = (last_a.get("home_team") == a_name)
            away_vals = _get_rolling(last_a, is_home)
            for k, v in away_vals.items():
                feat_row[k.replace("h_", "a_")] = v

    # --- FORM (computed from sofascore_full) ---
    def _compute_form_fallback(team_id, feat_data, prefix, form):
        """Fallback: features.parquet'tan form hesapla (sofascore yoksa).
        En guncel maci bul, ev/deplasman ayrimiyla rolling stats al."""
        home_matches = feat_data[feat_data["home_team_id"] == team_id].sort_values("date")
        away_matches = feat_data[feat_data["away_team_id"] == team_id].sort_values("date")
        if len(home_matches) == 0 and len(away_matches) == 0:
            return form

        last_h = home_matches.iloc[-1] if len(home_matches) > 0 else None
        last_a = away_matches.iloc[-1] if len(away_matches) > 0 else None

        h_date = last_h.get("date") if last_h is not None else None
        a_date = last_a.get("date") if last_a is not None else None
        h_date = h_date if (h_date is not None and pd.notna(h_date)) else pd.Timestamp.min
        a_date = a_date if (a_date is not None and pd.notna(a_date)) else pd.Timestamp.min

        h_is_newer = h_date >= a_date
        newer = last_h if h_is_newer else last_a
        older = last_a if h_is_newer else last_h
        new_side = "home" if h_is_newer else "away"

        if newer is not None:
            nd = newer.get("date")
            if nd is not None and pd.notna(nd):
                days = (pd.Timestamp.now() - nd).days
                form[f"{prefix}rest_days"] = min(max(days, 0), 30)

        for n in [3, 5, 8, 10, 20]:
            if newer is not None:
                gf = float(newer.get(f"{new_side}_gf_{n}", 0) or 0)
                ga = float(newer.get(f"{new_side}_ga_{n}", 0) or 0)
            else:
                gf = 0; ga = 0
            if older is not None and gf == 0:
                old_side = "away" if h_is_newer else "home"
                gf = float(older.get(f"{old_side}_gf_{n}", 0) or 0)
                ga = float(older.get(f"{old_side}_ga_{n}", 0) or 0)
            form[f"{prefix}gf_{n}"] = gf
            form[f"{prefix}ga_{n}"] = ga

        for n in [5, 10, 20]:
            if newer is not None:
                pts = float(newer.get(f"{new_side}_pts_{n}", 0) or 0)
            else:
                pts = 0
            if older is not None and pts == 0:
                old_side = "away" if h_is_newer else "home"
                pts = float(older.get(f"{old_side}_pts_{n}", 0) or 0)
            form[f"{prefix}pts_{n}"] = pts

        if last_h is not None:
            form[f"{prefix}hgf_5"] = float(last_h.get("home_hgf_5", 0) or 0)
            form[f"{prefix}hga_5"] = float(last_h.get("home_hga_5", 0) or 0)
            form[f"{prefix}hpts_5"] = float(last_h.get("home_hpts_5", 0) or 0)
        if last_a is not None:
            form[f"{prefix}agf_5"] = float(last_a.get("away_agf_5", 0) or 0)
            form[f"{prefix}aga_5"] = float(last_a.get("away_aga_5", 0) or 0)
            form[f"{prefix}apts_5"] = float(last_a.get("away_apts_5", 0) or 0)

        if newer is not None:
            ns = new_side
            form[f"{prefix}w_gf"] = float(newer.get(f"{ns}_w_gf", 0) or 0)
            form[f"{prefix}w_ga"] = float(newer.get(f"{ns}_w_ga", 0) or 0)
            form[f"{prefix}w_shots"] = float(newer.get(f"{ns}_w_shots", 0) or 0)
            form[f"{prefix}w_sot"] = float(newer.get(f"{ns}_w_sot", 0) or 0)
            form[f"{prefix}momentum"] = float(newer.get(f"{ns}_momentum", 0) or 0)
            form[f"{prefix}wins_last5"] = float(newer.get(f"{ns}_wins_last5", 0) or 0)
            form[f"{prefix}gdiff5"] = float(newer.get(f"{ns}_gdiff5", 0) or 0)
            form[f"{prefix}opp_elo"] = float(newer.get(f"{ns}_opp_elo", 1500) or 1500)

        if older is not None:
            os_ = "away" if h_is_newer else "home"
            if form.get(f"{prefix}w_gf", 0) == 0:
                form[f"{prefix}w_gf"] = float(older.get(f"{os_}_w_gf", 0) or 0)
                form[f"{prefix}w_ga"] = float(older.get(f"{os_}_w_ga", 0) or 0)
            if form.get(f"{prefix}w_shots", 0) == 0:
                form[f"{prefix}w_shots"] = float(older.get(f"{os_}_w_shots", 0) or 0)
                form[f"{prefix}w_sot"] = float(older.get(f"{os_}_w_sot", 0) or 0)
            if form.get(f"{prefix}momentum", 0) == 0:
                form[f"{prefix}momentum"] = float(older.get(f"{os_}_momentum", 0) or 0)
                form[f"{prefix}wins_last5"] = float(older.get(f"{os_}_wins_last5", 0) or 0)
                form[f"{prefix}gdiff5"] = float(older.get(f"{os_}_gdiff5", 0) or 0)
            if form.get(f"{prefix}opp_elo", 1500) == 1500:
                form[f"{prefix}opp_elo"] = float(older.get(f"{os_}_opp_elo", 1500) or 1500)

        form[f"{prefix}xg_r5"] = 0
        # Try to get xG from features data
        xg_col = f"{ns}_xg_real" if f"{ns}_xg_real" in newer.index else f"{ns}_xg"
        if xg_col in newer.index:
            xg_val = newer.get(xg_col)
            if pd.notna(xg_val) and float(xg_val) > 0:
                form[f"{prefix}xg_r5"] = float(xg_val)
        sf_xg_col = f"sf_{ns}_xg"
        if sf_xg_col in newer.index:
            sf_val = newer.get(sf_xg_col)
            if pd.notna(sf_val) and float(sf_val) > 0 and form[f"{prefix}xg_r5"] == 0:
                form[f"{prefix}xg_r5"] = float(sf_val)
        form[f"{prefix}sot_r5"] = 0
        form[f"{prefix}shots_r5"] = 0

        return form
    def _compute_form(team_name, team_id, prefix):
        """Form cache'den O(1) lookup ile form feature'larini al."""
        form_cache = STATE.get("form_cache", {})
        cached = form_cache.get(team_id)
        if cached is None:
            form = {}
            for n in [3, 5, 8, 10, 20]:
                form[f"{prefix}gf_{n}"] = 0
                form[f"{prefix}ga_{n}"] = 0
            for n in [5, 10, 20]:
                form[f"{prefix}pts_{n}"] = 0
            form[f"{prefix}rest_days"] = 5
            for k in ["xg_r5","sot_r5","shots_r5","w_gf","w_ga","w_shots","w_sot","momentum","wins_last5","gdiff5","opp_elo","hgf_5","hga_5","hpts_5","agf_5","aga_5","apts_5"]:
                form[f"{prefix}{k}"] = 0
            form[f"{prefix}opp_elo"] = 1500
            return form
        form = {}
        for k, v in cached.items():
            form[f"{prefix}{k}"] = v
        return form

        return form

    home_name = hd.get("name", "")
    away_name = ad.get("name", "")
    feat_row.update(_compute_form(home_name, home_id, "home_"))
    feat_row.update(_compute_form(away_name, away_id, "away_"))

    # --- DERIVED FEATURES (train_v8 uyumlu) ---
    feat_row["form_diff"] = feat_row.get("home_pts_5", 0) - feat_row.get("away_pts_5", 0)
    feat_row["gf_diff"] = feat_row.get("home_gf_5", 0) - feat_row.get("away_gf_5", 0)
    feat_row["ga_diff"] = feat_row.get("home_ga_5", 0) - feat_row.get("away_ga_5", 0)
    feat_row["elo_x_hform"] = feat_row.get("elo_diff", 0) * feat_row.get("home_pts_5", 0)
    feat_row["home_strong"] = 1 if feat_row.get("elo_diff", 0) > 100 else 0
    feat_row["away_strong"] = 1 if feat_row.get("elo_diff", 0) < -100 else 0
    draw_rate = feat_row.get("lg_draw_rate", 0.25)
    feat_row["draw_likely"] = 1 if (abs(feat_row.get("elo_diff", 0)) < 50 and draw_rate > 0.25) else 0
    feat_row["shots_diff_5"] = feat_row.get("home_shots_r5", 0) - feat_row.get("away_shots_r5", 0)
    feat_row["sot_diff_5"] = feat_row.get("home_sot_r5", 0) - feat_row.get("away_sot_r5", 0)
    feat_row["corner_diff_5"] = 0
    feat_row["home_pts_std"] = 0
    feat_row["away_pts_std"] = 0
    feat_row["data_completeness"] = 0.8

    # --- v11 DRAW FEATURES ---
    feat_row["elo_diff_abs"] = abs(feat_row.get("elo_diff", 0))
    feat_row["form_diff_abs"] = abs(feat_row.get("form_diff", 0))
    feat_row["gf_diff_abs"] = abs(feat_row.get("gf_diff", 0))
    feat_row["ga_diff_abs"] = abs(feat_row.get("ga_diff", 0))
    feat_row["teams_even"] = 1 if feat_row["elo_diff_abs"] < 30 else 0
    feat_row["teams_very_even"] = 1 if feat_row["elo_diff_abs"] < 15 else 0
    feat_row["form_similar"] = 1 if feat_row["form_diff_abs"] < 1.0 else 0
    feat_row["attack_similar"] = 1 if feat_row["gf_diff_abs"] < 0.5 else 0
    feat_row["h2h_draw_high"] = 1 if feat_row.get("h2h_draw", 0) > 0.3 else 0
    draw_rate_val = feat_row.get("lg_draw_rate", 0.25)
    feat_row["lg_draw_signal"] = 1 if draw_rate_val > 0.28 else 0

    # Rolling league stats (from feat data)
    lg_real_draw = feat_row.get("lg_draw_rate", 0.25)
    lg_real_home = feat_row.get("lg_home_goal_avg", 0)
    lg_real_goals = feat_row.get("lg_avg_goals", 2.5)
    if h_last is not None:
        if "lg_real_draw_rate" in h_last.index and pd.notna(h_last.get("lg_real_draw_rate")):
            lg_real_draw = float(h_last["lg_real_draw_rate"])
        if "lg_real_home_rate" in h_last.index and pd.notna(h_last.get("lg_real_home_rate")):
            lg_real_home = float(h_last["lg_real_home_rate"])
        if "lg_real_goal_avg" in h_last.index and pd.notna(h_last.get("lg_real_goal_avg")):
            lg_real_goals = float(h_last["lg_real_goal_avg"])
    feat_row["lg_real_draw_rate"] = lg_real_draw
    feat_row["lg_real_home_rate"] = lg_real_home
    feat_row["lg_real_goal_avg"] = lg_real_goals

    # v14 NEW MARKET rolling league stats
    feat_row["lg_real_btts_rate"] = feat_row.get("lg_btts_rate", 0.50)
    feat_row["lg_real_cs_home"] = 0.33
    feat_row["lg_real_cs_away"] = 0.24
    if h_last is not None:
        if "lg_real_btts_rate" in h_last.index and pd.notna(h_last.get("lg_real_btts_rate")):
            feat_row["lg_real_btts_rate"] = float(h_last["lg_real_btts_rate"])
        if "lg_real_cs_home" in h_last.index and pd.notna(h_last.get("lg_real_cs_home")):
            feat_row["lg_real_cs_home"] = float(h_last["lg_real_cs_home"])
        if "lg_real_cs_away" in h_last.index and pd.notna(h_last.get("lg_real_cs_away")):
            feat_row["lg_real_cs_away"] = float(h_last["lg_real_cs_away"])

    # --- v12 GOAL-SPECIFIC FEATURES ---
    feat_row["home_attack_power"] = feat_row.get("home_gf_5", 0) * feat_row.get("home_attack_elo", 1500) / 1500
    feat_row["away_attack_power"] = feat_row.get("away_gf_5", 0) * feat_row.get("away_attack_elo", 1500) / 1500
    feat_row["home_defence_power"] = feat_row.get("home_ga_5", 0) * feat_row.get("home_defence_elo", 1500) / 1500
    feat_row["away_defence_power"] = feat_row.get("away_ga_5", 0) * feat_row.get("away_defence_elo", 1500) / 1500
    feat_row["total_attack"] = feat_row["home_attack_power"] + feat_row["away_attack_power"]
    feat_row["total_defence"] = feat_row["home_defence_power"] + feat_row["away_defence_power"]
    feat_row["attack_defence_ratio"] = feat_row["total_attack"] / (feat_row["total_defence"] + 0.1)
    feat_row["home_scoring_rate"] = feat_row.get("home_gf_5", 0) / (feat_row.get("home_ga_5", 1) + 0.1)
    feat_row["away_scoring_rate"] = feat_row.get("away_gf_5", 0) / (feat_row.get("away_ga_5", 1) + 0.1)
    feat_row["scoring_rate_diff"] = feat_row["home_scoring_rate"] - feat_row["away_scoring_rate"]
    feat_row["both_score_potential"] = 1 if (feat_row.get("home_gf_5", 0) > 0.8 and feat_row.get("away_gf_5", 0) > 0.8) else 0
    feat_row["high_scoring_match"] = 1 if (feat_row.get("home_gf_5", 0) + feat_row.get("away_gf_5", 0)) > 2.5 else 0
    feat_row["low_scoring_match"] = 1 if (feat_row.get("home_gf_5", 0) + feat_row.get("away_gf_5", 0)) < 1.5 else 0
    feat_row["both_attack_strong"] = 1 if (feat_row.get("home_gf_5", 0) > 1.0 and feat_row.get("away_gf_5", 0) > 1.0) else 0
    feat_row["both_defence_weak"] = 1 if (feat_row.get("home_ga_5", 0) > 1.2 and feat_row.get("away_ga_5", 0) > 1.2) else 0
    feat_row["both_score_potential"] = 1 if (feat_row.get("home_gf_5", 0) > 0.8 and feat_row.get("away_gf_5", 0) > 0.8) else 0
    feat_row["home_clean_sheets"] = 1 if feat_row.get("home_ga_5", 1) < 0.5 else 0
    feat_row["away_clean_sheets"] = 1 if feat_row.get("away_ga_5", 1) < 0.5 else 0

    # --- BUILD FEATURE VECTOR ---
    X = pd.DataFrame([feat_row])[feat_cols].fillna(0)

    # --- GOAL FEATURES (separate) ---
    goal_feat_cols = wm.get("goal_features", [])
    if goal_feat_cols:
        X_goal = pd.DataFrame([feat_row])[goal_feat_cols].fillna(0)
    else:
        X_goal = X

    # --- PREDICT ---
    m1x2 = wm.get("m_1x2")
    if m1x2 is None:
        return None

    # v8 format: m1x2 = (model_a, model_b, cal_a, cal_b) where cal = [IsotonicRegression x3]
    if isinstance(m1x2, tuple) and len(m1x2) == 4:
        mA, mB, cal_a, cal_b = m1x2
        pA_raw = mA.predict_proba(X)
        pB_raw = mB.predict_proba(X)
        # Apply isotonic calibration per class
        if cal_a and cal_b and len(cal_a) == 3 and len(cal_b) == 3:
            pA_cal = np.column_stack([cal_a[c].predict(pA_raw[:, c]) for c in range(3)])
            pB_cal = np.column_stack([cal_b[c].predict(pB_raw[:, c]) for c in range(3)])
            sA = pA_cal.sum(axis=1, keepdims=True)
            sA = np.where(sA == 0, 1, sA)
            pA_cal /= sA
            sB = pB_cal.sum(axis=1, keepdims=True)
            sB = np.where(sB == 0, 1, sB)
            pB_cal /= sB
            prob = 0.5 * pA_cal + 0.5 * pB_cal
        else:
            prob = 0.5 * pA_raw + 0.5 * pB_raw
    else:
        m_model, m_cal, _ = m1x2
        prob = m_cal.predict_proba(X)

    s = prob.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1, s)
    prob /= s

    home_prob = float(prob[0][0])
    draw_prob = float(prob[0][1])
    away_prob = float(prob[0][2])

    # --- v11 LEAGUE-AWARE DRAW OVERRIDE ---
    draw_models = wm.get("m_draw")
    draw_thresholds = wm.get("draw_thresholds", {})
    if draw_models and isinstance(draw_models, list) and len(draw_models) > 0:
        try:
            draw_probs_all = []
            for dm_tuple in draw_models:
                if isinstance(dm_tuple, tuple) and len(dm_tuple) == 2:
                    dm, ir = dm_tuple
                    raw = dm.predict_proba(X)[:, 1][0]
                    cal = ir.predict([raw])[0]
                    draw_probs_all.append(cal)
            if draw_probs_all:
                draw_ml = float(np.mean(draw_probs_all))
                # Close match: home ve away olasiliklari cok yakinsa
                gap = abs(home_prob - away_prob)
                close_match = gap < draw_thresholds.get("close_match_gap", 0.25)
                # League-aware threshold
                t_high = draw_thresholds.get("threshold_high_draw", 0.32)
                t_mid = draw_thresholds.get("threshold_mid_draw", 0.38)
                t_low = draw_thresholds.get("threshold_low_draw", 0.45)
                rate_high = draw_thresholds.get("high_draw_rate", 0.28)
                rate_low = draw_thresholds.get("low_draw_rate", 0.22)

                override = False
                if lg_real_draw > rate_high and draw_ml > 0.22 and close_match:
                    override = True
                elif rate_low <= lg_real_draw <= rate_high and draw_ml > 0.26 and close_match:
                    override = True
                elif lg_real_draw < rate_low and draw_ml > 0.30 and close_match:
                    override = True
                elif draw_ml > 0.40:
                    override = True

                if override:
                    draw_prob = min(0.42, max(draw_prob, draw_ml) + max(0.0, 0.32 - draw_prob) * 0.4)
                    # Renormalize
                    total = home_prob + draw_prob + away_prob
                    if total > 0:
                        home_prob /= total
                        draw_prob /= total
                        away_prob /= total
        except Exception:
            pass

    # Poisson xG from goal models
    hxg_raw = max(feat_row.get("home_gf_5", 1.2), 0.3)
    axg_raw = max(feat_row.get("away_gf_5", 1.2), 0.3)
    # Use SofaScore xG rolling if available (from sofascore_rolling.parquet)
    h_xg_r5 = feat_row.get("home_xg_r5", 0)
    a_xg_r5 = feat_row.get("away_xg_r5", 0)
    if h_xg_r5 > 0:
        hxg_raw = 0.5 * hxg_raw + 0.5 * h_xg_r5
    if a_xg_r5 > 0:
        axg_raw = 0.5 * axg_raw + 0.5 * a_xg_r5
    # Also try sf_home_xg from features_combined (SofaScore per-match xG)
    sf_hxg = feat_row.get("sf_home_xg", 0)
    sf_axg = feat_row.get("sf_away_xg", 0)
    if sf_hxg > 0:
        hxg_raw = 0.6 * hxg_raw + 0.4 * sf_hxg
    if sf_axg > 0:
        axg_raw = 0.6 * axg_raw + 0.4 * sf_axg

    mh_pois = wm.get("m_home_goals")
    ma_pois = wm.get("m_away_goals")
    if mh_pois and ma_pois:
        hxg_pois = float(max(mh_pois.predict(X)[0], 0.3))
        axg_pois = float(max(ma_pois.predict(X)[0], 0.3))
        hxg = 0.7 * hxg_raw + 0.3 * hxg_pois
        axg = 0.7 * axg_raw + 0.3 * axg_pois
    else:
        hxg = hxg_raw
        axg = axg_raw
    hxg = min(max(hxg, 0.3), 3.0)
    axg = min(max(axg, 0.3), 3.0)
    total_lam = hxg + axg
    if total_lam > 5.0:
        scale = 5.0 / total_lam
        hxg *= scale
        axg *= scale
        total_lam = 5.0

    # Poisson helpers
    _poisson_pmf = lambda lam, k: math.exp(-lam) * (lam ** k) / math.factorial(k)
    def _poisson_overGoals(lam_h, lam_a, threshold):
        p_under = 0.0
        for gh in range(threshold + 1):
            for ga in range(threshold + 1 - gh):
                p_under += _poisson_pmf(lam_h, gh) * _poisson_pmf(lam_a, ga)
        return max(0.0, min(1.0, 1.0 - p_under))

    _CLIP = lambda v: min(max(float(v), 0.03), 0.88)

    # BTTS — 5-model ensemble + calibration
    mbtts = wm.get("m_btts")
    if mbtts and isinstance(mbtts, list) and len(mbtts) > 0:
        # BTTS = 1 - P(home=0) - P(away=0) + P(both=0)  [inc. exclusion]
        p_h0 = _poisson_pmf(hxg, 0)
        p_a0 = _poisson_pmf(axg, 0)
        bt_poisson = max(0.05, min(0.85, 1.0 - p_h0 - p_a0 + p_h0 * p_a0))
        if isinstance(mbtts[0], tuple):
            bt_cal = np.mean([ir.predict(bm.predict_proba(X_goal)[:, 1]) for bm, ir in mbtts])
            bt_ml = float(np.clip(bt_cal, 0.05, 0.85))
        else:
            bt_ml_a = _CLIP(mbtts[0].predict_proba(X_goal)[:, 1][0])
            bt_ml_b = _CLIP(mbtts[1].predict_proba(X_goal)[:, 1][0])
            bt_ml = 0.5 * bt_ml_a + 0.5 * bt_ml_b
        btts_p = 0.6 * bt_ml + 0.4 * bt_poisson
    else:
        btts_p = _CLIP(1.0 - _poisson_pmf(hxg, 0) - _poisson_pmf(axg, 0) + _poisson_pmf(hxg, 0) * _poisson_pmf(axg, 0))

    # Over 2.5 — 5-model ensemble + calibration
    mo25 = wm.get("m_over25")
    o25_poisson = _poisson_overGoals(hxg, axg, 2)
    if mo25 and isinstance(mo25, list) and len(mo25) > 0:
        if isinstance(mo25[0], tuple):
            o25_cal = np.mean([ir.predict(om.predict_proba(X_goal)[:, 1]) for om, ir in mo25])
            o25_ml = float(np.clip(o25_cal, 0.05, 0.85))
        else:
            o25_ml_a = _CLIP(mo25[0].predict_proba(X_goal)[:, 1][0])
            o25_ml_b = _CLIP(mo25[1].predict_proba(X_goal)[:, 1][0])
            o25_ml = 0.5 * o25_ml_a + 0.5 * o25_ml_b
        o25_p = 0.6 * o25_ml + 0.4 * o25_poisson
    else:
        o25_p = o25_poisson

    # Over 3.5 — 5-model ensemble + calibration
    o35_poisson = _poisson_overGoals(hxg, axg, 3)
    mo35 = wm.get("m_over35")
    if mo35 and isinstance(mo35, list) and len(mo35) > 0:
        if isinstance(mo35[0], tuple):
            o35_cal = np.mean([ir.predict(om.predict_proba(X_goal)[:, 1]) for om, ir in mo35])
            o35_ml = float(np.clip(o35_cal, 0.05, 0.85))
        else:
            o35_ml = _CLIP(mo35[0].predict_proba(X_goal)[:, 1][0])
        o35_p = 0.6 * o35_ml + 0.4 * o35_poisson
    else:
        o35_p = o35_poisson

    # Over 1.5 — 5-model ensemble + calibration
    o15_poisson = _poisson_overGoals(hxg, axg, 1)
    mo15 = wm.get("m_over15")
    if mo15 and isinstance(mo15, list) and len(mo15) > 0:
        if isinstance(mo15[0], tuple):
            o15_cal = np.mean([ir.predict(om.predict_proba(X_goal)[:, 1]) for om, ir in mo15])
            o15_ml = float(np.clip(o15_cal, 0.05, 0.85))
        else:
            o15_ml = _CLIP(mo15[0].predict_proba(X_goal)[:, 1][0])
        o15_p = 0.5 * o15_ml + 0.5 * o15_poisson
    else:
        o15_p = o15_poisson

    u25_p = 1.0 - o25_p
    btts_no_p = 1.0 - btts_p

    # Double chance — from 1X2 probs (consistency)
    d1X_p = min(max(home_prob + draw_prob, 0.05), 0.95)
    dX2_p = min(max(away_prob + draw_prob, 0.05), 0.95)
    d12_p = min(max(home_prob + away_prob, 0.05), 0.95)

    # --- NEW MARKETS (v14) ---
    new_feats_cols = wm.get("new_target_features", [])
    if new_feats_cols:
        X_new = pd.DataFrame([feat_row])[new_feats_cols].fillna(0)
    else:
        X_new = X

    def _new_market_prob(key):
        models = wm.get("new_"+key) or wm.get(key)
        if not models or not isinstance(models, list) or len(models) == 0:
            return None
        if isinstance(models[0], tuple):
            return float(np.mean([ir.predict(m.predict_proba(X_new)[:, 1]) for m, ir in models]))
        return None

    home_cs_p = _new_market_prob("m_home_cs")        # Ev sahibi gol yemez
    home_over15_p = _new_market_prob("m_home_over15")  # Ev sahibi 2+ gol atar
    away_over05_p = _new_market_prob("m_away_over05")  # Depasman gol atar
    home_over05_p = _new_market_prob("m_home_over05")  # Ev sahibi gol atar
    home_win_p = _new_market_prob("m_home_win")        # Ev sahibi kazanir

    # v15 NEW MARKETS — uses same features as main model
    def _v15_prob(key):
        models = wm.get(key)
        if not models or not isinstance(models, list) or len(models) == 0:
            return None
        return float(np.mean([ir.predict(m.predict_proba(X)[:, 1]) for m, ir in models]))

    corner_features = wm.get("corner_features", feat_cols)
    # Compute corner-specific features from sofascore data
    feat_row["home_corners_5"] = 0
    feat_row["away_corners_5"] = 0
    feat_row["corner_diff"] = 0
    if sofa_full is not None:
        sofa_name_h = sofa_name_map.get(home_id, home_name)
        sofa_name_a = sofa_name_map.get(away_id, away_name)
        h_home_matches = sofa_full[sofa_full["home_team"] == sofa_name_h].tail(5)
        if len(h_home_matches) > 0:
            hcol = "h_corners" if "h_corners" in h_home_matches.columns else "home_corners"
            feat_row["home_corners_5"] = float(pd.to_numeric(h_home_matches[hcol], errors="coerce").fillna(0).mean())
        a_away_matches = sofa_full[sofa_full["away_team"] == sofa_name_a].tail(5)
        if len(a_away_matches) > 0:
            acol = "a_corners" if "a_corners" in a_away_matches.columns else "away_corners"
            feat_row["away_corners_5"] = float(pd.to_numeric(a_away_matches[acol], errors="coerce").fillna(0).mean())
    feat_row["corner_diff"] = feat_row["home_corners_5"] - feat_row["away_corners_5"]
    X_corner = pd.DataFrame([feat_row])[corner_features].fillna(0)

    def _corner_prob(key):
        models = wm.get(key)
        if not models or not isinstance(models, list) or len(models) == 0:
            return None
        return float(np.mean([ir.predict(m.predict_proba(X_corner)[:, 1]) for m, ir in models]))

    iy_draw_p = _v15_prob("new_iy_draw")
    iy_ms_dh_p = _v15_prob("new_iy_ms_dh")
    iy_ms_da_p = _v15_prob("new_iy_ms_da")
    iy_over05_p = _v15_prob("new_iy_over05")
    two_y_over05_p = _v15_prob("new_2y_over05")
    two_y_over15_p = _v15_prob("new_2y_over15")
    two_y_more_p = _v15_prob("new_2y_more_goals")
    first_goal_home_p = _v15_prob("new_first_goal_home")
    hk1_home_p = _v15_prob("new_hk1_home")
    hk1_away_p = _v15_prob("new_hk1_away")
    corner_most_home_p = _corner_prob("new_corner_most_home")
    corner_even_p = _corner_prob("new_corner_even")
    corner_over95_p = _corner_prob("new_corner_over95")

    # Corners — Poisson from league avg + team data
    corner_league_avg = feat_row.get("lg_corner_avg", 9.0)
    if corner_league_avg <= 0 or corner_league_avg > 30:
        corner_league_avg = 9.0

    h_corner_for = 0.0
    a_corner_for = 0.0
    if sofa_full is not None:
        sofa_name_h = sofa_name_map.get(home_id, home_name)
        sofa_name_a = sofa_name_map.get(away_id, away_name)
        try:
            h_home_matches = sofa_full[sofa_full["home_team"] == sofa_name_h].tail(10)
            if len(h_home_matches) > 0:
                hcol = "h_corners" if "h_corners" in h_home_matches.columns else "home_corners"
                h_corner_for = float(pd.to_numeric(h_home_matches[hcol], errors="coerce").fillna(0).mean())
            a_away_matches = sofa_full[sofa_full["away_team"] == sofa_name_a].tail(10)
            if len(a_away_matches) > 0:
                acol = "a_corners" if "a_corners" in a_away_matches.columns else "away_corners"
                a_corner_for = float(pd.to_numeric(a_away_matches[acol], errors="coerce").fillna(0).mean())
        except Exception:
            pass
    # Fallback: use features_combined corner data
    if h_corner_for <= 0:
        h_corner_for = max(feat_row.get("home_corners_5", 0), feat_row.get("home_w_corners", 0))
    if a_corner_for <= 0:
        a_corner_for = max(feat_row.get("away_corners_5", 0), feat_row.get("away_w_corners", 0))
    # If still 0, estimate from league avg (balanced split)
    if h_corner_for <= 0:
        h_corner_for = corner_league_avg * 0.50
    if a_corner_for <= 0:
        a_corner_for = corner_league_avg * 0.50

    team_corner_est = h_corner_for + a_corner_for
    if team_corner_est <= 0:
        team_corner_est = corner_league_avg * 1.5
    corner_avg = 0.6 * team_corner_est + 0.4 * corner_league_avg
    corner_avg = float(np.clip(corner_avg, 7.0, 14.0))

    def _poisson_over(lam, threshold):
        return float(1.0 - sum(np.exp(-lam) * (lam ** k) / math.factorial(k) for k in range(threshold + 1)))

    cor75_p = _poisson_over(corner_avg, 7)
    cor85_p = _poisson_over(corner_avg, 8)
    cor95_p = _poisson_over(corner_avg, 9)
    cor105_p = _poisson_over(corner_avg, 10)

    # Score grid from Poisson
    maxg = 12
    ph = [np.exp(-hxg) * (hxg ** k) / math.factorial(k) for k in range(maxg)]
    pa = [np.exp(-axg) * (axg ** k) / math.factorial(k) for k in range(maxg)]
    mat = np.outer(ph, pa)
    mat_s = mat.sum()
    if mat_s > 0:
        mat /= mat_s
    grid = []
    for h in range(0, 8):
        for a in range(0, 8):
            p = mat[h][a]
            grid.append((p, h, a))
    grid.sort(reverse=True)
    scores = [{"score": f"{h}-{a}", "probability": round(p, 4)} for p, h, a in grid[:5]]

    home_prob = _CLIP(home_prob)
    draw_prob = _CLIP(draw_prob)
    away_prob = _CLIP(away_prob)
    total = home_prob + draw_prob + away_prob
    if total > 0:
        home_prob /= total
        draw_prob /= total
        away_prob /= total

    # Bias correction: ELO-aware regression toward base rates
    lg_real_home = feat_row.get("lg_real_home_rate", 0.44)
    if not (0.20 <= lg_real_home <= 0.70): lg_real_home = 0.44
    lg_real_draw = feat_row.get("lg_real_draw_rate", 0.26)
    if not (0.15 <= lg_real_draw <= 0.40): lg_real_draw = 0.26
    lg_real_away = 1.0 - lg_real_home - lg_real_draw
    if lg_real_away < 0.10: lg_real_away = 0.30

    # Reduce bias correction when ELO shows clear favorite
    elo_diff = feat_row.get("elo_diff", 0)
    if abs(elo_diff) > 50:
        bias_strength = 0.15
    elif abs(elo_diff) > 25:
        bias_strength = 0.25
    else:
        bias_strength = 0.35
    home_prob = home_prob * (1 - bias_strength) + lg_real_home * bias_strength
    draw_prob = draw_prob * (1 - bias_strength) + lg_real_draw * bias_strength
    away_prob = away_prob * (1 - bias_strength) + lg_real_away * bias_strength
    total = home_prob + draw_prob + away_prob
    if total > 0:
        home_prob /= total
        draw_prob /= total
        away_prob /= total

    # ELO-based correction: stronger adjustments for away favorites
    elo_diff = feat_row.get("elo_diff", 0)
    if elo_diff < -100:
        away_boost = min(0.18, abs(elo_diff) / 1200)
        away_prob += away_boost
        home_prob -= away_boost * 0.55
        draw_prob -= away_boost * 0.45
    elif elo_diff < -50:
        away_boost = min(0.10, abs(elo_diff) / 1500)
        away_prob += away_boost
        home_prob -= away_boost * 0.55
        draw_prob -= away_boost * 0.45
    elif elo_diff < -25:
        away_boost = min(0.05, abs(elo_diff) / 2500)
        away_prob += away_boost
        home_prob -= away_boost * 0.5
        draw_prob -= away_boost * 0.5
    elif elo_diff > 100:
        home_boost = min(0.12, elo_diff / 1500)
        home_prob += home_boost
        away_prob -= home_boost * 0.55
        draw_prob -= home_boost * 0.45
    elif elo_diff > 50:
        home_boost = min(0.06, elo_diff / 2500)
        home_prob += home_boost
        away_prob -= home_boost * 0.5
        draw_prob -= home_boost * 0.5

    # Final safety
    home_prob = max(home_prob, 0.03)
    draw_prob = max(draw_prob, 0.03)
    away_prob = max(away_prob, 0.03)
    total = home_prob + draw_prob + away_prob
    home_prob /= total
    draw_prob /= total
    away_prob /= total

    # FEEDBACK LOOP: Gecmis hatalardan ogrenerek tahmini ayarla
    feedback_adjustment = None
    try:
        from feedback.feedback_predictor import FeedbackAwarePredictor
        feedback_pred = FeedbackAwarePredictor(data_dir="data/feedback")
        feedback_adjustment = feedback_pred.adjust_prediction(
            home_team_id=home_id,
            away_team_id=away_id,
            league=league_name if 'league_name' in dir() else "",
            pred_home=home_prob,
            pred_draw=draw_prob,
            pred_away=away_prob,
            elo_diff=feat_row.get("elo_diff", 0),
            home_elo=feat_row.get("home_elo", 1500),
            away_elo=feat_row.get("away_elo", 1500),
        )
        # Ayarlanmis olasiliklari kullan
        if feedback_adjustment.adjustments_applied:
            home_prob = feedback_adjustment.adjusted_home
            draw_prob = feedback_adjustment.adjusted_draw
            away_prob = feedback_adjustment.adjusted_away
            total = home_prob + draw_prob + away_prob
            if total > 0:
                home_prob /= total
                draw_prob /= total
                away_prob /= total
    except Exception:
        pass

    # Winner selection: ELO-aware with draw detection
    elo_diff = feat_row.get("elo_diff", 0)
    gap_ha = abs(home_prob - away_prob)
    gap_hd = abs(home_prob - draw_prob)

    # Strong away team: predict away even if ML says home
    if elo_diff < -50 and away_prob > 0.25:
        winner, win_prob = "Deplasman", away_prob
    # Very strong away team: always predict away
    elif elo_diff < -100 and away_prob > 0.18:
        winner, win_prob = "Deplasman", away_prob
    # Strong home team: predict home
    elif elo_diff > 50 and home_prob > 0.25:
        winner, win_prob = "Ev Sahibi", home_prob
    # Close match: check for draw
    elif gap_ha < 0.10 and home_prob < 0.48:
        winner, win_prob = "Beraberlik", draw_prob
    # Default: pick highest
    elif home_prob >= away_prob and home_prob >= draw_prob:
        winner, win_prob = "Ev Sahibi", home_prob
    elif away_prob >= home_prob and away_prob >= draw_prob:
        winner, win_prob = "Deplasman", away_prob
    else:
        winner, win_prob = "Beraberlik", draw_prob

    recommendations = _build_recommendations(
        home_prob, draw_prob, away_prob, winner, win_prob,
        hxg, axg, total_lam,
        o15_p, o25_p, o35_p,
        btts_p, corner_avg, cor75_p, cor85_p,
        hd, ad,
        d1X_p, dX2_p, d12_p,
        u25_p, btts_no_p,
        cor95_p, cor105_p,
        home_cs_p=home_cs_p, home_over15_p=home_over15_p,
        away_over05_p=away_over05_p, home_over05_p=home_over05_p,
        home_win_p=home_win_p,
        iy_draw_p=iy_draw_p, iy_ms_dh_p=iy_ms_dh_p, iy_ms_da_p=iy_ms_da_p, iy_over05_p=iy_over05_p,
        two_y_over05_p=two_y_over05_p, two_y_over15_p=two_y_over15_p, two_y_more_p=two_y_more_p,
        first_goal_home_p=first_goal_home_p,
        hk1_home_p=hk1_home_p, hk1_away_p=hk1_away_p,
        corner_most_home_p=corner_most_home_p, corner_even_p=corner_even_p, corner_over95_p=corner_over95_p,
    )

    return {
        "home": hd, "away": ad,
        "home_name": hd.get("name", ""),
        "away_name": ad.get("name", ""),
        "home_win": round(home_prob * 100, 1),
        "draw": round(draw_prob * 100, 1),
        "away_win": round(away_prob * 100, 1),
        "winner": winner,
        "winner_prob": round(win_prob * 100, 1),
        "btts_yes": round(btts_p * 100, 1),
        "btts_no": round(btts_no_p * 100, 1),
        "new_markets": {
            "home_cs": round(home_cs_p * 100, 1) if home_cs_p else None,
            "home_over15": round(home_over15_p * 100, 1) if home_over15_p else None,
            "away_over05": round(away_over05_p * 100, 1) if away_over05_p else None,
            "home_over05": round(home_over05_p * 100, 1) if home_over05_p else None,
            "home_win": round(home_win_p * 100, 1) if home_win_p else None,
        },
        "v15_markets": {
            "iy_draw": round(iy_draw_p * 100, 1) if iy_draw_p else None,
            "iy_ms_dh": round(iy_ms_dh_p * 100, 1) if iy_ms_dh_p else None,
            "iy_ms_da": round(iy_ms_da_p * 100, 1) if iy_ms_da_p else None,
            "iy_over05": round(iy_over05_p * 100, 1) if iy_over05_p else None,
            "two_y_over05": round(two_y_over05_p * 100, 1) if two_y_over05_p else None,
            "two_y_over15": round(two_y_over15_p * 100, 1) if two_y_over15_p else None,
            "two_y_more": round(two_y_more_p * 100, 1) if two_y_more_p else None,
            "first_goal_home": round(first_goal_home_p * 100, 1) if first_goal_home_p else None,
            "hk1_home": round(hk1_home_p * 100, 1) if hk1_home_p else None,
            "hk1_away": round(hk1_away_p * 100, 1) if hk1_away_p else None,
            "corner_most_home": round(corner_most_home_p * 100, 1) if corner_most_home_p else None,
            "corner_even": round(corner_even_p * 100, 1) if corner_even_p else None,
            "corner_over95": round(corner_over95_p * 100, 1) if corner_over95_p else None,
        },
        "goals": {
            "home_lambda": round(hxg, 3),
            "away_lambda": round(axg, 3),
            "expected_total": round(hxg + axg, 3),
        },
        "total_goals": {
            "over15": round(o15_p * 100, 1),
            "under15": round((1 - o15_p) * 100, 1),
            "over25": round(o25_p * 100, 1),
            "under25": round(u25_p * 100, 1),
            "over35": round(o35_p * 100, 1),
            "under35": round((1 - o35_p) * 100, 1),
        },
        "double_chance": {
            "1X": round(d1X_p * 100, 1),
            "12": round(d12_p * 100, 1),
            "X2": round(dX2_p * 100, 1),
        },
        "corners": {
            "predicted_total": round(corner_avg, 1),
            "over75": round(cor75_p * 100, 1),
            "under75": round((1 - cor75_p) * 100, 1),
            "over85": round(cor85_p * 100, 1),
            "under85": round((1 - cor85_p) * 100, 1),
            "over95": round(cor95_p * 100, 1),
            "under95": round((1 - cor95_p) * 100, 1),
            "over105": round(cor105_p * 100, 1),
            "under105": round((1 - cor105_p) * 100, 1),
        },
        "top_scores": scores,
        "recommendations": recommendations,
        "feedback_info": {
            "adjustments_applied": feedback_adjustment.adjustments_applied if feedback_adjustment else [],
            "confidence_factor": feedback_adjustment.confidence_factor if feedback_adjustment else 1.0,
            "reasons": feedback_adjustment.reasons if feedback_adjustment else [],
        } if feedback_adjustment else None,
    }



@app.route("/api/stats")
def api_stats():
    return jsonify(_get_stats())


@app.route("/api/feedback/record-result", methods=["POST"])
def api_feedback_record_result():
    """Mac sonucunu feedback sistemine kaydet."""
    try:
        data = request.get_json()
        match_id = data.get("match_id")
        actual_result = data.get("actual_result")  # H/D/A

        if not match_id or actual_result not in ["H", "D", "A"]:
            return jsonify({"error": "Gecersiz veri"}), 400

        from feedback.feedback_predictor import FeedbackAwarePredictor
        feedback_pred = FeedbackAwarePredictor(data_dir="data/feedback")
        feedback_pred.record_result(match_id=match_id, actual_result=actual_result)

        return jsonify({"success": True, "message": f"Sonuc kaydedildi: {actual_result}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback/stats")
def api_feedback_stats():
    """Feedback istatistiklerini dondur."""
    try:
        from feedback.feedback_predictor import FeedbackAwarePredictor
        feedback_pred = FeedbackAwarePredictor(data_dir="data/feedback")
        return jsonify(feedback_pred.get_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback/team/<int:team_id>")
def api_feedback_team(team_id):
    """Takim feedback raporunu dondur."""
    try:
        from feedback.feedback_predictor import FeedbackAwarePredictor
        feedback_pred = FeedbackAwarePredictor(data_dir="data/feedback")
        return jsonify(feedback_pred.get_team_report(team_id))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback/league/<league>")
def api_feedback_league(league):
    """Lig feedback raporunu dondur."""
    try:
        from feedback.feedback_predictor import FeedbackAwarePredictor
        feedback_pred = FeedbackAwarePredictor(data_dir="data/feedback")
        return jsonify(feedback_pred.get_league_report(league))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sync/progress")
def api_sync_progress():
    """Anlik veri guncelleme ilerlemesini dondurur."""
    try:
        from jobs.daily_sync import get_progress
        return jsonify(get_progress())
    except Exception as e:
        return jsonify({"running": False, "stage": "error", "message": str(e)})


@app.route("/api/sync/start", methods=["POST"])
def api_sync_start():
    """Manuel veri guncelleme tetikler (arka planda)."""
    try:
        import threading
        from jobs.daily_sync import run_once, _progress_lock, _progress
        with _progress_lock:
            if _progress.get("running"):
                return jsonify({"started": False, "message": "Zaten calisiyor"})
        t = threading.Thread(target=run_once, daemon=True)
        t.start()
        return jsonify({"started": True, "message": "Veri guncelleme baslatildi"})
    except Exception as e:
        return jsonify({"started": False, "error": str(e)}), 500


@app.route("/api/model/info")
def api_model_info():
    fresh = _check_data_freshness()
    model_ok = bool(STATE.get("web_model") or STATE.get("pipeline"))
    
    # Model icinden metrikleri cikar
    metrics = {}
    wm = STATE.get("web_model")
    if wm:
        metrics["test_size"] = wm.get("train_n", 0)
        metrics["home_lambda_avg"] = None
        metrics["away_lambda_avg"] = None
        # Kalibrasyon metrikleri - eger features varsa hesapla
        feat = STATE.get("features")
        if feat is not None and len(feat) > 0:
            try:
                hg = feat["home_goals"].values
                ag = feat["away_goals"].values
                metrics["home_lambda_avg"] = round(float(np.mean(hg)), 3)
                metrics["away_lambda_avg"] = round(float(np.mean(ag)), 3)
                # Basit accuracy: ev sahibi galip ise 1, degilse 0
                hw = float(np.mean(hg > ag))
                dr = float(np.mean(hg == ag))
                aw = float(np.mean(hg < ag))
                metrics["home_win_rate"] = round(hw, 3)
                metrics["draw_rate"] = round(dr, 3)
                metrics["away_win_rate"] = round(aw, 3)
                # BTTS
                metrics["btts_rate"] = round(float(np.mean((hg > 0) & (ag > 0))), 3)
                # Over 2.5
                metrics["over25_rate"] = round(float(np.mean((hg + ag) >= 3)), 3)
            except Exception:
                pass
    
    return jsonify({
        "model_version": STATE.get("model_version"),
        "loaded_at": STATE.get("loaded_at"),
        "loading": STATE["loading"],
        "error": STATE.get("error"),
        "status": "ok" if model_ok else ("loading" if STATE["loading"] else "error"),
        "data_info": STATE.get("data_info", {}),
        "freshness": fresh,
        "metrics": metrics,
    })


@app.route("/api/team/<int:tid>")
def api_team(tid):
    det = TEAM_ID_MAP.get(tid)
    if not det:
        return jsonify({"error": "Takim bulunamadi"}), 404
    return jsonify(det)


@_memo
def _compute_team_stats(team_id):
    feat = STATE["features"]
    if feat is None:
        return None
    info = TEAM_ID_MAP.get(team_id, {})

    home_matches = feat[feat["home_team_id"] == team_id].copy()
    away_matches = feat[feat["away_team_id"] == team_id].copy()

    if len(home_matches) == 0 and len(away_matches) == 0:
        return None

    all_m = pd.concat([home_matches, away_matches]).sort_values("date")

    def _empty_side():
        return {"pl": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0,
                "btts": 0, "over25": 0, "cs": 0, "fts": 0, "xg_for": 0.0,
                "xg_against": 0.0, "shots_for": 0.0, "shots_against": 0.0,
                "corners_for": 0.0, "corners_against": 0.0, "yellow": 0.0,
                "red": 0, "ht_gf": 0.0, "ht_ga": 0.0, "st_gf": 0.0, "st_ga": 0.0,
                "over15": 0, "over35": 0}

    overall = _empty_side()
    home_s = _empty_side()
    away_s = _empty_side()
    goal_dist = {"for": [0, 0, 0, 0, 0], "against": [0, 0, 0, 0, 0]}
    monthly = {}
    opponents = {}
    matches = []
    form_history = []

    current_streak = 0
    current_type = ""
    max_win = 0
    max_unbeaten = 0
    unbeaten = 0
    best_diff = 0
    worst_diff = 0
    biggest_win = ""
    biggest_loss = ""
    best_attack = {"goals": 0, "opponent": ""}

    def _num(row, col, default=0.0):
        try:
            v = row.get(col, None)
            if v is None or pd.isna(v):
                return default
            return float(v)
        except Exception:
            return default

    for _, row in all_m.iterrows():
        hid = int(row["home_team_id"])
        aid = int(row["away_team_id"])
        hg = int(row["home_goals"])
        ag = int(row["away_goals"])
        res = row["result"]
        is_home = (hid == team_id)
        date_str = row["date"].strftime("%d.%m.%Y") if pd.notna(row.get("date")) else ""
        month_str = row["date"].strftime("%Y-%m") if pd.notna(row.get("date")) else ""

        my_g = hg if is_home else ag
        opp_g = ag if is_home else hg
        opp_id = aid if is_home else hid
        opp_name = TEAM_ID_MAP.get(opp_id, {}).get("name", f"Team {opp_id}")

        if is_home and res == "H" or not is_home and res == "A":
            my_res = "W"; overall["w"] += 1; overall["pts"] += 3
            if is_home: home_s["w"] += 1; home_s["pts"] += 3
            else: away_s["w"] += 1; away_s["pts"] += 3
        elif res == "D":
            my_res = "D"; overall["d"] += 1; overall["pts"] += 1
            if is_home: home_s["d"] += 1; home_s["pts"] += 1
            else: away_s["d"] += 1; away_s["pts"] += 1
        else:
            my_res = "L"; overall["l"] += 1
            if is_home: home_s["l"] += 1
            else: away_s["l"] += 1

        overall["pl"] += 1; overall["gf"] += my_g; overall["ga"] += opp_g
        side = home_s if is_home else away_s
        side["pl"] += 1; side["gf"] += my_g; side["ga"] += opp_g

        if hg > 0 and ag > 0:
            overall["btts"] += 1; side["btts"] += 1
        tot = hg + ag
        if tot >= 3:
            overall["over25"] += 1; side["over25"] += 1
        if tot >= 2:
            overall["over15"] += 1; side["over15"] += 1
        if tot >= 4:
            overall["over35"] += 1; side["over35"] += 1
        if opp_g == 0:
            overall["cs"] += 1
        if my_g == 0:
            overall["fts"] += 1

        # xG / shots / corners / cards
        xgf = _num(row, "home_xg_real") if is_home else _num(row, "away_xg_real")
        xga = _num(row, "away_xg_real") if is_home else _num(row, "home_xg_real")
        shf = _num(row, "home_shots") if is_home else _num(row, "away_shots")
        sha = _num(row, "away_shots") if is_home else _num(row, "home_shots")
        cf = _num(row, "home_corners") if is_home else _num(row, "away_corners")
        ca = _num(row, "away_corners") if is_home else _num(row, "home_corners")
        yel = _num(row, "home_yellow") if is_home else _num(row, "away_yellow")
        red = int(_num(row, "home_red")) if is_home else int(_num(row, "away_red"))
        overall["xg_for"] += xgf; overall["xg_against"] += xga
        overall["shots_for"] += shf; overall["shots_against"] += sha
        overall["corners_for"] += cf; overall["corners_against"] += ca
        overall["yellow"] += yel; overall["red"] += red
        side["xg_for"] += xgf; side["xg_against"] += xga
        side["shots_for"] += shf; side["shots_against"] += sha
        side["corners_for"] += cf; side["corners_against"] += ca
        side["yellow"] += yel; side["red"] += red

        # ilk yari / ikinci yari
        hth = _num(row, "ht_home_goals", float("nan"))
        hta = _num(row, "ht_away_goals", float("nan"))
        if hth == hth and hta == hta:
            my_ht = hth if is_home else hta
            opp_ht = hta if is_home else hth
            overall["ht_gf"] += my_ht; overall["ht_ga"] += opp_ht
            overall["st_gf"] += max(my_g - my_ht, 0); overall["st_ga"] += max(opp_g - opp_ht, 0)
            side["ht_gf"] += my_ht; side["ht_ga"] += opp_ht
            side["st_gf"] += max(my_g - my_ht, 0); side["st_ga"] += max(opp_g - opp_ht, 0)

        goal_dist["for"][min(my_g, 4)] += 1
        goal_dist["against"][min(opp_g, 4)] += 1

        if my_g > best_diff and my_g > opp_g:
            best_diff = my_g - opp_g
            biggest_win = f"{opp_name} {my_g}-{opp_g}"
        if opp_g > my_g and (opp_g - my_g) > worst_diff:
            worst_diff = opp_g - my_g
            biggest_loss = f"{opp_name} {opp_g}-{my_g}"
        if my_g > best_attack["goals"]:
            best_attack = {"goals": my_g, "opponent": opp_name}

        if my_res == current_type:
            current_streak += 1
        else:
            current_streak = 1; current_type = my_res
        if current_type == "W" and current_streak > max_win:
            max_win = current_streak
        if my_res in ("W", "D") and current_type in ("W", "D"):
            unbeaten += 1
        else:
            unbeaten = 0
        if unbeaten > max_unbeaten:
            max_unbeaten = unbeaten

        if month_str:
            if month_str not in monthly:
                monthly[month_str] = {"pl": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0}
            m = monthly[month_str]
            m["pl"] += 1; m["gf"] += my_g; m["ga"] += opp_g
            if my_res == "W": m["w"] += 1; m["pts"] += 3
            elif my_res == "D": m["d"] += 1; m["pts"] += 1
            else: m["l"] += 1

        if opp_id not in opponents:
            opponents[opp_id] = {"name": opp_name, "pl": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0}
        op = opponents[opp_id]
        op["pl"] += 1; op["gf"] += my_g; op["ga"] += opp_g
        if my_res == "W": op["w"] += 1
        elif my_res == "D": op["d"] += 1
        else: op["l"] += 1

        matches.append({
            "date": date_str, "month": month_str,
            "opponent": opp_name, "opp_id": opp_id,
            "is_home": is_home, "gf": my_g, "ga": opp_g,
            "result": my_res, "gd": my_g - opp_g,
            "home_xg": round(xgf, 2), "away_xg": round(xga, 2),
        })
        form_history.append({
            "date": date_str, "opponent": opp_name,
            "is_home": is_home, "gf": my_g, "ga": opp_g,
            "result": my_res, "xg_for": round(xgf, 2), "xg_against": round(xga, 2),
        })

    o = overall
    o["gd"] = o["gf"] - o["ga"]
    o["ppg"] = round(o["pts"] / max(o["pl"], 1), 2)
    o["gpg"] = round(o["gf"] / max(o["pl"], 1), 2)
    o["gapg"] = round(o["ga"] / max(o["pl"], 1), 2)
    o["btts_rate"] = round(o["btts"] / max(o["pl"], 1) * 100, 1)
    o["over25_rate"] = round(o["over25"] / max(o["pl"], 1) * 100, 1)
    o["over15_rate"] = round(o["over15"] / max(o["pl"], 1) * 100, 1)
    o["over35_rate"] = round(o["over35"] / max(o["pl"], 1) * 100, 1)
    o["cs_rate"] = round(o["cs"] / max(o["pl"], 1) * 100, 1)
    o["fts_rate"] = round(o["fts"] / max(o["pl"], 1) * 100, 1)
    o["win_rate"] = round(o["w"] / max(o["pl"], 1) * 100, 1)
    o["draw_rate"] = round(o["d"] / max(o["pl"], 1) * 100, 1)
    o["loss_rate"] = round(o["l"] / max(o["pl"], 1) * 100, 1)
    o["xg_for_avg"] = round(o["xg_for"] / max(o["pl"], 1), 2)
    o["xg_against_avg"] = round(o["xg_against"] / max(o["pl"], 1), 2)
    o["shots_for_avg"] = round(o["shots_for"] / max(o["pl"], 1), 1)
    o["shots_against_avg"] = round(o["shots_against"] / max(o["pl"], 1), 1)
    o["corners_for_avg"] = round(o["corners_for"] / max(o["pl"], 1), 1)
    o["corners_against_avg"] = round(o["corners_against"] / max(o["pl"], 1), 1)
    o["yellow_avg"] = round(o["yellow"] / max(o["pl"], 1), 1)
    o["red_total"] = int(o["red"])
    o["ht_gf_avg"] = round(o["ht_gf"] / max(o["pl"], 1), 2)
    o["ht_ga_avg"] = round(o["ht_ga"] / max(o["pl"], 1), 2)
    o["st_gf_avg"] = round(o["st_gf"] / max(o["pl"], 1), 2)
    o["st_ga_avg"] = round(o["st_ga"] / max(o["pl"], 1), 2)
    o["ht_share"] = round(o["ht_gf"] / max(o["gf"], 1) * 100, 1)
    o["st_share"] = round(o["st_gf"] / max(o["gf"], 1) * 100, 1)

    for side in [home_s, away_s]:
        s = side
        s["ppg"] = round(s["pts"] / max(s["pl"], 1), 2)
        s["gpg"] = round(s["gf"] / max(s["pl"], 1), 2)
        s["gapg"] = round(s["ga"] / max(s["pl"], 1), 2)
        s["btts_rate"] = round(s["btts"] / max(s["pl"], 1) * 100, 1)
        s["over25_rate"] = round(s["over25"] / max(s["pl"], 1) * 100, 1)
        s["over15_rate"] = round(s["over15"] / max(s["pl"], 1) * 100, 1)
        s["over35_rate"] = round(s["over35"] / max(s["pl"], 1) * 100, 1)
        s["xg_for_avg"] = round(s["xg_for"] / max(s["pl"], 1), 2)
        s["xg_against_avg"] = round(s["xg_against"] / max(s["pl"], 1), 2)
        s["corners_for_avg"] = round(s["corners_for"] / max(s["pl"], 1), 1)
        s["corners_against_avg"] = round(s["corners_against"] / max(s["pl"], 1), 1)
        s["ht_gf_avg"] = round(s["ht_gf"] / max(s["pl"], 1), 2)
        s["st_gf_avg"] = round(s["st_gf"] / max(s["pl"], 1), 2)

    n = max(overall["pl"], 1)
    stats = {
        "id": team_id,
        "name": info.get("name", f"Team {team_id}"),
        "league": info.get("league", "?"),
        "league_code": info.get("league_code", "?"),
        "elo": info.get("elo", 1500),
        "attack_elo": info.get("attack_elo", 1500),
        "defence_elo": info.get("defence_elo", 1500),
        "form": info.get("form", "?????"),
        "avatar": info.get("avatar") or _avatar_for(info.get("name", ""), team_id),
        "overall": o,
        "home": home_s,
        "away": away_s,
        "monthly": monthly,
        "matches": matches[-30:],
        "form_history": form_history[-10:],
        "opponents": opponents,
        "goal_dist": {
            "for": goal_dist["for"], "against": goal_dist["against"],
            "for_pct": [round(c / n * 100, 1) for c in goal_dist["for"]],
            "against_pct": [round(c / n * 100, 1) for c in goal_dist["against"]],
        },
        "streaks": {"current": current_streak, "current_type": current_type,
                    "longest_win": max_win, "longest_unbeaten": max_unbeaten},
        "highs": {"most_goals_scored": best_attack["goals"],
                  "most_goals_opponent": best_attack["opponent"],
                  "biggest_win": biggest_win, "biggest_loss": biggest_loss},
        "top_opponents": sorted(opponents.values(), key=lambda x: -x["pl"])[:10],
    }
    stats["monthly_list"] = sorted(monthly.items())[-12:]

    return stats


@app.route("/api/team/<int:tid>/detail")
def api_team_detail(tid):
    stats = _compute_team_stats(tid)
    if stats is None:
        return jsonify({"error": "Takim bulunamadi"}), 404
    return jsonify(stats)


@app.route("/team/<int:tid>")
def team_page(tid):
    return render_template("team.html", team_id=tid)


def _resolve_team_id(name):
    """Isim (normalize edilmis veya tam) ile takim id'sini bul."""
    q = _normalize_team_name(name)
    for tid, det in TEAM_ID_MAP.items():
        if det.get("norm") == q or _normalize_team_name(det.get("name", "")) == q:
            return tid
    # kismi eslesme: baslangic ile
    for tid, det in TEAM_ID_MAP.items():
        if det.get("norm", "").startswith(q) or _normalize_team_name(det.get("name", "")).startswith(q):
            return tid
    return None


@app.route("/team/name/<path:name>")
def team_page_by_name(name):
    tid = _resolve_team_id(name)
    if tid is None:
        return "Takim bulunamadi: " + name, 404
    return render_template("team.html", team_id=tid)


@app.route("/api/team/name/<path:name>")
def api_team_by_name(name):
    tid = _resolve_team_id(name)
    if tid is None:
        return jsonify({"error": "Takim bulunamadi"}), 404
    return jsonify({"id": tid, "detail": _compute_team_stats(tid)})


def _get_stats():
    m = STATE.get("matches")
    if m is None:
        m = STATE.get("features")
    if m is None:
        return {
            "total_matches": 0, "total_leagues": 0, "total_teams": 0,
            "avg_goals": 0, "btts_rate": 0, "over25_rate": 0,
            "home_win_rate": 0, "draw_rate": 0,
            "model_version": "N/A", "loaded_at": "Yukleniyor...", "date_range": "N/A",
            "recent_matches": [],
        }
    recent = []
    for _, row in m.tail(10).iterrows():
        recent.append({
            "date": row["date"].strftime("%d.%m") if pd.notna(row.get("date")) else "",
            "home_team": TEAM_ID_MAP.get(int(row["home_team_id"]), {}).get("name", f"Team {int(row['home_team_id'])}"),
            "away_team": TEAM_ID_MAP.get(int(row["away_team_id"]), {}).get("name", f"Team {int(row['away_team_id'])}"),
            "home_goals": int(row["home_goals"]),
            "away_goals": int(row["away_goals"]),
            "result": row["result"],
            "league": _resolve_league(str(row.get("league", ""))),
        })
    return {
        "total_matches": len(m), "total_leagues": m["league"].nunique(),
        "total_teams": len(set(m["home_team_id"].unique()) | set(m["away_team_id"].unique())),
        "avg_goals": round(float(m["home_goals"].mean() + m["away_goals"].mean()), 2),
        "btts_rate": round(float(((m["home_goals"] > 0) & (m["away_goals"] > 0)).mean()), 3),
        "over25_rate": round(float(((m["home_goals"] + m["away_goals"]) >= 3).mean()), 3),
        "home_win_rate": round(float((m["home_goals"] > m["away_goals"]).mean()), 3),
        "draw_rate": round(float((m["home_goals"] == m["away_goals"]).mean()), 3),
        "model_version": STATE.get("model_version", "N/A"),
        "loaded_at": STATE.get("loaded_at", "N/A"),
        "date_range": f"{m['date'].min().strftime('%d.%m.%Y')} - {m['date'].max().strftime('%d.%m.%Y')}",
        "recent_matches": recent,
    }


def _get_recent():
    if STATE["features"] is None:
        return []
    f = STATE["features"].tail(10)
    return [{
        "match_id": int(row.get("match_id", 0)),
        "home_team": TEAM_ID_MAP.get(int(row["home_team_id"]), {}).get("name", f"Team {int(row['home_team_id'])}"),
        "away_team": TEAM_ID_MAP.get(int(row["away_team_id"]), {}).get("name", f"Team {int(row['away_team_id'])}"),
        "date": row["date"].strftime("%d.%m.%Y") if pd.notna(row.get("date")) else "N/A",
        "result": row["result"],
        "home_goals": int(row["home_goals"]),
        "away_goals": int(row["away_goals"]),
        "home_elo": round(float(row["home_elo"]), 0),
        "away_elo": round(float(row["away_elo"]), 0),
    } for _, row in f.iterrows()]


@_memo
def _compute_league_stats(league_code):
    feat = STATE["features"]
    if feat is None:
        return None
    if isinstance(league_code, str) and "," in league_code:
        codes = [c.strip() for c in league_code.split(",")]
    elif isinstance(league_code, list):
        codes = league_code
    else:
        codes = [league_code]
    lf = feat[feat["league"].isin(codes)]
    if len(lf) == 0:
        return None
    lf = lf.sort_values("date")

    hg = lf["home_goals"].values
    ag = lf["away_goals"].values
    res = lf["result"].values
    hids = lf["home_team_id"].values.astype(int)
    aids = lf["away_team_id"].values.astype(int)
    dates = lf["date"].values
    n = len(lf)

    all_tids = np.unique(np.concatenate([hids, aids]))

    canonical_map = {}
    for raw_tid in all_tids:
        det = TEAM_ID_MAP.get(int(raw_tid))
        if det:
            canonical_map[int(raw_tid)] = det["id"]
        else:
            canonical_map[int(raw_tid)] = int(raw_tid)

    teams = {}
    for tid in all_tids:
        cid = canonical_map[int(tid)]
        if cid not in teams:
            info = TEAM_ID_MAP.get(int(tid), {})
            teams[cid] = {
                "id": cid, "name": info.get("name", f"Team {tid}"),
                "elo": info.get("elo", 1500), "form": [],
                "pl": 0, "w": 0, "d": 0, "l": 0,
                "gf": 0, "ga": 0, "gd": 0, "pts": 0,
                "home_pl": 0, "home_w": 0, "home_d": 0, "home_l": 0, "home_pts": 0,
                "away_pl": 0, "away_w": 0, "away_d": 0, "away_l": 0, "away_pts": 0,
                "btts": 0, "over15": 0, "over25": 0, "over35": 0,
                "clean_sheets": 0, "fts": 0, "xg_for": 0.0, "xg_against": 0.0,
                "matches": [],
            }
        else:
            existing = teams[cid]
            info = TEAM_ID_MAP.get(int(tid), {})
            if info.get("elo", 0) > existing["elo"]:
                existing["elo"] = info["elo"]
                existing["name"] = info["name"]

    for i in range(n):
        tid_h = canonical_map[int(hids[i])]
        tid_a = canonical_map[int(aids[i])]
        g_h = int(hg[i]); g_a = int(ag[i])
        r = res[i]

        th = teams[tid_h]; ta = teams[tid_a]
        th["pl"] += 1; th["gf"] += g_h; th["ga"] += g_a
        ta["pl"] += 1; ta["gf"] += g_a; ta["ga"] += g_h
        th["home_pl"] += 1; ta["away_pl"] += 1
        th["home_pts"] += 3 if r == "H" else (1 if r == "D" else 0)
        ta["away_pts"] += 3 if r == "A" else (1 if r == "D" else 0)

        if r == "H":
            th["w"] += 1; th["pts"] += 3; ta["l"] += 1
            th["home_w"] += 1; th["form"].append("W"); ta["form"].append("L")
        elif r == "D":
            th["d"] += 1; th["pts"] += 1; ta["d"] += 1; ta["pts"] += 1
            th["home_d"] += 1; ta["away_d"] += 1
            th["form"].append("D"); ta["form"].append("D")
        else:
            th["l"] += 1; ta["w"] += 1; ta["pts"] += 3
            th["home_l"] += 1; ta["away_w"] += 1
            th["form"].append("L"); ta["form"].append("W")

        tot = g_h + g_a
        if g_h > 0 and g_a > 0: th["btts"] += 1; ta["btts"] += 1
        if tot >= 2: th["over15"] += 1; ta["over15"] += 1
        if tot >= 3: th["over25"] += 1; ta["over25"] += 1
        if tot >= 4: th["over35"] += 1; ta["over35"] += 1
        if g_a == 0: th["clean_sheets"] += 1
        if g_h == 0: ta["clean_sheets"] += 1
        if g_h == 0: th["fts"] += 1
        if g_a == 0: ta["fts"] += 1

        try:
            th["xg_for"] += float(lf["home_xg_real"].iloc[i]) if "home_xg_real" in lf.columns else float(lf["home_xg"].iloc[i])
        except Exception:
            pass
        try:
            th["xg_against"] += float(lf["away_xg_real"].iloc[i]) if "away_xg_real" in lf.columns else float(lf["away_xg"].iloc[i])
            ta["xg_for"] += float(lf["away_xg_real"].iloc[i]) if "away_xg_real" in lf.columns else float(lf["away_xg"].iloc[i])
            ta["xg_against"] += float(lf["home_xg_real"].iloc[i]) if "home_xg_real" in lf.columns else float(lf["home_xg"].iloc[i])
        except Exception:
            pass

        d_str = str(dates[i])[:10] if pd.notna(dates[i]) else ""
        opp_a = teams.get(tid_a, {})
        opp_h = teams.get(tid_h, {})
        th["matches"].append({"date": d_str, "opponent": opp_a.get("name", "?"),
                              "is_home": True, "gf": g_h, "ga": g_a,
                              "result": "W" if r == "H" else ("D" if r == "D" else "L")})
        ta["matches"].append({"date": d_str, "opponent": opp_h.get("name", "?"),
                              "is_home": False, "gf": g_a, "ga": g_h,
                              "result": "W" if r == "A" else ("D" if r == "D" else "L")})

    for t in teams.values():
        t["gd"] = t["gf"] - t["ga"]
        t["form"] = t["form"][-5:]
        t["matches"] = t["matches"][-10:]
        pl = max(t["pl"], 1)
        t["ppg"] = round(t["pts"] / pl, 2)
        t["gpg"] = round(t["gf"] / pl, 2)
        t["gapg"] = round(t["ga"] / pl, 2)
        t["btts_rate"] = round(t["btts"] / pl * 100, 1)
        t["over25_rate"] = round(t["over25"] / pl * 100, 1)
        t["cs_rate"] = round(t["clean_sheets"] / pl * 100, 1)
        t["fts_rate"] = round(t["fts"] / pl * 100, 1)
        t["xg_for_avg"] = round(t["xg_for"] / pl, 2)
        t["xg_against_avg"] = round(t["xg_against"] / pl, 2)
        t["home_ppg"] = round(t["home_pts"] / max(t["home_pl"], 1), 2)
        t["away_ppg"] = round(t["away_pts"] / max(t["away_pl"], 1), 2)

    standings = sorted(teams.values(), key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))
    total_matches = n
    total_goals = int(np.sum(hg) + np.sum(ag))
    home_wins = int(np.sum(res == "H"))
    draws = int(np.sum(res == "D"))
    away_wins = int(np.sum(res == "A"))

    hxg_col = "home_xg_real" if "home_xg_real" in lf.columns else ("home_xg" if "home_xg" in lf.columns else None)
    axg_col = "away_xg_real" if "away_xg_real" in lf.columns else ("away_xg" if "away_xg" in lf.columns else None)
    home_xg = float(lf[hxg_col].fillna(0).sum()) if hxg_col else 0
    away_xg = float(lf[axg_col].fillna(0).sum()) if axg_col else 0

    tot_arr = hg + ag
    btts_n = int(np.sum((hg > 0) & (ag > 0)))
    over15_n = int(np.sum(tot_arr >= 2))
    over35_n = int(np.sum(tot_arr >= 4))
    goal_dist = [int(np.sum(tot_arr == k)) for k in range(5)]

    corners = 0.0
    cards = 0.0
    if "home_corners" in lf.columns and "away_corners" in lf.columns:
        corners = float(lf["home_corners"].fillna(0).sum() + lf["away_corners"].fillna(0).sum())
    if "home_yellow" in lf.columns:
        cards = float(lf["home_yellow"].fillna(0).sum() + lf["away_yellow"].fillna(0).sum())
        if "home_red" in lf.columns:
            cards += float(lf["home_red"].fillna(0).sum() + lf["away_red"].fillna(0).sum()) * 2

    recent = []
    for _, row in lf.tail(20).iterrows():
        recent.append({
            "date": row["date"].strftime("%d.%m.%Y") if pd.notna(row.get("date")) else "",
            "home": TEAM_ID_MAP.get(int(row["home_team_id"]), {}).get("name", "?"),
            "away": TEAM_ID_MAP.get(int(row["away_team_id"]), {}).get("name", "?"),
            "hg": int(row["home_goals"]), "ag": int(row["away_goals"]),
            "result": row["result"],
        })

    return {
        "league_code": (league_code[0] if isinstance(league_code, list) else league_code) if not isinstance(league_code, str) else league_code,
        "league_name": _resolve_league(league_code[0] if isinstance(league_code, list) else league_code),
        "total_matches": total_matches,
        "total_teams": len(teams),
        "total_goals": total_goals,
        "avg_goals": round(total_goals / max(total_matches, 1), 2),
        "home_win_pct": round(home_wins / max(total_matches, 1) * 100, 1),
        "draw_pct": round(draws / max(total_matches, 1) * 100, 1),
        "away_win_pct": round(away_wins / max(total_matches, 1) * 100, 1),
        "home_adv": round((home_wins - away_wins) / max(total_matches, 1) * 100, 1),
        "btts_rate": round(btts_n / max(total_matches, 1) * 100, 1),
        "over15_rate": round(over15_n / max(total_matches, 1) * 100, 1),
        "over25_rate": round(np.sum(tot_arr >= 3) / max(total_matches, 1) * 100, 1),
        "over35_rate": round(over35_n / max(total_matches, 1) * 100, 1),
        "avg_corners": round(corners / max(total_matches, 1), 1),
        "avg_cards": round(cards / max(total_matches, 1), 1),
        "goal_dist": {"counts": goal_dist, "pct": [round(c / max(total_matches, 1) * 100, 1) for c in goal_dist]},
        "avg_home_xg": round(home_xg / max(total_matches, 1), 2),
        "avg_away_xg": round(away_xg / max(total_matches, 1), 2),
        "standings": standings[:40],
        "recent": recent,
    }


@app.route("/api/league/<code>")
def api_league_detail(code):
    target_name = _resolve_league(code)
    all_codes = [c for c, teams in LEAGUE_TEAMS.items() if _resolve_league(c) == target_name]
    stats = _compute_league_stats(all_codes)
    if stats is None:
        return jsonify({"error": "Lig bulunamadi"}), 404
    return jsonify(stats)


@app.route("/league/<code>")
def league_page(code):
    return render_template("league.html", league_code=code)


@app.route("/results")
def results_page():
    return render_template("results.html")


@app.route("/api/results")
def api_results():
    """Gecmis maclari lig/takim filtresiyle dondur (Sonuclar sekmesi)."""
    feat = STATE.get("features")
    if feat is None:
        return jsonify({"error": "Veri yuklenmedi"}), 503
    try:
        league = request.args.get("league", "").strip()
        team = request.args.get("team", "").strip()
        limit = min(int(request.args.get("limit", 100)), 500)
        try:
            page = max(int(request.args.get("page", 0)), 0)
        except Exception:
            page = 0

        df = feat.sort_values("date").copy()
        if league:
            target_name = _resolve_league(league)
            all_codes = [c for c, teams in LEAGUE_TEAMS.items()
                         if _resolve_league(c) == target_name]
            df = df[df["league"].isin(all_codes if all_codes else [league])]
        if team:
            try:
                tid = int(team)
                df = df[(df["home_team_id"] == tid) | (df["away_team_id"] == tid)]
            except Exception:
                pass

        total = len(df)
        df = df.tail(limit + page * limit).head(limit)

        rows = []
        for _, row in df.iterrows():
            hid = int(row["home_team_id"])
            aid = int(row["away_team_id"])

            def _safe_xg(col):
                v = row.get(col, None)
                try:
                    if v is None or pd.isna(v):
                        return None
                    f = float(v)
                    if f != f:  # NaN
                        return None
                    return round(f, 2)
                except Exception:
                    return None

            rows.append({
                "date": row["date"].strftime("%d.%m.%Y") if pd.notna(row.get("date")) else "",
                "league": _resolve_league(str(row.get("league", ""))),
                "home_id": hid,
                "away_id": aid,
                "home": TEAM_ID_MAP.get(hid, {}).get("name", f"Team {hid}"),
                "away": TEAM_ID_MAP.get(aid, {}).get("name", f"Team {aid}"),
                "hg": int(row["home_goals"]),
                "ag": int(row["away_goals"]),
                "result": row["result"],
                "btts": int(row.get("btts", 0)),
                "over25": int(row.get("over25", 0)),
                "home_xg": _safe_xg("home_xg_real") if _safe_xg("home_xg_real") is not None else _safe_xg("home_xg"),
                "away_xg": _safe_xg("away_xg_real") if _safe_xg("away_xg_real") is not None else _safe_xg("away_xg"),
            })
        return jsonify({"total": total, "count": len(rows), "page": page, "matches": rows})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/league/summary")
def api_league_summary():
    """Dashboard icin: her ligde toplam mac, gol ort, BTTS/Over25 orani."""
    feat = STATE.get("features")
    if feat is None:
        return jsonify([])
    merged = {}
    for code, teams in LEAGUE_TEAMS.items():
        name = _resolve_league(code)
        if name not in merged:
            merged[name] = {"name": name, "codes": set(), "n": 0, "hg": 0, "ag": 0,
                            "btts": 0, "over25": 0, "teams": 0}
        merged[name]["codes"].add(code)
        merged[name]["teams"] += len(teams)
    # Toplulari data'dan hesap et
    out = []
    for name, info in merged.items():
        codes = list(info["codes"])
        sub = feat[feat["league"].isin(codes)]
        n = len(sub)
        hg = int(sub["home_goals"].sum()) if n else 0
        ag = int(sub["away_goals"].sum()) if n else 0
        btts = int(((sub["home_goals"] > 0) & (sub["away_goals"] > 0)).sum()) if n else 0
        over25 = int((sub["total_goals"] >= 3).sum()) if n else 0
        out.append({
            "name": name,
            "teams": info["teams"],
            "matches": n,
            "avg_goals": round((hg + ag) / max(n, 1), 2),
            "btts_rate": round(btts / max(n, 1) * 100, 1),
            "over25_rate": round(over25 / max(n, 1) * 100, 1),
        })
    out.sort(key=lambda x: -x["matches"])
    return jsonify(out)



