"""FBref provider - GERCEK mac istatistikleri (shot, SOT, kart, faul, hakem).

Kaynak: https://fbref.com via soccerdata kutuphanesi (ucretsiz scraping).
- FD'de olmayan ulkelere istatistik ekler:
  BRA, ARG, MEX, USA(MLS), JAP, KOR, COL, URU, PAR, ECU, CRO, SRB,
  CZE, POL, SUI, AUT, DEN, SWE, NOR, AUS, BUL, GRE, HUN, KSA, RUS...
- API key gerektirmez.
- Istatistikler GERCEK veridir: sahte uretim YOKTUR.
- Sonuclar data/gold/fbref_matches.parquet cache'ine kaydedilir.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lig -> sezon listesi (FBref formatinda)
# Avrupa ligleri: "2018-2019" formatinda, digerleri: "2019" formatinda
# SADECE non-EU ligler: EU istatistikleri zaten FD + xgabora'da mevcut
LEAGUE_SEASONS: dict[str, list[str]] = {
    # Guney Amerika
    "BRA-Serie A Brasil": ["2022", "2023", "2024"],
    "ARG-Primera División Argentina": ["2022", "2023", "2024"],
    "COL-Primera A Colombia": ["2022", "2023", "2024"],
    "URU-Uruguayan Primera División": ["2022", "2023", "2024"],
    "ECU-Liga Profesional Ecuador": ["2022", "2023", "2024"],
    "PER-Liga 1 Peru": ["2022", "2023", "2024"],
    "VEN-Venezuelan Primera División": ["2022", "2023", "2024"],
    # Kuzey Amerika
    "USA-Major League Soccer": ["2022", "2023", "2024"],
    "MEX-Liga MX": ["2022", "2023", "2024"],
    "CAN-Canadian Premier League": ["2022", "2023", "2024"],
    # Asya
    "JAP-J1 League": ["2022", "2023", "2024"],
    "KOR-K League 1": ["2022", "2023", "2024"],
    "CHN-Chinese Super League": ["2022", "2023", "2024"],
    "IND-Indian Super League": ["2022-2023", "2023-2024"],
    "AUS-A-League": ["2022-2023", "2023-2024"],
    "KSA-Saudi Pro League": ["2022-2023", "2023-2024"],
    "IRN-Persian Gulf Pro League": ["2022-2023", "2023-2024"],
    # Afrika
    "RSA-South African Premiership": ["2022-2023", "2023-2024"],
    # Avrupa (FD'de olmayanlar - son 2-3 sezon)
    "RUS-Russian Premier League": ["2022-2023", "2023-2024"],
    "UKR-Ukrainian Premier League": ["2022-2023", "2023-2024"],
    "GRE-Super League Greece": ["2022-2023", "2023-2024"],
    "CRO-Croatian Football League": ["2022-2023", "2023-2024"],
    "SRB-Serbian SuperLiga": ["2022-2023", "2023-2024"],
    "CZE-Czech First League": ["2022-2023", "2023-2024"],
    "POL-Ekstraklasa": ["2022-2023", "2023-2024"],
    "SUI-Swiss Super League": ["2022-2023", "2023-2024"],
    "AUT-Austrian Football Bundesliga": ["2022-2023", "2023-2024"],
    "DEN-Danish Superliga": ["2022-2023", "2023-2024"],
    "SWE-Allsvenskan": ["2022", "2023", "2024"],
    "NOR-Eliteserien": ["2022", "2023", "2024", "2025"],
    "BUL-First Professional Football League": ["2022-2023", "2023-2024", "2024-2025"],
    "HUN-Nemzeti Bajnokság I": ["2022-2023", "2023-2024", "2024-2025"],
}

# Ulke kodu -> league_name icin kisa kod
COUNTRY_MAP = {
    "BRA": "Brazil",
    "ARG": "Argentina",
    "COL": "Colombia",
    "URU": "Uruguay",
    "ECU": "Ecuador",
    "PER": "Peru",
    "VEN": "Venezuela",
    "USA": "USA",
    "MEX": "Mexico",
    "CAN": "Canada",
    "JAP": "Japan",
    "KOR": "Korea Republic",
    "CHN": "China",
    "IND": "India",
    "AUS": "Australia",
    "KSA": "Saudi Arabia",
    "IRN": "Iran",
    "RSA": "South Africa",
    "RUS": "Russia",
    "UKR": "Ukraine",
    "GRE": "Greece",
    "CRO": "Croatia",
    "SRB": "Serbia",
    "CZE": "Czech Republic",
    "POL": "Poland",
    "SUI": "Switzerland",
    "AUT": "Austria",
    "DEN": "Denmark",
    "SWE": "Sweden",
    "NOR": "Norway",
    "BUL": "Bulgaria",
    "HUN": "Hungary",
}


def _ensure_league_dict() -> None:
    """Custom league_dict.json'u soccerdata config'ine yaz."""
    import json
    import os

    cfg_dir = Path.home() / "soccerdata" / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    dict_path = cfg_dir / "league_dict.json"

    if dict_path.exists():
        return

    # hale46/fbref-scraper'daki 36 liglik liste (FBref adlariyla)
    base = {
        "NED-Eredivisie": {"FBref": "Eredivisie", "season_start": "Aug", "season_end": "May"},
        "BEL-Jupiler Pro League": {"FBref": "Belgian Pro League", "season_start": "Jul", "season_end": "Jun"},
        "AUT-Austrian Football Bundesliga": {"FBref": "Austrian Football Bundesliga", "season_start": "Jul", "season_end": "Jun"},
        "BRA-Serie A Brasil": {"FBref": "Campeonato Brasileiro Série A", "season_start": "Apr", "season_end": "Dec"},
        "COL-Primera A Colombia": {"FBref": "Categoría Primera A", "season_start": "Jan", "season_end": "Dec"},
        "CRO-Croatian Football League": {"FBref": "Croatian Football League", "season_start": "Jul", "season_end": "May"},
        "DEN-Danish Superliga": {"FBref": "Danish Superliga", "season_start": "Jul", "season_end": "Jun"},
        "NOR-Eliteserien": {"FBref": "Eliteserien", "season_start": "Apr", "season_end": "Nov"},
        "POR-Primeira Liga": {"FBref": "Primeira Liga", "season_start": "Aug", "season_end": "Jun"},
        "SUI-Swiss Super League": {"FBref": "Swiss Super League", "season_start": "Jul", "season_end": "Jun"},
        "SRB-Serbian SuperLiga": {"FBref": "Serbian SuperLiga", "season_start": "Jul", "season_end": "Jun"},
        "SWE-Allsvenskan": {"FBref": "Allsvenskan", "season_start": "Apr", "season_end": "Nov"},
        "ARG-Primera División Argentina": {"FBref": "Liga Profesional de Fútbol Argentina", "season_start": "Jun", "season_end": "Nov"},
        "CZE-Czech First League": {"FBref": "Czech First League", "season_start": "Jul", "season_end": "Jun"},
        "ECU-Liga Profesional Ecuador": {"FBref": "Liga Profesional Ecuador", "season_start": "Feb", "season_end": "Nov"},
        "PER-Liga 1 Peru": {"FBref": "Liga 1 de Fútbol Profesional", "season_start": "Feb", "season_end": "Nov"},
        "VEN-Venezuelan Primera División": {"FBref": "Venezuelan Primera División", "season_start": "Feb", "season_end": "Nov"},
        "POL-Ekstraklasa": {"FBref": "Ekstraklasa", "season_start": "Jul", "season_end": "Jun"},
        "URU-Uruguayan Primera División": {"FBref": "Liga AUF Uruguaya", "season_start": "May", "season_end": "Nov"},
        "JAP-J1 League": {"FBref": "J1 League", "season_start": "Feb", "season_end": "Nov"},
        "KOR-K League 1": {"FBref": "K League 1", "season_start": "Feb", "season_end": "Oct"},
        "MEX-Liga MX": {"FBref": "Liga MX", "season_start": "Jul", "season_end": "Jun"},
        "CHN-Chinese Super League": {"FBref": "Chinese Football Association Super League", "season_start": "Feb", "season_end": "Dec"},
        "IND-Indian Super League": {"FBref": "Indian Super League", "season_start": "Oct", "season_end": "May"},
        "CAN-Canadian Premier League": {"FBref": "Canadian Premier League", "season_start": "Apr", "season_end": "Dec"},
        "AUS-A-League": {"FBref": "A-League Men", "season_start": "Oct", "season_end": "May"},
        "BUL-First Professional Football League": {"FBref": "First Professional Football League", "season_start": "Jul", "season_end": "Apr"},
        "GRE-Super League Greece": {"FBref": "Super League Greece", "season_start": "Aug", "season_end": "Mar"},
        "HUN-Nemzeti Bajnokság I": {"FBref": "Nemzeti Bajnokság I", "season_start": "Jul", "season_end": "May"},
        "KSA-Saudi Pro League": {"FBref": "Saudi Pro League", "season_start": "Aug", "season_end": "May"},
        "RUS-Russian Premier League": {"FBref": "Russian Premier League", "season_start": "Jul", "season_end": "May"},
        "UKR-Ukrainian Premier League": {"FBref": "Ukrainian Premier League", "season_start": "Jul", "season_end": "May"},
        "IRN-Persian Gulf Pro League": {"FBref": "Persian Gulf Pro League", "season_start": "Jul", "season_end": "May"},
        "RSA-South African Premiership": {"FBref": "South African Premiership", "season_start": "Aug", "season_end": "May"},
        "USA-Major League Soccer": {"FBref": "Major League Soccer", "season_start": "Feb", "season_end": "Dec"},
    }
    dict_path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("FBref league_dict.json yazildi: %s", dict_path)


def _extract_game_hex(url) -> str | None:
    """match_report URL'sinden mac hex ID'sini cikar (ornek: 3f9c38cd)."""
    import re

    if url is None or pd.isna(url):
        return None
    m = re.search(r"/matches/([0-9a-f]+)/", str(url))
    return m.group(1) if m else None


def _scrape_league_season(league_key: str, season: str) -> pd.DataFrame:
    """Tek lig+sezon icin gercek mac istatistikleri cek."""
    import soccerdata as sd

    fbref = sd.FBref(league_key, season)

    sched = fbref.read_schedule()
    if sched is None or len(sched) == 0:
        return pd.DataFrame()

    stats_frames = {}
    for st, name in [("shooting", "shot"), ("misc", "misc")]:
        try:
            df = fbref.read_team_match_stats(stat_type=st)
            if df is not None and len(df) > 0:
                df = df.copy()
                df["_hex"] = df["match_report"].apply(_extract_game_hex)
                stats_frames[name] = df
        except Exception as e:
            logger.warning("%s %s %s basarisiz: %s", league_key, season, st, e)

    rows = []
    for game_key, g in sched.iterrows():
        if isinstance(game_key, tuple):
            game_str = game_key[-1]
        else:
            game_str = game_key
        if game_str is None or str(game_str) == "nan":
            continue

        date = g.get("date")
        home_team = g.get("home_team")
        away_team = g.get("away_team")
        if pd.isna(date) or home_team is None or away_team is None:
            continue

        # Skor parse: "0–3" -> (0, 3)
        score = g.get("score")
        if pd.isna(score) or not isinstance(score, str) or "–" not in score:
            continue
        try:
            hg, ag = int(score.split("–")[0]), int(score.split("–")[1])
        except (ValueError, TypeError):
            continue

        row = {
            "date": pd.to_datetime(date),
            "home_team": str(home_team).strip(),
            "away_team": str(away_team).strip(),
            "home_goals": hg,
            "away_goals": ag,
            "result": "H" if hg > ag else ("A" if ag > hg else "D"),
            "referee": g.get("referee") if pd.notna(g.get("referee")) else None,
        }

        hex_id = _extract_game_hex(g.get("match_report"))
        if hex_id is not None:
            for prefix, df in stats_frames.items():
                sub = df[df["_hex"] == hex_id]
                if len(sub) == 0:
                    continue
                home_row = sub[sub["venue"] == "Home"]
                away_row = sub[sub["venue"] == "Away"]
                if len(home_row) == 0 or len(away_row) == 0:
                    continue
                h = home_row.iloc[0]
                a = away_row.iloc[0]

                def _cell(series: pd.Series, level1: str):
                    try:
                        if isinstance(series.index, pd.MultiIndex):
                            for key, val in series.items():
                                if isinstance(key, tuple) and key[-1] == level1:
                                    if val is None or (isinstance(val, float) and np.isnan(val)):
                                        return None
                                    return val
                            return None
                        v = series.get(level1)
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            return None
                        return v
                    except Exception:
                        return None

                if prefix == "shot":
                    row["home_shots"] = _cell(h, "Sh")
                    row["away_shots"] = _cell(a, "Sh")
                    row["home_sot"] = _cell(h, "SoT")
                    row["away_sot"] = _cell(a, "SoT")
                else:
                    row["home_yellow"] = _cell(h, "CrdY")
                    row["away_yellow"] = _cell(a, "CrdY")
                    row["home_red"] = _cell(h, "CrdR")
                    row["away_red"] = _cell(a, "CrdR")
                    row["home_fouls"] = _cell(h, "Fls")
                    row["away_fouls"] = _cell(a, "Fls")

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_fbref_matches(leagues: dict[str, list[str]] | None = None,
                        cache_path: str = "data/gold/fbref_matches.parquet",
                        resume: bool = True,
                        scrape_new: bool = False) -> pd.DataFrame:
    """FBref'ten GERCEK istatistikli mac verisi ceker.

    leagues: lig -> sezon listesi. None ise LEAGUE_SEASONS kullanilir.
    resume: True ise cache'teki mevcut sonuclar tekrar cekilmez.
    scrape_new: False ise SADECE cache okur, hic scraping YAPMAZ.
        (canonical build icin: cache'te ne varsa o kullanilir, ag yok)
    """
    if leagues is None:
        leagues = LEAGUE_SEASONS

    _ensure_league_dict()

    # Cache yukle
    cache_file = Path(cache_path)
    cached: dict[tuple[str, str], pd.DataFrame] = {}
    if resume and cache_file.exists():
        try:
            prev = pd.read_parquet(cache_file)
            for (lg, season_col), grp in prev.groupby(["league", "season"]):
                cached[(lg, season_col)] = grp
            logger.info("FBref cache yuklendi: %d mac", len(prev))
        except Exception as e:
            logger.warning("FBref cache okunamadi: %s", e)

    all_rows: list[pd.DataFrame] = []
    total_planned = sum(len(s) for s in leagues.values())

    for i, (league_key, seasons) in enumerate(leagues.items()):
        country_code = league_key.split("-")[0]
        country = COUNTRY_MAP.get(country_code, country_code)
        league_name = league_key.split("-", 1)[1] if "-" in league_key else league_key
        league_code = f"FB_{country_code}_{''.join(c for c in league_name.replace(' ', '') if c.isalnum())[:12]}"

        for season in seasons:
            cache_key = (league_code, season)
            if resume and cache_key in cached:
                all_rows.append(cached[cache_key])
                logger.info("[%d/%d] %s %s: cache (%d mac)", i + 1, len(leagues), league_key, season, len(cached[cache_key]))
                continue

            if not scrape_new:
                # Scrape yok: cache'te olmayan lig-sezonlari atla (ag yok)
                logger.info("[%d/%d] %s %s: cache'te yok, scrape_new=False -> atlandi",
                            i + 1, len(leagues), league_key, season)
                continue

            logger.info("[%d/%d] %s %s cekiliyor... (%d sezon kaldi)",
                        i + 1, len(leagues), league_key, season, total_planned)
            try:
                df = _scrape_league_season(league_key, season)
            except Exception as e:
                logger.warning("%s %s basarisiz: %s", league_key, season, e)
                continue

            if df.empty:
                logger.warning("%s %s: veri yok", league_key, season)
                continue

            df["league"] = league_code
            df["league_name"] = f"{country} {league_name}"
            df["season"] = season

            # Eksik sutunlar
            for col in ["ht_home_goals", "ht_away_goals", "home_corners", "away_corners",
                        "avg_home_odds", "avg_draw_odds", "avg_away_odds",
                        "b365_home_odds", "b365_draw_odds", "b365_away_odds",
                        "avg_over25_odds", "avg_under25_odds"]:
                if col not in df.columns:
                    df[col] = None

            all_rows.append(df)
            n_stats = df["home_shots"].notna().sum() if "home_shots" in df.columns else 0
            logger.info("%s %s: %d mac (%d istatistikli)",
                        league_key, season, len(df), n_stats)

            # Her lig+sezon sonunda cache'e kaydet (resume destegi)
            if cache_file.exists() or True:
                combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
                if not combined.empty:
                    combined.to_parquet(cache_file, index=False)

    if not all_rows:
        logger.warning("FBref: hic veri cekilemedi")
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    result = result.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    result = result.sort_values("date").reset_index(drop=True)

    # Team ID uretimi (lig bazinda, tum sezonlar boyunca tutarli!)
    # Onceki hata: sezon bazinda factorize -> ayni takim sezonlar arasi farkli ID
    # aliyordu, Elo/form zinciri kopuyordu.
    next_id = 5000000
    team_map: dict[tuple[str, str], int] = {}
    for side in ["home", "away"]:
        for lg, name in zip(result["league"], result[side + "_team"]):
            key = (lg, str(name).strip().lower())
            if key not in team_map:
                team_map[key] = next_id
                next_id += 1
    result["home_team_id"] = [team_map[(lg, str(ht).strip().lower())]
                              for lg, ht in zip(result["league"], result["home_team"])]
    result["away_team_id"] = [team_map[(lg, str(at).strip().lower())]
                              for lg, at in zip(result["league"], result["away_team"])]

    result.to_parquet(cache_file, index=False)

    n_stats = result["home_shots"].notna().sum()
    logger.info("FBref toplam: %d mac, %d lig, %d mac istatistikli",
                len(result), result["league"].nunique(), n_stats)
    return result