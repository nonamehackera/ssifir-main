"""Point-in-time feature engine (ROADMAP bolum 24-26, 45-47).

CRITICAL PRINCIPLE: every feature for match i is computed ONLY from matches
j with kickoff_j < kickoff_i. No future result, no future form, no closing
odds of later matches. This is what makes out-of-sample evaluation honest.

Approach: process matches in strict chronological order. Maintain per-team
history (only past matches) plus per-league running baselines. For match i,
derive features from history, then append match i to history (so it is
visible to later matches only).

Features produced (home/away symmetric unless noted):
  - elo_before (point-in-time Elo, ROADMAP bolum 7)
  - form windows last_3/5/10: goals_for, goals_against, points, shots, sot, corners
  - home/away split form (ROADMAP bolum 6)
  - recency-weighted (exp decay, half_life 60d) means of the above
  - rest days since previous match (ROADMAP bolum 14)
  - H2H last 5 (ROADMAP bolum 16, max 5)
  - league baselines (ROADMAP bolum 15): avg goals, home/away goals, draw,
    btts, over25, corners, cards
  - opponent strength (avg elo of recent opponents) -- optional signal
  - market-implied probs (ONLY for with-odds models; odds are pre-match info)

Targets (ROADMAP bolum 35, 36):
  result_onehot (H/D/A), home_goals, away_goals, btts, over25, total_goals.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from configs import settings

logger = logging.getLogger(__name__)


def _safe_id(v):
    """Team id'yi int'e cevir; string (TM_xxxx) veya gecersizse hash ile stabil int uret."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v).strip()
    if s.isdigit():
        return int(s)
    # string id (ornek TM_3806) -> stabil pozitif int (hash)
    h = int("".join(str(ord(c)) for c in s)) % (2**31)
    return h


def _safe_int(v, default=0):
    """Sayiyi int'e cevir; None/NaN/string ise default."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


HALF_LIFE_DAYS = 60.0
ELO_INIT = 1500.0
ELO_K = 32.0
HOME_ADVANTAGE = 50.0  # rating points


@dataclass
class TeamState:
    elo: float = ELO_INIT
    attack_elo: float = ELO_INIT  # ROADMAP bolum 7: saldiri gucu
    defence_elo: float = ELO_INIT  # ROADMAP bolum 7: savunma gucu
    history: list[dict] = field(default_factory=list)  # past matches for this team
    season_goals_for: float = 0.0
    season_goals_against: float = 0.0
    season_n: int = 0
    last_season: object = None  # (season_id, ) tracking


@dataclass
class LeagueState:
    n: int = 0
    home_goals: float = 0.0
    away_goals: float = 0.0
    total_goals: float = 0.0
    draws: int = 0
    btts: int = 0
    over25: int = 0
    home_corners: float = 0.0
    away_corners: float = 0.0
    cards: int = 0  # yellow+red total

    def rates(self):
        n = max(self.n, 1)
        return {
            "lg_avg_home_goal": self.home_goals / n,
            "lg_avg_away_goal": self.away_goals / n,
            "lg_avg_goals": self.total_goals / n,
            "lg_draw_rate": self.draws / n,
            "lg_btts_rate": self.btts / n,
            "lg_over25_rate": self.over25 / n,
            "lg_corner_avg": (self.home_corners + self.away_corners) / n,
            "lg_card_avg": self.cards / n,
        }


def _weighted_mean(values, weights):
    vals = np.array([v for v, w in zip(values, weights) if v is not None], dtype=float)
    ws = np.array([w for v, w in zip(values, weights) if v is not None], dtype=float)
    if len(vals) == 0:
        return None
    return float(np.sum(vals * ws) / np.sum(ws))


def _simple_mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _last_season(team_hist, current_kickoff, season):
    """Return past matches for this team in the SAME season before current."""
    out = []
    for rec in reversed(team_hist):
        if rec["kickoff"] >= current_kickoff:
            continue
        if rec.get("season") != season:
            continue
        out.append(rec)
    return out


def _last(team_hist, current_kickoff, n, is_home=None):
    """Return up to n most-recent past match dicts for this team before current."""
    out = []
    for rec in reversed(team_hist):
        if rec["kickoff"] >= current_kickoff:
            continue
        if is_home is not None and rec["is_home"] != is_home:
            continue
        out.append(rec)
        if len(out) >= n:
            break
    return out  # oldest->newest within the slice


def _form_features(hist_slice, current_kickoff):
    if not hist_slice:
        return {}
    gf = [r["goals_for"] for r in hist_slice]
    ga = [r["goals_against"] for r in hist_slice]
    pts = [r["points"] for r in hist_slice]
    sh = [r["shots"] for r in hist_slice]
    sot = [r["sot"] for r in hist_slice]
    cor = [r["corners"] for r in hist_slice]
    cd = [r["cards"] for r in hist_slice if r.get("cards") is not None]
    days = [(current_kickoff - r["kickoff"]).total_seconds() / 86400.0 for r in hist_slice]
    w = [math.exp(-d / HALF_LIFE_DAYS) for d in days]
    return {
        "goals_for": _simple_mean(gf),
        "goals_against": _simple_mean(ga),
        "points": _simple_mean(pts),
        "shots": _simple_mean(sh),
        "sot": _simple_mean(sot),
        "corners": _simple_mean(cor),
        "cards": _simple_mean(cd) if cd else None,
        "w_goals_for": _weighted_mean(gf, w),
        "w_goals_against": _weighted_mean(ga, w),
        "w_shots": _weighted_mean(sh, w),
        "w_sot": _weighted_mean(sot, w),
        "w_corners": _weighted_mean(cor, w),
        "w_cards": _weighted_mean(cd, w) if cd else None,
    }


def build_features(
    matches: pd.DataFrame,
    out_path: str | None = None,
) -> pd.DataFrame:
    """Compute leakage-free features for every match. Returns feature frame.

    Her lig bagimsiz islenir (takim ID'leri ve lig state'i lige ozeldir):
    bellek tasarrufu icin parca parca, sonra birlestirilir.
    """
    df = matches.sort_values("date").reset_index(drop=True)

    frames = []
    for lg in df["league"].unique():
        grp = df[df["league"] == lg].reset_index(drop=True)
        sub = _build_league_features(grp)
        frames.append(sub)

    feats = pd.concat(frames, ignore_index=True)

    # forward-fill early NaNs in form features with league baseline where useful
    feats = _patch_early_nans(feats)

    # Eksik istatistikleri tahmin et (HF/TSDB maclari icin)
    # NOT: buyuk veride cok yavas + FD disi veriye uydurma deger yazar.
    # Hizli pipeline icin SKIP_STATS_PREDICTOR env ile atlanabilir.
    if not os.environ.get("SKIP_STATS_PREDICTOR"):
        try:
            from feature_engine.stats_predictor import build_enriched_matches
            feats = build_enriched_matches(feats)
        except Exception as e:
            logger.warning("Stats predictor basarisiz: %s", e)
    else:
        logger.info("Stats predictor SKIP edildi (SKIP_STATS_PREDICTOR=1)")

    out_path = out_path or (Path(settings.DATA_DIR) / "gold" / "features.parquet")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    logger.info("Features built: %d rows, %d cols -> %s", len(feats), feats.shape[1], out_path)
    return feats


def _build_league_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tek lig icin feature hesapla (kronolojik sirayla)."""
    teams: dict[int, TeamState] = {}
    leagues: dict[str, LeagueState] = {}
    h2h: dict[tuple[int, int], list[dict]] = {}  # unordered pair -> past meetings

    rows = []
    for m in df.to_dict("records"):
        hid = _safe_id(m["home_team_id"])
        aid = _safe_id(m["away_team_id"])
        kick = m["date"]
        lg = m["league"]
        season = m["season"]

        hs = teams.setdefault(hid, TeamState())
        as_ = teams.setdefault(aid, TeamState())
        lstate = leagues.setdefault(lg, LeagueState())

        # ---- point-in-time league baselines (from past matches only) ----
        lg_base = lstate.rates()

        # ---- form features (overall + home/away split) ----
        # ROADMAP bolum 25: coklu window (3/5/8/10/20 + season_to_date)
        h_form = _form_features(_last(hs.history, kick, 10), kick)
        a_form = _form_features(_last(as_.history, kick, 10), kick)
        h_form_h = _form_features(_last(hs.history, kick, 10, is_home=True), kick)
        a_form_a = _form_features(_last(as_.history, kick, 10, is_home=False), kick)
        h_form3 = _form_features(_last(hs.history, kick, 3), kick)
        a_form3 = _form_features(_last(as_.history, kick, 3), kick)
        h_form8 = _form_features(_last(hs.history, kick, 8), kick)
        a_form8 = _form_features(_last(as_.history, kick, 8), kick)
        h_form20 = _form_features(_last(hs.history, kick, 20), kick)
        a_form20 = _form_features(_last(as_.history, kick, 20), kick)
        # season_to_date (ayni sezon, kickoff'tan once)
        h_std = _form_features(_last_season(hs.history, kick, m["season"]), kick)
        a_std = _form_features(_last_season(as_.history, kick, m["season"]), kick)

        # ---- BTTS form features (son N macin BTTS olma durumu) ----
        h_btts_last5 = sum(1 for r in _last(hs.history, kick, 5) if r["goals_for"] > 0 and r["goals_against"] > 0)
        a_btts_last5 = sum(1 for r in _last(as_.history, kick, 5) if r["goals_for"] > 0 and r["goals_against"] > 0)

        # ---- rest days ----
        h_rest = _rest_days(hs.history, kick)
        a_rest = _rest_days(as_.history, kick)

        # ---- H2H last 5 ----
        key = (min(hid, aid), max(hid, aid))
        h2h_list = _last_h2h(h2h.get(key, []), kick, 5)
        h2h_feat = _h2h_features(h2h_list)

        # ---- elo before ----
        h_elo_b = hs.elo
        a_elo_b = as_.elo
        elo_diff = h_elo_b - a_elo_b

        # ---- opponent strength (avg elo of recent opponents) ----
        h_opp = _opp_strength(_last(hs.history, kick, 10))
        a_opp = _opp_strength(_last(as_.history, kick, 10))

        # ---- assemble feature row ----
        # momentum: son 3 mac gol farki trendi
        h_form3_raw = _last(hs.history, kick, 3)
        a_form3_raw = _last(as_.history, kick, 3)
        h_gf3 = [r["goals_for"] for r in h_form3_raw] if h_form3_raw else []
        h_ga3 = [r["goals_against"] for r in h_form3_raw] if h_form3_raw else []
        a_gf3 = [r["goals_for"] for r in a_form3_raw] if a_form3_raw else []
        a_ga3 = [r["goals_against"] for r in a_form3_raw] if a_form3_raw else []
        # BTTS form: son 5 macin her ikisinin de gol attigi sayisi
        h_btts_last5 = sum(1 for r in _last(hs.history, kick, 5) if r["goals_for"] > 0 and r["goals_against"] > 0)
        a_btts_last5 = sum(1 for r in _last(as_.history, kick, 5) if r["goals_for"] > 0 and r["goals_against"] > 0)
        # BTTS orani (son 5 mac)
        h_btts_last5_val = h_btts_last5 / max(len(_last(hs.history, kick, 5)), 1)
        a_btts_last5_val = a_btts_last5 / max(len(_last(as_.history, kick, 5)), 1)
        # momentum = son 3 gol farki ortalamasi - ilk 2 gol farki ortalamasi
        if len(h_gf3) >= 3:
            recent_diff = (h_gf3[2] - h_ga3[2])
            older_diff = np.mean([(h_gf3[0]-h_ga3[0]), (h_gf3[1]-h_ga3[1])])
            h_momentum = recent_diff - older_diff
        else:
            h_momentum = None
        if len(a_gf3) >= 3:
            recent_diff = (a_gf3[2] - a_ga3[2])
            older_diff = np.mean([(a_gf3[0]-a_ga3[0]), (a_gf3[1]-a_ga3[1])])
            a_momentum = recent_diff - older_diff
        else:
            a_momentum = None

        # win streak: son kac mac kazanildi/kaybedildi
        def _streak(hist, n=5):
            s = _last(hist, kick, n)
            if not s:
                return None, None
            wins = sum(1 for r in s if r["points"] == 3)
            draws = sum(1 for r in s if r["points"] == 1)
            return wins, draws

        h_wins, h_draws = _streak(hs.history)
        a_wins, a_draws = _streak(as_.history)

        # form consistency: gol standart sapmasi (ne kadar istikrarli)
        def _form_std(hist, n=5):
            sl = _last(hist, kick, n)
            if len(sl) < 3:
                return None
            gf = [r["goals_for"] for r in sl]
            return float(np.std(gf)) if gf else None

        h_form_std = _form_std(hs.history)
        a_form_std = _form_std(as_.history)

        # goal diff trend (son 5 mac ortalama gol farki)
        def _goal_diff_avg(hist, n=5):
            sl = _last(hist, kick, n)
            if not sl:
                return None
            diffs = [r["goals_for"] - r["goals_against"] for r in sl]
            return float(np.mean(diffs))

        h_gdiff5 = _goal_diff_avg(hs.history, 5)
        a_gdiff5 = _goal_diff_avg(as_.history, 5)

        # home form trend: evde son 5 vs deplasmanda son 5
        h_hgf5_feat = _form_features(_last(hs.history, kick, 5, is_home=True), kick)
        a_agf5_feat = _form_features(_last(as_.history, kick, 5, is_home=False), kick)

        rec = {
            "match_id": _safe_id(m["match_id"]),
            "league": lg,
            "season": m["season"],
            "date": kick,
            "home_team_id": hid,
            "away_team_id": aid,
            # elo
            "home_elo": h_elo_b,
            "away_elo": a_elo_b,
            "elo_diff": elo_diff,
            # home form
            "home_gf_5": h_form.get("goals_for"),
            "home_ga_5": h_form.get("goals_against"),
            "home_pts_5": h_form.get("points"),
            "home_shots_5": h_form.get("shots"),
            "home_sot_5": h_form.get("sot"),
            "home_corners_5": h_form.get("corners"),
            "home_cards_5": h_form.get("cards"),
            "home_w_gf": h_form.get("w_goals_for"),
            "home_w_ga": h_form.get("w_goals_against"),
            "home_w_shots": h_form.get("w_shots"),
            "home_w_sot": h_form.get("w_sot"),
            "home_w_corners": h_form.get("w_corners"),
            "home_w_cards": h_form.get("w_cards"),
            # BTTS form features
            "home_btts_last5": h_btts_last5,
            "home_btts_last5_val": h_btts_last5_val,
            # momentum & streaks
            "home_momentum": h_momentum,
            "home_wins_last5": h_wins,
            "home_draws_last5": h_draws,
            "home_form_std": h_form_std,
            "home_gdiff5": h_gdiff5,
            "home_gf_5_real": h_hgf5_feat.get("goals_for"),
            "home_ga_5_real": h_hgf5_feat.get("goals_against"),
            "away_gf_5_real": a_agf5_feat.get("goals_for"),
            "away_ga_5_real": a_agf5_feat.get("goals_against"),
            "home_hgf_5": h_form_h.get("goals_for"),
            "home_hga_5": h_form_h.get("goals_against"),
            "home_hpts_5": h_form_h.get("points"),
            "home_gf_3": h_form3.get("goals_for"),
            "home_ga_3": h_form3.get("goals_against"),
            "home_pts_3": h_form3.get("points"),
            "home_gf_8": h_form8.get("goals_for"),
            "home_ga_8": h_form8.get("goals_against"),
            "home_gf_20": h_form20.get("goals_for"),
            "home_ga_20": h_form20.get("goals_against"),
            "home_gf_std": h_std.get("goals_for"),
            "home_ga_std": h_std.get("goals_against"),
            "home_pts_std": h_std.get("points"),
            # away form
            "away_gf_5": a_form.get("goals_for"),
            "away_ga_5": a_form.get("goals_against"),
            "away_pts_5": a_form.get("points"),
            "away_shots_5": a_form.get("shots"),
            "away_sot_5": a_form.get("sot"),
            "away_corners_5": a_form.get("corners"),
            "away_cards_5": a_form.get("cards"),
            "away_w_gf": a_form.get("w_goals_for"),
            "away_w_ga": a_form.get("w_goals_against"),
            "away_w_shots": a_form.get("w_shots"),
            "away_w_sot": a_form.get("w_sot"),
            "away_w_corners": a_form.get("w_corners"),
            "away_w_cards": a_form.get("w_cards"),
            # BTTS form features
            "away_btts_last5": a_btts_last5,
            "away_btts_last5_val": a_btts_last5_val,
            # momentum & streaks
            "away_momentum": a_momentum,
            "away_wins_last5": a_wins,
            "away_draws_last5": a_draws,
            "away_form_std": a_form_std,
            "away_gdiff5": a_gdiff5,
            "away_agf_5": a_form_a.get("goals_for"),
            "away_aga_5": a_form_a.get("goals_against"),
            "away_apts_5": a_form_a.get("points"),
            "away_gf_3": a_form3.get("goals_for"),
            "away_ga_3": a_form3.get("goals_against"),
            "away_pts_3": a_form3.get("points"),
            "away_gf_8": a_form8.get("goals_for"),
            "away_ga_8": a_form8.get("goals_against"),
            "away_gf_20": a_form20.get("goals_for"),
            "away_ga_20": a_form20.get("goals_against"),
            "away_gf_std": a_std.get("goals_for"),
            "away_ga_std": a_std.get("goals_against"),
            "away_pts_std": a_std.get("points"),
            # attack/defence elo (ROADMAP bolum 7)
            "home_attack_elo": hs.attack_elo,
            "home_defence_elo": hs.defence_elo,
            "away_attack_elo": as_.attack_elo,
            "away_defence_elo": as_.defence_elo,
            "attack_elo_diff": hs.attack_elo - as_.attack_elo,
            "defence_elo_diff": hs.defence_elo - as_.defence_elo,
            # rest
            "home_rest_days": h_rest,
            "away_rest_days": a_rest,
            # h2h
            "h2h_home_win": h2h_feat.get("home_win"),
            "h2h_draw": h2h_feat.get("draw"),
            "h2h_away_win": h2h_feat.get("away_win"),
            "h2h_goals_avg": h2h_feat.get("goals_avg"),
            "h2h_btts": h2h_feat.get("btts"),
            # opponent strength
            "home_opp_elo": h_opp,
            "away_opp_elo": a_opp,
            # league baselines
            "lg_avg_goals": lg_base["lg_avg_goals"],
            "lg_home_goal_avg": lg_base["lg_avg_home_goal"],
            "lg_away_goal_avg": lg_base["lg_avg_away_goal"],
            # league baselines
            "lg_draw_rate": lg_base["lg_draw_rate"],
            "lg_btts_rate": lg_base["lg_btts_rate"],
            "lg_over25_rate": lg_base["lg_over25_rate"],
            "lg_corner_avg": lg_base["lg_corner_avg"],
            "lg_card_avg": lg_base["lg_card_avg"],
            # market-implied (pre-match info -> only with_odds models)
            "mkt_home_prob": _implied(m["avg_home_odds"]),
            "mkt_draw_prob": _implied(m["avg_draw_odds"]),
            "mkt_away_prob": _implied(m["avg_away_odds"]),
            "mkt_over25_prob": _implied(m["avg_under25_odds"]),  # under25 odds->over25
            # RAW ODDS (betting katmani icin SART - implied prob yetmez, oran lazim)
            "avg_home_odds": _safe_odds(m["avg_home_odds"]),
            "avg_draw_odds": _safe_odds(m["avg_draw_odds"]),
            "avg_away_odds": _safe_odds(m["avg_away_odds"]),
            "avg_over25_odds": _safe_odds(m["avg_over25_odds"]),
            # CLOSING ODDS (daha dogru, mac oncesi son oran)
            "mkt_close_home_prob": _implied(m.get("avg_close_home_odds")),
            "mkt_close_draw_prob": _implied(m.get("avg_close_draw_odds")),
            "mkt_close_away_prob": _implied(m.get("avg_close_away_odds")),
            "mkt_close_over25_prob": _implied(m.get("avg_close_under25_odds")),
            "avg_close_home_odds": _safe_odds(m.get("avg_close_home_odds")),
            "avg_close_draw_odds": _safe_odds(m.get("avg_close_draw_odds")),
            "avg_close_away_odds": _safe_odds(m.get("avg_close_away_odds")),
            "avg_close_over25_odds": _safe_odds(m.get("avg_close_over25_odds")),
            # ---- targets ----
            "home_goals": _safe_int(m["home_goals"]),
            "away_goals": _safe_int(m["away_goals"]),
            "ht_home_goals": (_safe_int(m["ht_home_goals"]) if pd.notna(m["ht_home_goals"]) else None),
            "ht_away_goals": (_safe_int(m["ht_away_goals"]) if pd.notna(m["ht_away_goals"]) else None),
            "result": m["result"],
            "result_H": 1 if m["result"] == "H" else 0,
            "result_D": 1 if m["result"] == "D" else 0,
            "result_A": 1 if m["result"] == "A" else 0,
            "btts": 1 if (_safe_int(m["home_goals"]) > 0 and _safe_int(m["away_goals"]) > 0) else 0,
            "over25": 1 if (_safe_int(m["home_goals"]) + _safe_int(m["away_goals"]) >= 3) else 0,
            "total_goals": _safe_int(m["home_goals"]) + _safe_int(m["away_goals"]),
            "home_shots": m["home_shots"],
            "away_shots": m["away_shots"],
            "home_sot": m["home_sot"],
            "away_sot": m["away_sot"],
            "home_corners": m["home_corners"],
            "away_corners": m["away_corners"],
            "home_xg": m.get("home_xg"),  # GERCEK xG (HF xg_training'den)
            "away_xg": m.get("away_xg"),
            "home_yellow": m["home_yellow"],
            "away_yellow": m["away_yellow"],
            "home_red": m["home_red"],
            "away_red": m["away_red"],
            "referee": m["referee"],
        }
        rows.append(rec)

        # ---- update state for future matches (AFTER feature extraction) ----
        hg = _safe_int(m["home_goals"])
        ag = _safe_int(m["away_goals"])
        pts_h = 3 if hg > ag else (1 if hg == ag else 0)
        pts_a = 3 if ag > hg else (1 if hg == ag else 0)
        # elo update uses pre-match elo
        eh = _expected(h_elo_b, a_elo_b, True)
        ea = _expected(a_elo_b, h_elo_b, False)
        hs.elo = h_elo_b + ELO_K * (pts_h / 3.0 - eh)
        as_.elo = a_elo_b + ELO_K * (pts_a / 3.0 - ea)
        # attack/defence elo: attack = kendi attigi gol beklentisi vs rakip
        # defence = rakibin attigi gol beklentisi (dusukse iyi)
        # basit: attack_elo gol farki, defence_elo yenilen gol farki ile guncellenir
        h_att_exp = _expected(hs.attack_elo, as_.defence_elo, True)
        a_att_exp = _expected(as_.attack_elo, hs.defence_elo, False)
        gf_h = hg / max(hs.season_n + 1, 1)  # normalize edilmis gol
        # attack: attigi gol sayisi ile guncelle
        hs.attack_elo += ELO_K * (float(hg) / 3.0 - h_att_exp)
        as_.attack_elo += ELO_K * (float(ag) / 3.0 - a_att_exp)
        # defence: yenilen gol (az olmasi iyi) -> negatif yonde
        hs.defence_elo += ELO_K * (float(ag) / 3.0 - (1 - h_att_exp))
        as_.defence_elo += ELO_K * (float(hg) / 3.0 - (1 - a_att_exp))
        # season_to_date tracking
        if hs.last_season != m["season"]:
            hs.season_goals_for = 0.0
            hs.season_goals_against = 0.0
            hs.season_n = 0
            hs.last_season = m["season"]
        hs.season_goals_for += hg
        hs.season_goals_against += ag
        hs.season_n += 1
        if as_.last_season != m["season"]:
            as_.season_goals_for = 0.0
            as_.season_goals_against = 0.0
            as_.season_n = 0
            as_.last_season = m["season"]
        as_.season_goals_for += ag
        as_.season_goals_against += hg
        as_.season_n += 1
        # store match in each team's history
        hrec = {
            "kickoff": kick, "is_home": True, "season": m["season"],
            "goals_for": hg, "goals_against": ag, "points": pts_h,
            "shots": m["home_shots"], "sot": m["home_sot"], "corners": m["home_corners"],
            "cards": (m["home_yellow"] or 0) + (m["home_red"] or 0),
            "opp_elo": a_elo_b, "opp_id": aid,
            "ht_home_goals": m.get("ht_home_goals"),
            "ht_away_goals": m.get("ht_away_goals"),
        }
        arec = {
            "kickoff": kick, "is_home": False, "season": m["season"],
            "goals_for": ag, "goals_against": hg, "points": pts_a,
            "shots": m["away_shots"], "sot": m["away_sot"], "corners": m["away_corners"],
            "cards": (m["away_yellow"] or 0) + (m["away_red"] or 0),
            "opp_elo": h_elo_b, "opp_id": hid,
            "ht_home_goals": m.get("ht_home_goals"),
            "ht_away_goals": m.get("ht_away_goals"),
        }
        hs.history.append(hrec)
        as_.history.append(arec)
        # h2h
        h2h.setdefault(key, []).append({
            "kickoff": kick, "hg": hg, "ag": ag,
            "home_id": hid, "away_id": aid,
        })
        # league baseline update
        lstate.n += 1
        lstate.home_goals += hg
        lstate.away_goals += ag
        lstate.total_goals += hg + ag
        lstate.draws += 1 if hg == ag else 0
        lstate.btts += 1 if (hg > 0 and ag > 0) else 0
        lstate.over25 += 1 if (hg + ag >= 3) else 0
        lstate.home_corners += (m["home_corners"] or 0)
        lstate.away_corners += (m["away_corners"] or 0)
        lstate.cards += (
            (m["home_yellow"] or 0) + (m["away_yellow"] or 0)
            + (m["home_red"] or 0) + (m["away_red"] or 0)
        )

    feats = pd.DataFrame(rows)
    # None-only kolonlar object dtype olur (per-league frame'lerde);
    # concat sonrasi numeric kalsin diye object kolonlari sayisala cevir.
    # (String kolonlar - league, season, result - dokunulmaz.)
    for c in feats.columns:
        if feats[c].dtype == object:
            num = pd.to_numeric(feats[c], errors="coerce")
            if num.notna().any() or not feats[c].notna().any():
                feats[c] = num
    feats = feats.sort_values("date").reset_index(drop=True)
    return feats


def _expected(rating_a, rating_b, a_is_home):
    adv = HOME_ADVANTAGE if a_is_home else -HOME_ADVANTAGE
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b + adv) / 400.0))


def _rest_days(history, current_kickoff):
    last = None
    for rec in reversed(history):
        if rec["kickoff"] < current_kickoff:
            last = rec["kickoff"]
            break
    if last is None:
        return None
    return (current_kickoff - last).total_seconds() / 86400.0


def _last_h2h(h2h_list, current_kickoff, n):
    out = []
    for rec in reversed(h2h_list):
        if rec["kickoff"] >= current_kickoff:
            continue
        out.append(rec)
        if len(out) >= n:
            break
    return out


def _h2h_features(h2h_list):
    if not h2h_list:
        return {}
    n = len(h2h_list)
    home_win = sum(1 for r in h2h_list if r["hg"] > r["ag"]) / n
    draw = sum(1 for r in h2h_list if r["hg"] == r["ag"]) / n
    away_win = sum(1 for r in h2h_list if r["hg"] < r["ag"]) / n
    goals_avg = np.mean([r["hg"] + r["ag"] for r in h2h_list])
    btts = np.mean([1 if (r["hg"] > 0 and r["ag"] > 0) else 0 for r in h2h_list])
    return {"home_win": home_win, "draw": draw, "away_win": away_win,
            "goals_avg": goals_avg, "btts": btts}


def _opp_strength(hist_slice):
    elo = [r["opp_elo"] for r in hist_slice if r.get("opp_elo") is not None]
    return float(np.mean(elo)) if elo else None


def _safe_odds(odds):
    """Ham orani guvenli float'a cevir (betting katmani icin)."""
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return None
    try:
        v = float(odds)
        return v if v > 1.0 else None
    except (ValueError, TypeError):
        return None


def _implied(odds):
    if odds is None or odds <= 1.0 or (isinstance(odds, float) and np.isnan(odds)):
        return None
    return 1.0 / odds


def _patch_early_nans(feats: pd.DataFrame) -> pd.DataFrame:
    """Early matches have no history; fill form NaNs with league baseline /
    global priors so models still train, flagged by data_completeness."""
    # Fill elo already numeric. Fill form goals with league avg.
    for col, base in [
        ("home_gf_5", "lg_home_goal_avg"), ("away_gf_5", "lg_away_goal_avg"),
        ("home_ga_5", "lg_away_goal_avg"), ("away_ga_5", "lg_home_goal_avg"),
        ("home_w_gf", "lg_home_goal_avg"), ("away_w_gf", "lg_away_goal_avg"),
        ("home_w_ga", "lg_away_goal_avg"), ("away_w_ga", "lg_home_goal_avg"),
    ]:
        feats[col] = feats[col].fillna(feats[base])
    # points priors
    feats["home_pts_5"] = feats["home_pts_5"].fillna(1.4)
    feats["away_pts_5"] = feats["away_pts_5"].fillna(1.0)
    feats["home_pts_3"] = feats["home_pts_3"].fillna(feats["home_pts_5"])
    feats["away_pts_3"] = feats["away_pts_3"].fillna(feats["away_pts_5"])
    feats["home_ga_3"] = feats["home_ga_3"].fillna(feats["home_ga_5"])
    feats["away_ga_3"] = feats["away_ga_3"].fillna(feats["away_ga_5"])
    feats["home_gf_3"] = feats["home_gf_3"].fillna(feats["home_gf_5"])
    feats["away_gf_3"] = feats["away_gf_3"].fillna(feats["away_gf_5"])
    # home/away split
    feats["home_hgf_5"] = feats["home_hgf_5"].fillna(feats["home_gf_5"])
    feats["home_hga_5"] = feats["home_hga_5"].fillna(feats["home_ga_5"])
    feats["home_hpts_5"] = feats["home_hpts_5"].fillna(feats["home_pts_5"])
    feats["away_agf_5"] = feats["away_agf_5"].fillna(feats["away_gf_5"])
    feats["away_aga_5"] = feats["away_aga_5"].fillna(feats["away_ga_5"])
    feats["away_apts_5"] = feats["away_apts_5"].fillna(feats["away_pts_5"])
    # shots/sot/corners: league ort ile doldur (0 degil!)
    for col, base in [
        ("home_shots_5", "lg_avg_goals"), ("away_shots_5", "lg_avg_goals"),
        ("home_sot_5", "lg_avg_goals"), ("away_sot_5", "lg_avg_goals"),
        ("home_corners_5", "lg_corner_avg"), ("away_corners_5", "lg_corner_avg"),
        ("home_cards_5", "lg_card_avg"), ("away_cards_5", "lg_card_avg"),
    ]:
        # league avg * carpan ile doldur (shots ~ 13, sot ~ 5, corners ~ 5)
        if base in feats.columns:
            feats[col] = feats[col].fillna(feats[base])
    for col in ["home_w_shots", "away_w_shots", "home_w_sot", "away_w_sot",
                "home_w_corners", "away_w_corners", "home_w_cards", "away_w_cards"]:
        feats[col] = feats[col].fillna(feats[col].fillna(0).mean())
    # rest days: assume 7 if unknown
    feats["home_rest_days"] = feats["home_rest_days"].fillna(7.0)
    feats["away_rest_days"] = feats["away_rest_days"].fillna(7.0)
    feats["home_opp_elo"] = feats["home_opp_elo"].fillna(ELO_INIT)
    feats["away_opp_elo"] = feats["away_opp_elo"].fillna(ELO_INIT)
    # corners/cards absent entirely in some leagues (B1/T1/D1 free CSV) ->
    # fallback to global mean (ROADMAP bolum 79: feature fallback).
    for col in ["lg_corner_avg", "lg_card_avg"]:
        feats[col] = feats[col].fillna(feats[col].mean())
    feats["lg_corner_avg"] = feats["lg_corner_avg"].fillna(10.0)
    feats["lg_card_avg"] = feats["lg_card_avg"].fillna(5.0)
    # season_to_date + 8/20 windows
    for col in ["home_gf_8", "away_gf_8", "home_ga_8", "away_ga_8",
                "home_gf_20", "away_gf_20", "home_ga_20", "away_ga_20",
                "home_gf_std", "away_gf_std", "home_ga_std", "away_ga_std",
                "home_pts_std", "away_pts_std"]:
        feats[col] = feats[col].fillna(feats[col.replace("_8", "_5").replace("_20", "_5").replace("_std", "_5")])
    # attack/defence elo
    for col in ["home_attack_elo", "away_attack_elo", "home_defence_elo", "away_defence_elo",
                "attack_elo_diff", "defence_elo_diff"]:
        feats[col] = feats[col].fillna(ELO_INIT)
    # data completeness: fraction of core pre-match features present
    core = ["home_gf_5", "away_gf_5", "home_ga_5", "away_ga_5",
            "home_shots_5", "away_shots_5", "h2h_home_win", "lg_corner_avg",
            "home_attack_elo", "away_defence_elo"]
    feats["data_completeness"] = 1.0 - feats[core].isna().mean(axis=1)

    # Market-implied probs are only ~13% present. Fill missing with
    # league-conditional mean so with_odds models get a usable signal
    # (league favorite rate) instead of NaN.
    mkt_cols = ["mkt_home_prob", "mkt_draw_prob", "mkt_away_prob", "mkt_over25_prob"]
    for col in mkt_cols:
        if col in feats.columns:
            feats[col] = feats.groupby("league")[col].transform(
                lambda s: s.fillna(s.mean())
            )
            feats[col] = feats[col].fillna(feats[col].mean())
    if all(c in feats.columns for c in ["mkt_home_prob", "mkt_draw_prob", "mkt_away_prob"]):
        s = feats[["mkt_home_prob", "mkt_draw_prob", "mkt_away_prob"]].sum(axis=1)
        s = s.replace(0, 1.0)
        feats["mkt_home_prob"] /= s
        feats["mkt_draw_prob"] /= s
        feats["mkt_away_prob"] /= s

    # CLOSING ODDS (daha dogru oranlar)
    close_mkt_cols = ["mkt_close_home_prob", "mkt_close_draw_prob", "mkt_close_away_prob", "mkt_close_over25_prob"]
    for col in close_mkt_cols:
        if col in feats.columns:
            feats[col] = feats.groupby("league")[col].transform(
                lambda s: s.fillna(s.mean())
            )
            feats[col] = feats[col].fillna(feats[col].mean())
    if all(c in feats.columns for c in ["mkt_close_home_prob", "mkt_close_draw_prob", "mkt_close_away_prob"]):
        s = feats[["mkt_close_home_prob", "mkt_close_draw_prob", "mkt_close_away_prob"]].sum(axis=1)
        s = s.replace(0, 1.0)
        feats["mkt_close_home_prob"] /= s
        feats["mkt_close_draw_prob"] /= s
        feats["mkt_close_away_prob"] /= s

    # xG: sadece GERCEK degerleri kullan, proxy SIFIRLA (leak onleme)
    # proxy = mac sonrasi shots/sot verisi, mac oncesi bilinemez
    feats["xg_is_proxy"] = (~feats["home_xg"].notna() | ~feats["away_xg"].notna()).astype(float)
    feats["home_xg_real"] = feats["home_xg"].fillna(0)
    feats["away_xg_real"] = feats["away_xg"].fillna(0)
    feats["total_xg_real"] = feats["home_xg_real"] + feats["away_xg_real"]
    feats["xg_diff_real"] = feats["home_xg_real"] - feats["away_xg_real"]
    feats["shots_diff"] = feats["home_shots"] - feats["away_shots"]
    feats["sot_diff"] = feats["home_sot"] - feats["away_sot"]
    # Score probabilities - sadece gercek xG varsa hesapla
    feats["home_score_prob"] = np.where(feats["xg_is_proxy"] == 0,
        1.0 / (1.0 + np.exp(-1.5 * (feats["home_xg_real"] - 1.0))), 0.5)
    feats["away_score_prob"] = np.where(feats["xg_is_proxy"] == 0,
        1.0 / (1.0 + np.exp(-1.5 * (feats["away_xg_real"] - 1.0))), 0.5)
    feats["btts_xprob"] = feats["home_score_prob"] * feats["away_score_prob"]
    feats["over25_xprob"] = np.where(feats["xg_is_proxy"] == 0,
        1.0 - np.exp(-feats["total_xg_real"] * 0.8), 0.5)
    feats["xg_diff_abs"] = np.abs(feats["xg_diff_real"])
    feats["draw_xprob"] = 1.0 / (1.0 + feats["xg_diff_abs"] * 3.0)

    # ---- NEW FEATURES: halftime, momentum fill, form consistency ----
    # halftime: 0-0 degilse = "ikinci yari golu olma ihtimali"
    if "ht_home_goals" in feats.columns and "ht_away_goals" in feats.columns:
        ht_h = feats["ht_home_goals"].fillna(0)
        ht_a = feats["ht_away_goals"].fillna(0)
        feats["ht_total_goals"] = ht_h + ht_a
        feats["ht_result_is_draw"] = (ht_h == ht_a).astype(float)
        feats["ht_home_leading"] = (ht_h > ht_a).astype(float)
        feats["ht_second_half_goals_expected"] = feats["lg_avg_goals"] * 0.45
        # second half goals = total - ht_total (if available)
        total = feats["home_goals"].fillna(0) + feats["away_goals"].fillna(0)
        feats["second_half_goals"] = np.where(
            feats["ht_total_goals"].notna(),
            np.maximum(total - feats["ht_total_goals"], 0),
            np.nan
        )

    # momentum fill with league avg
    for col in ["home_momentum", "away_momentum", "home_form_std", "away_form_std",
                "home_gdiff5", "away_gdiff5"]:
        if col in feats.columns:
            feats[col] = feats[col].fillna(0)

    # streak fill
    for col in ["home_wins_last5", "home_draws_last5", "away_wins_last5", "away_draws_last5"]:
        if col in feats.columns:
            feats[col] = feats[col].fillna(0)

    # home/away form real fill
    for col in ["home_gf_5_real", "home_ga_5_real", "away_gf_5_real", "away_ga_5_real"]:
        if col in feats.columns:
            base_col = col.replace("_real", "")
            if base_col in feats.columns:
                feats[col] = feats[col].fillna(feats[base_col])

    return feats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ingestion.football_data.canonical import build_matches

    matches = build_matches()
    feats = build_features(matches)
    print(feats.iloc[1000:1003].T)
    print("\nNull counts (sample):\n", feats.isna().sum().sort_values(ascending=False).head(10))
    print("\nColumns:", list(feats.columns))
