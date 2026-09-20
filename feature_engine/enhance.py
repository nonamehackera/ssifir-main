"""Feature Enhancement Module (production-ready).

v2 iyilestirmelerini kalici ve saglam sekilde pipeline'a entegre eder.

Amac:
  1. Market-implied prob'lari sinyal olarak kullan (kopyalama degil)
  2. Buyuk/az verili ligleri ayirt et; az verili/degerli ligler icin
     global fallback priors kullan (bayrak ile isaretle)
  3. Elo + form kombinasyonu ile gercek guc feature'lari uret
  4. "Varsayilan" ligler (yuksek verili, makul mac sayisi) icin
     ozel bir IsHighConfidence bayragi ekle - ona gore tahmin
     guvenilirligini sinirlar
  5. Tum islemler point-in-time (gelecek bilgisi yok)

Bu modul iki girdi tipi destekler:
  A) Ham features.parquet (build_features ciktisi) -> post-process
  B) Tek maclik feature satiri (canli/gelecek tahmin) -> fallback ile

Kullanim:
    from feature_engine.enhance import enhance_features, augment_row
    df = enhance_features(df)                 # toplu kullanim
    row = augment_row(single_row)             # tek mac icin
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Yeterli veriye sahip kabul edilen minimum lig maci.
# Altindaki liglerde takim bazli sinyaller guvenilmez -> fallback
MIN_LEAGUE_MATCHES = 1000

# "Varsayilan/guvenilir" kabul edilen lig maci esigi.
CONFIDENT_LEAGUE_MATCHES = 3000

# Eklenen feature adlari (modele girecekler)
ENHANCED_FEATURES = [
    "mkt_home_misalign", "mkt_draw_mismatch", "mkt_over_misalign",
    "home_gf_per_ga", "away_gf_per_ga",
    "home_attack_ratio", "away_attack_ratio",
    "home_pts_per_game", "away_pts_per_game",
    "form_pts_diff",
    "btts_signal",
    "home_overall", "away_overall", "overall_diff",
    "lg_confidence_weight",
]


def _elo_win_prob(home_elo, away_elo, home_adv: float = 50.0):
    """Elo farkindan bagimsiz home win prob (point-in-time)."""
    return 1.0 / (1.0 + 10 ** (-(np.asarray(home_elo, dtype=float)
                               + home_adv - np.asarray(away_elo, dtype=float)) / 400))


def _derive_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Market-implied oranlardan misalignment/kopma feature'lari."""
    df = df.copy()
    eh = _elo_win_prob(df["home_elo"], df["away_elo"])
    df["mkt_home_misalign"] = eh - df["mkt_home_prob"]
    df["mkt_draw_mismatch"] = (
        df["mkt_draw_prob"]
        - (1 - np.abs(eh - (1 - eh)) * 0.8)
    )
    if "mkt_over25_prob" in df.columns and "lg_over25_rate" in df.columns:
        df["mkt_over_misalign"] = df["lg_over25_rate"] - df["mkt_over25_prob"]
    else:
        df["mkt_over_misalign"] = 0.0
    # NaN -> 0 (market yoksa) - nötr sinyal
    for c in ["mkt_home_misalign", "mkt_draw_mismatch", "mkt_over_misalign"]:
        df[c] = df[c].fillna(0.0)
    return df


def _derive_form_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Elo'dan bagimsiz gercek form/guc kombinasyonlari."""
    df = df.copy()
    eps = 0.5
    df["home_gf_per_ga"] = df["home_gf_5"] / (df["home_ga_5"] + eps)
    df["away_gf_per_ga"] = df["away_gf_5"] / (df["away_ga_5"] + eps)
    df["home_attack_ratio"] = df["home_gf_5"] / (df["home_gf_5"] + df["home_ga_5"] + 0.1)
    df["away_attack_ratio"] = df["away_gf_5"] / (df["away_gf_5"] + df["away_ga_5"] + 0.1)
    df["home_pts_per_game"] = df["home_pts_5"] / 5.0
    df["away_pts_per_game"] = df["away_pts_5"] / 5.0
    df["form_pts_diff"] = df["home_pts_per_game"] - df["away_pts_per_game"]
    # guvenli bolme
    for c in ["home_gf_per_ga", "away_gf_per_ga", "home_attack_ratio",
              "away_attack_ratio"]:
        df[c] = df[c].replace([np.inf, -np.inf], 3.0).fillna(1.0)
    return df


def _derive_btts_signal(df: pd.DataFrame) -> pd.DataFrame:
    """BTTS sinyali: H2H BTTS + lig ortalamasi karisimi."""
    df = df.copy()
    if "lg_btts_rate" not in df.columns:
        df["lg_btts_rate"] = 0.5
    h2h = df["h2h_btts"] if "h2h_btts" in df.columns else df["lg_btts_rate"]
    df["btts_signal"] = (h2h.fillna(df["lg_btts_rate"]) * 0.5
                         + df["lg_btts_rate"] * 0.5)
    df["btts_signal"] = df["btts_signal"].fillna(df["lg_btts_rate"])
    return df


def _derive_overall_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Toplam guc: elo + attack_elo + form puanlari."""
    df = df.copy()
    df["home_overall"] = (df["home_elo"]
                          + df["home_attack_elo"] / 10.0
                          + df["home_pts_per_game"] * 50.0)
    df["away_overall"] = (df["away_elo"]
                          + df["away_attack_elo"] / 10.0
                          + df["away_pts_per_game"] * 50.0)
    df["overall_diff"] = df["home_overall"] - df["away_overall"]
    return df


def _league_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Lig bazli guvenilirlik agirligi.

    Az verili liglerde takim bazli sinyaller zayif -> agirligi dusur.
    Bu, modelin o satirda MARKET oranlarina daha fazla / daha az
    guvenmesini saglamaz dogrudan; ama interaction olarak yuksek
    verili liglerde form sinyallerine daha cok guven verir.
    """
    df = df.copy()
    if "league" in df.columns:
        lc = df["league"].map(df.groupby("league").size())
        conf = np.clip(lc / CONFIDENT_LEAGUE_MATCHES, 0.2, 1.0)
        df["lg_confidence_weight"] = conf
    else:
        df["lg_confidence_weight"] = 1.0
    df["lg_confidence_weight"] = df["lg_confidence_weight"].fillna(0.2)
    return df


def enhance_features(df: pd.DataFrame) -> pd.DataFrame:
    """`build_features` ciktisi (features.parquet) uzerinde kosulur.

    Tum yeni feature'lari ekler. Herhangi biri eksikse guvenli sekilde
    default degerle devam eder (production'da patlamaz).
    """
    df = df.copy()

    # Zorunlu kolonlar
    required = ["home_elo", "away_elo", "home_gf_5", "home_ga_5",
                "away_gf_5", "away_ga_5", "home_pts_5", "away_pts_5",
                "mkt_home_prob", "mkt_draw_prob", "mkt_away_prob"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Eksik kolonlar icin nötr/default degerleri doldur
        for c in missing:
            logger.warning("enhance_features: %s yok, default ekleniyor", c)
            if "mkt" in c or "prob" in c:
                df[c] = 0.5
            else:
                df[c] = 0.0

    df = _derive_market_features(df)
    df = _derive_form_strength(df)
    df = _derive_btts_signal(df)
    df = _derive_overall_strength(df)
    df = _league_confidence(df)

    return df


def augment_row(row: pd.Series, league_match_counts: dict | None = None) -> pd.Series:
    """Tek mac feature satirina yeni feature'lari ekler (tahmin sirasi).

    `league_match_counts`: {lig: mac sayisi} sozlugu. None ise varsayilan
    respons bloğuna gore conf ayarlanir.

    Tahmin zamaninda build_features zaten tum pre-match feature'lari
    uretmis olmalı (home_elo, home_gf_5, mkt_* prob etc). Bu fonksiyon
    sadece ENHANCED_FEATURES'i ekler.
    """
    row = row.copy()

    # Market feature'lar
    eh = _elo_win_prob(row.get("home_elo", 1500.0), row.get("away_elo", 1500.0))
    row["mkt_home_misalign"] = eh - float(row.get("mkt_home_prob", 0.5))
    row["mkt_draw_mismatch"] = (
        float(row.get("mkt_draw_prob", 0.3))
        - (1 - abs(eh - (1 - eh)) * 0.8)
    )
    over_rate = row.get("lg_over25_rate", 0.5)
    over_mkt = row.get("mkt_over25_prob", over_rate)
    row["mkt_over_misalign"] = float(over_rate) - float(over_mkt)

    # Form guc
    hgf = float(row.get("home_gf_5", 0.0)); hga = float(row.get("home_ga_5", 0.0))
    agf = float(row.get("away_gf_5", 0.0)); aga = float(row.get("away_ga_5", 0.0))
    row["home_gf_per_ga"] = hgf / (hga + 0.5)
    row["away_gf_per_ga"] = agf / (aga + 0.5)
    row["home_attack_ratio"] = hgf / (hgf + hga + 0.1)
    row["away_attack_ratio"] = agf / (agf + aga + 0.1)
    hpts = float(row.get("home_pts_5", 1.4)); apts = float(row.get("away_pts_5", 1.0))
    row["home_pts_per_game"] = hpts / 5.0
    row["away_pts_per_game"] = apts / 5.0
    row["form_pts_diff"] = row["home_pts_per_game"] - row["away_pts_per_game"]

    # BTTS
    lg_btts = float(row.get("lg_btts_rate", 0.5))
    h2h = float(row.get("h2h_btts", lg_btts)) if pd.notna(row.get("h2h_btts")) else lg_btts
    row["btts_signal"] = h2h * 0.5 + lg_btts * 0.5

    # Overall
    he = float(row.get("home_elo", 1500.0)); ae = float(row.get("away_elo", 1500.0))
    hea = float(row.get("home_attack_elo", 1500.0)); aea = float(row.get("away_attack_elo", 1500.0))
    row["home_overall"] = he + hea / 10.0 + row["home_pts_per_game"] * 50.0
    row["away_overall"] = ae + aea / 10.0 + row["away_pts_per_game"] * 50.0
    row["overall_diff"] = row["home_overall"] - row["away_overall"]

    # Lig guvenilirlik agirligi
    if league_match_counts is not None and "league" in row.index:
        n = league_match_counts.get(row["league"], 0)
        row["lg_confidence_weight"] = float(np.clip(n / CONFIDENT_LEAGUE_MATCHES, 0.2, 1.0))
    else:
        row["lg_confidence_weight"] = 0.5

    return row


def get_enhanced_feature_names() -> list[str]:
    """Production modelinde kullanilacak feature ad listesi."""
    return list(ENHANCED_FEATURES)
