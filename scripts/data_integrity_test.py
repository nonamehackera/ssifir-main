"""VERI BUNTUNLUK + LEAKAGE TESTI (model degil, veri sorgulaniyor).

Testler:
  1. LEAKAGE: home_elo / home_xg gibi feature'lar o macin SONRASINI iceriyor mu?
     -> home_elo, macin oldugu andaki elo olmali (gelecek maclarin ortalamasi degil)
  2. BASIT MODEL: sadece elo_diff ile %50+ yapabiliyor muyuz?
     -> Yapamiyorsak veri BOZUK (model degil)
  3. FEATURE ANLAM: home_xg vs home_goals korelasyonu mantikli mi?
"""
import sys, time, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from models.elo.model import EloModel
from evaluation.metrics import accuracy_1x2, log_loss_1x2

R2I = {"H": 0, "D": 1, "A": 2}


def main():
    t0 = time.time()
    f = pd.read_parquet("data/gold/features.parquet")
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values("date").reset_index(drop=True)
    print(f"Veri: {len(f)} mac, {f['date'].min().date()} -> {f['date'].max().date()}")

    # ---- TEST 1: LEAKAGE KONTROL ----
    print("\n" + "=" * 60)
    print("TEST 1: LEAKAGE (feature == o macin SONRASI verisi mi?)")
    print("=" * 60)
    # home_elo, mac tarihindeki elo olmali. Bir takimin maclari chrono siralı.
    # Eger home_elo, o macin AKABINDEKI maclarin elo'sunu iceriyorsa leakage var.
    # Basit kontrol: home_elo, mac sonucuyla (home_goals) pozitif korelasyon gostermeli
    # (iyi takim kazanir) ama bu yeterli degil. Daha guclu test:
    # home_elo - away_elo farki ile home_goals - away_goals farki korelasyonu.
    elodiff = (f["home_elo"] - f["away_elo"]).to_numpy()
    goaldiff = (f["home_goals"] - f["away_goals"]).to_numpy()
    corr = np.corrcoef(elodiff, goaldiff)[0, 1]
    print(f"  elo_diff vs goal_diff korelasyonu: {corr:.3f}")
    print(f"  (pozitif olmali; dusukse elo zayif/leakage)")
    # xG kontrol: home_xg vs home_goals korelasyonu (gercek xG, golle uyumlu olmali)
    if "home_xg" in f.columns and f["home_xg"].notna().any():
        xg_corr = np.corrcoef(f["home_xg"].to_numpy(), f["home_goals"].to_numpy())[0, 1]
        print(f"  home_xg vs home_goals korelasyonu: {xg_corr:.3f} (yuksek olmali, xG golun ongorusudur)")

    # ---- TEST 2: BASIT MODEL (sadece elo) ----
    print("\n" + "=" * 60)
    print("TEST 2: BASIT MODEL (sadece elo_diff) - veri yeterli mi?")
    print("=" * 60)
    seasons = [
        ("2022-07-01", "2023-07-01", "2022/23"),
        ("2023-07-01", "2024-07-01", "2023/24"),
        ("2024-07-01", "2025-07-01", "2024/25"),
    ]
    accs = []
    for (vs, ve, label) in seasons:
        v_start = pd.Timestamp(vs)
        valid = f[(f["date"] >= v_start) & (f["date"] < pd.Timestamp(ve))].copy()
        train = f[f["date"] < v_start]
        elo = EloModel().fit(train, train.tail(2000))  # elo sadece train kullanir
        p = elo.predict(valid)
        acc = accuracy_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        accs.append(acc)
        print(f"  {label}: elo-only acc = {acc*100:.1f}%")
    print(f"  ORTALAMA elo-only acc = {np.mean(accs)*100:.1f}%")

    # ---- TEST 3: BASIT RULE (favori = elo_diff>0 -> H, <0 -> A) ----
    print("\n" + "=" * 60)
    print("TEST 3: EN BASIT KURAL (elo_diff > 0 => H tahmini)")
    print("=" * 60)
    # tum veride: elo_diff pozitifse H, negatifse A, ~0 ise D
    mask_h = f["elo_diff"] > 50
    mask_a = f["elo_diff"] < -50
    mask_d = (~mask_h) & (~mask_a)
    pred = np.where(mask_h, "H", np.where(mask_a, "A", "D"))
    true = f["result"].to_numpy()
    acc_simple = (pred == true).mean()
    print(f"  elo_diff kurali ile acc = {acc_simple*100:.1f}%")
    print(f"  (eger bu %50'nin altindaysa veri BOZUK)")

    print("\n" + "=" * 60)
    print("SONUC:")
    print(f"  - elo_diff vs goal_diff corr: {corr:.3f}")
    print(f"  - elo-only model acc: {np.mean(accs)*100:.1f}%")
    print(f"  - basit kural acc: {acc_simple*100:.1f}%")
    print(f"  Sure: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
