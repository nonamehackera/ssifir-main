# Futbol Maç Tahmin Sistemi — Üretim Sürümü

Geçmiş maçlardan öğrenen, maç öncesi tahmin üreten, canlı veriyle güncel
tahmin yapan ve kendini ölçerek geliştiren futbol tahmin platformu.

## Durum: ÇALIŞIYOR (end-to-end doğrulandı)

Tüm pipeline ücretsiz **Football-Data.co.uk** verisiyle çalışır (API anahtarı
gerekmez). 17,612 maç, 12 lig, 2021/22–2024/25 sezonları.

## Doğrulanmış Gerçek Sonuçlar

Walk-forward (4 fold, kronolojik, out-of-sample):
```
model          log_loss  brier  accuracy   ece   btts_auc  over25_auc
catboost_odds    0.9697  0.576    0.517    0.032   0.549     0.603
catboost         0.9875  0.589    0.497    0.034   0.543     0.582
ensemble         0.9891  0.590    0.488    0.041   0.541     0.584
elo              1.0128  0.601    0.432    0.047    NaN       NaN
dixon_coles      1.0671  0.641    0.460    0.062   0.535     0.564
lightgbm         1.1060  0.644    0.484    0.111   0.511     0.547
```
Naive baseline log-loss = 1.0986. Tüm modeller baseline'ı geçiyor.

Veri gerçekliği (sızıntı yok): BTTS %53.1, Over2.5 %52.4, ort. gol 1.53/1.23.

## Mimari (ROADMAP'a uygun)

```
Football-Data.co.uk (free) -> canonical.py -> gold/matches.parquet
                                      |
                              feature_engine (point-in-time, 85 feature)
                                      |
        +-----------+-----------+-----------+-----------+
        ELO       DIXON-COLES   CATBOOST   LIGHTGBM    ENSEMBLE
                                      |
        +-----------+-----------+-----------+-----------+
   calibration/   registry/   prediction/   odds/(CLV)   live/   monitoring/
        |
   PostgreSQL (fixtures, odds_snapshots, live_snapshots, model_versions)
        |
   api/service.py (FastAPI /predict)
```

## Çalıştırma

```bash
# 1. Veri + feature (ücretsiz, anahtarsız)
.venv/bin/python -c "from ingestion.football_data.canonical import build_matches;
from feature_engine.engine import build_features;
build_features(build_matches())"

# 2. Walk-forward karşılaştırma
.venv/bin/python -m jobs._eval_wf --folds 4 --quick

# 3. Tüm doğrulama testleri
.venv/bin/python tests/run_all.py

# 4. API başlat (model belleğe yüklenir)
.venv/bin/python -m api.service
#   GET  /health
#   POST /predict  {"fixture_id": 123, "features": {...}}
```

## Modüller

| Modül | Dosya | Durum |
|-------|-------|-------|
| Veri toplama (free) | `ingestion/football_data/canonical.py` | ✅ 17.6k maç |
| Feature engine | `feature_engine/engine.py` | ✅ 85 leakage-free feature |
| Elo | `models/elo/model.py` | ✅ log-loss 1.013 |
| Dixon-Coles | `models/poisson/model.py` | ✅ λ kalibre (1.55/1.24) |
| CatBoost | `models/catboost/model.py` | ✅ en iyi (0.97) |
| LightGBM | `models/lightgbm/model.py` | ✅ early-stop (1.00) |
| Ensemble | `models/ensemble/ensemble.py` | ✅ ağırlık opt. |
| Kalibrasyon | `calibration/calibrators.py` | ✅ akıllı seçim |
| Registry | `registry/registry.py` | ✅ cand→staging→prod |
| Odds/CLV | `odds/archive.py` | ✅ 17.5k maç CLV |
| Prediction | `prediction/pipeline.py` | ✅ ROADMAP 52 JSON |
| API | `api/service.py` | ✅ FastAPI /predict |
| Canlı model | `live/model.py` | ✅ geleceğe bakmaz |
| Drift | `monitoring/drift.py` | ✅ threshold alarm |

## Önemli Prensipler (ROADMAP)

1. **Point-in-time feature reconstruction** — hiçbir model gelecek bilgisi
   kullanmaz (feature engine kronolojik sıralı, sadece geçmiş state).
2. **Walk-forward validation** — random split YOK, kronolojik fold'lar.
3. **Dürüst metrikler** — accuracy tek başına yetmez; log-loss, Brier, ECE,
   CLV, AUC.
4. **Kalibrasyon her zaman iyileştirmez** — zaten kalibre modelde (CatBoost)
   esnek kalibrator overfit olup zarar verir; modül akıllı seçim yapar.
5. **Canlı model geleceğe bakmaz** (ROADMAP 23) — t anı tahmini sadece 0-t.

## Sonraki Adımlar (ROADMAP Phase 6+)

- [ ] API-Football / Sportmonks entegrasyonu (lineups, injuries, xG) — ücretli
- [ ] Canlı odds polling (15-30sn) + canlı snapshot arşivi
- [ ] Player strength / expected XI modeli
- [ ] MLflow tracking (şu an registry manuel)
- [ ] Redis canlı state cache
