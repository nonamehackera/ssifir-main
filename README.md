# ssifir-main

Futbol mac tahmin sistemi - LightGBM + XGB ensemble model, feedback loop ve market validator ile.

## Ozellikler

- **138 feature** ile mac sonucu tahmini (1X2, gol, korner, BTTS, over/under)
- **3x LightGBM + 3x XGB** ensemble (seed=42,123,789)
- **Isotonic Regression** kalibrasyonu
- **Market Validator**: Takim/lig istatistikleriyle tum pazarlari dogruluyor
- **Error Memory**: Yanlis tahminleri ogrenen hafiza sistemi
- **Web arayuzu**: Flask ile gercek zamanli tahmin + feedback API

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanim

### Tahmin
```bash
python predict.py
```

Ornek:
```
Galatasaray - Fenerbahce
```

### Web Sunucusu
```bash
python run_web.py
```

### Toplu Tahmin
```bash
python predict.py
> toplu tahminler/matches.json
```

## Feedback Loop

Her mac sonucunu kaydederek sistem kendini gelistirir:

```python
from predict import tahmin, sonuc_kaydet

# Tahmin yap
result, err = tahmin(home_id, away_id)

# Mac bittiginde sonucu kaydet
sonuc_kaydet(home_id, away_id, league,
             result["home_win"], result["draw"], result["away_win"],
             actual_result="H")  # H=Ev sahibi, D=Beraberlik, A=Deplasman
```

### Market Validator
Her tahmini takim istatistikleriyle karsilastirarak duzeltiyor:
- **1X2**: Gol oranlarina gore ev/deplansman gucu
- **BTTS**: Takimlarin gol yeme/atlama oranlari
- **Over/Under**: Gol beklentisi
- **Korner**: Takim korner ortalamalari

### Error Memory
Yanlis tahminleri ogrenen hafiza:
- Her takimin hata orani takip ediliyor
- Cok hata yapan takimlara "dikkat" uyarisi
- Bir dahaki sefere daha temkinli tahmin

## Proje Yapisi

```
ssifir-main/
├── predict.py              # Ana tahmin arayuzu
├── run_web.py              # Web sunucusu baslatma
├── feedback/               # Feedback loop sistemi
│   ├── market_validator.py # Pazar dogrulama
│   ├── error_memory.py     # Hata ogrenme hafizasi
│   ├── feedback_predictor.py
│   ├── error_tracker.py
│   ├── pattern_detector.py
│   └── self_correction.py
├── web/                    # Web arayuzu
│   └── app.py
├── models/                 # Model modulleri
│   ├── lightgbm/
│   ├── xgboost/
│   ├── poisson/
│   ├── elo/
│   └── ensemble/
├── scripts/                # Yardimci scriptler
├── tests/                  # Test dosyalari
├── data/                   # Veri dosyalari
├── configs/                # Ayarlar
└── tahminler/              # Kupon ve sonuc verileri
```

## Veri

- `data/gold/features_enhanced_v5.parquet`: 528K satir, 192 feature, 2015-2026
- `data/gold/team_id_to_name.json`: Takim ID -> isim eslesmesi

## Test

```bash
python test_feedback_comprehensive.py
```

11 kapsamli test - hepsi basarili.

## Teknolojiler

- Python 3.14
- LightGBM, XGBoost
- scikit-learn (IsotonicRegression)
- Flask (web)
- pandas, numpy
- pyarrow (parquet)
