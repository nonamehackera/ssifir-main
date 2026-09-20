# ssifir-main

Futbol mac tahmin sistemi. LightGBM + XGB ensemble model, 138 feature, otomatik veri pipeline'i ve feedback loop ile.

## Nasil Calisir?

```
Veri Toplama -> Feature Uretimi -> Model Egitimi -> Tahmin -> Feedback Loop
     |              |                  |               |            |
  9 kaynak     138 feature      3xLGB+3xXGB     1X2/Gol/Korner  Ogrenme
```

### Tahmin Ornegi

```bash
python predict.py
> Galatasaray - Fenerbahce

  Galatasaray vs Fenerbahce
  ─────────────────────────
  1X2:   Ev %46  Beraberlik %22  Dis %32
  Gol:   Ust 2.5: %57  BTTS: %54
  Korner: 7.2  Ust 8.5: %40
  Guven: %46
```

## Kurulum

```bash
# 1. Bagimliliklari yukle
pip install -r requirements.txt

# 2. Veritabani ayarlari (opsiyonel)
cp .env.example .env
# .env dosyasini duzenle

# 3. Modelleri egit
python predict.py
```

## Kullanim

### Tahmin

```bash
python predict.py

# Komutlar:
# Galatasaray - Fenerbahce     -> Tek mac tahmini
# ara galatasaray              -> Takim ara
# value GS - FB 2.1/3.4/3.2   -> Value bet analizi
# toplu tahminler/match.json   -> Toplu tahmin
# cikis                        -> Cikis
```

### Web Arayuzu

```bash
python run_web.py
# http://localhost:5000
```

### Programatik Kullanim

```python
from predict import tahmin, sonuc_kaydet

# Tahmin yap
result, err = tahmin(home_id, away_id)
print(result["home_win"], result["draw"], result["away_win"])

# Sonucu kaydet (feedback loop icin)
sonuc_kaydet(home_id, away_id, league,
             result["home_win"], result["draw"], result["away_win"],
             actual_result="H")
```

## Feedback Loop

Sistem her tahmini ve sonucunu kaydeder. Yanlis tahminlerden ogrenir.

```python
from feedback.error_memory import ErrorMemory

em = ErrorMemory(data_dir="data/feedback")
em.get_team_error_rate(team_id)  # Takimin hata orani
em.get_stats()                    # Genel istatistikler
```

### Market Validator

Her tahmini takim/lig istatistikleriyle karsilastirir:

| Pazar | Kontrol |
|-------|---------|
| 1X2 | Gol oranlari, ev/deplasman gucu |
| BTTS | Gol yeme/atlama oranlari |
| Over/Under | Gol beklentisi |
| Korner | Takim korner ortalamalari |

Ornek: Model "Korner 9.0" dediginde, takim ortalamasi 6.5 ise duzeltir.

### Error Memory

Yanlis tahminleri ogrenen hafiza:
- Her takimin hata orani takip ediliyor
- 50+ mac kaydindan sonra anlamli ogrenme baslar
- Bir dahaki sefere daha temkinli tahmin yapar

## Proje Yapisi

```
ssifir-main/
├── predict.py                 # Ana tahmin arayuzu
├── run_web.py                 # Web sunucusu
│
├── feedback/                  # Feedback loop
│   ├── market_validator.py    #   Pazar dogrulama (1X2, gol, korner)
│   ├── error_memory.py        #   Hata ogrenme hafizasi
│   ├── feedback_predictor.py  #   Feedback tahminci
│   ├── error_tracker.py       #   Hata takibi
│   ├── pattern_detector.py    #   Kalip tespiti
│   └── self_correction.py     #   Otomatik duzeltme
│
├── feature_engine/            # Feature uretimi
│   ├── engine.py              #   Ana motor (138 feature)
│   ├── enhance.py             #   Ek feature'lar
│   └── stats_predictor.py     #   Eksik istatistik tahmini
│
├── models/                    # ML modelleri
│   ├── lightgbm/              #   LightGBM modeli
│   ├── xgboost/               #   XGBoost modeli
│   ├── ensemble/              #   Ensemble (kalibre edilmis)
│   ├── poisson/               #   Poisson skor modeli
│   ├── elo/                   #   Elo rating
│   └── trained/               #   Egitilmis modeller
│
├── prediction/                # Tahmin pipeline'i
│   ├── pipeline.py            #   Ana pipeline
│   ├── production_pipeline.py #   Production ensemble
│   ├── persistence.py         #   DB'ye yazma
│   └── *_scraper.py           #   Skor kaynaklari
│
├── ingestion/                 # Veri toplama (9 kaynak)
│   ├── api_football/          #   API-Football
│   ├── fbref/                 #   FBref
│   ├── football_data/         #   Football-Data.co.uk
│   └── ...                    #   Diger kaynaklar
│
├── web/                       # Web arayuzu
│   ├── app.py                 #   Flask uygulamasi
│   ├── templates/             #   HTML sablonlari
│   └── static/                #   CSS/JS dosyalari
│
├── api/                       # FastAPI uretim API'si
├── betting/                   # Value betting + Kelly stake
├── calibration/               # Olasilik kalibrasyonu
├── configs/                   # Merkezi ayarlar
├── db/                        # SQLAlchemy ORM
├── jobs/                      # Scheduler + otomasyon
├── live/                      # Canli mac verisi
├── monitoring/                # Drift monitoring
├── normalization/             # Veri donusumu
├── odds/                      # Odds arsivi
├── registry/                  # Model versiyon yonetimi
│
├── scripts/                   # Yardimci scriptler
├── tests/                     # Test dosyalari
├── tahminler/                 # Kupon ve sonuc verileri
└── data/                      # Veri dosyalari
    ├── bronze/                #   Ham veri
    ├── silver/                #   Normalize edilmis
    ├── gold/                  #   Feature'lar (parquet)
    └── feedback/              #   Feedback loglari
```

## Veri

| Dosya | Icerik |
|-------|--------|
| `data/gold/features_enhanced_v5.parquet` | 528K satir, 192 feature, 2015-2026 |
| `data/gold/team_id_to_name.json` | Takim ID -> isim eslesmesi |
| `data/bronze/football_data/` | Ham mac verileri (20+ lig, 20 yil) |

## Teknolojiler

| Katman | Teknoloji |
|--------|-----------|
| Model | LightGBM, XGBoost, Poisson, Elo |
| Kalibrasyon | Isotonic Regression, Platt Scaling |
| Feature | pandas, numpy, scikit-learn |
| Veri | pyarrow (parquet), SQLAlchemy (PostgreSQL) |
| API | Flask (web), FastAPI (production) |
| Ornek | APScheduler, Playwright |

## Test

```bash
# Kapsamli feedback testi
python test_feedback_comprehensive.py

# Tum testler
python -m pytest tests/
```

## Lisans

Proje sahibi tarafindan kullanilir.
