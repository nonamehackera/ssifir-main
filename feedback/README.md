# Feedback Loop Sistemi

Modelin yanlış tahminlerinden otomatik olarak öğrenme ve kendini geliştirme mekanizması.

## Mimari

```
┌─────────────────────────────────────────────────────────────┐
│                    FEEDBACK LOOP                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │ Error       │    │ Pattern     │    │ Self        │    │
│  │ Tracker     │───▶│ Detector    │───▶│ Correction  │    │
│  │             │    │             │    │ Engine      │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│         │                  │                  │            │
│         │                  │                  │            │
│         ▼                  ▼                  ▼            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Feature Generator                      │  │
│  │  (Hata kalıplarından yeni özellikler üretir)       │  │
│  └─────────────────────────────────────────────────────┘  │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Feedback Orchestrator                   │  │
│  │  (Tüm bileşenleri koordine eder)                   │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Bileşenler

### 1. Error Tracker (`feedback/error_tracker.py`)
- Tahminleri ve gerçek sonuçları kaydeder
- Periyodik analiz yapar
- Büyük hatalı tahminleri tespit eder

### 2. Pattern Detector (`feedback/pattern_detector.py`)
Tespit edilen kalıp türleri:
- **Lig bazında önyargı**: Bir ligde sürekli yanlış tahmin
- **Takım bazında önyargı**: Belirli bir takımı yanlış tahmin etme
- **Sonuç bazında önyargı**: Beraberlikleri kaçırma gibi
- **Gol aralığı önyargısı**: Yüksek skorlu maçları tahmin edememe
- **Güven önyargısı**: Yüksek güvenli tahminlerde hata
- **Zamanlama önyargısı**: Dönemsel sapmalar

### 3. Self-Correction Engine (`feedback/self_correction.py`)
Düzeltme mekanizmaları:
- **Ağırlık Ayarlama**: Ensemble ağırlıklarını güncelleme
- **Kalibrasyon Düzeltme**: Olasılık kalibrasyonu
- **Sınıf Ağırlığı**: Azınlık sınıfların ağırlığını artırma
- **Çıkış Düzeltme**: Model çıkışlarını ayarlama

### 4. Feature Generator (`feedback/feature_generator.py`)
Üretilen özellikler:
- Takım hata profilleri
- Lig hata profilleri
- Sonuç hata profilleri
- Gol aralığı hata profilleri
- Güven hata profilleri
- Zamanlama hata profilleri
- Kombinasyon özellikleri

### 5. Feedback Orchestrator (`feedback/orchestrator.py`)
- Tüm bileşenleri koordine eder
- Periyodik döngü çalıştırır
- Geçmiş takibi yapar
- Raporlar oluşturur

## Kullanım

### Temel Kullanım

```python
from feedback.integration import get_feedback

# Geri besleme örneğini al
feedback = get_feedback()

# Tahmin kaydet
feedback.record_prediction(
    match_id="match_001",
    home_team="Galatasaray",
    away_team="Fenerbahce",
    league="Super Lig",
    pred_home=0.45,
    pred_draw=0.25,
    pred_away=0.30,
    actual_result="H",
    home_goals=2,
    away_goals=1,
)

# Döngü çalıştır
result = feedback.run_feedback_cycle()
print(f"Tespit edilen kalıp: {result['patterns_detected']}")

# Rapor al
report = feedback.get_report()
print(report)
```

### Ayarlanmış Tahmin

```python
# Ham tahmin
raw_prediction = {
    "home_win": 0.55,
    "draw": 0.25,
    "away_win": 0.20,
}

# Geri besleme ile ayarla
adjusted = feedback.get_adjusted_prediction(
    home_team="Galatasaray",
    away_team="Fenerbahce",
    league="Super Lig",
    raw_prediction=raw_prediction,
)

# Sonuç
print(f"Güven faktörü: {adjusted['confidence_factor']:.3f}")
print(f"Ayarlanmış: {adjusted['home_win']:.3f}")
```

### Takım Düzeltme Önerileri

```python
# Takım için düzeltme önerileri
features = feedback.get_team_correction("Galatasaray")
print(f"Hata oranı: {features.get('Galatasaray_error_rate', 'N/A')}")

# Lig için düzeltme önerileri
features = feedback.get_league_correction("Super Lig")
print(f"Hata oranı: {features.get('Super_Lig_error_rate', 'N/A')}")
```

## Dosya Yapısı

```
data/feedback/
├── prediction_records.parquet    # Tahmin kayıtları
├── cycle_history.json           # Döngü geçmişi
├── analysis/                    # Analiz sonuçları
│   └── error_analysis.json
├── corrections/                 # Düzeltme geçmişi
│   └── correction_0001.json
└── features/                    # Üretilen özellikler
    └── error_features.parquet
```

## Entegrasyon

### Mevcut predict.py ile Kullanım

```python
# predict.py dosyasının başına ekle
from feedback.integration import get_feedback

# Tahmin fonksiyonunun sonunda
def tahmin(home_id, away_id):
    # ... mevcut kod ...
    
    # Geri besleme için kaydet
    feedback = get_feedback()
    feedback.record_prediction(
        match_id=f"{home_id}_{away_id}",
        home_team=home_name,
        away_team=away_name,
        league=league_name,
        pred_home=cp[0],
        pred_draw=cp[1],
        pred_away=cp[2],
        actual_result=actual_result,  # Gerçek sonuç biliniyorsa
        home_goals=home_goals,
        away_goals=away_goals,
    )
    
    return result, None
```

### API ile Kullanım

```python
# api/service.py'ye ekle
from feedback.integration import get_feedback

@app.post("/predict_with_feedback")
async def predict_with_feedback(match: MatchRequest):
    # Tahmin yap
    prediction = make_prediction(match)
    
    # Geri besleme için kaydet
    feedback = get_feedback()
    feedback.record_prediction(
        match_id=match.id,
        home_team=match.home_team,
        away_team=match.away_team,
        league=match.league,
        pred_home=prediction["home_win"],
        pred_draw=prediction["draw"],
        pred_away=prediction["away_win"],
        actual_result=match.actual_result,  # Gerçek sonuç biliniyorsa
    )
    
    return prediction
```

## Otomatik Düzeltme

Sistem otomatik olarak şu düzeltmeleri yapar:

1. **Yeterli veri olduğunda** (50+ tahmin)
2. **Hata kalıplarını tespit ettiğinde**
3. **İstatistiksel olarak anlamlı olduğunda** (p < 0.05)

Düzeltme türleri:
- Ensemble ağırlıklarını güncelleme
- Sıcaklık kalibrasyonunu ayarlama
- Sınıf ağırlıklarını dengeleme
- Yeni özellikler üretme

## Metrikler

Sistem şu metrikleri takip eder:
- **Doğruluk (Accuracy)**: Doğru tahmin oranı
- **Log Loss**: Olasılık kalitesi
- **Brier Score**: Tahmin kalitesi
- **Hata Oranı**: Toplam hata oranı
- **İyileştirme**: Düzeltmelerden sonra beklenen kazanç

## Güvenlik

- Aşırı düzeltme koruması (maksimum %30 değişiklik)
- İstatistiksel anlamlılık testi
- Minimum örnek sayısı gereksinimi
- Geri alınabilir değişiklikler

## Çalışma Zamanı

Önerilen çalışma zamanı:
- **Günlük**: Yeni tahminleri kaydet
- **Haftalık**: Döngü çalıştır
- **Aylık**: Tam rapor oluştur

```python
# jobs/feedback_daily.py
from feedback.integration import get_feedback

def daily_feedback():
    feedback = get_feedback()
    feedback.run_feedback_cycle()

# jobs/feedback_weekly.py
def weekly_report():
    feedback = get_feedback()
    report = feedback.get_report()
    # Raporu kaydet veya gönder
```
