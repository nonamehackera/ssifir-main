# Otomatik Tahmin Sistemi

## Ne Yapıldı?

Sistem **kafasından sallamıyor** - gerçek makine öğrenimi modelleri kullanıyor ama **otomatik tahmin oluşturma** yoktu. Şimdi eklendi!

## Yeni Özellikler

### 1. **Takım Eşleştirme Sistemi** (`prediction/team_matcher.py`)
- Flashscore'dan gelen takım isimlerini local DB'deki ID'lerle eşleştirir
- Fuzzy matching algoritması (benzerlik oranı %75+)
- Türkçe karakter normalizasyonu
- Bilinen takım eşleştirmeleri (Manuel override)

**Örnek:**
```python
from prediction.team_matcher import find_team_id

team_id = find_team_id("Manchester City", TEAM_ID_MAP)
# Output: 8 (veya başka bir ID)
```

### 2. **Otomatik Tahmin Oluşturma** (`jobs/auto_predictions.py`)
- Yaklaşan maçları Flashscore'dan çeker
- Takım isimlerini eşleştirir
- Gerçek verilerle tahmin oluşturur
- Her şey otomatik!

**Kullanım (Command Line):**
```bash
# 3 gün için tahmin oluştur
python -m jobs.auto_predictions

# 7 gün için tahmin oluştur
python -m jobs.auto_predictions --days 7

# Sadece eşleşmeleri göster (tahmin oluşturma)
python -m jobs.auto_predictions --dry-run
```

### 3. **Web Arayüzü Butonu**
- "Tahminlerim" sayfasına **"Otomatik Tahmin"** butonu eklendi
- Tek tıkla yaklaşan maçlar için tahmin oluşturur
- İlerleme durumu gösterir

### 4. **Backend API Endpoint** (`/api/predictions/auto-generate`)
- POST isteği ile otomatik tahmin oluşturur
- Parametreler:
  - `days`: Kaç gün sonrasına kadar (default: 3)
- Response:
  ```json
  {
    "success": true,
    "total": 50,
    "matched": 35,
    "created": 30,
    "skipped": 5
  }
  ```

## Nasıl Çalışır?

1. **Flashscore Scraping**: Yaklaşan maçları Flashscore.com'dan çeker
2. **Takım Eşleştirme**: Takım isimlerini local DB'deki ID'lerle eşleştirir
3. **Tahmin Oluşturma**: Gerçek makine öğrenimi modelleriyle tahmin yapar
4. **Kaydetme**: Tahminleri `tahminler/predictions.json`'a kaydeder

## Örnek Akış

```
1. Kullanıcı "Otomatik Tahmin" butonuna tıklar
   ↓
2. Sistem Flashscore'dan yaklaşan maçları çeker
   - Manchester City vs Arsenal (Yarın, 20:00)
   - Real Madrid vs Barcelona (3 gün sonra)
   ↓
3. Takım isimlerini eşleştirir
   - "Manchester City" → ID: 8
   - "Arsenal" → ID: 12
   ↓
4. Her maç için tahmin oluşturur
   - ELO ratings, form verileri, istatistikler
   - Makine öğrenimi modelleri ile hesaplama
   ↓
5. Tahminleri kaydeder
   - 1X2, BTTS, Over/Under, Korner, vb.
   ↓
6. Kullanıcıya bildirim gösterir
   "✓ 30 yeni tahmin oluşturuldu!"
```

## Avantajlar

✅ **Kafasından sallamıyor** - Gerçek verilerle çalışıyor
✅ **Otomatik** - Manuel olarak her maç için tahmin yapmaya gerek yok
✅ **Hızlı** - Dakikalar içinde onlarca tahmin oluşturur
✅ **Güvenilir** - Takım eşleştirme %75+ doğruluk oranı
✅ **Esnek** - 1-7 gün arası tahmin oluşturabilir

## Sorun Giderme

### "Takım bulunamadı" Hatası
- Takım isimleri DB'de yoksa eşleşmez
- `KNOWN_MAPPINGS` dict'ine manuel olarak eklenebilir
- Örnek: `team_matcher.py` → `KNOWN_MAPPINGS`

### "Flashscore bağlantı hatası"
- Internet bağlantısını kontrol et
- Rate limiting olabilir (çok sık istek)
- Birkaç dakika bekleyip tekrar dene

### "Model yüklenemedi"
- `data/gold/models/web_model_vXX.pkl` dosyası olmalı
- Model eğitilmemişse: `python train_vX.py`

## Test

### Command Line ile Test:
```bash
# Dry run - sadece eşleşmeleri göster
python -m jobs.auto_predictions --dry-run

# Gerçek tahmin oluştur
python -m jobs.auto_predictions
```

### Web Arayüzü ile Test:
1. http://localhost:5011/predictions adresine git
2. "Otomatik Tahmin" butonuna tıkla
3. Birkaç dakika bekle
4. Yeni tahminleri gör

## Sonraki Adımlar

- ✅ Kupon kontrol sistemi düzeltildi (3 gün önce oynanan maçlar)
- ✅ Otomatik tahmin sistemi eklendi
- 🔄 Scheduler ile günlük otomatik tahmin (opsiyonel)
- 🔄 Takım eşleştirme doğruluğunu artır (%85+)
- 🔄 Daha fazla lig desteği

## Önemli Notlar

1. **Rate Limiting**: Flashscore'dan çok sık istek yapma (2-3 saniye ara ver)
2. **Veri Güncelliği**: Model verileri güncel olmalı (`features.parquet`)
3. **Duplikasyon**: Aynı maç için 7 gün içinde tekrar tahmin oluşturmaz
4. **Log Takibi**: Console'da detaylı loglar var

## Log Örnekleri

```
[AUTO_PRED] Otomatik tahmin başlatıldı: 3 gün
[AUTO_PRED] 45 yaklaşan maç bulundu
[AUTO_PRED] ✓ Eşleşti: Manchester City (8) vs Arsenal (12)
[AUTO_PRED] ✓ Tahmin oluşturuldu: Manchester City vs Arsenal
[AUTO_PRED] ✗ Eşleşmedi: Obscure Team vs Unknown FC
[AUTO_PRED] Tamamlandı: 30 oluşturuldu, 5 atlandı, 35/45 eşleşti
```

## Katkıda Bulunma

Yeni takım eşleştirmeleri eklemek için:
```python
# prediction/team_matcher.py
KNOWN_MAPPINGS = {
    "yeni takım adı": "Canonical Takım Adı",
    # ...
}
```
