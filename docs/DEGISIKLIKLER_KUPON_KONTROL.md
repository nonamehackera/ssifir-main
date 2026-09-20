# Kupon Kontrol Sistemi İyileştirmeleri

## Problem
- 3 gün önce oynanan maçlar "SONUÇ BEKLENİYOR" olarak gösteriliyordu
- Sistem yalnızca bugünkü maçları Flashscore'dan kontrol ediyordu
- Geçmiş maçlar için canlı kontrol çalışmıyordu
- Maçları kafasından buluyordu ama sonuçları kaydetmiyordu

## Çözüm

### 1. Canlı Maç Kontrolü İyileştirildi (`_check_match_liveness`)
**Dosya:** `web/app.py`

- ✅ Son 7 gün içindeki **TÜM** maçlar için canlı ve bitmiş maç kontrolü
- ✅ Bitmiş maçlar cache'inden (`_FLASHSCORE_FINISHED_CACHE`) de arama
- ✅ Daha güçlü takım adı eşleştirme (kelime bazlı kontrol)
- ✅ Maç tarihi eşleştirme toleransı (±2 gün)

**Değişiklikler:**
```python
# ÖNCE: Sadece bugün için kontrol
elif days_diff == 0:
    # Bugun - Flashscore cache'inden canli maclari kontrol et

# SONRA: Son 7 gün için kontrol
elif days_diff >= 0 and days_diff <= 7:
    # Son 7 gun icindeki maclar - hem canli hem bitmis maclari kontrol et
    live_matches = _FLASHSCORE_LIVE_CACHE or []
    finished_matches = _FLASHSCORE_FINISHED_CACHE or []
```

### 2. Flashscore Cache Kapasitesi Artırıldı (`_refresh_flashscore_cache`)
**Dosya:** `web/app.py`

- ✅ Bitmiş maçlar limiti: **200 → 1000**
- ✅ Daha fazla geçmiş maç cache'leniyor

```python
# ÖNCE
_FLASHSCORE_FINISHED_CACHE = get_fixtures_flashscore(kind="finished", limit=200) or []

# SONRA
_FLASHSCORE_FINISHED_CACHE = get_fixtures_flashscore(kind="finished", limit=1000) or []
```

### 3. Maç Sonucu Arama Optimize Edildi (`_find_match_result`)
**Dosya:** `web/app.py`

- ✅ Flashscore bitmiş maçları **cache'den** alınıyor (her seferinde yeni API çağrısı yok)
- ✅ Daha güçlü takım adı eşleştirme
- ✅ Hata logları eklendi

```python
# ÖNCE: Her seferinde API çağrısı
from prediction.flashscore_scraper import get_fixtures_flashscore
finished = get_fixtures_flashscore(kind="finished", limit=10000) or []

# SONRA: Cache'den al
finished = _FLASHSCORE_FINISHED_CACHE or []
```

### 4. Otomatik Sonuç Kaydetme Eklendi (`api_check_live_coupons`)
**Dosya:** `web/app.py`

- ✅ Bulunan sonuçlar otomatik olarak `match_results.json`'a kaydediliyor
- ✅ Log mesajları: kontrol edilen, bulunan ve kaydedilen maç sayıları
- ✅ Response'a `saved` parametresi eklendi
- ✅ Performans optimizasyonu: önce hızlı kontrol, sonra tam arama

```python
# Sonucu kaydet (match_results.json'a)
cid = f"{hid}-{aid}"
score_fetcher.save_local_match_result(cid, {
    "home_goals": actual.get("home_goals"),
    "away_goals": actual.get("away_goals"),
    # ...
})
saved += 1
```

### 5. Frontend İyileştirmeleri
**Dosya:** `web/templates/coupons.html`

- ✅ Otomatik canlı kontrol: **30 saniye → 2 dakika** (daha performanslı)
- ✅ Kaydedilen sonuç sayısı gösterimi
- ✅ Daha iyi kullanıcı geri bildirimleri

```javascript
// ÖNCE: 30 saniye
setInterval(()=>{checkLiveCoupons();},30000);

// SONRA: 2 dakika
setInterval(()=>{checkLiveCoupons();},120000);
```

## Nasıl Kullanılır

1. **Otomatik:** Sistem her 2 dakikada bir otomatik olarak canlı kontrol yapar
2. **Manuel:** "Canlı Kontrol" butonuna tıklayarak manuel kontrol yapabilirsiniz
3. **Sonuçlar:** Bulunan sonuçlar otomatik olarak kaydedilir ve kupon durumu güncellenir

## Beklenen Sonuçlar

- ✅ 3 gün önce oynanan maçlar artık "TUTTU/TUTMADI" olarak gösterilecek
- ✅ Geçmiş maçlar Flashscore bitmiş maçlar cache'inden bulunacak
- ✅ Bulunan sonuçlar otomatik kaydedilecek, bir daha aranmayacak
- ✅ Sistem daha hızlı ve performanslı çalışacak

## Test

Servisi yeniden başlatın:
```bash
# Servisi durdur
taskkill /F /IM python.exe

# Servisi başlat
cd c:\Users\furka\Desktop\ssifir-main
.venv_run\Scripts\activate.bat
python web\app.py
```

Web arayüzünde:
1. Kuponlar sayfasına gidin
2. "Canlı Kontrol" butonuna tıklayın
3. Bekleyen kuponların sonuçlarını kontrol edin
4. Console log'larına bakın (kaç maç bulundu/kaydedildi)
