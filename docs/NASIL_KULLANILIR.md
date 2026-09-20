# 🚀 OTOMATİK TAHMİN SİSTEMİ - KULLANIM KILAVUZU

## 📌 HIZLI BAŞLANGIÇ

### 1. Web Server'ı Başlat
```cmd
baslat.bat
```
Tarayıcıda otomatik olarak açılacak: http://localhost:5000

### 2. Otomatik Tahmin Oluştur

#### Yöntem A: Web Arayüzünden (Önerilen)
1. http://localhost:5000/predictions adresine git
2. Sağ üstte **"Otomatik Tahmin"** butonuna tıkla
3. Sistem otomatik olarak:
   - Flashscore'dan yaklaşan 3 gün için maçları çeker
   - Takım isimlerini veritabanındaki ID'lerle eşleştirir
   - Her maç için ML modeliyle tahmin oluşturur
   - Tahminleri listeye ekler

#### Yöntem B: Komut Satırından
```cmd
cd c:\Users\furka\Desktop\ssifir-main

# Sadece eşleşmeleri gör (tahmin oluşturma)
python -m jobs.auto_predictions --dry-run --days 1

# Gerçekten tahmin oluştur
python -m jobs.auto_predictions --days 3
```

### 3. Kuponları Kontrol Et
1. http://localhost:5000/coupons adresine git
2. **"Canlı Kontrol"** butonuna tıkla
3. Sistem:
   - Son 7 gündeki tüm maçları Flashscore'da arar
   - Bitmiş maçların sonuçlarını bulur
   - Tahminleri gerçek sonuçlarla karşılaştırır
   - Doğruluk oranını gösterir

## 📊 SİSTEM NASIL ÇALIŞIR?

### Otomatik Tahmin Akışı:

```
1. Flashscore Scraper
   ↓
   Yaklaşan maçları çek (7 gün)
   
2. Team Matcher
   ↓
   Takım isimlerini DB ID'leriyle eşleştir
   - "Manchester City" → 8000017
   - "Galatasaray" → 729
   - %75+ benzerlik oranı
   
3. Prediction Generator
   ↓
   Her maç için ML modeli çalıştır
   - ELO skorları
   - Tarihsel veriler
   - Form analizi
   
4. Sonuç
   ↓
   predictions.json'a kaydet
   - 1X2 tahminleri
   - Gol tahminleri
   - Korner tahminleri
   - En iyi bahis önerileri
```

### Kupon Kontrol Akışı:

```
1. Predictions.json'dan Tahminleri Yükle
   ↓
   
2. Flashscore Cache Oluştur
   ↓
   - Canlı maçlar
   - Son 7 gün bitmiş maçlar (1000 maç)
   
3. Her Tahmin İçin:
   ↓
   - Takım isimlerini eşleştir
   - Flashscore cache'inde ara
   - Sonuç varsa karşılaştır
   
4. Sonuç Görüntüle
   ↓
   - Yeşil: Tuttu
   - Kırmızı: Tutmadı
   - Gri: Sonuç bekleniyor
```

## 🔧 GELİŞMİŞ KULLANIM

### API Endpoint'leri

#### 1. Otomatik Tahmin Oluştur
```bash
curl -X POST http://localhost:5000/api/predictions/auto-generate \
  -H "Content-Type: application/json" \
  -d '{"days": 3}'
```

**Yanıt:**
```json
{
  "success": true,
  "total": 50,          // Bulunan toplam maç
  "matched": 35,        // Eşleşen maç
  "created": 28,        // Oluşturulan tahmin
  "skipped": 7          // Zaten var olan
}
```

#### 2. Kuponları Kontrol Et
```bash
curl -X POST http://localhost:5000/api/coupons/check_live \
  -H "Content-Type: application/json" \
  -d '{"only_pending": true}'
```

#### 3. Tüm Tahminleri Listele
```bash
curl http://localhost:5000/api/predictions
```

### Komut Satırı Parametreleri

```cmd
# 1 gün için dry-run
python -m jobs.auto_predictions --dry-run --days 1

# 7 gün için gerçek tahmin
python -m jobs.auto_predictions --days 7

# Varsayılan: 3 gün
python -m jobs.auto_predictions
```

## 🎨 WEB ARAYÜZÜ

### Tahminler Sayfası (`/predictions`)
- **Otomatik Tahmin Butonu**: Yeni tahminler oluştur
- **Filtreler**: 
  - En yeni / En güvenli / En çok gol
  - Güçlü (50%+) / Orta (35-50%) / Tümü
- **Tahmin Kartları**:
  - Takım isimleri ve ELO skorları
  - 1X2 olasılıkları
  - xG (beklenen gol)
  - Korner tahminleri
  - En iyi bahis önerisi
  - Detaylı analiz (tıklayınca)

### Kuponlar Sayfası (`/coupons`)
- **Canlı Kontrol Butonu**: Flashscore'dan sonuçları çek
- **Otomatik Yenileme**: Her 2 dakikada bir
- **Kupon Kartları**:
  - Maç bilgisi ve tarih
  - Tahmin edilen: 1X2, Gol, BTTS, vb.
  - Gerçek sonuç
  - ✅ Tuttu / ❌ Tutmadı
  - Detaylı piyasa analizi

## ⚙️ YAPILANDIRMA

### Takım Eşleştirmeyi Geliştirme

`prediction/team_matcher.py` dosyasındaki `KNOWN_MAPPINGS` dict'ine yeni takımlar ekle:

```python
KNOWN_MAPPINGS = {
    "man city": "Manchester City",
    "fener": "Fenerbahçe",
    # Yeni ekle:
    "beşiktaş jk": "Beşiktaş",
    "trabzon": "Trabzonspor",
}
```

### Cache Boyutlarını Ayarlama

`web/app.py` dosyasında:

```python
# Flashscore cache limitleri
_FLASHSCORE_LIVE_CACHE_LIMIT = 200      # Canlı maçlar
_FLASHSCORE_FINISHED_CACHE_LIMIT = 1000  # Bitmiş maçlar
```

### Otomatik Yenileme Süresi

`web/templates/coupons.html` dosyasında:

```javascript
// 2 dakika = 120000ms
setTimeout(autoRefresh, 120000);
```

## 🐛 SORUN GİDERME

### "team_index.pkl bulunamadı" Hatası
```cmd
# Dosyayı doğru yere kopyala
copy team_index.pkl data\gold\team_index.pkl
```

### "404 Not Found" - API Endpoint Hatası
```cmd
# Web server'ı yeniden başlat
baslat.bat
```

### Flashscore Scraper Çok Yavaş
- **Normal**: İlk çalıştırmada cache oluşturuluyor (2-3 dakika)
- **Çözüm**: Sonraki çalıştırmalarda cache kullanılıyor, çok daha hızlı

### Takımlar Eşleşmiyor
1. Logları kontrol et: Hangi takımlar eşleşmedi?
2. `prediction/team_matcher.py` dosyasına ekle:
   ```python
   KNOWN_MAPPINGS = {
       "flashscore_ismi": "DB_Ismi",
   }
   ```

### Tahminler Oluşturulmuyor
1. TEAM_ID_MAP'in yüklendiğini kontrol et
2. Model dosyalarının varlığını kontrol et
3. Log dosyalarını incele

## 📈 PERFORMANS İPUÇLARI

### Hızlı Test İçin
```cmd
# Sadece 1 gün, dry-run (hızlı)
python -m jobs.auto_predictions --dry-run --days 1
```

### Haftalık Tahminler İçin
```cmd
# 7 gün için tam tahmin
python -m jobs.auto_predictions --days 7
```

### Otomasyonla Çalıştırma
Windows Task Scheduler ile her gün otomatik çalıştır:
1. Task Scheduler'ı aç
2. Yeni görev oluştur
3. Trigger: Her gün 08:00
4. Action: `python -m jobs.auto_predictions --days 3`
5. Start in: `c:\Users\furka\Desktop\ssifir-main`

## 📝 LOG VE DEBUG

### Log Dosyaları
- Web server logs: Terminal'de görünür
- Auto predictions logs: Komut satırında görünür
- Flashscore cache: `web/app.py` içinde `logger.info()` ile loglanır

### Debug Mode
```python
# web/app.py içinde
logging.basicConfig(level=logging.DEBUG)  # INFO yerine DEBUG
```

## ✅ BAŞARILI KURULUM KONTROL LİSTESİ

- [ ] `baslat.bat` çalışıyor mu?
- [ ] http://localhost:5000 açılıyor mu?
- [ ] Tahminler sayfasında "Otomatik Tahmin" butonu var mı?
- [ ] Butona tıklayınca yeni tahminler oluşuyor mu?
- [ ] Kuponlar sayfasında "Canlı Kontrol" çalışıyor mu?
- [ ] Sonuçlar doğru gösteriliyor mu?

Tüm checkboxlar ✅ ise sistem tamamen çalışır durumda!

## 🎯 SONUÇ

Artık sisteminiz:
- ✅ Flashscore'dan otomatik maç çekebiliyor
- ✅ ML modelleriyle tahmin oluşturabiliyor
- ✅ Gerçek sonuçlarla karşılaştırma yapabiliyor
- ✅ Web arayüzünden tek tıkla kullanılabiliyor

**Keyifli tahminler! 🎲⚽**
