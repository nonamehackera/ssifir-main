# 📅 KUPON KONTROL: 7 GÜN → 14 GÜN (2 HAFTA)

## 🎯 DEĞİŞİKLİK:

**Kupon kontrol sistemi artık son 2 HAFTAYI (14 gün) tarıyor!**

### ÖNCE:
- ❌ Son 7 gün (1 hafta)
- ❌ 8 gün önceki maçlar "old" (eski) sayılıyordu
- ❌ Kontrol edilmiyordu

### SONRA:
- ✅ Son 14 gün (2 hafta)
- ✅ 14 gün içindeki tüm maçlar kontrol edilecek
- ✅ Daha fazla maç sonucu bulunacak!

---

## 🔧 YAPILAN DEĞİŞİKLİKLER

### 1. `_check_match_liveness()` - Maç Canlılık Kontrolü

**Satır: ~2763-2820**

```python
# ESKİ:
elif days_diff >= 0 and days_diff <= 7:
    # Son 7 gun icindeki maclar

# YENİ:
elif days_diff >= 0 and days_diff <= 14:
    # Son 14 gun (2 hafta) icindeki maclar
```

**Durum Etiketleri:**
```python
if days_diff == 0:
    return "today"           # Bugün
elif days_diff == 1:
    return "yesterday"       # Dün
elif days_diff <= 7:
    return "recent"          # Son hafta (2-7 gün)
elif days_diff <= 14:
    return "last_week"       # 1-2 hafta arası (8-14 gün)
else:
    return "old"             # 2 haftadan eski (15+ gün)
```

### 2. Flashscore Finished Match Eşleştirme

**Satır: ~2810**

```python
# ESKİ: Mac tarihi eslesmesi (mac_date +/- 2 gun toleransi)
if abs((ko_dt.date() - md.date()).days) <= 2:

# YENİ: Mac tarihi eslesmesi (mac_date +/- 7 gun toleransi)
if abs((ko_dt.date() - md.date()).days) <= 7:
```

**Sebep:** Flashscore'daki maç tarihi predictions.json'daki tahmin tarihinden birkaç gün farklı olabilir (saat dilimi, yanlış tarih, vb.)

### 3. `_find_match_result()` - Sonuç Bulma Fonksiyonu

**Satır: ~1789**

```python
# ESKİ:
def _date_close(source_date, max_days=10):

# YENİ:
def _date_close(source_date, max_days=14):
```

**Satır: ~2013**

```python
# ESKİ:
if date_diff > 7:  # 7 günden fazla fark varsa atla

# YENİ:
if date_diff > 14:  # 14 günden fazla fark varsa atla
```

**Satır: ~1955**

```python
# ESKİ:
# 4.2. Flashscore bitmis mac sonuclari (cache'den, en yakin tarihi tercih et, maks 7 gun)

# YENİ:
# 4.2. Flashscore bitmis mac sonuclari (cache'den, en yakin tarihi tercih et, maks 14 gun / 2 hafta)
```

### 4. `_find_existing_prediction()` - Mevcut Tahmin Kontrolü

**Satır: ~1597**

```python
# ESKİ:
if (datetime.now() - pred_time).days < 7:  # Son 7 gün

# YENİ:
if (datetime.now() - pred_time).days < 14:  # Son 14 gün (2 hafta)
```

**Amaç:** Otomatik tahmin sistemi aynı maç için 14 gün içinde tekrar tahmin oluşturmasın.

---

## 📊 ETKİ ANALİZİ

### Senaryo 1: 10 Gün Önceki Maç

```
Maç Tarihi: 27 Ağustos 2026
Bugün: 6 Eylül 2026
Fark: 10 gün

ÖNCE:
- days_diff = 10 > 7 → "old"
- Kontrol edilmedi ❌

SONRA:
- days_diff = 10 <= 14 → "last_week"
- Flashscore'da aranır ✅
- Sonuç bulunur ✅
```

### Senaryo 2: 3 Gün Önceki Maç

```
Maç Tarihi: 3 Eylül 2026
Bugün: 6 Eylül 2026
Fark: 3 gün

ÖNCE:
- days_diff = 3 <= 7 → "recent"
- Kontrol edilir ✅

SONRA:
- days_diff = 3 <= 7 → "recent"
- Kontrol edilir ✅
- (Değişmedi, zaten çalışıyordu)
```

### Senaryo 3: 15 Gün Önceki Maç

```
Maç Tarihi: 22 Ağustos 2026
Bugün: 6 Eylül 2026
Fark: 15 gün

ÖNCE:
- days_diff = 15 > 7 → "old"
- Kontrol edilmedi ❌

SONRA:
- days_diff = 15 > 14 → "old"
- Kontrol edilmeyecek ❌
- (Flashscore cache'i zaten 30 gün, ama 15+ gün eskisi "old" sayılır)
```

---

## 🚀 PERFORMANS

### Flashscore Cache:
```python
# Zaten 30 gün veri çekiyordu (değişiklik YOK)
_FLASHSCORE_FINISHED_CACHE = get_fixtures_flashscore(kind="finished", limit=1000)
# 30 günlük bitmiş maçlar (flashscore_scraper.py içinde range(-30, 1))
```

### Kontrol Kapsamı:
```
ÖNCE: Son 7 gün → ~50-100 maç kontrol
SONRA: Son 14 gün → ~100-200 maç kontrol

Flashscore cache: 30 gün → ~1000 maç
Yeterli veri var! ✅
```

### Süre:
```
İlk çalıştırma: 2-3 dakika (değişmedi)
İkinci çalıştırma: 60 saniye (cache kullanır)
```

---

## 📝 ÖZET

### Değiştirilen Değerler:

| Parametre | ÖNCE | SONRA | Sebep |
|-----------|------|-------|-------|
| `days_diff <= X` | 7 | 14 | 2 hafta kontrol |
| `max_days` | 10 | 14 | Tarih toleransı |
| `date_diff > X` | 7 | 14 | Eşleştirme aralığı |
| `kickoff toleransı` | ±2 gün | ±7 gün | Tarih esnekliği |
| `pred_time days <` | 7 | 14 | Tekrar tahmin önleme |

### Durum Etiketleri:

| Gün Farkı | Durum | Açıklama |
|-----------|-------|----------|
| -2+ | `future` | Gelecek |
| -1 | `tomorrow` | Yarın |
| 0 | `today` | Bugün |
| 1 | `yesterday` | Dün |
| 2-7 | `recent` | Son hafta |
| 8-14 | `last_week` | 1-2 hafta arası **[YENİ]** |
| 15+ | `old` | Eski (kontrol edilmez) |

---

## ✅ SONUÇ:

**ARTIK 2 HAFTAYA KADAR (14 GÜN) GERİYE GİDİP MAÇ SONUÇLARINI BULACAK!**

### Kullanıcı İçin:
```
1. Web server'ı yeniden başlat:
   baslat.bat

2. Tarayıcıda:
   http://localhost:5000/coupons
   → "Canlı Kontrol" tıkla

3. Bekle (2-3 dakika ilk çalıştırma)

4. Sonuçları gör:
   - 3 gün önceki maçlar ✅
   - 7 gün önceki maçlar ✅
   - 10 gün önceki maçlar ✅ [YENİ!]
   - 14 gün önceki maçlar ✅ [YENİ!]
```

**DAHA FAZLA MAÇ SONUCU BULUNACAK!** 🎯
