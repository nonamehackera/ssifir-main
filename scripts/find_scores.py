"""
Tahminlere özel skor arayıcı — match_score_fetcher ile arar ve match_results.json dosyasına kaydeder.
"""
import os
import json
import prediction.match_score_fetcher as score_fetcher

def main():
    preds_file = os.path.join("tahminler", "predictions.json")
    results_file = os.path.join("tahminler", "match_results.json")
    
    if not os.path.exists(preds_file):
        print(f"Hata: {preds_file} dosyasi bulunamadi!")
        return

    with open(preds_file, encoding="utf-8") as f:
        preds = json.load(f)
    
    results = score_fetcher.load_local_match_results()
    
    print("=== TAHMİN SKOR ARAYICI & SENKRONİZASYON ===")
    print(f"Toplam tahmin: {len(preds)}")
    print(f"Mevcut kayıtlı sonuç: {len(results)}\n")
    
    found_new = 0
    for i, p in enumerate(preds):
        hid = p.get("home_id")
        aid = p.get("away_id")
        cid = f"{hid}-{aid}"
        hname = p.get("home_name", "?")
        aname = p.get("away_name", "?")
        mdate = p.get("match_date", "")
        
        if cid in results and isinstance(results[cid], dict) and results[cid].get("home_goals") is not None:
            r = results[cid]
            hg = r.get("home_goals")
            ag = r.get("away_goals")
            src = r.get("source", "cache")
            print(f"  {i+1:2d}. [{mdate}] {hname} vs {aname} -> {hg}-{ag} ({src}) [OK]")
            continue
        
        print(f"  {i+1:2d}. [{mdate}] {hname} vs {aname} -> Aranıyor...", end="", flush=True)
        res = score_fetcher.get_match_result(hid, aid, hname, aname, mdate)
        
        if res:
            hg = res["home_goals"]
            ag = res["away_goals"]
            src = res.get("source", "web")
            print(f" BULUNDU: {hg}-{ag} ({src})")
            found_new += 1
        else:
            print(" BULUNAMADI")
    
    updated_results = score_fetcher.load_local_match_results()
    print("\n=== ÖZET ===")
    print(f"Toplam Tahmin: {len(preds)}")
    print(f"Tamamlanan Skorlar: {len(updated_results)}")
    print(f"Yeni Bulunan: {found_new}")
    print(f"Eksik Kalan: {len(preds) - len(updated_results)}")

if __name__ == "__main__":
    main()
