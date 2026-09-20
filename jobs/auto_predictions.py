"""Otomatik Tahmin Oluşturma Sistemi

Yaklaşan maçlar için otomatik tahmin oluşturur.
Flashscore'dan yaklaşan maçları çeker, takım ID'lerini eşleştirir
ve gerçek verilerle tahmin yapar.

Kullanım:
    python -m jobs.auto_predictions           # Tek seferlik çalıştır
    python -m jobs.auto_predictions --days 3   # 3 gün için tahmin oluştur
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [AUTO_PRED]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("auto_predictions")


def load_team_id_map():
    """TEAM_ID_MAP'i yükle."""
    try:
        import pickle
        team_index_file = Path("data/gold/team_index.pkl")
        if not team_index_file.exists():
            logger.error("team_index.pkl bulunamadı!")
            return {}
        
        with open(team_index_file, "rb") as f:
            team_index = pickle.load(f)
        
        # Convert to TEAM_ID_MAP format
        team_id_map = {}
        for tid, info in team_index.items():
            team_id_map[tid] = {
                "name": info.get("name", ""),
                "league": info.get("league", ""),
                "league_code": info.get("league_code", ""),
                "elo": info.get("elo", 1500),
                "attack_elo": info.get("attack_elo", 1500),
                "defence_elo": info.get("defence_elo", 1500),
            }
        
        logger.info(f"TEAM_ID_MAP yüklendi: {len(team_id_map)} takım")
        return team_id_map
    except Exception as e:
        logger.error(f"TEAM_ID_MAP yüklenirken hata: {e}")
        return {}


def get_upcoming_matches(days: int = 3):
    """Yaklaşan maçları Flashscore'dan çek."""
    try:
        from prediction.flashscore_scraper import get_fixtures_flashscore
        
        logger.info(f"Yaklaşan {days} gün için maçlar çekiliyor...")
        
        all_matches = []
        for day in range(days + 1):
            logger.info(f"Gün {day} maçları çekiliyor...")
            matches = get_fixtures_flashscore(kind="upcoming", limit=1000, refresh=True)
            if matches:
                # Sadece bugünden sonraki maçları al
                today = datetime.now(timezone.utc).date()
                target_date = today + timedelta(days=day)
                
                for match in matches:
                    kickoff = match.get("kickoff", "")
                    if kickoff:
                        try:
                            match_date = datetime.fromisoformat(kickoff.replace("Z", "")).date()
                            if match_date == target_date:
                                all_matches.append(match)
                        except:
                            pass
            
            time.sleep(2)  # Rate limiting
        
        logger.info(f"Toplam {len(all_matches)} yaklaşan maç bulundu")
        return all_matches
    except Exception as e:
        logger.error(f"Maç çekme hatası: {e}")
        return []


def match_teams(home_name: str, away_name: str, team_id_map: dict):
    """Takım isimlerini ID'lerle eşleştir."""
    from prediction.team_matcher import find_team_ids
    
    home_id, away_id = find_team_ids(home_name, away_name, team_id_map)
    
    if home_id and away_id:
        logger.info(f"✓ Eşleşti: {home_name} ({home_id}) vs {away_name} ({away_id})")
        return home_id, away_id
    elif home_id:
        logger.warning(f"⚠ Kısmi eşleşme: {home_name} ({home_id}) bulundu, {away_name} bulunamadı")
        return None, None
    elif away_id:
        logger.warning(f"⚠ Kısmi eşleşme: {away_name} ({away_id}) bulundu, {home_name} bulunamadı")
        return None, None
    else:
        logger.warning(f"✗ Eşleşmedi: {home_name} vs {away_name}")
        return None, None


def create_prediction(home_id: int, away_id: int, match_date: str, home_name: str, away_name: str):
    """Tahmin oluştur ve kaydet."""
    try:
        import requests
        
        # Web API'ye tahmin isteği gönder
        response = requests.post(
            "http://localhost:5011/api/predict",
            json={"home_id": home_id, "away_id": away_id},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✓ Tahmin oluşturuldu: {home_name} vs {away_name}")
            logger.info(f"  1X2: {result.get('home_win', 0):.1f}% / {result.get('draw', 0):.1f}% / {result.get('away_win', 0):.1f}%")
            logger.info(f"  Kazanan: {result.get('winner', '?')} ({result.get('winner_prob', 0):.1f}%)")
            return True
        else:
            logger.error(f"✗ Tahmin hatası: HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"✗ Tahmin oluşturma hatası: {e}")
        return False


def run_auto_predictions(days: int = 3, dry_run: bool = False):
    """
    Otomatik tahmin oluşturma ana fonksiyonu.
    
    Args:
        days: Kaç gün sonrasına kadar tahmin oluşturulsun
        dry_run: True ise sadece eşleşmeleri gösterir, tahmin oluşturmaz
    """
    logger.info("=" * 60)
    logger.info("OTOMATİK TAHMİN SİSTEMİ BAŞLADI")
    logger.info("=" * 60)
    
    # 1. TEAM_ID_MAP'i yükle
    team_id_map = load_team_id_map()
    if not team_id_map:
        logger.error("TEAM_ID_MAP yüklenemedi, işlem durduruluyor!")
        return
    
    # 2. Yaklaşan maçları çek
    upcoming_matches = get_upcoming_matches(days=days)
    if not upcoming_matches:
        logger.warning("Yaklaşan maç bulunamadı!")
        return
    
    # 3. Her maç için tahmin oluştur
    total = len(upcoming_matches)
    matched = 0
    predicted = 0
    failed = 0
    
    logger.info(f"\n{total} maç için tahmin oluşturuluyor...\n")
    
    for i, match in enumerate(upcoming_matches, 1):
        home_name = match.get("home_name", "")
        away_name = match.get("away_name", "")
        kickoff = match.get("kickoff", "")
        league = match.get("league_name", "")
        
        logger.info(f"[{i}/{total}] {home_name} vs {away_name}")
        logger.info(f"  Lig: {league}")
        logger.info(f"  Tarih: {kickoff}")
        
        # Takımları eşleştir
        home_id, away_id = match_teams(home_name, away_name, team_id_map)
        
        if home_id and away_id:
            matched += 1
            
            if not dry_run:
                # Tahmin oluştur
                success = create_prediction(home_id, away_id, kickoff, home_name, away_name)
                if success:
                    predicted += 1
                else:
                    failed += 1
                
                # Rate limiting
                time.sleep(1)
        
        logger.info("")
    
    # 4. Özet
    logger.info("=" * 60)
    logger.info("ÖZET")
    logger.info("=" * 60)
    logger.info(f"Toplam maç: {total}")
    logger.info(f"Eşleşen maç: {matched} (%{matched/total*100:.1f})")
    
    if not dry_run:
        logger.info(f"Tahmin oluşturulan: {predicted}")
        logger.info(f"Başarısız: {failed}")
    else:
        logger.info("DRY RUN - Tahmin oluşturulmadı")
    
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Otomatik tahmin oluşturma sistemi")
    parser.add_argument("--days", type=int, default=3, help="Kaç gün sonrasına kadar tahmin oluşturulsun (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Sadece eşleşmeleri göster, tahmin oluşturma")
    
    args = parser.parse_args()
    
    run_auto_predictions(days=args.days, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
