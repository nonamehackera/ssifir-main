"""Takım Adı Eşleştirme Modülü

Flashscore, API-Football ve diğer kaynaklardan gelen takım isimlerini
yerel veritabanındaki takım ID'leriyle eşleştirir.

Kullanım:
    from prediction.team_matcher import find_team_id
    
    team_id = find_team_id("Manchester City")
    if team_id:
        print(f"Takım bulundu: {team_id}")
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional


def normalize_team_name(name: str) -> str:
    """Takım adını normalize et (Türkçe karakterler, büyük/küçük harf, boşluklar)."""
    if not name:
        return ""
    
    # Türkçe karakterleri normalize et - GENİŞLETİLMİŞ
    tr_map = str.maketrans(
        "çğıöşüÇĞİÖŞÜâîûÂÎÛéàèùäëïöüÄËÏÖÜ",
        "cgiosuCGIOSUaiuAIUeaeUaeiouAEIOU"
    )
    normalized = name.translate(tr_map).lower().strip()
    
    # NOT: Gereksiz kelimeleri KALDIRMAYIN! 
    # "Manchester City" -> "manchester city" (city kalmalı, United ile karışmasın)
    # "Çaykur Rizespor" -> "caykur rizespor" (caykur kalmalı, tanıma için)
    
    # Sadece çoklu boşlukları tek boşluğa çevir
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def calculate_similarity(name1: str, name2: str) -> float:
    """İki takım adı arasındaki benzerlik oranını hesapla (0-1).
    
    AKILLI EŞLEŞTİRME - Kupon kontrol sistemiyle aynı mantık:
    1. Tam eşleşme → 1.0
    2. Uzun string içerme (5+ kar) → 0.95
    3. Kelime bazlı eşleşme (2+ ortak kelime) → 0.85-0.95
    4. Sequence matcher → 0.0-0.85
    """
    n1 = normalize_team_name(name1)
    n2 = normalize_team_name(name2)
    
    if not n1 or not n2:
        return 0.0
    
    # 1. Tam eşleşme
    if n1 == n2:
        return 1.0
    
    # 2. Uzun string içerme (5+ karakter)
    if (n1 in n2 and len(n1) > 5) or (n2 in n1 and len(n2) > 5):
        return 0.95
    
    # 3. Kelime bazlı eşleşme (4+ karakter kelimeleri say)
    words1 = set([w for w in n1.split() if len(w) > 3])
    words2 = set([w for w in n2.split() if len(w) > 3])
    
    if words1 and words2:
        common_words = words1 & words2
        if common_words:
            # En az 2 ortak kelime VEYA tek kelime takımsa (Galatasaray gibi)
            min_words_needed = min(2, min(len(words1), len(words2)))
            if len(common_words) >= min_words_needed:
                word_score = len(common_words) / max(len(words1), len(words2))
                return 0.85 + (word_score * 0.10)  # 0.85-0.95 arası
    
    # 4. Sequence matcher (fallback)
    seq_score = SequenceMatcher(None, n1, n2).ratio()
    # Sequence matcher'ı düşük tut (max 0.80) ki kelime bazlı üstün olsun
    return min(seq_score, 0.80)


# Bilinen takım eşleştirmeleri (manuel override)
KNOWN_MAPPINGS = {
    # İngiltere
    "manchester city": "Manchester City",
    "man city": "Manchester City",
    "manchester united": "Manchester United",
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "tottenham": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "newcastle united": "Newcastle",
    "newcastle": "Newcastle",
    "brighton": "Brighton",
    "brighton & hove albion": "Brighton",
    "west ham united": "West Ham",
    "west ham": "West Ham",
    "wolverhampton": "Wolverhampton",
    "wolves": "Wolverhampton",
    "nottingham forest": "Nottingham Forest",
    "nottm forest": "Nottingham Forest",
    "sheffield united": "Sheffield United",
    "sheffield utd": "Sheffield United",
    "leeds united": "Leeds United",
    "leeds": "Leeds United",
    
    # Türkiye - GENİŞLETİLMİŞ
    "galatasaray": "Galatasaray",
    "gala": "Galatasaray",
    "gs": "Galatasaray",
    "fenerbahce": "Fenerbahçe",
    "fenerbahçe": "Fenerbahçe",
    "fener": "Fenerbahçe",
    "fb": "Fenerbahçe",
    "besiktas": "Beşiktaş",
    "beşiktaş": "Beşiktaş",
    "bjk": "Beşiktaş",
    "trabzonspor": "Trabzonspor",
    "trabzon": "Trabzonspor",
    "ts": "Trabzonspor",
    "istanbul basaksehir": "İstanbul Başakşehir",
    "basaksehir": "İstanbul Başakşehir",
    "başakşehir": "İstanbul Başakşehir",
    "ibfk": "İstanbul Başakşehir",
    "caykur rizespor": "Çaykur Rizespor",
    "rizespor": "Çaykur Rizespor",
    "rize": "Çaykur Rizespor",
    "alanyaspor": "Alanyaspor",
    "alanya": "Alanyaspor",
    "goztepe": "Göztepe",
    "göztepe": "Göztepe",
    "goztep": "Göztepe",
    "gaziantepspor": "Gaziantepspor",
    "gaziantep": "Gaziantepspor",
    "gaziantep fk": "Gaziantepspor",
    "konyaspor": "Konyaspor",
    "konya": "Konyaspor",
    "antalyaspor": "Antalyaspor",
    "antalya": "Antalyaspor",
    "sivasspor": "Sivasspor",
    "sivas": "Sivasspor",
    "kasimpasa": "Kasımpaşa",
    "kasımpaşa": "Kasımpaşa",
    "karagumruk": "Fatih Karagümrük",
    "fatih karagumruk": "Fatih Karagümrük",
    "fatih karagümrük": "Fatih Karagümrük",
    "genclerbirligi": "Gençlerbirliği",
    "gençlerbirliği": "Gençlerbirliği",
    "gencler": "Gençlerbirliği",
    "eyupspor": "Eyüpspor",
    "eyüpspor": "Eyüpspor",
    "eyup": "Eyüpspor",
    "corum fk": "Çorum FK",
    "çorum fk": "Çorum FK",
    "corum": "Çorum FK",
    "hatayspor": "Hatayspor",
    "hatay": "Hatayspor",
    "kayserispor": "Kayserispor",
    "kayseri": "Kayserispor",
    "samsunspur": "Samsunspor",
    "samsun": "Samsunspor",
    "adana demirspor": "Adana Demirspor",
    "adana": "Adana Demirspor",
    "erzurum bb": "Erzurum BB",
    "erzurumspor": "Erzurum BB",
    "erzurum": "Erzurum BB",
    "bodrumspor": "Bodrumspor",
    "bodrum": "Bodrumspor",
    
    # Avrupa
    "bayern munich": "Bayern Munich",
    "bayern munchen": "Bayern Munich",
    "bayern": "Bayern Munich",
    "borussia dortmund": "Borussia Dortmund",
    "dortmund": "Borussia Dortmund",
    "bvb": "Borussia Dortmund",
    "real madrid": "Real Madrid",
    "barcelona": "Barcelona",
    "barca": "Barcelona",
    "atletico madrid": "Atletico Madrid",
    "atleti": "Atletico Madrid",
    "inter milan": "Inter Milan",
    "inter": "Inter Milan",
    "ac milan": "AC Milan",
    "milan": "AC Milan",
    "juventus": "Juventus",
    "juve": "Juventus",
    "psg": "Paris SG",
    "paris saint germain": "Paris SG",
    "paris sg": "Paris SG",
}


def find_team_id(team_name: str, team_id_map: dict, threshold: float = 0.85) -> Optional[int]:
    """
    Takım adından takım ID'sini bul - AKILLI EŞLEŞTİRME.
    
    Args:
        team_name: Aranacak takım adı (örn: "Manchester City", "Çaykur Rizespor")
        team_id_map: TEAM_ID_MAP dict'i (ID -> {name, league, elo, ...})
        threshold: Minimum benzerlik oranı (0-1) - 0.85'e yükseltildi (daha kesin eşleşme)
    
    Returns:
        Takım ID'si veya None
        
    Eşleştirme Mantığı:
        1. Bilinen eşleştirmeler (KNOWN_MAPPINGS)
        2. Tam eşleşme (1.0)
        3. Uzun string içerme (0.95) - "Rizespor" in "Çaykur Rizespor"
        4. Kelime bazlı (0.85-0.95) - 2+ ortak kelime
        5. Sequence matcher (max 0.80) - benzerlik
        
    Threshold 0.85 sayesinde sadece güçlü eşleşmeler kabul edilir.
    """
    if not team_name or not team_id_map:
        return None
    
    # Önce bilinen eşleştirmelere bak
    normalized_query = normalize_team_name(team_name)
    if normalized_query in KNOWN_MAPPINGS:
        canonical_name = KNOWN_MAPPINGS[normalized_query]
        # Canonical name ile tam eşleşme ara
        for tid, info in team_id_map.items():
            if normalize_team_name(info.get("name", "")) == normalize_team_name(canonical_name):
                return tid
    
    # Tüm takımları tara ve en iyi eşleşmeyi bul
    best_match_id = None
    best_score = 0.0
    best_match_name = None
    
    for team_id, info in team_id_map.items():
        db_name = info.get("name", "")
        if not db_name:
            continue
        
        score = calculate_similarity(team_name, db_name)
        
        if score > best_score:
            best_score = score
            best_match_id = team_id
            best_match_name = db_name
    
    # Threshold'u geçen eşleşme varsa döndür
    if best_score >= threshold:
        # Debug log (opsiyonel)
        if best_score < 1.0:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[TeamMatcher] '{team_name}' → '{best_match_name}' (score: {best_score:.2f})")
        return best_match_id
    
    return None


def find_team_ids(home_name: str, away_name: str, team_id_map: dict) -> tuple[Optional[int], Optional[int]]:
    """
    Her iki takımın ID'sini bul.
    
    Args:
        home_name: Ev sahibi takım adı
        away_name: Deplasman takım adı
        team_id_map: TEAM_ID_MAP dict'i
    
    Returns:
        (home_id, away_id) tuple'ı
    """
    home_id = find_team_id(home_name, team_id_map)
    away_id = find_team_id(away_name, team_id_map)
    
    return home_id, away_id


def get_team_name_by_id(team_id: int, team_id_map: dict) -> Optional[str]:
    """Takım ID'sinden takım adını al."""
    if not team_id or not team_id_map:
        return None
    
    info = team_id_map.get(team_id)
    if info:
        return info.get("name")
    
    return None
