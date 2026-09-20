"""1. The Odds API - ucretsiz 500 req/ay, tarihsel oranlar."""
import os, json, urllib.request, urllib.error
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

CACHE = Path("data/raw/theoddsapi/odds_cache.parquet")
API_KEY = os.environ.get("ODDS_API_KEY", "")

# Bahis piyasalari
SPORTS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_turkey_super_lig",
    "soccer_netherlands_eredivisie",
    "soccer_portugal_liga_portugal",
    "soccer_belgium_jupiler_league",
    "soccer_scotland_premiership",
    "soccer_brazil_campeonato",
    "soccer_usa_mls",
    "soccer_japan_j_league",
    "soccer_korea_kleague1",
]

SPORT_MAP = {
    "soccer_epl": "E0", "soccer_spain_la_liga": "SP1",
    "soccer_germany_bundesliga": "D1", "soccer_italy_serie_a": "I1",
    "soccer_france_ligue_one": "F1", "soccer_turkey_super_lig": "T1",
    "soccer_netherlands_eredivisie": "N1", "soccer_portugal_liga_portugal": "P1",
    "soccer_belgium_jupiler_league": "B1", "soccer_scotland_premiership": "SC0",
    "soccer_brazil_campeonato": "BR1", "soccer_usa_mls": "US1",
    "soccer_japan_j_league": "JP1", "soccer_korea_kleague1": "KR1",
}

def fetch_odds_sport(sport: str, api_key: str, days_back: int = 30) -> list[dict]:
    """Tek spor icin tarihsel oranlari cek."""
    # Tum tarih araligini cek
    all_odds = []
    current = datetime.now()
    start = current - timedelta(days=days_back)

    # Her hafta icin cek (API limiti: 1 gun = 1 request)
    for day_offset in range(days_back):
        dt = start + timedelta(days=day_offset)
        date_str = dt.strftime("%Y-%m-%d")
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds-history/?apiKey={api_key}&date={date_str}&regions=eu,uk&markets=h2h,totals"
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read())
                for game in data:
                    for bookmaker in game.get("bookmakers", []):
                        for market in bookmaker.get("markets", []):
                            if market["key"] == "h2h":
                                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                                all_odds.append({
                                    "sport": sport,
                                    "date": game.get("commence_time", ""),
                                    "home_team": game.get("home_team", ""),
                                    "away_team": game.get("away_team", ""),
                                    "bookmaker": bookmaker["key"],
                                    "home_odds": outcomes.get(game.get("home_team")),
                                    "draw_odds": outcomes.get("Draw"),
                                    "away_odds": outcomes.get(game.get("away_team")),
                                })
                            elif market["key"] == "totals":
                                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                                # totals market: e.g. "Over 2.5" / "Under 2.5"
                                for o in market["outcomes"]:
                                    if "Over" in o.get("name", ""):
                                        all_odds.append({
                                            "sport": sport,
                                            "date": game.get("commence_time", ""),
                                            "home_team": game.get("home_team", ""),
                                            "away_team": game.get("away_team", ""),
                                            "bookmaker": bookmaker["key"],
                                            "over25_odds": o["price"],
                                        })
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"  Hata {date_str}: {e}")
            continue

    return all_odds


def main():
    if not API_KEY:
        print("ODDS_API_KEY yok! .env dosyasina ekle.")
        print("Ucretsiz kayit: https://the-odds-api.com/")
        print("500 request/ay ucretsiz.")
        return

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    all_data = []

    for sport in SPORTS:
        print(f"Cekiliyor: {sport}...")
        odds = fetch_odds_sport(sport, API_KEY, days_back=30)
        all_data.extend(odds)
        print(f"  {len(odds)} kayit")

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_parquet(CACHE, index=False)
        print(f"\nToplam: {len(df)} kayit -> {CACHE}")
    else:
        print("Veri cekilemedi.")


if __name__ == "__main__":
    main()
