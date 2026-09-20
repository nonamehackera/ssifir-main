import logging
from typing import Any

from ingestion.api_football.client import ApiFootballClient
from ingestion.base import FootballDataProvider

logger = logging.getLogger(__name__)


class ApiFootballProvider(FootballDataProvider):
    """API-Football (api-sports.io) v3 provider'ı.

    Not (ROADMAP bölüm 61): canlı veri için polling 15-60 sn bandında
    dinamik olarak ayarlanmalı; canlı odds tarihsel saklanmadığı için
    sistem kendi snapshot arşivini oluşturmalı (bölüm 62).
    """

    source_name = "api_football"

    def __init__(self, client: ApiFootballClient | None = None) -> None:
        self.client = client or ApiFootballClient()

    def get_leagues(self, **kwargs) -> list[dict[str, Any]]:
        params = {k: v for k, v in kwargs.items() if v is not None}
        return self.client.get_list("/leagues", params)

    def get_seasons(self, league_id: int, **kwargs) -> list[dict[str, Any]]:
        params = {"id": league_id}
        params.update({k: v for k, v in kwargs.items() if v is not None})
        data = self.client.get_list("/leagues", params)
        seasons: list[dict[str, Any]] = []
        for league in data:
            for season in league.get("seasons", []):
                seasons.append(
                    {
                        "league_id": league["league"]["id"],
                        "year": season["year"],
                        "start": season.get("start"),
                        "end": season.get("end"),
                    }
                )
        return seasons

    def get_teams(self, league_id: int, season: int, **kwargs) -> list[dict[str, Any]]:
        params = {"league": league_id, "season": season}
        params.update({k: v for k, v in kwargs.items() if v is not None})
        data = self.client.get_list("/teams", params)
        result: list[dict[str, Any]] = []
        for item in data:
            team = item.get("team", {})
            venue = item.get("venue", {})
            result.append(
                {
                    "id": team["id"],
                    "name": team.get("name"),
                    "code": team.get("code"),
                    "country": team.get("country"),
                    "founded": team.get("founded"),
                    "logo": team.get("logo"),
                    "venue_id": venue.get("id"),
                    "venue_name": venue.get("name"),
                    "venue_city": venue.get("city"),
                    "venue_capacity": venue.get("capacity"),
                    "venue_surface": venue.get("surface"),
                }
            )
        return result

    def get_fixtures(self, league_id: int, season: int, **kwargs) -> list[dict[str, Any]]:
        params = {"league": league_id, "season": season}
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return self.client.get_list("/fixtures", params)

    def get_fixtures_by_date(self, date_str: str, **kwargs) -> list[dict[str, Any]]:
        """Belirli bir tarihteki tüm maçları çeker (YYYY-AA-GG)."""
        params = {"date": date_str}
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return self.client.get_list("/fixtures", params)

    def get_fixtures_live(self, **kwargs) -> list[dict[str, Any]]:
        """Şu an devam eden tüm canlı maçları çeker (API 'live' all seçeneği)."""
        params = {"live": "all"}
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return self.client.get_list("/fixtures", params)

    def get_fixtures_by_status(self, status: str, **kwargs) -> list[dict[str, Any]]:
        """Belirli bir kısa/uzun durum kodu ile maçları çeker
        (örn: 'NS' = Not Started, '1H' = First Half, '2H' = Second Half,
        'HT' = Halftime, 'FT' = Finished, 'AET', 'PEN')."""
        params = {"status": status}
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return self.client.get_list("/fixtures", params)

    def get_fixture_by_id(self, fixture_id: int) -> dict[str, Any] | None:
        """Tek bir maçı ID ile çeker."""
        items = self.client.get_list("/fixtures", {"id": fixture_id})
        return items[0] if items else None

    def get_standings(self, league_id: int, season: int, **kwargs) -> list[dict[str, Any]]:
        params = {"league": league_id, "season": season}
        params.update({k: v for k, v in kwargs.items() if v is not None})
        data = self.client.get_list("/standings", params)
        rows: list[dict[str, Any]] = []
        for item in data:
            league = item.get("league", {})
            for standing_list in league.get("standings", []):
                for row in standing_list:
                    rows.append(
                        {
                            "league_id": league["id"],
                            "season": item.get("league", {}).get("season"),
                            "team_id": row["team"]["id"],
                            "position": row.get("rank"),
                            "points": row.get("points"),
                            "played": row.get("all", {}).get("played"),
                        }
                    )
        return rows

    def get_statistics(self, fixture_id: int, **kwargs) -> dict[str, Any]:
        params = {"fixture": fixture_id}
        params.update({k: v for k, v in kwargs.items() if v is not None})
        data = self.client.get_list("/fixtures/statistics", params)
        return {"fixture_id": fixture_id, "teams": data}

    def get_odds(self, fixture_id: int, **kwargs) -> list[dict[str, Any]]:
        params = {"fixture": fixture_id}
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return self.client.get_list("/odds", params)

    def get_lineups(self, fixture_id: int, **kwargs) -> list[dict[str, Any]]:
        params = {"fixture": fixture_id}
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return self.client.get_list("/fixtures/lineups", params)

    def get_injuries(self, fixture_id: int, **kwargs) -> list[dict[str, Any]]:
        params = {"fixture": fixture_id}
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return self.client.get_list("/injuries", params)