from abc import ABC, abstractmethod
from typing import Any


class FootballDataProvider(ABC):
    """Tüm veri sağlayıcılarının uygulaması gereken arayüz.

    Amaç: ana sağlayıcıyı (API-Football) değiştirmek veya ikincil kaynak
    (Sportmonks) eklemek, ingestion katmanını etkilemeden mümkün olmalı.
    """

    source_name: str = "unknown"

    @abstractmethod
    def get_leagues(self, **kwargs) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_seasons(self, league_id: int, **kwargs) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_teams(self, league_id: int, season: int, **kwargs) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_fixtures(self, league_id: int, season: int, **kwargs) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_standings(self, league_id: int, season: int, **kwargs) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_statistics(self, fixture_id: int, **kwargs) -> dict[str, Any]:
        ...

    @abstractmethod
    def get_odds(self, fixture_id: int, **kwargs) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_lineups(self, fixture_id: int, **kwargs) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_injuries(self, fixture_id: int, **kwargs) -> list[dict[str, Any]]:
        ...