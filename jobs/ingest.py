"""Phase 1 CLI: veritabanı kurulumu ve veri ingestion.

Kullanım:
    python -m jobs.ingest init-db
    python -m jobs.ingest sync-leagues
    python -m jobs.ingest sync-seasons 39
    python -m jobs.ingest sync-teams 39 2024
    python -m jobs.ingest sync-fixtures 39 2024
    python -m jobs.ingest sync-all              # .env'deki DEFAULT_LEAGUES x DEFAULT_SEASONS
    python -m jobs.ingest fd-download E0 --all
    python -m jobs.ingest fd-load E0 2425
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")

from sqlalchemy.orm import Session  # noqa: E402

from configs import settings  # noqa: E402
from db.engine import SessionLocal, init_db  # noqa: E402
from ingestion.api_football.provider import ApiFootballProvider  # noqa: E402
from ingestion.storage import save_raw  # noqa: E402
from normalization.silver import (  # noqa: E402
    upsert_fixture,
    upsert_league,
    upsert_season,
    upsert_team,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest")


def cmd_init_db(_args) -> None:
    init_db()
    logger.info("Şema oluşturuldu (veya güncellendi).")


def _upsert_league_with_seasons(session: Session, provider: ApiFootballProvider, league_id: int) -> None:
    raw_leagues = provider.get_leagues(id=league_id)
    for raw in raw_leagues:
        league_obj = upsert_league(session, raw)
        seasons = provider.get_seasons(league_id=league_id)
        for season in seasons:
            upsert_season(session, league_obj.id, season)
        save_raw(provider.source_name, "leagues", {"id": league_id}, raw_leagues)
        save_raw(provider.source_name, "seasons", {"id": league_id}, seasons)
        session.commit()
        logger.info("Lig '%s' ve %d sezon kaydedildi", league_obj.name, len(seasons))


def cmd_sync_leagues(args) -> None:
    provider = ApiFootballProvider()
    with SessionLocal() as session:
        ids = [int(x) for x in args.league_ids.split(",")] if args.league_ids else settings.DEFAULT_LEAGUES
        for league_id in ids:
            _upsert_league_with_seasons(session, provider, league_id)


def cmd_sync_seasons(args) -> None:
    provider = ApiFootballProvider()
    with SessionLocal() as session:
        _upsert_league_with_seasons(session, provider, args.league_id)


def cmd_sync_teams(args) -> None:
    provider = ApiFootballProvider()
    with SessionLocal() as session:
        raw_teams = provider.get_teams(args.league_id, args.season)
        for raw in raw_teams:
            upsert_team(session, raw)
        save_raw(provider.source_name, "teams", {"league": args.league_id, "season": args.season}, raw_teams)
        session.commit()
        logger.info("%d takım kaydedildi (lig=%s sezon=%s)", len(raw_teams), args.league_id, args.season)


def cmd_sync_fixtures(args) -> None:
    provider = ApiFootballProvider()
    with SessionLocal() as session:
        raw_fixtures = provider.get_fixtures(args.league_id, args.season)
        save_raw(provider.source_name, "fixtures", {"league": args.league_id, "season": args.season}, raw_fixtures)
        ok = fail = 0
        for raw in raw_fixtures:
            if upsert_fixture(session, raw):
                ok += 1
            else:
                fail += 1
        session.commit()
        logger.info("Fixture: %d kaydedildi, %d hatalı", ok, fail)


def cmd_sync_all(args) -> None:
    provider = ApiFootballProvider()
    leagues = args.league_ids.split(",") if args.league_ids else [str(x) for x in settings.DEFAULT_LEAGUES]
    seasons = [int(x) for x in args.seasons.split(",")] if args.seasons else settings.DEFAULT_SEASONS
    for league_id in leagues:
        lid = int(league_id)
        with SessionLocal() as session:
            _upsert_league_with_seasons(session, provider, lid)
        for season in seasons:
            cmd_sync_teams(argparse.Namespace(league_id=lid, season=season))
            cmd_sync_fixtures(argparse.Namespace(league_id=lid, season=season))


def cmd_fd_download(args) -> None:
    from ingestion.football_data.loader import download_all, download_season

    if args.all:
        result = download_all([args.league_code] if args.league_code else None)
        total = sum(len(v) for v in result.values())
        logger.info("Toplam %d sezon dosyası indirildi", total)
    else:
        if not args.season_codes:
            raise SystemExit("fd-download için --season-code veya --all gerekli")
        for code in args.season_codes:
            download_season(args.league_code, code)


def cmd_fd_load(args) -> None:
    from ingestion.football_data.loader import load_season

    with SessionLocal() as session:
        total = 0
        for code in args.season_codes:
            total += load_season(session, args.league_code, code)
        logger.info("Toplam %d satır yüklendi", total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Futbol tahmin sistemi ingestion CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="PostgreSQL şemasını oluştur").set_defaults(func=cmd_init_db)

    p = sub.add_parser("sync-leagues", help="Lig + sezonları senkronize et")
    p.add_argument("--league-ids", help="Virgülle ayrılmış league ID'leri (.env'deki varsayılan kullanılır)")
    p.set_defaults(func=cmd_sync_leagues)

    p = sub.add_parser("sync-seasons", help="Tek lig için sezonları senkronize et")
    p.add_argument("league_id", type=int)
    p.set_defaults(func=cmd_sync_seasons)

    p = sub.add_parser("sync-teams", help="Lig+sezon takımlarını senkronize et")
    p.add_argument("league_id", type=int)
    p.add_argument("season", type=int)
    p.set_defaults(func=cmd_sync_teams)

    p = sub.add_parser("sync-fixtures", help="Lig+sezon fikstürlerini senkronize et")
    p.add_argument("league_id", type=int)
    p.add_argument("season", type=int)
    p.set_defaults(func=cmd_sync_fixtures)

    p = sub.add_parser("sync-all", help="Varsayılan ligler x sezonlar (takım + fikstür)")
    p.add_argument("--league-ids", help="Virgülle ayrılmış ID'ler")
    p.add_argument("--seasons", help="Virgülle ayrılmış sezon yılları")
    p.set_defaults(func=cmd_sync_all)

    p = sub.add_parser("fd-download", help="Football-Data.co.uk CSV indir (bronze)")
    p.add_argument("league_code", nargs="?", help="Örn. E0, SP1, T1")
    p.add_argument("--season-code", dest="season_codes", action="append", help="Örn. 2425 (tekrarlanabilir)")
    p.add_argument("--all", action="store_true", help="Bilinen tüm sezonları indir")
    p.set_defaults(func=cmd_fd_download)

    p = sub.add_parser("fd-load", help="Bronze CSV'yi football_data_matches tablosuna yükle")
    p.add_argument("league_code", help="Örn. E0")
    p.add_argument("season_codes", nargs="+", help="Örn. 2425 2324")
    p.set_defaults(func=cmd_fd_load)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()