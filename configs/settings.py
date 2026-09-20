import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://football:football_dev@localhost:5432/football",
)

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
API_FOOTBALL_BASE_URL = os.getenv(
    "API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io"
)

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
REQUEST_MAX_RETRIES = int(os.getenv("REQUEST_MAX_RETRIES", "5"))
REQUEST_BACKOFF_FACTOR = float(os.getenv("REQUEST_BACKOFF_FACTOR", "2.0"))

DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

DEFAULT_LEAGUES = [int(x) for x in os.getenv("DEFAULT_LEAGUES", "").split(",") if x]
DEFAULT_SEASONS = [int(x) for x in os.getenv("DEFAULT_SEASONS", "2024").split(",") if x]