#!/usr/bin/env python3
"""Test Flashscore scraper."""
from prediction.flashscore_scraper import get_fixtures_flashscore
import json

print("Testing Flashscore scraper...")
print("-" * 60)

# Test finished matches
print("\n=== FINISHED MATCHES (last 30 days) ===")
matches = get_fixtures_flashscore(kind='finished', limit=10, refresh=True)
print(f"Found {len(matches)} matches")

if matches:
    for i, m in enumerate(matches[:5], 1):
        score = f"{m['home_score'] if m['home_score'] is not None else '-'} - {m['away_score'] if m['away_score'] is not None else '-'}"
        print(f"{i}. {m['home_name']} vs {m['away_name']} = {score} [{m['status_short']}]")
        print(f"   League: {m['league_name']} ({m['league_country']})")
        print(f"   Kickoff: {m['kickoff']}")
else:
    print("NO MATCHES FOUND! Scraper is broken.")

# Test live matches
print("\n=== LIVE MATCHES ===")
live = get_fixtures_flashscore(kind='live', limit=5, refresh=True)
print(f"Found {len(live)} live matches")

if live:
    for i, m in enumerate(live[:3], 1):
        score = f"{m['home_score'] if m['home_score'] is not None else '-'} - {m['away_score'] if m['away_score'] is not None else '-'}"
        print(f"{i}. {m['home_name']} vs {m['away_name']} = {score} [{m['status_short']}]")
else:
    print("No live matches right now (normal if no games are playing)")

print("\n" + "=" * 60)
print("TEST COMPLETE")
