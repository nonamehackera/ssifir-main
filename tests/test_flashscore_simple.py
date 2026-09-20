#!/usr/bin/env python3
"""Simple Flashscore scraper test - just check yesterday."""
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from prediction.flashscore_scraper import _scrape_mobile_day, _normalize_match
from playwright.sync_api import sync_playwright

print("Testing single day scrape...")
print("-" * 60)

pw = None
browser = None
ctx = None
page = None

try:
    print("1. Starting Playwright...")
    pw = sync_playwright().start()
    
    print("2. Launching browser...")
    browser = pw.chromium.launch(headless=True)
    
    print("3. Creating context...")
    ctx = browser.new_context(
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
        locale="tr-TR",
        viewport={"width": 414, "height": 896},
        timezone_id="Europe/Istanbul",
        is_mobile=True,
        has_touch=True,
    )
    
    print("4. Creating page...")
    page = ctx.new_page()
    
    print("5. Scraping yesterday (d=-1, s=3 = finished)...")
    raw_matches = _scrape_mobile_day(day_offset=-1, status=3, page=page)
    print(f"   Found {len(raw_matches)} raw matches")
    
    if raw_matches:
        print("\nFirst 3 raw matches:")
        for i, m in enumerate(raw_matches[:3], 1):
            print(f"  {i}. {m}")
        
        print("\nNormalizing first match...")
        normalized = _normalize_match(raw_matches[0], day_offset=-1, status_kind="finished")
        if normalized:
            print(f"  Home: {normalized['home_name']}")
            print(f"  Away: {normalized['away_name']}")
            print(f"  Score: {normalized['home_score']}-{normalized['away_score']}")
            print(f"  League: {normalized['league_name']} ({normalized['league_country']})")
        else:
            print("  Failed to normalize!")
    else:
        print("\n*** NO MATCHES FOUND - SCRAPER IS BROKEN! ***")
        print("This means the mobile site structure has changed.")
        
except Exception as e:
    print(f"\n*** ERROR: {e} ***")
    import traceback
    traceback.print_exc()
    
finally:
    print("\n6. Cleaning up...")
    if page:
        page.close()
    if ctx:
        ctx.close()
    if browser:
        browser.close()
    if pw:
        pw.stop()

print("\nTEST COMPLETE")
