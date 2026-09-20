"""Probe Football-Data.co.uk column availability across seasons/leagues."""
import csv
import io
import urllib.request

BASE = "https://www.football-data.co.uk/mmz4281"
WANT = ["Div","Date","Time","HomeTeam","AwayTeam","FTHG","FTAG","FTR","HTHG","HTAG",
        "HS","AS","HST","AST","HF","AF","HC","AC","HY","AY","HR","AR","Referee",
        "AvgH","AvgD","AvgA","Avg>2.5","Avg<2.5","B365H","B365D","B365A"]

def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.read().decode("utf-8-sig", "replace")
    except Exception as e:
        return f"ERR:{e}"

def header_cols(text):
    first = text.splitlines()[0] if text else ""
    return first.split(",")

print("=== E0 column presence by season ===")
for s in ["2425","2324","2223","2122","2021","1920","1819","1415","0506"]:
    t = fetch(f"{BASE}/{s}/E0.csv")
    cols = header_cols(t)
    have = [c for c in WANT if c in cols]
    print(f"E0 {s}: ncols={len(cols)} has_stats={'HS' in cols} sample={have[:6]}")

print("\n=== league availability (HTTP code per season) ===")
for code in ["E0","E1","E2","SP1","D1","I1","F1","N1","T1","B1","P1","SC0"]:
    codes = []
    for s in ["2425","2324","2021","1415"]:
        t = fetch(f"{BASE}/{s}/{code}.csv")
        if t.startswith("ERR"):
            codes.append(f"{s}:ERR")
        elif t.strip():
            codes.append(f"{s}:OK")
        else:
            codes.append(f"{s}:EMPTY")
    print(f"{code}: {codes}")
