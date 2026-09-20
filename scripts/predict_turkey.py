import pandas as pd
import json
from models.catboost.model import CatBoostModel

# Load data and model
feats = pd.read_parquet('data/gold/features.parquet')
tid_map = json.load(open('data/gold/team_id_unified.json', encoding='utf-8'))

# Load trained model
model = CatBoostModel(with_odds=False, iterations=400, verbose=False, time_decay=1.0).fit(feats.iloc[:int(len(feats)*0.8)])

# User's Turkey Super League teams (mapping from their list)
user_teams = ['basaksehir', 'galatasaray', 'fenerbahce', 'besiktas',
              'erzurumspor', 'konyaspor', 'corum', 'eyupspor',
              'kassimpa', 'amed', 'goztepe', 'gaziantep',
              'rizespor', 'alanyaspor', 'samsunspor', 'genclerbir']

# Find team IDs from mapping
found_teams = {}
for tid_str, name in tid_map.items():
    name_lower = name.lower()
    for user_team in user_teams:
        if user_team in name_lower:
            found_teams[user_team] = (int(tid_str), name)
            break

print('Found teams:', found_teams)
print()

# Get team IDs
found_ids = set()
for tid_str, name in found_teams.values():
    found_ids.add(tid)

# Find matches involving these teams
tr_mask = feats['home_team_id'].isin(found_ids) | feats['away_team_id'].isin(found_ids)
tr_matches = feats[tr_mask].copy()
print('Turkey matches in data:', len(tr_matches))

# Show some matches with found teams
for _, row in tr_matches.head(20).iterrows():
    h_name = tid_map.get(str(int(row['home_team_id'])), 'Unknown')
    a_name = tid_map.get(str(int(row['away_team_id'])), 'Unknown')
    print(f"  {h_name} vs {a_name} | {row['league']} | {row['date']}")

# Now predict on a few of these matches
print("\n--- Predictions ---")
predict_mask = tr_matches.index[:5]  # First 5 matches
for idx in predict_mask:
    row = tr_matches.iloc[[idx]]
    try:
        pred = model.predict(row)
        h_name = tid_map.get(str(int(row['home_team_id'].iloc[0])), 'Unknown')
        a_name = tid_map.get(str(int(row['away_team_id'].iloc[0])), 'Unknown')
        print(f"\n{h_name} vs {a_name}:")
        print(f"  1X2: Home={pred.result.home[0]:.3f} Draw={pred.result.draw[0]:.3f} Away={pred.result.away[0]:.3f}")
        print(f"  BTTS Yes={pred.btts.yes[0]:.3f} Over2.5={pred.over_under_2_5.over[0]:.3f}")
        print(f"  Lam: {pred.goals.home_lambda[0]:.2f}-{pred.goals.away_lambda[0]:.2f}")
    except Exception as e:
        print(f"Error predicting match {idx}: {e}")