import json

with open(r'C:\Users\furka\Desktop\ssifir-main\tahminler\predictions.json', 'r', encoding='utf-8') as f:
    preds = json.load(f)

# Deduplicate by id - keep latest
seen = {}
for p in preds:
    pid = p.get('id')
    if pid not in seen:
        seen[pid] = p
deduped = list(seen.values())

export = []
for p in deduped:
    home = p.get('home_name', '?')
    away = p.get('away_name', '?')
    r = p.get('result', {})
    tg = p.get('total_goals', {})
    cr = p.get('corners', {})
    dc = p.get('double_chance', {})
    g = p.get('goals', {})
    export.append({
        'id': p.get('id'),
        'timestamp': p.get('timestamp'),
        'match': f'{home} vs {away}',
        'home_team': home,
        'away_team': away,
        'home_elo': p.get('home_elo', 0),
        'away_elo': p.get('away_elo', 0),
        'predictions': {
            '1X2': {
                'home': r.get('home_win'),
                'draw': r.get('draw'),
                'away': r.get('away_win'),
                'winner': r.get('winner'),
                'winner_prob': r.get('winner_prob')
            },
            'double_chance': {'1X': dc.get('1X'), '12': dc.get('12'), 'X2': dc.get('X2')},
            'btts': {'yes': p.get('btts_yes'), 'no': p.get('btts_no')},
            'goals': {
                'expected': {'home_xg': g.get('home_lambda'), 'away_xg': g.get('away_lambda'), 'total_xg': g.get('expected_total')},
                'over_under': {'1.5': tg.get('over15'), '2.5': tg.get('over25'), '3.5': tg.get('over35')}
            },
            'corners': {
                'average': cr.get('predicted_total') or cr.get('avg'),
                'over75': cr.get('over75'),
                'over85': cr.get('over85')
            },
            'top_scores': [{'score': s.get('score'), 'probability': s.get('probability')} for s in p.get('top_scores', [])]
        },
        'prediction_summary': p.get('prediction_summary'),
        'recommendations': p.get('recommendations')
    })

outpath = r'C:\Users\furka\Desktop\ssifir-main\tahminler\ftms_tahminler_2026-09-02.json'
with open(outpath, 'w', encoding='utf-8') as f:
    json.dump(export, f, ensure_ascii=False, indent=2)

print(f'{len(export)} tahmin kaydedildi (benzersiz)')

# Show unique matches
unique = {}
for p in preds:
    key = (p.get('home_name'), p.get('away_name'))
    unique[key] = True
print(f'\nBenzersiz mac sayisi: {len(unique)}')
for k in unique:
    print(f'  {k[0]} vs {k[1]}')
