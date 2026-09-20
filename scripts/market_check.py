import json

with open('tahminler/predictions.json','r',encoding='utf-8') as f:
    preds = json.load(f)

# Bugunku 3 mac
targets = ['Basaksehir', 'Stuttgart', 'Ipswich']

for p in preds:
    h = p['home_name']
    a = p['away_name']
    found = False
    for t in targets:
        if t.lower() in h.lower() or t.lower() in a.lower():
            found = True
            break
    if not found:
        continue

    r = p['result']
    tg = p.get('total_goals',{})
    btts_y = p.get('btts_yes',0)
    btts_n = p.get('btts_no',0)
    dc = p.get('double_chance',{})
    corners = p.get('corners',{})
    goals = p.get('goals',{})
    
    hw = r['home_win']
    dr = r['draw']
    aw = r['away_win']
    
    o15 = tg.get('over15',0)
    o25 = tg.get('over25',0)
    o35 = tg.get('over35',0)
    
    dc_1x = dc.get('1X',0)
    dc_12 = dc.get('12',0)
    dc_x2 = dc.get('X2',0)
    
    c_total = corners.get('predicted_total',0)
    c_o75 = corners.get('over75',0)
    c_o85 = corners.get('over85',0)
    c_o95 = corners.get('over95',0)
    c_o105 = corners.get('over105',0)
    
    hl = goals.get('home_lambda',0)
    al = goals.get('away_lambda',0)
    exp = goals.get('expected_total',0)
    
    print('='*70)
    print(f'  {h} vs {a}')
    print('='*70)
    print()
    print('  1X2 TAHMIN:')
    print(f'    Ev={hw:.1f}%  Beraberlik={dr:.1f}%  Dep={aw:.1f}%')
    fav = 'Ev' if hw >= dr and hw >= aw else ('Berabere' if dr >= hw and dr >= aw else 'Dep')
    print(f'    -> Favori: {fav}')
    print()
    print('  TOPLAM GOL:')
    print(f'    Ev Lambda={hl:.2f}  Dep Lambda={al:.2f}  Beklenen Toplam={exp:.2f}')
    print(f'    Ust 1.5: {o15:.1f}%  |  Alt 1.5: {100-o15:.1f}%')
    print(f'    Ust 2.5: {o25:.1f}%  |  Alt 2.5: {100-o25:.1f}%')
    print(f'    Ust 3.5: {o35:.1f}%  |  Alt 3.5: {100-o35:.1f}%')
    print()
    print('  BTTS (Iki Takim da Gol Atar):')
    print(f'    Evet: {btts_y:.1f}%  |  Hayir: {btts_n:.1f}%')
    print()
    print('  CIFTE SANS:')
    print(f'    1X (Ev veya Berabere): {dc_1x:.1f}%')
    print(f'    12 (Ev veya Dep):      {dc_12:.1f}%')
    print(f'    X2 (Berabere veya Dep): {dc_x2:.1f}%')
    print()
    print('  TOPLAM KORNER:')
    print(f'    Tahmini Toplam: {c_total:.1f}')
    print(f'    Ust 7.5:  {c_o75:.1f}%  |  Alt 7.5:  {100-c_o75:.1f}%')
    print(f'    Ust 8.5:  {c_o85:.1f}%  |  Alt 8.5:  {100-c_o85:.1f}%')
    print(f'    Ust 9.5:  {c_o95:.1f}%  |  Alt 9.5:  {100-c_o95:.1f}%')
    print(f'    Ust 10.5: {c_o105:.1f}% |  Alt 10.5: {100-c_o105:.1f}%')
    print()
    
    # Oneri
    print('  ** MODELE GORE OYNANABILIR MI? **')
    if o25 > 60:
        print(f'    Gol Ust 2.5: EVET ({o25:.1f}%)')
    else:
        print(f'    Gol Ust 2.5: HAYIR ({o25:.1f}%)')
    if btts_y > 55:
        print(f'    BTTS Evet:   EVET ({btts_y:.1f}%)')
    else:
        print(f'    BTTS Evet:   HAYIR ({btts_y:.1f}%)')
    if dc_1x > 70:
        print(f'    CiftSans 1X: EVET ({dc_1x:.1f}%)')
    elif dc_12 > 70:
        print(f'    CiftSans 12: EVET ({dc_12:.1f}%)')
    elif dc_x2 > 70:
        print(f'    CiftSans X2: EVET ({dc_x2:.1f}%)')
    else:
        print(f'    CiftSans:    HICBIRI GUCLU DEGIL')
    if c_o75 > 60:
        print(f'    Korner Ust 7.5: EVET ({c_o75:.1f}%)')
    elif c_o85 > 60:
        print(f'    Korner Ust 8.5: EVET ({c_o85:.1f}%)')
    else:
        print(f'    Korner: DUSUK GUVEN')
    print()
