import warnings, pickle, time
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier

t0=time.time()
print('='*70)
print('  v14: YENI BAHIS PIYASALARI')
print('='*70)

# Load and prepare features
feat=pd.read_parquet('data/gold/features.parquet')
feat['date']=pd.to_datetime(feat['date'],errors='coerce')
feat=feat.sort_values('date').reset_index(drop=True)
for c in ['home_goals','away_goals']: feat[c]=feat[c].fillna(0).astype(int)
feat['total_goals']=feat['home_goals']+feat['away_goals']
feat['result']=feat['result'].fillna('D')
feat=feat.dropna(subset=['result','elo_diff'])
feat['elo_diff_abs']=feat['elo_diff'].abs()
feat['form_diff']=feat.get('home_pts_5',0).fillna(0)-feat.get('away_pts_5',0).fillna(0)
feat['form_diff_abs']=feat['form_diff'].abs()
feat['gf_diff']=feat.get('home_gf_5',0).fillna(0)-feat.get('away_gf_5',0).fillna(0)
feat['gf_diff_abs']=feat['gf_diff'].abs()
feat['ga_diff']=feat.get('home_ga_5',0).fillna(0)-feat.get('away_ga_5',0).fillna(0)
feat['ga_diff_abs']=feat['ga_diff'].abs()
feat['home_attack_power']=feat.get('home_gf_5',0).fillna(0)*feat.get('home_attack_elo',1500).fillna(1500)/1500
feat['away_attack_power']=feat.get('away_gf_5',0).fillna(0)*feat.get('away_attack_elo',1500).fillna(1500)/1500
feat['home_defence_power']=feat.get('home_ga_5',0).fillna(0)*feat.get('home_defence_elo',1500).fillna(1500)/1500
feat['away_defence_power']=feat.get('away_ga_5',0).fillna(0)*feat.get('away_defence_elo',1500).fillna(1500)/1500
feat['total_attack']=feat['home_attack_power']+feat['away_attack_power']
feat['total_defence']=feat['home_defence_power']+feat['away_defence_power']
feat['attack_defence_ratio']=feat['total_attack']/(feat['total_defence']+0.1)
feat['home_scoring_rate']=feat.get('home_gf_5',0).fillna(0)/(feat.get('home_ga_5',1).fillna(1)+0.1)
feat['away_scoring_rate']=feat.get('away_gf_5',0).fillna(0)/(feat.get('away_ga_5',1).fillna(1)+0.1)
feat['scoring_rate_diff']=feat['home_scoring_rate']-feat['away_scoring_rate']
feat['both_attack_strong']=((feat.get('home_gf_5',0).fillna(0)>1.0)&(feat.get('away_gf_5',0).fillna(0)>1.0)).astype(int)
feat['both_defence_weak']=((feat.get('home_ga_5',0).fillna(0)>1.2)&(feat.get('away_ga_5',0).fillna(0)>1.2)).astype(int)
feat['high_scoring_match']=((feat.get('home_gf_5',0).fillna(0)+feat.get('away_gf_5',0).fillna(0))>2.5).astype(int)
feat['low_scoring_match']=((feat.get('home_gf_5',0).fillna(0)+feat.get('away_gf_5',0).fillna(0))<1.5).astype(int)

# Rolling league stats
feat['lg_real_draw_rate']=np.nan; feat['lg_real_home_rate']=np.nan; feat['lg_real_goal_avg']=np.nan
feat['lg_real_btts_rate']=np.nan; feat['lg_real_cs_home']=np.nan; feat['lg_real_cs_away']=np.nan
for league in feat['league'].unique():
    mask=feat['league']==league; idx=feat.index[mask]
    lg=feat.loc[idx].sort_values('date')
    if len(lg)<50: continue
    r=lg['result'].eq('D').astype(int).rolling(200,min_periods=50).mean()
    g=lg['total_goals'].rolling(200,min_periods=50).mean()
    feat.loc[idx,'lg_real_draw_rate']=r.values
    feat.loc[idx,'lg_real_home_rate']=(lg['result']=='H').rolling(200,min_periods=50).mean().values
    feat.loc[idx,'lg_real_goal_avg']=g.values
    btts_lg=((lg['home_goals']>0)&(lg['away_goals']>0)).astype(int)
    feat.loc[idx,'lg_real_btts_rate']=btts_lg.rolling(200,min_periods=50).mean().values
    cs_home=(lg['away_goals']==0).astype(int)
    cs_away=(lg['home_goals']==0).astype(int)
    feat.loc[idx,'lg_real_cs_home']=cs_home.rolling(200,min_periods=50).mean().values
    feat.loc[idx,'lg_real_cs_away']=cs_away.rolling(200,min_periods=50).mean().values
feat=feat.dropna(subset=['lg_real_draw_rate'])

# TARGETS
targets = {
    'home_cs':      (feat['away_goals']==0).astype(int),         # Ev sahibi gol yemez
    'away_cs':      (feat['home_goals']==0).astype(int),         # Depasman gol yemez
    'home_over15':  (feat['home_goals']>1.5).astype(int),        # Ev sahibi 2+ gol atar
    'away_over05':  (feat['away_goals']>0.5).astype(int),        # Depasman gol atar
    'home_over05':  (feat['home_goals']>0.5).astype(int),        # Ev sahibi gol atar
    'goals_2to3':   ((feat['total_goals']>=2)&(feat['total_goals']<=3)).astype(int),  # 2-3 gol
    'goals_0to1':   (feat['total_goals']<=1).astype(int),        # 0-1 gol
    'goals_4plus':  (feat['total_goals']>=4).astype(int),        # 4+ gol
    'home_win':     (feat['result']=='H').astype(int),           # Ev sahibi kazanir
    'draw':         (feat['result']=='D').astype(int),           # Beraberlik
}

# Feature set
base_feats=[
    'home_elo','away_elo','elo_diff','elo_diff_abs',
    'home_attack_elo','home_defence_elo','away_attack_elo','away_defence_elo',
    'attack_elo_diff','defence_elo_diff',
    'home_pts_5','away_pts_5','form_diff_abs',
    'home_gf_5','away_gf_5','home_ga_5','away_ga_5',
    'lg_avg_goals','lg_home_goal_avg','lg_away_goal_avg','lg_draw_rate',
    'lg_btts_rate','lg_over25_rate',
    'lg_real_draw_rate','lg_real_home_rate','lg_real_goal_avg',
    'lg_real_btts_rate','lg_real_cs_home','lg_real_cs_away',
    'home_rest_days','away_rest_days','data_completeness',
    'home_attack_power','away_attack_power','home_defence_power','away_defence_power',
    'total_attack','total_defence','attack_defence_ratio',
    'home_scoring_rate','away_scoring_rate','scoring_rate_diff',
    'both_attack_strong','both_defence_weak','high_scoring_match','low_scoring_match',
]
base_feats=[c for c in base_feats if c in feat.columns]

# Train/test split
N=len(feat)
n_train=int(N*0.70); n_cal=int(N*0.15)
train=feat.iloc[:n_train]; cal=feat.iloc[n_train:n_train+n_cal]; test=feat.iloc[n_train+n_cal:]

results={}

for tname, y_all in targets.items():
    ytr=train.index.map(y_all).values
    yca=cal.index.map(y_all).values
    yte=test.index.map(y_all).values
    
    Xtr=train[base_feats].fillna(0); Xca=cal[base_feats].fillna(0); Xte=test[base_feats].fillna(0)
    
    base_rate=yte.mean()*100
    
    models=[]
    for seed in [42,99,123,456,789]:
        m=LGBMClassifier(
            n_estimators=400,num_leaves=40,max_depth=6,
            learning_rate=0.05,min_child_samples=30,
            subsample=0.8,colsample_bytree=0.7,
            reg_alpha=0.1,reg_lambda=0.1,
            random_state=seed,verbose=-1,n_jobs=-1
        )
        m.fit(Xtr,ytr)
        ir=IsotonicRegression(out_of_bounds='clip')
        ir.fit(m.predict_proba(Xca)[:,1],yca)
        models.append((m,ir))
    
    ens=np.mean([ir.predict(m.predict_proba(Xte)[:,1]) for m,ir in models],axis=0)
    
    res_line='  %-20s (base: %4.1f%%)' % (tname,base_rate)
    best_entry=None
    for t in [0.50,0.55,0.60,0.65,0.70,0.75]:
        mask=ens>=t; n=mask.sum()
        if n>0:
            corr=yte[mask].mean()*100
            res_line+='  t=%.0f:%d(%.0f%%)' % (t,n,corr)
            if t>=0.65 and (best_entry is None or corr>best_entry[1]):
                best_entry=(t,n,corr)
    print(res_line)
    results[tname]={'models':models,'base_rate':base_rate,'best':best_entry,'ens':ens,'y':yte}

print()
print('='*70)
print('  OZET: Hangi piyasalar calisiyor?')
print('='*70)
print()
print('%-20s %6s  %15s  %s' % ('PIYASA','BASE','EN IYI ESIK','DOGRULUK'))
print('-'*65)
for tname,r in sorted(results.items(),key=lambda x:-(x[1]['best'][2] if x[1]['best'] else 0)):
    b=r['best']
    if b:
        print('%-20s %5.1f%%  t>%.0f: %5d picks  %.1f%%' % (tname,r['base_rate'],b[0],b[1],b[2]))
    else:
        print('%-20s %5.1f%%  yetersiz' % (tname,r['base_rate']))

# Save all new models
wm_path='web_model.pkl'
with open(wm_path,'rb') as f: wm=pickle.load(f)
for tname,r in results.items():
    wm['m_'+tname]=r['models']
wm['new_target_features']=base_feats
wm['version']='v14_new_markets'
with open(wm_path,'wb') as f: pickle.dump(wm,f)
print()
print('Saved: %s (v14)' % wm_path)
print('Done: %.0fs' % (time.time()-t0))
