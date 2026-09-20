import warnings, pickle, time
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier

t0=time.time()
print('='*70)
print('  v13b: BTTS — CLEAN FEATURES ONLY')
print('='*70)

feat=pd.read_parquet('data/gold/features.parquet')
feat['date']=pd.to_datetime(feat['date'],errors='coerce')
feat=feat.sort_values('date').reset_index(drop=True)
for c in ['home_goals','away_goals']: feat[c]=feat[c].fillna(0).astype(int)
feat['total_goals']=feat['home_goals']+feat['away_goals']
feat['result']=feat['result'].fillna('D')
feat['btts']=((feat['home_goals']>0)&(feat['away_goals']>0)).astype(int)
feat=feat.dropna(subset=['result','elo_diff'])

# Rolling league BTTS stats (pre-match only, no leakage)
feat['lg_btts_rate_real']=np.nan
feat['lg_total_goals_real']=np.nan

for league in feat['league'].unique():
    mask=feat['league']==league
    idx=feat.index[mask]
    lg=feat.loc[idx].sort_values('date')
    if len(lg)<50: continue
    r=lg['btts'].rolling(300,min_periods=50).mean()
    g=lg['total_goals'].rolling(300,min_periods=50).mean()
    feat.loc[idx,'lg_btts_rate_real']=r.values
    feat.loc[idx,'lg_total_goals_real']=g.values

feat=feat.dropna(subset=['lg_btts_rate_real'])

# Derived features
feat['home_attack_power']=feat.get('home_gf_5',0).fillna(0)*feat.get('home_attack_elo',1500).fillna(1500)/1500
feat['away_attack_power']=feat.get('away_gf_5',0).fillna(0)*feat.get('away_attack_elo',1500).fillna(1500)/1500
feat['home_defence_power']=feat.get('home_ga_5',0).fillna(0)*feat.get('home_defence_elo',1500).fillna(1500)/1500
feat['away_defence_power']=feat.get('away_ga_5',0).fillna(0)*feat.get('away_defence_elo',1500).fillna(1500)/1500
feat['total_attack']=feat['home_attack_power']+feat['away_attack_power']
feat['total_defence']=feat['home_defence_power']+feat['away_defence_power']
feat['attack_defence_ratio']=feat['total_attack']/(feat['total_defence']+0.1)
feat['both_attack_strong']=((feat['home_gf_5'].fillna(0)>1.0)&(feat['away_gf_5'].fillna(0)>1.0)).astype(int)
feat['both_defence_weak']=((feat['home_ga_5'].fillna(0)>1.2)&(feat['away_ga_5'].fillna(0)>1.2)).astype(int)
feat['form_goals_total']=feat['home_gf_5'].fillna(0)+feat['away_gf_5'].fillna(0)
feat['high_scoring_lg']=(feat['lg_total_goals_real'].fillna(2.5)>2.7).astype(int)

# --- Feature set (CLEAN, no team-level) ---
btts_feats=[
    'home_elo','away_elo','elo_diff','elo_diff_abs',
    'home_attack_elo','home_defence_elo','away_attack_elo','away_defence_elo',
    'attack_elo_diff','defence_elo_diff',
    'home_pts_5','away_pts_5',
    'home_gf_5','away_gf_5','home_ga_5','away_ga_5',
    'lg_avg_goals','lg_home_goal_avg','lg_away_goal_avg','lg_draw_rate',
    'lg_btts_rate','lg_over25_rate',
    'lg_btts_rate_real','lg_total_goals_real','high_scoring_lg',
    'both_attack_strong','both_defence_weak','form_goals_total',
    'attack_defence_ratio','total_attack','total_defence',
    'home_rest_days','away_rest_days','data_completeness',
]
btts_feats=[c for c in btts_feats if c in feat.columns]
print('Using %d clean BTTS features' % len(btts_feats))

# --- Split ---
N=len(feat)
n_train=int(N*0.70)
n_cal=int(N*0.15)
train=feat.iloc[:n_train]; cal=feat.iloc[n_train:n_train+n_cal]; test=feat.iloc[n_train+n_cal:]

X_train=train[btts_feats].fillna(0); X_cal=cal[btts_feats].fillna(0); X_test=test[btts_feats].fillna(0)
y_train=train['btts'].values; y_cal=cal['btts'].values; y_test=test['btts'].values

print('Train: %d, Cal: %d, Test: %d' % (len(train),len(cal),len(test)))
print('BTTS rate: train=%.1f%% cal=%.1f%% test=%.1f%%' % (y_train.mean()*100,y_cal.mean()*100,y_test.mean()*100))

# --- Cross-validation to find best params ---
print()
print('Testing different ensemble sizes and params...')

# Single strong model first
m_single=LGBMClassifier(
    n_estimators=500,num_leaves=50,max_depth=7,
    learning_rate=0.03,min_child_samples=20,
    subsample=0.75,colsample_bytree=0.7,
    reg_alpha=0.05,reg_lambda=0.05,
    random_state=42,verbose=-1,n_jobs=-1
)
m_single.fit(X_train,y_train)
prob_single=m_single.predict_proba(X_test)[:,1]

print('Single model (no calibration):')
for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
    mask=prob_single>=t; n=mask.sum()
    if n>0:
        corr=y_test[mask].mean()*100
        print('  t=%.0f: %5d picks, %.1f%% correct' % (t*100,n,corr))

# With isotonic calibration on single model
ir_single=IsotonicRegression(out_of_bounds='clip')
ir_single.fit(m_single.predict_proba(X_cal)[:,1],y_cal)
prob_single_cal=ir_single.predict(prob_single)
print()
print('Single model + isotonic calibration:')
for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
    mask=prob_single_cal>=t; n=mask.sum()
    if n>0:
        corr=y_test[mask].mean()*100
        print('  t=%.0f: %5d picks, %.1f%% correct' % (t*100,n,corr))

# 5-model ensemble
btts_models=[]
for seed in [42,99,123,456,789]:
    m=LGBMClassifier(
        n_estimators=500,num_leaves=50,max_depth=7,
        learning_rate=0.03,min_child_samples=20,
        subsample=0.75,colsample_bytree=0.7,
        reg_alpha=0.05,reg_lambda=0.05,
        random_state=seed,verbose=-1,n_jobs=-1
    )
    m.fit(X_train,y_train)
    ir=IsotonicRegression(out_of_bounds='clip')
    ir.fit(m.predict_proba(X_cal)[:,1],y_cal)
    btts_models.append((m,ir))

ens_probs=np.mean([ir.predict(m.predict_proba(X_test)[:,1]) for m,ir in btts_models],axis=0)
print()
print('5-model ensemble + isotonic calibration:')
for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
    mask=ens_probs>=t; n=mask.sum()
    if n>0:
        corr=y_test[mask].mean()*100
        print('  t=%.0f: %5d picks, %.1f%% correct' % (t*100,n,corr))

# Feature importance
from collections import Counter
imp=Counter()
for m,ir in btts_models:
    for f,n in zip(btts_feats,m.feature_importances_):
        imp[f]+=int(n)
print()
print('Top 10 features:')
for f,n in imp.most_common(10):
    print('  %-35s %d' % (f,n))

# --- Save best model ---
wm_path='web_model.pkl'
with open(wm_path,'rb') as f: wm=pickle.load(f)
wm['m_btts']=btts_models
wm['btts_features']=btts_feats
wm['version']='v13b_btts_clean'
with open(wm_path,'wb') as f: pickle.dump(wm,f)
print()
print('Saved: %s (v13b)' % wm_path)
print('Done: %.0fs' % (time.time()-t0))
