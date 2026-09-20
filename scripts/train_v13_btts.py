import warnings, pickle, time
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier

t0=time.time()
print('='*70)
print('  v13: BTTS OZEL MODEL')
print('='*70)

feat=pd.read_parquet('data/gold/features.parquet')
feat['date']=pd.to_datetime(feat['date'],errors='coerce')
feat=feat.sort_values('date').reset_index(drop=True)
for c in ['home_goals','away_goals']: feat[c]=feat[c].fillna(0).astype(int)
feat['total_goals']=feat['home_goals']+feat['away_goals']
feat['result']=feat['result'].fillna('D')
feat['btts']=((feat['home_goals']>0)&(feat['away_goals']>0)).astype(int)
feat['is_btts']=feat['btts']
feat=feat.dropna(subset=['result','elo_diff'])

# --- Rolling league BTTS stats ---
feat['lg_btts_rate_real']=np.nan
feat['lg_btts_high']=np.nan
feat['lg_total_goals_real']=np.nan

for league in feat['league'].unique():
    mask=feat['league']==league
    idx=feat.index[mask]
    lg=feat.loc[idx].sort_values('date')
    if len(lg)<50: continue
    r=lg['is_btts'].rolling(300,min_periods=50).mean()
    g=lg['total_goals'].rolling(300,min_periods=50).mean()
    feat.loc[idx,'lg_btts_rate_real']=r.values
    feat.loc[idx,'lg_btts_high']=(r>0.55).astype(int).values
    feat.loc[idx,'lg_total_goals_real']=g.values

feat=feat.dropna(subset=['lg_btts_rate_real'])

# --- Team BTTS rolling (vectorized via melt+groupby) ---
# Create a long-format table: team_id, date, did_btts_happen
home_df=feat[['date','home_team_id','away_team_id','btts']].copy()
home_df.columns=['date','team_id','opp_id','btts']

away_df=feat[['date','away_team_id','home_team_id','btts']].copy()
away_df.columns=['date','team_id','opp_id','btts']

teams=pd.concat([home_df,away_df],ignore_index=True).sort_values('date')

# Rolling BTTS rate per team
teams['team_btts_5']=teams.groupby('team_id')['btts'].transform(lambda x: x.rolling(5,min_periods=2).mean())
teams['team_btts_10']=teams.groupby('team_id')['btts'].transform(lambda x: x.rolling(10,min_periods=3).mean())

# Map back to home/away
# For home matches: first occurrence of team_id in home_df
home_idx=feat.index
away_idx=feat.index

feat['team_btts_rate_5']=teams.groupby('team_id')['team_btts_5'].transform('first').values[:len(feat)]
feat['team_btts_rate_10']=teams.groupby('team_id')['team_btts_10'].transform('first').values[:len(feat)]

# Simpler approach: just compute from feat directly
# For each match, home_team_id's rolling BTTS from their home matches
home_btts_5=[]; away_btts_5=[]; home_btts_10=[]; away_btts_10=[]

home_data={}  # team_id -> list of btts values
away_data={}

for i in range(len(feat)):
    row=feat.iloc[i]
    ht=int(row['home_team_id'])
    at=int(row['away_team_id'])
    b=row['btts']

    if ht not in home_data: home_data[ht]=[]
    if at not in away_data: away_data[at]=[]

    h_list=home_data[ht]
    a_list=away_data[at]

    h5=np.mean(h_list[-5:]) if len(h_list)>=2 else 0.5
    h10=np.mean(h_list[-10:]) if len(h_list)>=3 else 0.5
    a5=np.mean(a_list[-5:]) if len(a_list)>=2 else 0.5
    a10=np.mean(a_list[-10:]) if len(a_list)>=3 else 0.5

    home_btts_5.append(h5)
    home_btts_10.append(h10)
    away_btts_5.append(a5)
    away_btts_10.append(a10)

    home_data[ht].append(b)
    away_data[at].append(b)

feat['team_btts_rate_5']=home_btts_5
feat['team_btts_rate_10']=home_btts_10
feat['opp_btts_rate_5']=away_btts_5
feat['opp_btts_rate_10']=away_btts_10

# --- Derived BTTS features ---
feat['btts_potential']=feat['team_btts_rate_5']+feat['opp_btts_rate_5']
feat['both_attack_strong']=((feat.get('home_gf_5',0).fillna(0)>1.0)&(feat.get('away_gf_5',0).fillna(0)>1.0)).astype(int)
feat['both_defence_weak']=((feat.get('home_ga_5',0).fillna(0)>1.2)&(feat.get('away_ga_5',0).fillna(0)>1.2)).astype(int)
feat['attack_vs_defence_home']=feat.get('home_attack_elo',1500).fillna(1500)/(feat.get('away_defence_elo',1500).fillna(1500)+1)
feat['attack_vs_defence_away']=feat.get('away_attack_elo',1500).fillna(1500)/(feat.get('home_defence_elo',1500).fillna(1500)+1)
feat['btts_elo_signal']=feat['attack_vs_defence_home']*feat['attack_vs_defence_away']
feat['high_scoring_lg']=(feat.get('lg_total_goals_real',2.5).fillna(2.5)>2.7).astype(int)
feat['low_scoring_lg']=(feat.get('lg_total_goals_real',2.5).fillna(2.5)<2.2).astype(int)
feat['form_goals_both']=feat.get('home_gf_5',0).fillna(0)+feat.get('away_gf_5',0).fillna(0)
feat['goals_diff_small']=((feat.get('home_gf_5',0).fillna(0)-feat.get('away_gf_5',0).fillna(0)).abs()<0.8).astype(int)
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

print('BTTS features ready:', len(feat))

# --- Feature set ---
btts_feats=[
    'home_elo','away_elo','elo_diff','elo_diff_abs',
    'home_attack_elo','home_defence_elo','away_attack_elo','away_defence_elo',
    'attack_elo_diff','defence_elo_diff',
    'home_pts_5','away_pts_5','form_diff_abs',
    'home_gf_5','away_gf_5','home_ga_5','away_ga_5',
    'lg_avg_goals','lg_home_goal_avg','lg_away_goal_avg','lg_draw_rate',
    'lg_btts_rate','lg_over25_rate',
    'lg_btts_rate_real','lg_btts_high','lg_total_goals_real',
    'team_btts_rate_5','team_btts_rate_10','opp_btts_rate_5','opp_btts_rate_10',
    'btts_potential','both_attack_strong','both_defence_weak',
    'attack_vs_defence_home','attack_vs_defence_away','btts_elo_signal',
    'high_scoring_lg','low_scoring_lg','form_goals_both','goals_diff_small',
    'home_rest_days','away_rest_days','data_completeness',
    'home_attack_power','away_attack_power','home_defence_power','away_defence_power',
    'total_attack','total_defence','attack_defence_ratio',
    'home_scoring_rate','away_scoring_rate','scoring_rate_diff',
]
btts_feats=[c for c in btts_feats if c in feat.columns]
print('Using %d BTTS features' % len(btts_feats))

# --- Split ---
N=len(feat)
n_train=int(N*0.70)
n_cal=int(N*0.15)
train=feat.iloc[:n_train]
cal=feat.iloc[n_train:n_train+n_cal]
test=feat.iloc[n_train+n_cal:]

X_train=train[btts_feats].fillna(0)
X_cal=cal[btts_feats].fillna(0)
X_test=test[btts_feats].fillna(0)
y_train=train['btts'].values
y_cal=cal['btts'].values
y_test=test['btts'].values

print('Train: %d, Cal: %d, Test: %d' % (len(train),len(cal),len(test)))
print('BTTS rate: train=%.1f%% cal=%.1f%% test=%.1f%%' % (y_train.mean()*100,y_cal.mean()*100,y_test.mean()*100))

# --- Train 5-model ensemble ---
btts_models=[]
for seed in [42,99,123,456,789]:
    m=LGBMClassifier(
        n_estimators=400,num_leaves=40,max_depth=6,
        learning_rate=0.05,min_child_samples=30,
        subsample=0.8,colsample_bytree=0.7,
        reg_alpha=0.1,reg_lambda=0.1,
        random_state=seed,verbose=-1,n_jobs=-1
    )
    m.fit(X_train,y_train)
    prob_cal=m.predict_proba(X_cal)[:,1]
    ir=IsotonicRegression(out_of_bounds='clip')
    ir.fit(prob_cal,y_cal)
    btts_models.append((m,ir))

    prob_test=ir.predict(m.predict_proba(X_test)[:,1])
    res='  seed=%d' % seed
    for t in [0.55,0.60,0.65,0.70,0.75]:
        mask=prob_test>=t
        n=mask.sum()
        if n>0:
            corr=y_test[mask].mean()*100
            res+='  t=%.0f:%d(%.0f%%)' % (t,n,corr)
    print(res)

# --- Ensemble ---
ens_probs=np.mean([ir.predict(m.predict_proba(X_test)[:,1]) for m,ir in btts_models],axis=0)

print()
print('ENSEMBLE:')
for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
    mask=ens_probs>=t
    n=mask.sum()
    if n>0:
        corr=y_test[mask].mean()*100
        print('  t=%.0f: %4d picks, %.1f%% correct' % (t*100,n,corr))

# --- Feature importance ---
from collections import Counter
imp=Counter()
for m,ir in btts_models:
    for f,n in zip(btts_feats,m.feature_importances_):
        imp[f]+=int(n)
print()
print('Top 15 features:')
for f,n in imp.most_common(15):
    print('  %-35s %d' % (f,n))

# --- Save ---
wm_path='web_model.pkl'
with open(wm_path,'rb') as f: wm=pickle.load(f)
wm['m_btts']=btts_models
wm['btts_features']=btts_feats
wm['version']='v13_btts_enhanced'
with open(wm_path,'wb') as f: pickle.dump(wm,f)
print()
print('Saved: %s (v13)' % wm_path)
print('Done: %.0fs' % (time.time()-t0))
