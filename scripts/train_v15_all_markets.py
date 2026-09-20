import warnings, pickle, time
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier

t0=time.time()
print('='*70)
print('  v15: SECILI PAZARLARI HIZLI TEST')
print('='*70)

feat=pd.read_parquet('data/gold/features.parquet')
feat['date']=pd.to_datetime(feat['date'],errors='coerce')
feat=feat.sort_values('date').reset_index(drop=True)
for c in ['home_goals','away_goals']: feat[c]=feat[c].fillna(0).astype(int)
feat['total_goals']=feat['home_goals']+feat['away_goals']
feat['result']=feat['result'].fillna('D')
feat=feat.dropna(subset=['result','elo_diff']).reset_index(drop=True)
feat['elo_diff_abs']=feat['elo_diff'].abs()
feat['form_diff']=feat.get('home_pts_5',0).fillna(0)-feat.get('away_pts_5',0).fillna(0)
feat['form_diff_abs']=feat['form_diff'].abs()
feat['gf_diff']=feat.get('home_gf_5',0).fillna(0)-feat.get('away_gf_5',0).fillna(0)
feat['gf_diff_abs']=feat['gf_diff'].abs()
feat['ga_diff']=feat.get('home_ga_5',0).fillna(0)-feat.get('away_ga_5',0).fillna(0)
feat['ga_diff_abs']=feat['ga_diff'].abs()
feat['teams_even']=(feat['elo_diff_abs']<30).astype(int)
feat['teams_very_even']=(feat['elo_diff_abs']<15).astype(int)
feat['form_similar']=(feat['form_diff_abs']<1.0).astype(int)
feat['attack_similar']=(feat['gf_diff_abs']<0.5).astype(int)
feat['h2h_draw_high']=(feat.get('h2h_draw',0).fillna(0)>0.3).astype(int)
feat['lg_real_draw_rate']=np.nan; feat['lg_real_home_rate']=np.nan; feat['lg_real_goal_avg']=np.nan
feat['lg_real_btts_rate']=np.nan; feat['lg_real_cs_home']=np.nan; feat['lg_real_cs_away']=np.nan
for league in feat['league'].unique():
    mask=feat['league']==league; idx=feat.index[mask]
    lg=feat.loc[idx].sort_values('date')
    if len(lg)<50: continue
    feat.loc[idx,'lg_real_draw_rate']=lg['result'].eq('D').astype(int).rolling(200,min_periods=50).mean().values
    feat.loc[idx,'lg_real_home_rate']=(lg['result']=='H').rolling(200,min_periods=50).mean().values
    feat.loc[idx,'lg_real_goal_avg']=lg['total_goals'].rolling(200,min_periods=50).mean().values
    btts_lg=((lg['home_goals']>0)&(lg['away_goals']>0)).astype(int)
    feat.loc[idx,'lg_real_btts_rate']=btts_lg.rolling(200,min_periods=50).mean().values
    cs_h=(lg['away_goals']==0).astype(int); cs_a=(lg['home_goals']==0).astype(int)
    feat.loc[idx,'lg_real_cs_home']=cs_h.rolling(200,min_periods=50).mean().values
    feat.loc[idx,'lg_real_cs_away']=cs_a.rolling(200,min_periods=50).mean().values
feat=feat.dropna(subset=['lg_real_draw_rate']).reset_index(drop=True)
feat['lg_draw_signal']=(feat['lg_real_draw_rate']>0.28).astype(int)
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
feat['both_score_potential']=((feat.get('home_gf_5',0).fillna(0)>0.8)&(feat.get('away_gf_5',0).fillna(0)>0.8)).astype(int)
feat['home_clean_sheets']=(feat.get('home_ga_5',1).fillna(1)<0.5).astype(int)
feat['away_clean_sheets']=(feat.get('away_ga_5',1).fillna(1)<0.5).astype(int)

# Corner/Shot features
feat['home_corners_5']=feat.get('home_corners_5',pd.Series(0,index=feat.index)).fillna(0)
feat['away_corners_5']=feat.get('away_corners_5',pd.Series(0,index=feat.index)).fillna(0)
feat['corner_diff']=feat['home_corners_5']-feat['away_corners_5']
feat['home_corners']=feat.get('home_corners',pd.Series(0,index=feat.index)).fillna(0)
feat['away_corners']=feat.get('away_corners',pd.Series(0,index=feat.index)).fillna(0)
feat['total_corners']=feat['home_corners']+feat['away_corners']
feat['home_sot']=feat.get('home_sot',pd.Series(0,index=feat.index)).fillna(0)
feat['away_sot']=feat.get('away_sot',pd.Series(0,index=feat.index)).fillna(0)
feat['home_shots']=feat.get('home_shots',pd.Series(0,index=feat.index)).fillna(0)
feat['away_shots']=feat.get('away_shots',pd.Series(0,index=feat.index)).fillna(0)

# HT features
feat['ht_home_goals']=feat['ht_home_goals'].fillna(0).astype(int)
feat['ht_away_goals']=feat['ht_away_goals'].fillna(0).astype(int)
feat['ht_total_goals']=feat['ht_home_goals']+feat['ht_away_goals']
feat['ht_result']=np.where(feat['ht_home_goals']>feat['ht_away_goals'],'H',np.where(feat['ht_home_goals']<feat['ht_away_goals'],'A','D'))
feat['ht_btts']=((feat['ht_home_goals']>0)&(feat['ht_away_goals']>0)).astype(int)
feat['ht_over05']=(feat['ht_total_goals']>0.5).astype(int)
feat['ht_over15']=(feat['ht_total_goals']>1.5).astype(int)
feat['sh_home_goals']=feat['home_goals']-feat['ht_home_goals']
feat['sh_away_goals']=feat['away_goals']-feat['ht_away_goals']
feat['sh_total']=feat['sh_home_goals']+feat['sh_away_goals']
feat['sh_btts']=((feat['sh_home_goals']>0)&(feat['sh_away_goals']>0)).astype(int)
feat['sh_over05']=(feat['sh_total']>0.5).astype(int)
feat['sh_over15']=(feat['sh_total']>1.5).astype(int)
feat['goal_diff']=feat['home_goals']-feat['away_goals']
feat['hk1_0_win']=((feat['goal_diff']-1)>0).astype(int)
feat['hk1_0_lose']=((feat['goal_diff']-1)<0).astype(int)
feat['first_goal_home']=((feat['ht_home_goals']>feat['ht_away_goals'])|((feat['home_goals']>0)&(feat['away_goals']==0))).astype(int)
feat['home_win_margin']=feat['goal_diff'].clip(lower=0)

base_feats=[c for c in [
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
] if c in feat.columns]

corner_feats=[c for c in base_feats+['home_corners_5','away_corners_5','corner_diff','lg_corner_avg'] if c in feat.columns]
shot_feats=[c for c in base_feats+['home_sot','away_sot','home_shots','away_shots'] if c in feat.columns]

N=len(feat); n_train=int(N*0.70); n_cal=int(N*0.15)
train=feat.iloc[:n_train]; cal=feat.iloc[n_train:n_train+n_cal]; test=feat.iloc[n_train+n_cal:]

def train_and_eval(name, y_all, features):
    Xtr=train[features].fillna(0); Xca=cal[features].fillna(0); Xte=test[features].fillna(0)
    if isinstance(y_all, pd.Series): y_all=y_all.values
    ytr=y_all[train.index]; yca=y_all[cal.index]; yte=y_all[test.index]
    if ytr.mean()==0 or ytr.mean()==1:
        return
    models=[]
    for seed in [42,99,123]:
        m=LGBMClassifier(n_estimators=300,num_leaves=40,max_depth=6,learning_rate=0.05,min_child_samples=30,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=0.1,random_state=seed,verbose=-1,n_jobs=-1)
        m.fit(Xtr,ytr)
        ir=IsotonicRegression(out_of_bounds='clip')
        ir.fit(m.predict_proba(Xca)[:,1],yca)
        models.append((m,ir))
    ens=np.mean([ir.predict(m.predict_proba(Xte)[:,1]) for m,ir in models],axis=0)
    base=yte.mean()*100
    results=[]
    for t in [0.65,0.70,0.75,0.80]:
        mask=ens>=t; n=mask.sum()
        if n>10:
            c=yte[mask].mean()*100
            results.append((t,n,c))
    if results:
        best=max(results,key=lambda x:x[2] if x[2]>=70 else 0)
        if best[2]>=70:
            print('  %-35s base=%4.1f%%  t>%.0f: %5d picks, %.1f%%' % (name,base,best[0],best[1],best[2]))
            return
    print('  %-35s base=%4.1f%%  YETERSIZ' % (name,base))

print()
print('--- 1. HANDIKAP ---')
train_and_eval('HK1:0 Ev Sahibi',feat['hk1_0_win'],base_feats)
train_and_eval('HK1:0 Depasman',feat['hk1_0_lose'],base_feats)
train_and_eval('Ev 1 farkla',feat['home_win_margin']==1,base_feats)

print()
print('--- 2. ILK YARI ---')
train_and_eval('IY MS Ev',feat['ht_result']=='H',base_feats)
train_and_eval('IY MS Beraberlik',feat['ht_result']=='D',base_feats)
train_and_eval('IY Ust 0.5',feat['ht_over05'],base_feats)
train_and_eval('IY BTTS',feat['ht_btts'],base_feats)
train_and_eval('IY/MS D/H',(feat['ht_result']=='D')&(feat['result']=='H'),base_feats)
train_and_eval('IY/MS D/A',(feat['ht_result']=='D')&(feat['result']=='A'),base_feats)

print()
print('--- 3. 2. YARI ---')
train_and_eval('2Y Ust 0.5',feat['sh_over05'],base_feats)
train_and_eval('2Y Ust 1.5',feat['sh_over15'],base_feats)
train_and_eval('2Y BTTS',feat['sh_btts'],base_feats)
train_and_eval('2Y da daha cok gol',feat['sh_total']>feat['ht_total_goals'],base_feats)
train_and_eval('Ilk golu Ev atar',feat['first_goal_home'],base_feats)

print()
print('--- 4. KORNER ---')
for th in [9.5,10.5,11.5]:
    train_and_eval('Korner Ust %.1f' % th,(feat['total_corners']>th),corner_feats)
train_and_eval('En cok korner Ev',(feat['home_corners']>feat['away_corners']),corner_feats)
train_and_eval('Korner Cift',(feat['total_corners']%2==0),corner_feats)

print()
print('--- 5. SUT ---')
for th in [11,12,13]:
    train_and_eval('Ev Sut %d+' % th,(feat['home_shots']>=th),shot_feats)
for th in [6,7]:
    train_and_eval('Ev Isabetli Sut %d+' % th,(feat['home_sot']>=th),shot_feats)

print()
print('Done: %.0fs' % (time.time()-t0))
