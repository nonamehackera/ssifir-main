"""1X2 tahminlerinin gercek accuracy'sini olc."""
import sys, os, warnings, json
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
os.environ["SKIP_STATS_PREDICTOR"] = "1"
import pandas as pd, numpy as np, pickle

feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals","away_goals"]: feat[c]=feat[c].fillna(0).astype(int)
feat["total_goals"]=feat["home_goals"]+feat["away_goals"]
feat["result"]=feat["result"].fillna("D")
feat["btts"]=((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over25"]=(feat["total_goals"]>2.5).astype(int)
feat=feat.dropna(subset=["result","elo_diff"])

with open("web_model.pkl","rb") as f: wm=pickle.load(f)
F=wm["features"]

idmap={}
try:
    with open("data/gold/team_id_to_name.json") as fh: idmap={int(k):v for k,v in json.load(fh).items()}
except: pass

test=feat[feat["date']>="2026-08-01"].copy()
Xt=test[F].fillna(0)
y=test["result"].map({"H":0,"D":1,"A":2}).values

mA,mB,ca,cb=wm["m_1x2"]
pA=mA.predict_proba(Xt)
pB=mB.predict_proba(Xt)
pAc=np.column_stack([ca[c].predict(pA[:,c]) for c in range(3)])
pBc=np.column_stack([cb[c].predict(pB[:,c]) for c in range(3)])
p=0.5*pAc+0.5*pBc
s=p.sum(axis=1,keepdims=True); s=np.where(s==0,1,s); p/=s

pred=np.argmax(p,axis=1)
conf=np.max(p,axis=1)
label_map={0:"1",1:"X",2:"2"}
res_map={0:"H",1:"D",2:"A"}

print("="*70)
print("  SON 1 AY (Agustos 2026) - 1X2 TAHMINLERI")
print("="*70)
print("  Toplam mac:", len(test))

acc=(pred==y).mean()
print("  1X2 GENEL ACCURACY: %.1f%% (%d/%d)" % (acc*100, (pred==y).sum(), len(y)))

for cls,name in [(0,"Ev Sahibi"),(1,"Beraberlik"),(2,"Dep. Takimi")]:
    n=(y==cls).sum()
    c=((pred==cls)&(y==cls)).sum()
    print("  %s: %d mac, %d dogru (%.1f%%)" % (name, n, c, c/n*100))

print("")
print("  GUVEN ESIGINE GORE:")
print("  %8s %7s %7s %7s %7s" % ("Esik","Picks","Oran","Dogru","Acc"))
print("  " + "-"*42)
for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
    m=conf>=t; n=m.sum()
    if n>0:
        c=(pred[m]==y[m]).sum()
        print("  %6.0f%% %7d %6.1f%% %7d %6.1f%%" % (t*100, n, n/len(y)*100, c, c/n*100))

print("")
print("  SON 30 MAC:")
print("  %10s %18s %18s %5s %7s %7s %5s %3s" % ("Tarih","Ev","Dep","Skor","Tahmin","Gercek","Guv","OK"))
print("  " + "-"*78)
for i in range(min(30,len(test))):
    row=test.iloc[i]
    dt=row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "?"
    hn=idmap.get(int(row["home_team_id"]),str(int(row["home_team_id"])))[:18]
    an=idmap.get(int(row["away_team_id"]),str(int(row["away_team_id"])))[:18]
    hg=int(row["home_goals"]); ag=int(row["away_goals"])
    pr=label_map[pred[i]]
    gr=res_map[y[i]]
    ok="Y" if pred[i]==y[i] else "N"
    print("  %10s %18s %18s %d-%d %7s %7s %4.0f%% %3s" % (dt, hn, an, hg, ag, pr, gr, conf[i]*100, ok))

# Draw analiz
print("")
print("  BERABERLIK ANALIZI:")
draw_pred=pred==1
draw_true=y==1
n_d=draw_pred.sum()
if n_d>0:
    c_d=((draw_pred)&(draw_true)).sum()
    print("  X tahmin edilen: %d mac, %d gercekten X (%.1f%%)" % (n_d, c_d, c_d/n_d*100))
print("  Gercek X sayisi: %d / %d mac (%.1f%%)" % (draw_true.sum(), len(y), draw_true.sum()/len(y)*100))

# Conf>=60 detail
print("")
print("  YUKSEK GUVEN (>=60%) TAHMINLER:")
m60=conf>=0.60
n60=m60.sum()
if n60>0:
    c60=(pred[m60]==y[m60]).sum()
    print("  %d tahmin, %d dogru (%.1f%%)" % (n60, c60, c60/n60*100))
    for i in np.where(m60)[0][:20]:
        row=test.iloc[i]
        dt=row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "?"
        hn=idmap.get(int(row["home_team_id"]),str(int(row["home_team_id"])))[:18]
        an=idmap.get(int(row["away_team_id"]),str(int(row["away_team_id"])))[:18]
        hg=int(row["home_goals"]); ag=int(row["away_goals"])
        pr=label_map[pred[i]]
        gr=res_map[y[i]]
        ok="Y" if pred[i]==y[i] else "N"
        print("  %10s %18s vs %-18s %d-%d  %s(guv=%.0f%%) gercek=%s %s" % (dt, hn, an, hg, ag, pr, conf[i]*100, gr, ok))

# Her sonuc icin model ne dedi
print("")
print("  SONUC BAZLI MODEL DAVRANISI:")
for actual_cls,actual_name in [(0,"H sonuclu"),(1,"D sonuclu"),(2,"A sonuclu")]:
    mask=y==actual_cls
    n=mask.sum()
    if n==0: continue
    preds_this=pred[mask]
    conf_this=conf[mask]
    for p_cls,p_name in [(0,"1 dedi"),(1,"X dedi"),(2,"2 dedi")]:
        cnt=(preds_this==p_cls).sum()
        print("  %s %s mac: %s -> %d (%.1f%%) ort_guv=%.0f%%" % (actual_name, n, p_name, cnt, cnt/n*100, conf_this[preds_this==p_cls].mean()*100 if (preds_this==p_cls).sum()>0 else 0))
