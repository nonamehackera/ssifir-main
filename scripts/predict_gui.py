"""Tahmin Masaustu Uygulamasi - Tkinter ile."""
import sys, os, json
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"
import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import ttk
import pandas as pd, numpy as np
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
import threading

# ═══════════════════════════════════════════════════════════════════════════
# RENKLER
# ═══════════════════════════════════════════════════════════════════════════
BG = "#0f1117"
CARD = "#1a1d27"
CARD2 = "#232738"
BORDER = "#2d3148"
TEXT = "#e4e6f0"
MUTED = "#8b8fa3"
GREEN = "#00d97e"
RED = "#e63757"
YELLOW = "#f6c343"
BLUE = "#2c7be5"

# ═══════════════════════════════════════════════════════════════════════════
# MODEL + DATA
# ═══════════════════════════════════════════════════════════════════════════
print("[1/3] Veri yukleniyor...")

feat = pd.read_parquet("data/gold/features_enhanced.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals","away_goals"]: feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["result"] = feat["result"].fillna("D")
feat["btts"] = ((feat["home_goals"] > 0) & (feat["away_goals"] > 0)).astype(int)
feat["over25"] = (feat["total_goals"] > 2.5).astype(int)
feat = feat.dropna(subset=["result","elo_diff"])

# Lig mac sayisi (playable filtresi icin)
LEAGUE_MATCH_COUNTS = feat["league"].value_counts().to_dict()

LEAK = {"second_half_goals","ht_total_goals","ht_result_is_draw","ht_home_leading",
    "ht_home_goals","ht_away_goals","home_xg","away_xg","total_xg_real","xg_diff_real",
    "home_score_prob","away_score_prob","btts_xprob","over25_xprob","draw_xprob","data_completeness",
    "home_shots","away_shots","home_sot","away_sot","home_corners","away_corners",
    "home_yellow","away_yellow","home_red","away_red","avg_home_odds","avg_draw_odds",
    "avg_away_odds","avg_over25_odds","avg_close_home_odds","avg_close_draw_odds",
    "avg_close_away_odds","avg_close_over25_odds","mkt_close_home_prob","mkt_close_draw_prob",
    "mkt_close_away_prob","mkt_close_over25_prob","shots_diff","sot_diff",
    "home_goals","away_goals","result","result_H","result_D","result_A",
    "btts","over25","over15","over35","total_goals","match_id","season","date",
    "referee","home_xg_real","away_xg_real"}

CAT_COLS = ["league","home_team_id","away_team_id"]
# Enhanced feature listesi (buyuk/orta liglerde eklenir)
ENHANCED_COLS = ["mkt_home_misalign","mkt_draw_mismatch","mkt_over_misalign",
    "home_gf_per_ga","away_gf_per_ga","home_attack_ratio","away_attack_ratio",
    "home_pts_per_game","away_pts_per_game","form_pts_diff",
    "btts_signal","home_overall","away_overall","overall_diff","lg_confidence_weight"]
ACO = [c for c in feat.columns if c not in LEAK and feat[c].dtype in ["float64","int64","float32","int32"]]
ALL_COLS = CAT_COLS + [c for c in ACO if c not in CAT_COLS]
ALL_COLS_ENHANCED = ALL_COLS + [c for c in ENHANCED_COLS if c not in ALL_COLS and c in feat.columns]
cat_idx = [ALL_COLS.index(c) for c in CAT_COLS]
cat_idx_enh = [ALL_COLS_ENHANCED.index(c) for c in CAT_COLS]

with open("data/gold/team_id_to_name.json") as fh:
    NAME_MAP = {int(k): v for k, v in json.load(fh).items()}

# Son feature'lar + detay
latest = {}
team_details = {}
for tid in feat["home_team_id"].unique():
    s = feat[feat["home_team_id"] == tid].sort_values("date")
    if len(s) == 0: continue
    last = s.iloc[-1]
    latest[tid] = last
    recent = s.tail(10)
    form = recent["result"].map({"H":"G","D":"B","A":"M"}).tolist()
    team_details[tid] = {
        "id": int(tid),
        "name": NAME_MAP.get(tid, str(tid)),
        "league": last.get("league", "?"),
        "elo": float(last.get("home_elo", 1500)),
        "n_matches": len(s),
        "form": "".join(form[-5:]),
    }

team_list = sorted(team_details.values(), key=lambda x: -x["elo"])
print(f"  {len(team_list)} takim hazir")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL EGIT
# ═══════════════════════════════════════════════════════════════════════════
print("[2/3] Model egitimi...")

train = feat[(feat["date"] >= "2020-01-01") & (feat["date"] < "2026-01-01")].copy()
calib = feat[(feat["date"] >= "2026-01-01") & (feat["date"] < "2026-04-01")].copy()

# Buyuk+orta liglerde egitim (enhanced model icin)
lc = feat["league"].value_counts()
_train_big = train[train["league"].map(lc).fillna(0) >= 1000].copy()
_cal_big = calib[calib["league"].map(lc).fillna(0) >= 1000].copy()
print(f"  Train: {len(train)} | Buyuk/orta: {len(_train_big)} | Calib: {len(calib)}")

Xtr = train[ALL_COLS]; ytr = train["result"].map({"H":0,"D":1,"A":2}).values

# === BASE MODEL (eski - tum ligler) ===
clf_1x2 = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=10,
    border_count=64, random_strength=2, bagging_temperature=1, cat_features=cat_idx,
    loss_function="MultiClass", class_weights={0:1, 1:1.3, 2:1}, random_seed=42,
    verbose=False, allow_writing_files=False)
clf_1x2.fit(Xtr, ytr)

y_bt = train["btts"].values
clf_bt = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=10,
    cat_features=cat_idx, loss_function="Logloss", random_seed=42, verbose=False,
    allow_writing_files=False)
clf_bt.fit(Xtr, y_bt)

y_o25 = train["over25"].values
clf_o25 = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=10,
    cat_features=cat_idx, loss_function="Logloss", random_seed=42, verbose=False,
    allow_writing_files=False)
clf_o25.fit(Xtr, y_o25)

# === ENHANCED MODEL (buyuk+orta ligler) ===
if len(_train_big) > 5000 and len(_cal_big) > 500:
    Xtr_enh = _train_big[ALL_COLS_ENHANCED]
    ytr_enh = _train_big["result"].map({"H":0,"D":1,"A":2}).values
    clf_1x2_enh = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=10,
        border_count=64, random_strength=2, bagging_temperature=1, cat_features=cat_idx_enh,
        loss_function="MultiClass", class_weights={0:1, 1:1.3, 2:1}, random_seed=42,
        verbose=False, allow_writing_files=False)
    clf_1x2_enh.fit(Xtr_enh, ytr_enh)
    y_bt_enh = _train_big["btts"].values
    clf_bt_enh = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=10,
        cat_features=cat_idx_enh, loss_function="Logloss", random_seed=42, verbose=False,
        allow_writing_files=False)
    clf_bt_enh.fit(Xtr_enh, y_bt_enh)
    y_o25_enh = _train_big["over25"].values
    clf_o25_enh = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=10,
        cat_features=cat_idx_enh, loss_function="Logloss", random_seed=42, verbose=False,
        allow_writing_files=False)
    clf_o25_enh.fit(Xtr_enh, y_o25_enh)
    HAS_ENHANCED = True
    print(f"  Enhanced model: {len(_train_big)} mac ile egitildi ({len(ALL_COLS_ENHANCED)} feature)")
else:
    HAS_ENHANCED = False
    print(f"  Enhanced model icin yeterli veri yok")

# Calibration
Xcal = calib[ALL_COLS]; ycal = calib["result"].map({"H":0,"D":1,"A":2}).values
raw_cal = clf_1x2.predict_proba(Xcal)
cal_ir = []
for c in range(3):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_cal[:, c], (ycal == c).astype(float))
    cal_ir.append(ir)
ir_bt = IsotonicRegression(out_of_bounds="clip")
ir_bt.fit(clf_bt.predict_proba(Xcal)[:, 1], calib["btts"].values.astype(float))
ir_o25 = IsotonicRegression(out_of_bounds="clip")
ir_o25.fit(clf_o25.predict_proba(Xcal)[:, 1], calib["over25"].values.astype(float))

# Enhanced calibration
if HAS_ENHANCED and len(_cal_big) > 500:
    Xcal_enh = _cal_big[ALL_COLS_ENHANCED]
    ycal_enh = _cal_big["result"].map({"H":0,"D":1,"A":2}).values
    raw_cal_enh = clf_1x2_enh.predict_proba(Xcal_enh)
    cal_ir_enh = []
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_cal_enh[:, c], (ycal_enh == c).astype(float))
        cal_ir_enh.append(ir)
    ir_bt_enh = IsotonicRegression(out_of_bounds="clip")
    ir_bt_enh.fit(clf_bt_enh.predict_proba(Xcal_enh)[:, 1], _cal_big["btts"].values.astype(float))
    ir_o25_enh = IsotonicRegression(out_of_bounds="clip")
    ir_o25_enh.fit(clf_o25_enh.predict_proba(Xcal_enh)[:, 1], _cal_big["over25"].values.astype(float))
else:
    cal_ir_enh = None

print("[3/3] Hazir!")

# ═══════════════════════════════════════════════════════════════════════════
# TAHMIN
# ═══════════════════════════════════════════════════════════════════════════
def predict_match(home_id, away_id):
    if home_id not in latest or away_id not in latest:
        return None
    h = latest[home_id]; a = latest[away_id]
    row = {}
    for col in ALL_COLS:
        if col in CAT_COLS:
            if col == "league": row[col] = h.get("league", "Unknown")
            elif col == "home_team_id": row[col] = home_id
            elif col == "away_team_id": row[col] = away_id
        else:
            row[col] = float(h.get(col, 0)) if col in h.index else 0.0
    row["elo_diff"] = float(h.get("home_elo", 1500) - a.get("home_elo", 1500) + 50)
    row["attack_elo_diff"] = float(h.get("home_attack_elo", 1500) - a.get("home_attack_elo", 1500))
    row["defence_elo_diff"] = float(h.get("home_defence_elo", 1500) - a.get("away_defence_elo", 1500))

    # Enhanced feature'lar (buyuk/orta ligde mevcutsa kullan)
    league = row.get("league", "")
    n_lg = LEAGUE_MATCH_COUNTS.get(league, 0)
    use_enhanced = HAS_ENHANCED and n_lg >= 1000 and cal_ir_enh is not None

    if use_enhanced:
        for col in ENHANCED_COLS:
            if col not in row:
                row[col] = float(h.get(col, 0)) if col in h.index else 0.0
        X = pd.DataFrame([row])[ALL_COLS_ENHANCED]
        raw = clf_1x2_enh.predict_proba(X)[0]
        cp = np.column_stack([cal_ir_enh[c].predict([raw[c]]) for c in range(3)])[0]
        s = cp.sum(); s = s if s > 0 else 1; cp /= s
        bt = float(ir_bt_enh.predict([clf_bt_enh.predict_proba(X)[0,1]])[0])
        o25 = float(ir_o25_enh.predict([clf_o25_enh.predict_proba(X)[0,1]])[0])
    else:
        X = pd.DataFrame([row])[ALL_COLS]
        raw = clf_1x2.predict_proba(X)[0]
        cp = np.column_stack([cal_ir[c].predict([raw[c]]) for c in range(3)])[0]
        s = cp.sum(); s = s if s > 0 else 1; cp /= s
        bt = float(ir_bt.predict([clf_bt.predict_proba(X)[0,1]])[0])
        o25 = float(ir_o25.predict([clf_o25.predict_proba(X)[0,1]])[0])

    confidence = max(float(cp[0]), float(cp[1]), float(cp[2]))
    is_playable = confidence >= 0.60 and n_lg >= 1000

    return {
        "home_win": round(float(cp[0]) * 100, 1),
        "draw": round(float(cp[1]) * 100, 1),
        "away_win": round(float(cp[2]) * 100, 1),
        "btts_yes": round(bt * 100, 1),
        "btts_no": round((1 - bt) * 100, 1),
        "over25_yes": round(o25 * 100, 1),
        "over25_no": round((1 - o25) * 100, 1),
        "playable": is_playable,
        "model_used": "enhanced" if use_enhanced else "base",
        "confidence": round(confidence * 100, 1),
        "league_matches": n_lg,
    }

# ═══════════════════════════════════════════════════════════════════════════
# ARAYUZ
# ═══════════════════════════════════════════════════════════════════════════
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Futbol Tahmin Sistemi")
        self.root.geometry("900x800")
        self.root.configure(bg=BG)
        self.root.minsize(800, 700)

        self.selected_home = None
        self.selected_away = None

        self.build_ui()

    def build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=CARD, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="\u26bd  Futbol Tahmin Sistemi", bg=CARD, fg=TEXT,
                 font=("Segoe UI", 16, "bold")).pack(side="left", padx=20)
        tk.Label(hdr, text=f"{len(team_list)} takim  |  CatBoost AI", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="right", padx=20)

        # Main frame
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=16)

        # --- SEARCH + TEAM SELECT ---
        top = tk.Frame(main, bg=BG)
        top.pack(fill="x", pady=(0, 16))

        # Search
        srch_frame = tk.Frame(top, bg=BG)
        srch_frame.pack(fill="x", pady=(0, 12))
        tk.Label(srch_frame, text="\U0001F50D Takim Ara:", bg=BG, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(srch_frame, textvariable=self.search_var, bg=CARD2,
                                     fg=TEXT, insertbackground=TEXT, font=("Segoe UI", 12),
                                     relief="flat", bd=0, highlightthickness=1, highlightcolor=GREEN)
        self.search_entry.pack(fill="x", ipady=6)
        self.search_entry.bind("<KeyRelease>", self.on_search)
        self.search_entry.bind("<FocusOut>", lambda e: self.hide_results())
        self.search_entry.bind("<FocusIn>", lambda e: self.on_search(None))

        # Search results listbox
        self.results_frame = tk.Frame(main, bg=CARD, bd=0, highlightthickness=1, highlightcolor=BORDER)
        self.results_list = tk.Listbox(self.results_frame, bg=CARD, fg=TEXT, selectbackground=GREEN,
                                       selectforeground="#000", font=("Segoe UI", 11),
                                       relief="flat", bd=0, activestyle="none", height=8)
        self.results_list.bind("<<ListboxSelect>>", self.on_select)
        self.results_list.pack(fill="x")
        self.results_frame.place_forget()

        # Team slots
        slots = tk.Frame(main, bg=BG)
        slots.pack(fill="x", pady=(0, 16))
        slots.columnconfigure(0, weight=1)
        slots.columnconfigure(1, weight=0)
        slots.columnconfigure(2, weight=1)

        # Home slot
        self.home_frame = tk.Frame(slots, bg=CARD, bd=1, relief="solid", highlightbackground=BORDER)
        self.home_frame.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        self.home_inner = tk.Frame(self.home_frame, bg=CARD)
        self.home_inner.pack(expand=True, fill="both", padx=16, pady=16)
        self.home_inner.pack_forget()
        self.home_placeholder = tk.Frame(self.home_frame, bg=CARD)
        self.home_placeholder.pack(expand=True, fill="both")
        tk.Label(self.home_placeholder, text="\u26bd", bg=CARD, fg=MUTED, font=("Segoe UI", 28)).pack(pady=(16,4))
        tk.Label(self.home_placeholder, text="Ev Sahibi Sec", bg=CARD, fg=MUTED, font=("Segoe UI", 11)).pack()

        # VS
        tk.Label(slots, text="VS", bg=BG, fg=MUTED, font=("Segoe UI", 20, "bold")).grid(row=0, column=1, padx=12)

        # Away slot
        self.away_frame = tk.Frame(slots, bg=CARD, bd=1, relief="solid", highlightbackground=BORDER)
        self.away_frame.grid(row=0, column=2, sticky="nsew", padx=(8,0))
        self.away_inner = tk.Frame(self.away_frame, bg=CARD)
        self.away_inner.pack(expand=True, fill="both", padx=16, pady=16)
        self.away_inner.pack_forget()
        self.away_placeholder = tk.Frame(self.away_frame, bg=CARD)
        self.away_placeholder.pack(expand=True, fill="both")
        tk.Label(self.away_placeholder, text="\u26bd", bg=CARD, fg=MUTED, font=("Segoe UI", 28)).pack(pady=(16,4))
        tk.Label(self.away_placeholder, text="Dep. Takim Sec", bg=CARD, fg=MUTED, font=("Segoe UI", 11)).pack()

        # Buttons
        btns = tk.Frame(main, bg=BG)
        btns.pack(fill="x", pady=(0, 16))
        self.predict_btn = tk.Button(btns, text="TAHMIN YAP", bg=GREEN, fg="#000",
                                     font=("Segoe UI", 13, "bold"), relief="flat", bd=0,
                                     activebackground="#00b86b", activeforeground="#000",
                                     cursor="hand2", state="disabled", command=self.do_predict)
        self.predict_btn.pack(ipady=6, ipadx=30)

        # --- RESULTS ---
        self.results_outer = tk.Frame(main, bg=BG)

        # 1X2 cards
        self.r1x2 = tk.Frame(self.results_outer, bg=BG)
        self.r1x2.pack(fill="x", pady=(0, 12))
        self.r1x2.columnconfigure(0, weight=1)
        self.r1x2.columnconfigure(1, weight=1)
        self.r1x2.columnconfigure(2, weight=1)

        self.val_home = self._make_result_card(self.r1x2, "Ev Kazanir", GREEN, 0)
        self.val_draw = self._make_result_card(self.r1x2, "Beraberlik", YELLOW, 1)
        self.val_away = self._make_result_card(self.r1x2, "Dep. Kazanir", RED, 2)

        # Market cards
        mkt = tk.Frame(self.results_outer, bg=BG)
        mkt.pack(fill="x", pady=(0, 12))
        mkt.columnconfigure(0, weight=1)
        mkt.columnconfigure(1, weight=1)

        # BTTS
        btts_card = tk.Frame(mkt, bg=CARD, bd=1, relief="solid", highlightbackground=BORDER)
        btts_card.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        tk.Label(btts_card, text="BTTS (Iki Takim da Gol Atar)", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10, "bold")).pack(pady=(12,8), padx=12, anchor="w")
        self.btts_vals = self._make_market_rows(btts_card)

        # Over/Under
        ou_card = tk.Frame(mkt, bg=CARD, bd=1, relief="solid", highlightbackground=BORDER)
        ou_card.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        tk.Label(ou_card, text="Ust / Alt 2.5 Gol", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10, "bold")).pack(pady=(12,8), padx=12, anchor="w")
        self.ou_vals = self._make_market_rows(ou_card)

        # Form
        frm = tk.Frame(self.results_outer, bg=BG)
        frm.pack(fill="x", pady=(0, 8))
        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)
        self.form_home_frame = tk.Frame(frm, bg=CARD, bd=1, relief="solid", highlightbackground=BORDER)
        self.form_home_frame.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        tk.Label(self.form_home_frame, text="Ev Sahibi Son 5 Mac", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10, "bold")).pack(pady=(8,4), padx=12, anchor="w")
        self.form_home_dots = tk.Frame(self.form_home_frame, bg=CARD)
        self.form_home_dots.pack(pady=(0,12), padx=12, anchor="w")

        self.form_away_frame = tk.Frame(frm, bg=CARD, bd=1, relief="solid", highlightbackground=BORDER)
        self.form_away_frame.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        tk.Label(self.form_away_frame, text="Dep. Takimi Son 5 Mac", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10, "bold")).pack(pady=(8,4), padx=12, anchor="w")
        self.form_away_dots = tk.Frame(self.form_away_frame, bg=CARD)
        self.form_away_dots.pack(pady=(0,12), padx=12, anchor="w")

        # Playable status bar
        self.playable_frame = tk.Frame(self.results_outer, bg=BG)
        self.playable_frame.pack(fill="x", pady=(4, 0))

    def _make_result_card(self, parent, label, color, col):
        card = tk.Frame(parent, bg=CARD, bd=1, relief="solid", highlightbackground=BORDER)
        card.grid(row=0, column=col, sticky="nsew", padx=4)
        tk.Label(card, text=label, bg=CARD, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(pady=(10,4))
        val_label = tk.Label(card, text="0%", bg=CARD, fg=color, font=("Segoe UI", 28, "bold"))
        val_label.pack()
        bar_frame = tk.Frame(card, bg=BORDER, height=6)
        bar_frame.pack(fill="x", padx=12, pady=(4,12))
        bar_frame.pack_propagate(False)
        bar_fill = tk.Frame(bar_frame, bg=color, height=6)
        bar_fill.place(x=0, y=0, relheight=1.0, relwidth=0)
        return {"val": val_label, "bar": bar_fill}

    def _make_market_rows(self, parent):
        vals = {}
        for key, label in [("yes","Evet"),("no","Hayir")]:
            row = tk.Frame(parent, bg=CARD)
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=label, bg=CARD, fg=TEXT, font=("Segoe UI", 11),
                     width=8, anchor="w").pack(side="left")
            vl = tk.Label(row, text="0%", bg=CARD, fg=GREEN if key=="yes" else RED,
                          font=("Segoe UI", 13, "bold"))
            vl.pack(side="right")
            vals[key] = vl
        return vals

    # --- SEARCH ---
    def on_search(self, event):
        q = self.search_var.get().strip().lower()
        if len(q) < 1:
            self.hide_results()
            return
        matches = [(t["id"], t["name"], t["league"], t["elo"])
                   for t in team_list if q in t["name"].lower() or q in t["league"].lower()]
        self.results_list.delete(0, "end")
        for tid, name, lg, elo in matches[:15]:
            self.results_list.insert("end", f"  {name}  |  {lg}  |  ELO {elo:.0f}")
            self.results_list.insert("end", tid)
        # Store IDs
        self._search_ids = [m[0] for m in matches[:15]]
        if matches:
            self.results_frame.place(in_=self.search_entry, x=0, y=38, relwidth=1.0)
            self.results_frame.lift()
        else:
            self.hide_results()

    def hide_results(self):
        self.results_frame.place_forget()

    def on_select(self, event):
        sel = self.results_list.curselection()
        if not sel: return
        idx = sel[0]
        # Listbox has pairs: display + id
        real_idx = idx // 2
        if real_idx >= len(self._search_ids): return
        tid = self._search_ids[real_idx]
        det = team_details.get(tid)
        if not det: return

        if self.selected_home is None:
            self.set_team("home", det)
        elif self.selected_away is None:
            self.set_team("away", det)
        else:
            self.set_team("home", det)

        self.search_var.set("")
        self.hide_results()

    def set_team(self, side, det):
        if side == "home":
            self.selected_home = det
            self._fill_slot(self.home_frame, self.home_inner, self.home_placeholder, det)
        else:
            self.selected_away = det
            self._fill_slot(self.away_frame, self.away_inner, self.away_placeholder, det)
        self.predict_btn.configure(state="normal" if self.selected_home and self.selected_away else "disabled")

    def _fill_slot(self, frame, inner, placeholder, det):
        placeholder.pack_forget()
        inner.pack(expand=True, fill="both")
        for w in inner.winfo_children(): w.destroy()
        initials = det["name"][:2].upper()
        tk.Label(inner, text=initials, bg=GREEN, fg="#000", font=("Segoe UI", 16, "bold"),
                 width=4, height=1).pack(pady=(0,8))
        tk.Label(inner, text=det["name"], bg=CARD, fg=TEXT, font=("Segoe UI", 14, "bold")).pack()
        tk.Label(inner, text=f"{det['league']}  |  {det['n_matches']} mac", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10)).pack()
        tk.Label(inner, text=f"ELO {det['elo']:.0f}", bg=CARD, fg=GREEN,
                 font=("Segoe UI", 10, "bold")).pack(pady=(4,0))
        # Form
        frm = tk.Frame(inner, bg=CARD)
        frm.pack(pady=(6,0))
        for ch in det["form"]:
            clr = GREEN if ch == "G" else (YELLOW if ch == "B" else RED)
            tk.Label(frm, text=ch, bg=clr, fg="#000", font=("Segoe UI", 9, "bold"),
                     width=2).pack(side="left", padx=1)

    def do_predict(self):
        if not self.selected_home or not self.selected_away: return
        self.predict_btn.configure(state="disabled", text="Hesaplaniyor...")
        self.root.update()
        threading.Thread(target=self._run_predict, daemon=True).start()

    def _run_predict(self):
        res = predict_match(self.selected_home["id"], self.selected_away["id"])
        self.root.after(0, self._show_results, res)

    def _show_results(self, res):
        self.predict_btn.configure(state="normal", text="TAHMIN YAP")
        if res is None:
            return
        # 1X2
        self._set_card(self.val_home, res["home_win"])
        self._set_card(self.val_draw, res["draw"])
        self._set_card(self.val_away, res["away_win"])
        # BTTS
        self.btts_vals["yes"].configure(text=f"{res['btts_yes']}%")
        self.btts_vals["no"].configure(text=f"{res['btts_no']}%")
        # OU
        self.ou_vals["yes"].configure(text=f"{res['over25_yes']}%")
        self.ou_vals["no"].configure(text=f"{res['over25_no']}%")
        # Form
        self._set_form(self.form_home_dots, self.selected_home["form"])
        self._set_form(self.form_away_dots, self.selected_away["form"])
        # Playable status
        playable = res.get("playable", False)
        model = res.get("model_used", "base")
        conf = res.get("confidence", 0)
        n_lg = res.get("league_matches", 0)
        self._set_playable_status(playable, model, conf, n_lg)
        # Show results
        self.results_outer.pack(fill="both", expand=True)

    def _set_playable_status(self, playable, model, conf, n_lg):
        for w in self.playable_frame.winfo_children(): w.destroy()
        if playable:
            bg = GREEN; txt = f"OYNANABILIR  |  Guven: {conf:.0f}%  |  Model: {model.upper()}  |  Lig: {n_lg} mac"
        else:
            bg = RED; txt = f"OYNANMAZ  |  Guven: {conf:.0f}%  |  Model: {model.upper()}  |  Lig: {n_lg} mac"
        tk.Label(self.playable_frame, text=txt, bg=bg, fg="#000",
                 font=("Segoe UI", 10, "bold"), padx=12, pady=4).pack(fill="x")

    def _set_card(self, card, val):
        card["val"].configure(text=f"{val}%")
        card["bar"].place(x=0, y=0, relheight=1.0, relwidth=val/100)

    def _set_form(self, frame, form):
        for w in frame.winfo_children(): w.destroy()
        if not form: return
        for ch in form[-1::-1]:
            clr = GREEN if ch == "G" else (YELLOW if ch == "B" else RED)
            tk.Label(frame, text=ch, bg=clr, fg="#000", font=("Segoe UI", 10, "bold"),
                     width=3, height=1).pack(side="left", padx=2)

# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
