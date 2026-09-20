# GERÇEK FUTBOL MAÇ TAHMİN SİSTEMİ — YOL HARİTASI

> Bu doküman önceki herhangi bir JSON veri yapısına bağlı değildir.
> Proje tamamen yeni bir veri platformu olarak tasarlanmıştır.
>
> Amaç: Geçmiş maçlardan öğrenen, maç başlamadan önce tahmin üreten, maç
> sırasında yeni bilgiler geldikçe canlı tahmin yapan ve yaptığı tahminleri
> ölçerek zaman içinde kendini geliştiren üretim kalitesinde bir futbol
> tahmin sistemi kurmak.

---

## 1. PROJENİN GERÇEK HEDEFİ

Sistem yalnızca *"Bu maç kim kazanır?"* sorusunu cevaplamamalıdır. Aynı veri
omurgasından birden fazla hedef üretilmelidir:

| Pazar | Hedefler |
|---|---|
| Maç sonucu | Home Win, Draw, Away Win |
| Gol | Ev sahibi gol beklentisi, deplasman gol beklentisi, toplam gol, Over 0.5/1.5/2.5/3.5, Under 1.5/2.5/3.5, exact score dağılımı |
| BTTS | Both Teams To Score: Yes / No |
| Diğer | Double Chance, Draw No Bet, Asian Handicap, Corners, Cards, İlk yarı sonucu, İlk yarı golü, İlk yarı Over/Under |

Bütün bunlar tek bir modele rastgele feature doldurularak yapılmamalıdır.
Doğru yaklaşım:

```
                    DATA PLATFORM
                         |
        +----------------+----------------+
        |                |                |
   Team Strength      Match Context    Market Data
        |                |                |
        +----------------+----------------+
                         |
                  FEATURE ENGINE
                         |
             +-----------+-----------+
             |                       |
       PRE-MATCH MODEL          LIVE MODEL
             |                       |
             +-----------+-----------+
                         |
                PROBABILITY LAYER
                         |
                  CALIBRATION
                         |
                   PREDICTIONS
                         |
                    EVALUATION
                         |
                  MODEL REGISTRY
                         |
                   RE-TRAINING
```

---

## 2. VERİ KAYNAĞI STRATEJİSİ

### 2.1 Sportmonks (aday ana kaynak)

Tarihsel maçlar, canlı skorlar, istatistikler, kadrolar, line-up, xG, oranlar,
oyuncu/takım verileri. xG tarafında takım ve oyuncu bazında xG, xGA, xGD,
npxG, xPTS, xGoT metrikleri sunuyor.

- https://www.sportmonks.com/football-api/
- https://www.sportmonks.com/football-api/xg-data/

> Kapsam, ücretli plan ve tarihsel veri derinliği satın almadan önce
> doğrulanmalıdır.

### 2.2 API-Football / API-Sports (ana kaynak — anahtar mevcut)

Endpoints: fixtures, standings, teams, players, lineups, fixture statistics,
events, injuries, odds, live odds, live fixtures. Coverage: 1.200+ lig/kupa.

- https://www.api-football.com/
- https://www.api-football.com/documentation
- https://www.api-football.com/coverage

> Uyarı: Canlı odds endpoint'i tarihsel olarak saklanmıyor. Canlı odds
> kullanılacaksa sistem kendi snapshot arşivini oluşturmalı (bkz. bölüm 62).

### 2.3 Football-Data.co.uk (ücretsiz tarihsel validation)

FT/HT sonuç, maç istatistikleri, 1X2 odds, totals odds, Asian handicap odds.
Bazı liglerde 1990'lı sezonlara uzanan arşiv.

- https://www.football-data.co.uk/data.php

Ana API'den bağımsız **historical validation source** olarak kullanılır.

### 2.4 StatsBomb Open Data (araştırma)

Event-level veri (competitions, seasons, matches, events, lineups, kısmen
StatsBomb 360). Üretim kaynağı değil; feature geliştirme/event modelleme için.

- https://github.com/statsbomb/open-data

### 2.5 Kaggle

Üretim veri kaynağı olarak kabul edilmez. Her dataset için kaynak, toplama
tarihi, duplicate, tarih formatı, missing data, lisans ve data leakage
kontrolü yapılmalıdır.

---

## 3. ÖNERİLEN VERİ KOMBİNASYONU

```
                    API-FOOTBALL
                         |
             +-----------+-----------+
             |                       |
        Main Historical          Live / Current
             |
             v
                     INTERNAL DB
                         ^
             +-----------+-----------+
             |                       |
       SPORTMONKS             FOOTBALL-DATA
       secondary source       historical validation
             |
             +----------------------+
```

Amaçlar:
1. Tek sağlayıcıya bağımlılığı azaltmak.
2. Veri hatalarını çapraz kontrol etmek.
3. Eksik feature olduğunda fallback sağlamak.
4. Geçmiş model ile yeni veriyi aynı şemaya bağlamak.

---

## 4. TOPLANACAK VERİ GRUPLARI

### 4.1 Fixture / Match (zorunlu)

```text
fixture_id, league_id, season_id, date, kickoff_timestamp,
home_team_id, away_team_id, venue_id, referee_id, round, match_status
```

Sonuç geldikten sonra:

```text
home_goals, away_goals, half_time_home_goals, half_time_away_goals
```

### 5. Takım gücü (zamanla değişen state)

- Hücum: `goals_for, xg, npxg, shots, shots_on_target, big_chances, touches_in_box`
- Savunma: `goals_against, xga, shots_against, shots_on_target_against, big_chances_against`
- Oyun: `possession, passes, pass_accuracy, corners_for, corners_against, fouls, offsides`
- Form windowları: `last_3, last_5, last_8, last_10`
  - Örnek: `home_team_goals_avg_last_5, home_team_xg_avg_last_5, home_team_xga_avg_last_5, home_team_shots_avg_last_5`
  - Aynıları `away_*` için.

### 6. Ev / deplasman ayrı feature'lar

```text
home_team_home_goals_avg_5, home_team_home_xg_avg_5, home_team_home_xga_avg_5
away_team_away_goals_avg_5, away_team_away_xg_avg_5, away_team_away_xga_avg_5
```

Genel form ile saha formu aynı olmayabilir.

### 7. Elo ve takım gücü state'i

```text
elo_rating, home_elo, away_elo, attack_rating, defence_rating
```

Maç sonrası `rating_before_match` ve `rating_after_match` saklanır.
Model **yalnızca `elo_before_match`** kullanır. `elo_after_match` = data leakage.

### 8. xG katmanı

Sportmonks xG metrikleri: `xG, xGA, xGD, xGoT, xPTS, npxG, xG Open Play,
xG Set Play, xG Free Kick, xG Corners, xG Penalties`.

Kırılımlar: `season, last_10, last_5, home, away`.

```text
home_xg_for_last_5, home_xg_against_last_5
away_xg_for_last_5, away_xg_against_last_5
home_xg_diff, away_xg_diff
```

### 9. Oyuncu ve kadro verisi

Oyuncu: `player_id, position, age, minutes, starts, goals, assists, xg, xa,
shots, shots_on_target, key_passes`.

Maç öncesi availability: `injured, suspended, questionable, available`.

Takım seviyesinde aggregate:

```text
missing_minutes, missing_goals, missing_xg, missing_assists, missing_defensive_actions
```

İleri sürüm:

```text
expected_starting_xi_strength, bench_strength, attack_strength_missing, defence_strength_missing
```

### 10. Lineup

Maç öncesi snapshot'ları:

```text
T-24h, T-90m, T-45m, T-30m, T-kickoff
```

Resmi lineup geldiğinde model yeniden tahmin üretir.
**Lineup maçtan sonra oluşuyorsa geçmiş training sample'a eklenmez.**

### 11. Sakatlık / ceza

```text
player_id, team_id, status, reason, reported_at, start_date, expected_return
```

Feature'lar:

```text
home_missing_players, away_missing_players
home_missing_xg, away_missing_xg
home_missing_minutes, away_missing_minutes
home_defensive_absence, away_defensive_absence
```

Geçmiş modelde yalnızca maçtan önce kamuya açık olan snapshot kullanılır;
"şu anki injury listesi" kullanılmaz.

### 12. Teknik direktör

```text
coach_id, coach_tenure_matches, new_coach_flag, coach_points_per_game,
coach_attack_strength, coach_defence_strength
```

Yeni hocada `new_coach_flag = 1`; ilk 3-5 maçta eski/yeni dönem ayrı tutulur.

### 13. Hakem

```text
referee_id, yellow_cards_avg, red_cards_avg, fouls_avg, penalties_avg
```

Cards / penalty / foul modellerinde kullanılır; gol modeline düşük ağırlık.

### 14. Fikstür yoğunluğu

```text
days_since_last_match, matches_last_7_days, matches_last_14_days,
travel_distance, days_until_next_match
```

### 15. Lig feature'ları

```text
league_avg_goals, league_home_goal_avg, league_away_goal_avg,
league_draw_rate, league_btts_rate, league_over25_rate,
league_corner_avg, league_card_avg, league_strength, country_strength, competition_type
```

### 16. H2H

```text
h2h_last_5_home_win_rate, h2h_last_5_draw_rate, h2h_last_5_away_win_rate,
h2h_goals_avg, h2h_btts_rate
```

Limit: `maximum_history = 5`. H2H güçlü bir temel feature değildir.

### 17. Market odds

İki ayrı model:

- **Model A (NO ODDS):** gerçek futbol tahmin gücünü ölçer.
- **Model B (WITH ODDS):** market ile birlikte tahmin üretir.

Karşılaştırma: `football_model_probability` vs `market_probability` → `edge`.

Odds feature'ları:

```text
opening_home_odds, opening_draw_odds, opening_away_odds
current_home_odds, current_draw_odds, current_away_odds
closing_home_odds, closing_draw_odds, closing_away_odds
```

Canlı sistemde:

```text
odds_timestamp, bookmaker, market_type, selection, line, price
```

### 18. Odds'ta kritik kural — POINT-IN-TIME DATA

T-60m tahmininde `closing odds` henüz bilinmez; yalnızca o ana kadar görülen
odds kullanılabilir. Closing odds yalnızca evaluation/CLV için kullanılır.

---

## 19-21. VERİ ŞEMASI (PostgreSQL)

Önerilen tablolar:

```text
leagues, seasons, teams, players, coaches, venues, referees,
fixtures, fixture_events, fixture_statistics, fixture_lineups,
fixture_player_statistics, team_match_stats, player_match_stats,
injuries, suspensions, odds_snapshots, team_ratings,
team_form_snapshots, standings_snapshots, weather_snapshots,
model_predictions, prediction_results, data_quality_log,
model_versions
```

```sql
CREATE TABLE fixtures (
    fixture_id BIGINT PRIMARY KEY,
    league_id BIGINT NOT NULL,
    season_id BIGINT,
    home_team_id BIGINT NOT NULL,
    away_team_id BIGINT NOT NULL,
    kickoff_at TIMESTAMPTZ NOT NULL,
    status VARCHAR(30),
    home_goals INTEGER,
    away_goals INTEGER,
    ht_home_goals INTEGER,
    ht_away_goals INTEGER,
    venue_id BIGINT,
    referee_id BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

```sql
CREATE TABLE odds_snapshots (
    id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT NOT NULL,
    bookmaker_id BIGINT,
    captured_at TIMESTAMPTZ NOT NULL,
    market VARCHAR(50),
    selection VARCHAR(100),
    line NUMERIC,
    price NUMERIC,
    source VARCHAR(50)
);
```

---

## 22-23. CANLI VERİ

Her değişimde snapshot (12:00, 12:01, 12:02...). Canlı feature'lar:

```text
minute, score, xg_home, xg_away, shots_home, shots_away, sot_home, sot_away,
corners_home, corners_away, red_cards_home, red_cards_away,
yellow_cards_home, yellow_cards_away, possession_home, possession_away
```

**Geleceğe bakma yasağı:** 65. dakika tahmini yalnızca 0-65 dk verisini
kullanabilir. 70./78. dk bilgisi ve maç sonucu kullanılamaz.

---

## 24-26. FEATURE ENGINEERING

```text
fixture_id, prediction_time, home_team_id, away_team_id,
home_elo, away_elo, elo_diff,
home_form_5, away_form_5, home_xg_5, away_xg_5,
home_xga_5, away_xga_5, home_shots_5, away_shots_5,
home_sot_5, away_sot_5, home_corners_5, away_corners_5,
home_position, away_position, home_rest_days, away_rest_days,
home_missing_xg, away_missing_xg, home_lineup_strength, away_lineup_strength,
home_odds, draw_odds, away_odds
```

Windowlar: `last_3, last_5, last_8, last_10, last_20, season_to_date`.
Ağırlıkları model öğrenir. Recency decay (örn. `weight = exp(-days_old / 60)`),
ancak tüm sistemde agresif decay uygulanmaz.

---

## 27-33. MODEL MİMARİSİ

```
                 RAW DATA -> VALIDATION -> NORMALIZED DATA
                          -> FEATURE GENERATOR
        +-----------+-----------+
        |           |           |
       ELO      GOAL MODEL   ML MODEL
        |           |           |
        +-----------+-----------+
              ENSEMBLE LAYER -> CALIBRATION -> OUTPUT
```

- **Model 1 — ELO:** takım gücü baseline'ı; `P(home), P(draw), P(away)` üretir.
- **Model 2 — Dixon-Coles / Poisson:** `lambda_home, lambda_away` üretir,
  skor olasılık matrisi çıkarır (0-0, 1-0, 1-1, 2-1...). Dixon-Coles düşük
  skor bağımlılığını düzeltir. Çıktı: `expected_home_goals,
  expected_away_goals, score_matrix`.
- **Model 3 — CatBoost:** tabular + categorical (team IDs, league, coach,
  referee). `CatBoostClassifier` (1X2, BTTS, Over 2.5) +
  `CatBoostRegressor` (home/away goals).
- **Model 4 — LightGBM/XGBoost:** numeric feature'lar için benchmark.

İlk sürümde **Transformer/LSTM/DNN yok** (bkz. bölüm 32). Önce Elo,
Poisson, CatBoost; sonra derin model.

**Ensemble:** validation'dan öğrenilen ağırlıklarla, örn.
`0.20 * ELO + 0.30 * Poisson + 0.50 * CatBoost`. Ağırlıklar kafadan
belirlenip kalıcı yapılmaz; validation setinde optimize edilir.

---

## 34-39. CALIBRATION VE ALT MODELLER

- Calibration: Platt Scaling, Isotonic Regression, Beta Calibration
  (validation ile seçilir). Ana metrikler: **Log Loss, Brier Score,
  Calibration Error**. Accuracy tek başına yeterli değildir.
- 1X2 hedef: `0=Home, 1=Draw, 2=Away`; output `{home_win, draw, away_win}`.
- Gol modelinden `total_goals, BTTS, over_under, exact_score` türetilir.
- Exact score: skor matrisinden (örn. 1-1 %13, 2-1 %10, 1-0 %11).
- Corners: Poisson / Negative Binomial / LightGBM benchmark (aşırı saçılma).
- Cards: Negative Binomial / LightGBM / CatBoost; hakem feature'ları.

---

## 40-44. MAÇ ÖNCESİ VE CANLI MODEL

Tahmin lifecycle: `T-48h, T-24h, T-6h, T-90m, T-60m, T-45m (lineup), T-30m,
T-15m, KICKOFF`. Her snapshot ayrı `prediction_id`
(örn. `12345_T-24H`).

Canlı model segmentlerden öğrenir: `0-15, 15-30, 30-45, 45-60, 60-75, 75-90`.

Canlı feature'lar:

```text
minute, score_diff, home_goals, away_goals,
current_xg_diff, current_shot_diff, current_sot_diff, current_corner_diff,
red_card_diff, yellow_card_diff, substitutions_home, substitutions_away,
pre_match_home_strength, pre_match_away_strength, pre_match_expected_goals,
live_odds, odds_movement
```

Canlı model pre-match gücünü ve dakika-içi istatistiği birlikte kullanır.
Market referanstır; hedef "model marketten farklı ve güvenilir bir olasılık
çıkarabiliyor mu?" sorusudur. İlk testte odds'suz model şarttır.

---

## 45-49. DATA LEAKAGE VE VALİDASYON

Kullanılmayacaklar (maçtan önce bilinmiyorsa): `final_score, final_xg,
final_shots, closing odds, post-match lineup/injury/rating, future
standings, future form`.

- Standings snapshot'ı maçtan **önce** (örn. Saturday 17:59) alınır.
- Feature store: `feature_name, value, valid_at, source, created_at`;
  `valid_at` tahmin zamanını geçemez.
- Split: random değil — zaman serisi. `2020-2022 → Train, 2023 → Valid,
  2024 → Test`.
- Walk-forward:

```text
Fold 1: Train 2020-2022, Valid 2023 Q1
Fold 2: Train 2020-2023 Q1, Valid 2023 Q2
Fold 3: Train 2020-2023 Q2, Valid 2023 Q3
```

---

## 50-53. METRİKLER VE OUTPUT

| Alan | Metrikler |
|---|---|
| 1X2 | Log Loss, Brier, Calibration; ikincil: Accuracy, Macro F1, Balanced Accuracy |
| Gol | MAE, RMSE, Poisson Deviance |
| BTTS / O-U | Log Loss, Brier, AUC, Calibration |
| Betting | ROI, Yield, CLV, Max Drawdown, Hit Rate (yalnızca out-of-sample) |

CLV: tahmin anındaki odds ile closing odds karşılaştırması.

```json
{
  "fixture_id": 123456,
  "prediction_time": "2026-08-14T17:30:00Z",
  "result": { "home": 0.481, "draw": 0.281, "away": 0.238 },
  "goals": { "home_lambda": 1.74, "away_lambda": 0.91, "expected_total": 2.65 },
  "btts": { "yes": 0.552, "no": 0.448 },
  "over_under_2_5": { "over": 0.571, "under": 0.429 },
  "top_scores": [
    { "score": "1-0", "probability": 0.118 },
    { "score": "1-1", "probability": 0.104 },
    { "score": "2-0", "probability": 0.093 }
  ]
}
```

Güven skoru: uydurma "confidence %93" üretilmez; `probability,
calibration, model_agreement, data_completeness, prediction_stability`
temelinde reliability raporu verilir.

---

## 54-56. VERİ KALİTESİ VE ALTYAPI

- Data quality engine: duplicate fixture, invalid team, unknown league,
  future event, missing date, negative shots/goals, impossible score,
  duplicate odds, bad timestamp kontrolleri.
- Entity resolution: `Man Utd / Manchester United / Manchester United FC`
  → tek `canonical_team_id`. `providers, provider_team_id, canonical_team_id,
  valid_from, valid_to`.
- Altyapı: PostgreSQL (metadata/sorgu) + Parquet (training/history) +
  Redis (canlı state/cache).

---

## 57-62. DOSYA MİMARİSİ VE INGESTION

```
football_prediction/
├── data/          raw/ bronze/ silver/ gold/
├── ingestion/     sportmonks/ api_football/ football_data/
├── normalization/
├── feature_engine/
├── models/        elo/ poisson/ catboost/ lightgbm/ ensemble/
├── calibration/
├── live/
├── evaluation/
├── registry/
├── monitoring/
├── api/
├── jobs/
├── configs/
└── tests/
```

- Bronze: API yanıtı olduğu gibi (`bronze/source/date/response.json`).
- Silver: normalize tablolar. Gold: `features.parquet` (ML-ready).
- Provider arayüzü:

```python
class FootballDataProvider:
    def get_fixtures(...)
    def get_statistics(...)
    def get_odds(...)
    def get_lineups(...)
    def get_injuries(...)
```

- API çağrıları: timeout, retry, exponential backoff, rate-limit handling,
  circuit breaker, cache.
- API-Football canlı polling: score/events 15-30 sn, statistics 30-60 sn
  (kendi rate limitine göre dinamik ayarlanır).

---

## 63-65. DÜŞÜK ÖNCELİKLİ / İLERİ KATMANLAR

- Weather: düşük öncelik.
- Transferler: ilk sürümde `squad_strength`, `minutes_weighted_strength`;
  sonra `player_strength` (`player_attack_rating, player_defence_rating,
  player_creation_rating, player_goalkeeping_rating` → `starting_xi_attack,
  starting_xi_defence`).

---

## 66-72. TRAIN PIPELINE, REGISTRY, DEPLOYMENT

```
extract → validate → normalize → deduplicate → snapshot reconstruction
→ feature generation → time split → train → calibrate → evaluate
→ register model → deploy
```

- Registry: `model_id, version, training_start/end, features_hash,
  dataset_version, algorithm, hyperparameters, metrics,
  calibration_method, created_at` (örn. `pre_match_1x2_v17`).
- Deployment: `candidate → staging → production` (yalnızca `approved`).
- Retraining: haftalık full retrain; ileri: günlük incremental, aylık
  feature audit.
- Drift: feature/prediction/calibration/league/team drift izlenir (örn.
  Brier 0.192 → 0.234 = alarm).
- Lig: başlangıçta global model + `league_id, league_strength,
  league_avg_goals`; yeterli veri olan büyük liglere özel modeller yalnızca
  validation destekliyorsa.
- Multi-task (shared encoder + 1X2/Goals/BTTS/Corners/Cards head): ilk
  sürümde değil; önce ayrı güçlü modeller + ensemble.

---

## 73-77. AŞAMALAR

| Aşama | Kapsam |
|---|---|
| MVP | 10-15 güçlü lig, 2020-2026; results, goals, shots, SOT, corners, xG, xGA, home/away, form, Elo, standings, rest, odds; Elo + Dixon-Coles + CatBoost; 1X2, home/away goals, BTTS, Over 2.5 |
| 2 | injuries, lineup, players, coach, referee, weather, transfer |
| 3 | live events, live xG, live shots/SOT/corners/cards/odds |
| 4 | player-aware: expected XI, player strength, missing player impact, formation |
| 5 | Temporal Transformer / TabTransformer / TFT (yalnızca yeterli snapshot sonrası) |

**MVP sağlam çalışmadan player transformer, RL, LLM, computer vision eklemek anlamsızdır.**

---

## 78-80. VERİ HACMİ VE TEK KAYNAK SORUNU

Öncelik: `fixtures → results → league → team IDs → standings → match stats
→ xG → odds → lineups → injuries → player stats → referee → weather →
event-level coordinates`.

Hedef: 6+ sezon, 10+ güçlü lig (ideal 8-12 sezon). Feature coverage matrisi:

```text
league | season | feature | coverage %
```

Eksik coverage'ta ikinci kaynak veya fallback gerekir. Tek kaynağın
farklı ID'leri için internal mapping şarttır.

---

## 81-84. SNAPSHOT / TRAINING VERİSİ

```python
def validate_fixture(row):
    if row["home_goals"] is not None:
        assert row["home_goals"] >= 0
    if row["away_goals"] is not None:
        assert row["away_goals"] >= 0
    assert row["home_team_id"] != row["away_team_id"]
    assert row["kickoff_at"] is not None
```

Feature snapshot (model input):

```json
{
  "fixture_id": 123,
  "prediction_time": "2026-08-14T15:00:00Z",
  "home": { "elo": 1687, "form_5": 0.72, "xg_5": 1.91, "xga_5": 0.93, "shots_5": 14.8 },
  "away": { "elo": 1602, "form_5": 0.48, "xg_5": 1.31, "xga_5": 1.27, "shots_5": 10.7 },
  "context": { "home_rest_days": 6, "away_rest_days": 3, "home_missing_xg": 0.15, "away_missing_xg": 0.42 }
}
```

Training data:

```text
fixture_id, prediction_time, feature_001..., target_home_win, target_draw,
target_away_win, target_home_goals, target_away_goals, target_btts, target_over25
```

Model geçmişi ezberlemez; takım gücü + form + xG + kadro + bağlam + lig +
market → gelecek maçın olasılık dağılımını öğrenir. Çok boyutlu state, tek
bir "son 10 maçta 7 galibiyet" sayısından değerlidir.

---

## 85-90. FİNAL MİMARİ VE TEMEL PRENSİPLER

```
DATA PROVIDERS (API-Football, Sportmonks, Football-Data)
  → RAW DATA LAKE → NORMALIZATION → ENTITY RESOLUTION → POSTGRESQL
  → SNAPSHOT RECONSTRUCTION → FEATURE ENGINE
  → ELO | DIXON-COLES | CATBOOST → ENSEMBLE → CALIBRATION
  → PRE-MATCH OUTPUT → KICKOFF → LIVE DATA → LIVE MODEL → FINAL OUTPUT
  → RESULT INGESTION → EVALUATION → MODEL MONITORING → RE-TRAINING
```

Teknoloji: Python 3.13, FastAPI, pandas/polars, scikit-learn, CatBoost,
LightGBM, SciPy, statsmodels, MLflow, APScheduler/Celery/Prefect.
Başlangıçta ağır dağıtık sistem kurulmaz.

**Her tahmin için zorunlu kayıt:**

```text
prediction_id, fixture_id, prediction_time, model_version,
dataset_version, feature_version, source_versions, probabilities,
odds_at_prediction, actual_result, evaluation
```

**En önemli prensip:** Sistem "en yüksek accuracy" üzerine kurulmaz;
"doğru zaman damgasındaki bilgiyle güvenilir probability dağılımı" üzerine
kurulur. İyi model "bu kesin kazanır" demez:

```text
Home: 51.2%  Draw: 27.4%  Away: 21.4%
Calibration: Good  Data completeness: 97%  Model agreement: High
```

İki kritik nokta:
1. **Point-in-time feature reconstruction**
2. **Walk-forward validation**

Bunlar düzgün yapılmadan elde edilen yüksek skorlar gerçek başarı değildir.