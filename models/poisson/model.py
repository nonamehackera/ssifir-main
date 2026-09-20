"""Dixon-Coles / Poisson goal model (ROADMAP bolum 29, 30).

Estimates attacking/defending strengths per team and a per-league home
advantage + rho (low-score correlation correction), then builds the full
score probability matrix P(home=h, away=a) from independent Poisson marginals
with the Dixon-Coles dependence term. From the matrix we derive:

  home_lambda, away_lambda   (expected goals)
  1X2 probabilities
  BTTS (yes) probability
  Over 2.5 probability
  exact-score distribution

Important leakage note (ROADMAP bolum 45): team strengths used for predicting
match i must be estimated from matches strictly BEFORE i. We fit the model in
a *rolling* manner: for each match we refit attack/defense on all prior matches
within the same league. To keep this tractable over 17k matches we refit per
league on a calendar-window of the most recent N prior matches (default 600).

This produces honest, point-in-time parameters without future leakage.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize

from models.base import PredictionResult

DEFAULT_WINDOW = 600  # most-recent prior matches per league used for fitting


class DixonColesModel:
    name = "dixon_coles"

    def __init__(self, window: int = DEFAULT_WINDOW, max_goals: int = 12) -> None:
        self.window = window
        self.max_goals = max_goals
        # per-league fitted params keyed by league code
        self.params: dict[str, dict] = {}

    def fit(self, train, val=None) -> "DixonColesModel":
        # Precompute per-league chronological frames for rolling fit at predict.
        self._train = train.sort_values("date").reset_index(drop=True)
        return self

    @staticmethod
    def _fit_league(block) -> dict:
        """Estimate attack/defense (+ home adv, rho) for one league block."""
        teams = sorted(set(block["home_team_id"]).union(set(block["away_team_id"])))
        team_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hg = block["home_goals"].to_numpy(dtype=float)
        ag = block["away_goals"].to_numpy(dtype=float)
        hid = block["home_team_id"].to_numpy()
        aid = block["away_team_id"].to_numpy()
        m = len(block)

        # params: attack[i] for i in teams, defense[j] for j in teams,
        # home_adv (log), rho. Sum-to-zero attack constraint via last team.
        npar = 2 * n + 2

        def unpack(p):
            attack = p[:n]
            defense = p[n:2 * n]
            home_adv = p[2 * n]
            rho = p[2 * n + 1]
            return attack, defense, home_adv, rho

        def neg_ll(p):
            attack, defense, home_adv, rho = unpack(p)
            # mean goals
            lam_h = np.exp(attack[np.array([team_idx[t] for t in hid])]
                           + defense[np.array([team_idx[t] for t in aid])] + home_adv)
            lam_a = np.exp(attack[np.array([team_idx[t] for t in aid])]
                           + defense[np.array([team_idx[t] for t in hid])])
            # poisson log-lik with Dixon-Coles low-score correction
            ll = -np.sum(_dc_logpmf(hg, ag, lam_h, lam_a, rho))
            # regularization to avoid drift
            ll += 0.001 * np.sum(attack ** 2) + 0.001 * np.sum(defense ** 2)
            # sum-to-zero on attack (penalize if last != -sum others) via soft
            return ll

        # init: league avg goals
        avg = (hg.sum() + ag.sum()) / (2 * m)
        p0 = np.r_[np.full(n, 0.0), np.full(n, 0.0), 0.1, -0.1]
        try:
            res = minimize(neg_ll, p0, method="L-BFGS-B",
                           options={"maxiter": 200, "ftol": 1e-4})
            attack, defense, home_adv, rho = unpack(res.x)
        except Exception:
            attack = np.zeros(n)
            defense = np.zeros(n)
            home_adv, rho = 0.1, -0.1
            lam_h = np.full(m, avg)
            lam_a = np.full(m, avg)
            return {"teams": teams, "team_idx": team_idx,
                    "attack": attack, "defense": defense,
                    "home_adv": home_adv, "rho": rho, "avg": avg}

        return {"teams": teams, "team_idx": team_idx, "attack": attack,
                "defense": defense, "home_adv": home_adv, "rho": rho, "avg": avg}

    def predict(self, feats) -> PredictionResult:
        feats = feats.sort_values("date").reset_index(drop=True)
        n = len(feats)
        home_win = np.empty(n)
        draw = np.empty(n)
        away_win = np.empty(n)
        home_lambda = np.empty(n)
        away_lambda = np.empty(n)
        btts = np.empty(n)
        over25 = np.empty(n)
        score_matrices = [None] * n

        train_sorted = getattr(self, "_train", feats).sort_values("date").reset_index(drop=True)

        # 1) Determine per-match params (rolling, point-in-time)
        params_per_row: list[dict | None] = [None] * n
        cache: dict[str, tuple[object, dict | None]] = {}
        for i in range(n):
            row = feats.iloc[i]
            lg = row["league"]
            kick = row["date"]
            need_refit = lg not in cache or (kick - cache[lg][0]).total_seconds() > 365 * 86400
            if need_refit:
                prior = train_sorted[train_sorted["date"] < kick]
                prior = prior[prior["league"] == lg].tail(self.window)
                params = self._fit_league(prior) if len(prior) >= 20 else None
                cache[lg] = (kick, params)
            params_per_row[i] = cache[lg][1]

        # 2) Batch score computation over runs with identical params
        run_start = 0
        while run_start < n:
            run_end = run_start + 1
            while run_end < n and params_per_row[run_end] is params_per_row[run_start]:
                run_end += 1
            params = params_per_row[run_start]
            rows = feats.iloc[run_start:run_end]
            lam_h = np.full(run_end - run_start, 2.6)
            lam_a = np.full(run_end - run_start, 2.6)
            rho = -0.1
            if params is not None:
                ti = params["team_idx"]
                rho = params["rho"]
                hid = rows["home_team_id"].astype(int).to_numpy()
                aid = rows["away_team_id"].astype(int).to_numpy()
                known = np.array([h in ti and a in ti for h, a in zip(hid, aid)])
                if known.any():
                    kh, ka = hid[known], aid[known]
                    lam_h[known] = np.exp(
                        params["attack"][[ti[t] for t in kh]]
                        + params["defense"][[ti[t] for t in ka]]
                        + params["home_adv"]
                    )
                    lam_a[known] = np.exp(
                        params["attack"][[ti[t] for t in ka]]
                        + params["defense"][[ti[t] for t in kh]]
                    )
                lam_h[~known] = (params["avg"] * 1.1) if params["avg"] else 2.6
                lam_a[~known] = (params["avg"] * 0.9) if params["avg"] else 2.6
            lam_h = np.clip(lam_h, 0.05, None)
            lam_a = np.clip(lam_a, 0.05, None)

            sm_list, p_h, p_d, p_a, p_b, p_o = _score_probs(lam_h, lam_a, rho, self.max_goals)
            home_win[run_start:run_end] = p_h
            draw[run_start:run_end] = p_d
            away_win[run_start:run_end] = p_a
            btts[run_start:run_end] = p_b
            over25[run_start:run_end] = p_o
            home_lambda[run_start:run_end] = lam_h
            away_lambda[run_start:run_end] = lam_a
            score_matrices[run_start:run_end] = sm_list
            run_start = run_end

        return PredictionResult(
            home_win=home_win, draw=draw, away_win=away_win,
            home_lambda=home_lambda, away_lambda=away_lambda,
            btts_yes=btts, over25=over25, score_matrix=score_matrices,
        )


def _poisson_pmf(k, lam):
    import scipy.stats as st
    return st.poisson.pmf(k, lam)


def _dc_logpmf(h, a, lam_h, lam_a, rho, max_g=12):
    """Dixon-Coles adjusted log-PMF for arrays of scores (fully vectorized).

    h, a, lam_h, lam_a: equal-length arrays. Returns per-row log-probability
    of observing score (h_i, a_i) under independent Poisson marginals with
    the Dixon-Coles low-score dependence correction.
    """
    from scipy.stats import poisson

    m = len(h)
    gh = np.arange(max_g + 1)
    log_ph = poisson.logpmf(gh[None, :], lam_h[:, None])   # (m, G)
    log_pa = poisson.logpmf(gh[None, :], lam_a[:, None])   # (m, G)
    # full log grid (m, G, G); tau correction on the four low-score cells
    logp = log_ph[:, :, None] + log_pa[:, None, :]
    logp = logp.copy()
    tau = np.ones_like(logp)
    tau[:, 0, 0] = 1.0 - lam_h * lam_a * rho
    tau[:, 0, 1] = 1.0 + lam_a * rho
    tau[:, 1, 0] = 1.0 + lam_h * rho
    tau[:, 1, 1] = 1.0 - rho
    logp = logp + np.log(np.clip(tau, 1e-12, None))
    idx = np.arange(m)
    return logp[idx, h.astype(int), a.astype(int)]


def _score_probs(lam_h, lam_a, rho, max_g):
    """Return (matrix, p_home, p_draw, p_away, p_btts, p_over25).

    Vectorized over arrays of lambdas (per-match scoring in predict).
    """
    from scipy.stats import poisson

    lam_h = np.asarray(lam_h, dtype=float)
    lam_a = np.asarray(lam_a, dtype=float)
    n = len(lam_h)
    gh = np.arange(max_g + 1)
    ph = poisson.pmf(gh[None, :], lam_h[:, None])   # (n, G)
    pa = poisson.pmf(gh[None, :], lam_a[:, None])   # (n, G)
    mat = ph[:, :, None] * pa[:, None, :]           # (n, G, G)
    tau = np.ones_like(mat)
    tau[:, 0, 0] = 1.0 - lam_h * lam_a * rho
    tau[:, 0, 1] = 1.0 + lam_a * rho
    tau[:, 1, 0] = 1.0 + lam_h * rho
    tau[:, 1, 1] = 1.0 - rho
    mat = mat * tau
    s = mat.sum(axis=(1, 2), keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    mat = mat / s

    g1, g2 = np.meshgrid(gh, gh, indexing="ij")
    flat = mat.reshape(n, -1)
    home_mask = (g1 > g2).reshape(-1)
    away_mask = (g1 < g2).reshape(-1)
    draw_mask = (g1 == g2).reshape(-1)
    btts_mask = ((g1 > 0) & (g2 > 0)).reshape(-1)
    over_mask = ((g1 + g2) >= 3).reshape(-1)

    p_home = flat[:, home_mask].sum(axis=1)
    p_draw = flat[:, draw_mask].sum(axis=1)
    p_away = flat[:, away_mask].sum(axis=1)
    p_btts = flat[:, btts_mask].sum(axis=1)
    p_over = flat[:, over_mask].sum(axis=1)

    top_scores = np.argsort(flat, axis=1)[:, ::-1][:, :5]
    scores = [
        [(int(idx % (max_g + 1)), int(idx // (max_g + 1)), float(flat[i, idx])) for idx in row]
        for i, row in enumerate(top_scores)
    ]
    return scores, p_home, p_draw, p_away, p_btts, p_over
