"""2. DIXON-COLES POISSON MODELI - FULLY VECTORIZED
Dixon & Coles (1997) - her sey numpy vektörize.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
import time


def neg_log_likelihood_vec(params, h_idx, a_idx, hg, ag, n_teams):
    """Vektörize negatif log-likelihood."""
    home_adv = np.exp(params[0])
    rho = params[1]
    att = params[2:2 + n_teams]
    def_ = params[2 + n_teams:2 + 2 * n_teams]

    lambda_h = np.exp(att[h_idx] + def_[a_idx] + np.log(home_adv))
    lambda_a = np.exp(att[a_idx] + def_[h_idx])

    ll = hg * np.log(lambda_h) - lambda_h - gammaln(hg + 1)
    ll += ag * np.log(lambda_a) - lambda_a - gammaln(ag + 1)

    tau = np.ones(len(hg))
    c00 = (hg == 0) & (ag == 0)
    c01 = (hg == 0) & (ag == 1)
    c10 = (hg == 1) & (ag == 0)
    c11 = (hg == 1) & (ag == 1)
    tau[c00] = 1 - lambda_h[c00] * lambda_a[c00] * rho
    tau[c01] = 1 + lambda_h[c01] * rho
    tau[c10] = 1 + lambda_a[c10] * rho
    tau[c11] = 1 - rho
    tau = np.clip(tau, 1e-10, None)
    ll += np.log(tau)

    return -np.sum(ll)


class DixonColesModel:

    def __init__(self):
        self.team_ids = {}
        self.home_adv = None
        self.rho = None
        self.att = None
        self.defense = None
        self.n_teams = 0

    def fit(self, matches_df, home_adv_init=1.25, rho_init=-0.13, max_teams=300):
        t0 = time.time()

        team_counts = (matches_df["home_team_id"].value_counts() +
                       matches_df["away_team_id"].value_counts())
        top_teams = team_counts.nlargest(max_teams).index.tolist()
        other_teams = (set(matches_df["home_team_id"].unique()) |
                       set(matches_df["away_team_id"].unique())) - set(top_teams)

        self.team_ids = {t: i for i, t in enumerate(top_teams)}
        other_idx = len(top_teams)
        for t in other_teams:
            self.team_ids[t] = other_idx
        self.n_teams = other_idx + 1

        h_idx = matches_df["home_team_id"].map(self.team_ids).fillna(other_idx).astype(int).values
        a_idx = matches_df["away_team_id"].map(self.team_ids).fillna(other_idx).astype(int).values
        hg = matches_df["home_goals"].values.astype(int)
        ag = matches_df["away_goals"].values.astype(int)

        x0 = np.zeros(2 + 2 * self.n_teams)
        x0[0] = np.log(home_adv_init)
        x0[1] = rho_init

        result = minimize(
            neg_log_likelihood_vec, x0,
            args=(h_idx, a_idx, hg, ag, self.n_teams),
            method="L-BFGS-B",
            options={"maxiter": 50, "disp": False}
        )

        self.params = result.x
        self.home_adv = np.exp(self.params[0])
        self.rho = self.params[1]
        self.att = self.params[2:2 + self.n_teams]
        self.defense = self.params[2 + self.n_teams:2 + 2 * self.n_teams]
        self.att[0] = 0
        self.defense[0] = 0

        print(f"    Dixon-Coles fitted: {self.n_teams} teams, {len(matches_df)} matches [{time.time()-t0:.0f}s]")
        print(f"    home_adv={self.home_adv:.3f}, rho={self.rho:.3f}")

    def predict_batch(self, matches_df):
        """Full vektörize batch prediction - kapalı formül ile."""
        h_idx = matches_df["home_team_id"].map(self.team_ids).fillna(self.n_teams - 1).astype(int).values
        a_idx = matches_df["away_team_id"].map(self.team_ids).fillna(self.n_teams - 1).astype(int).values
        n = len(matches_df)

        lambda_h = np.exp(self.att[h_idx] + self.defense[a_idx] + np.log(self.home_adv))
        lambda_a = np.exp(self.att[a_idx] + self.defense[h_idx])

        # Basit Poisson: P(H>D) = sum over all h>a: poisson(h,lh)*poisson(a,la)
        # Use CDF approach: P(H>A) = sum_a poisson(a,la) * (1 - CDF(a, lh))
        max_g = 10
        g = np.arange(max_g + 1)

        # P(X=k) for each lambda: shape (n, max_g+1)
        po_h = np.exp(g[np.newaxis, :] * np.log(np.maximum(lambda_h[:, np.newaxis], 1e-10)) 
                       - lambda_h[:, np.newaxis] - gammaln(g[np.newaxis, :] + 1))
        po_a = np.exp(g[np.newaxis, :] * np.log(np.maximum(lambda_a[:, np.newaxis], 1e-10))
                       - lambda_a[:, np.newaxis] - gammaln(g[np.newaxis, :] + 1))

        # Cumulative sums for CDF
        cdf_a = np.cumsum(po_a, axis=1)  # P(A <= k)
        cdf_h = np.cumsum(po_h, axis=1)  # P(H <= k)

        # P(draw) = sum_k po_h[k] * po_a[k]
        p_draw = np.sum(po_h * po_a, axis=1)

        # P(home win) = sum_h po_h[h] * P(A < h) = sum_h po_h[h] * cdf_a[h-1]
        p_home = np.zeros(n)
        p_home = po_h[:, 0] * 0  # h=0: no wins
        for h in range(1, max_g + 1):
            p_home += po_h[:, h] * cdf_a[:, h - 1]

        # P(away win) = sum_a po_a[a] * P(H < a) = sum_a po_a[a] * cdf_h[a-1]
        p_away = np.zeros(n)
        for a in range(1, max_g + 1):
            p_away += po_a[:, a] * cdf_h[:, a - 1]

        # Apply Dixon-Coles tau correction (low-score correction)
        tau = np.ones(n)
        tau_corr = np.ones(n)

        # tau for 0-0
        tau_corr = (1 - lambda_h * lambda_a * self.rho)
        # tau for 0-1
        tau_corr_01 = (1 + lambda_h * self.rho)
        # tau for 1-0
        tau_corr_10 = (1 + lambda_a * self.rho)
        # tau for 1-1
        tau_corr_11 = (1 - self.rho)

        # Recalculate with correction for exact low scores
        # P_exact(0,0) = po_h[0]*po_a[0] * (1 - lh*la*rho)
        # P_exact(0,1) = po_h[0]*po_a[1] * (1 + lh*rho)
        # P_exact(1,0) = po_h[1]*po_a[0] * (1 + la*rho)
        # P_exact(1,1) = po_h[1]*po_a[1] * (1 - rho)
        p_exact_00 = po_h[:, 0] * po_a[:, 0] * tau_corr
        p_exact_01 = po_h[:, 0] * po_a[:, 1] * tau_corr_01
        p_exact_10 = po_h[:, 1] * po_a[:, 0] * tau_corr_10
        p_exact_11 = po_h[:, 1] * po_a[:, 1] * tau_corr_11
        p_exact_rest = p_draw - po_h[:, 0] * po_a[:, 0] - po_h[:, 1] * po_a[:, 1]
        p_draw_corr = p_exact_00 + p_exact_01 + p_exact_10 + p_exact_11 + p_exact_rest

        # Fix home/away with low-score corrections
        p_home_corr = p_home - p_exact_10 + po_h[:, 1] * po_a[:, 0] * tau_corr_10
        p_away_corr = p_away - p_exact_01 + po_h[:, 0] * po_a[:, 1] * tau_corr_01

        probs = np.column_stack([p_home_corr, p_draw_corr, p_away_corr])
        total = probs.sum(axis=1, keepdims=True)
        total = np.where(total > 0, total, 1)
        probs /= total
        probs = np.clip(probs, 1e-10, 1)

        return probs
