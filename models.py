"""Model katmanı.

- DixonColes: her takım için hücum/savunma gücü, ev sahibi avantajı ve düşük skor
  düzeltmesi (rho). Eski maçlar üstel olarak daha az ağırlık alır (form etkisi).
- Monte Carlo: parametre belirsizliği eklenerek maç binlerce kez simüle edilir.
- Elo: gol farkı çarpanlı, sezon başında ortalamaya çekilen klasik puanlama.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


class DixonColes:
    def __init__(self, xi: float = 0.0019, reg: float = 1.0, sot: float = 0.0, bc: float = 0.0):
        # xi: günlük ağırlık azalması. 0.0019 => bir maçın ağırlığı ~1 yılda yarıya iner.
        # reg: az maçı olan takımların (ör. yeni çıkanlar) aşırı uç değer almasını engeller.
        # sot: isabetli şut ağırlığı, bc: büyük şans ağırlığı (toplamları en fazla 1).
        #      Takım gücü, gerçek goller ile şut/şans sayılarından türetilen "beklenen gol"ün
        #      karışımından öğrenilir. Bunlar golden daha az şansa bağlı olduğu için gücü daha
        #      istikrarlı gösterir (xG'nin basit bir yerine geçer). Verisi olmayan maçta sadece
        #      gerçek gol kullanılır.
        if sot + bc > 1:
            raise ValueError("sot + bc en fazla 1 olabilir")
        self.xi = xi
        self.reg = reg
        self.sot = sot
        self.bc = bc
        self.teams: list[str] = []

    # ---------- eğitim ----------
    def fit(self, df: pd.DataFrame, ref_date=None) -> "DixonColes":
        if ref_date is None:
            ref_date = df["Date"].max() + pd.Timedelta(days=1)
        ref_date = pd.Timestamp(ref_date)
        d = df[df["Date"] < ref_date]
        days = (ref_date - d["Date"]).dt.days.to_numpy(float)
        w = np.exp(-self.xi * days)
        keep = w > 1e-3
        d, w = d[keep], w[keep]
        if len(d) < 50:
            raise RuntimeError("Model için yeterli maç yok.")

        teams = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
        ix = {t: i for i, t in enumerate(teams)}
        h = d["HomeTeam"].map(ix).to_numpy()
        a = d["AwayTeam"].map(ix).to_numpy()
        x = d["FTHG"].to_numpy(float)
        y = d["FTAG"].to_numpy(float)
        n = len(teams)
        # düşük skor düzeltmesi (rho) her zaman GERÇEK skorlara göre
        m00, m01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
        m10, m11 = (x == 1) & (y == 0), (x == 1) & (y == 1)
        self.sot_used, self.bc_used = 0.0, 0.0
        base_w = np.ones_like(x)        # gerçek golün payı (maç bazında)
        add_x, add_y = np.zeros_like(x), np.zeros_like(y)
        for weight, hc, ac, attr in ((self.sot, "HST", "AST", "sot"), (self.bc, "HBC", "ABC", "bc")):
            if weight <= 0 or not {hc, ac} <= set(d.columns):
                continue
            hv = pd.to_numeric(d[hc], errors="coerce").to_numpy(float)
            av = pd.to_numeric(d[ac], errors="coerce").to_numpy(float)
            ok = ~np.isnan(hv) & ~np.isnan(av)
            if ok.sum() < 100:
                continue
            # şutu/şansı gole çeviren lig oranı (ör. 3 isabetli şutta 1 gol)
            conv = np.sum(w[ok] * (x[ok] + y[ok])) / max(np.sum(w[ok] * (hv[ok] + av[ok])), 1e-9)
            add_x += np.where(ok, weight * conv * np.nan_to_num(hv), 0.0)
            add_y += np.where(ok, weight * conv * np.nan_to_num(av), 0.0)
            base_w -= np.where(ok, weight, 0.0)
            setattr(self, f"{attr}_used", weight)
            setattr(self, f"{attr}_conv", float(conv))
        x = base_w * x + add_x
        y = base_w * y + add_y
        reg = self.reg

        def nll_grad(p):
            r, dd, g, rho = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
            att = r - r.mean()
            lam = np.exp(g + att[h] + dd[a])
            mu = np.exp(att[a] + dd[h])
            t = np.ones_like(lam)
            tl, tm, tr = np.zeros_like(lam), np.zeros_like(lam), np.zeros_like(lam)

            lm = lam[m00] * mu[m00]
            t[m00] = np.maximum(1 - lm * rho, 1e-10)
            tl[m00] = -lm * rho / t[m00]
            tm[m00] = tl[m00]
            tr[m00] = -lm / t[m00]

            t[m01] = np.maximum(1 + lam[m01] * rho, 1e-10)
            tl[m01] = lam[m01] * rho / t[m01]
            tr[m01] = lam[m01] / t[m01]

            t[m10] = np.maximum(1 + mu[m10] * rho, 1e-10)
            tm[m10] = mu[m10] * rho / t[m10]
            tr[m10] = mu[m10] / t[m10]

            t[m11] = max(1 - rho, 1e-10)
            tr[m11] = -1 / t[m11]

            ll = w * (np.log(t) + x * np.log(lam) - lam + y * np.log(mu) - mu)
            gl = w * (x - lam + tl)
            gm = w * (y - mu + tm)
            ga = np.bincount(h, gl, n) + np.bincount(a, gm, n)
            gd = np.bincount(a, gl, n) + np.bincount(h, gm, n)
            ga -= ga.mean()
            dc = dd - dd.mean()
            nll = -ll.sum() + reg * (att @ att + dc @ dc)
            grad = np.concatenate([-ga + 2 * reg * att, -gd + 2 * reg * dc,
                                   [-gl.sum(), -(w * tr).sum()]])
            return nll, grad

        p0 = np.concatenate([np.zeros(n), np.full(n, np.log(max(y.mean(), 0.2))), [0.25, -0.05]])
        bounds = [(None, None)] * (2 * n + 1) + [(-0.25, 0.25)]
        res = minimize(nll_grad, p0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000})
        p = res.x
        att = p[:n] - p[:n].mean()
        self.teams = teams
        self.att = dict(zip(teams, att))
        self.dfn = dict(zip(teams, p[n:2 * n]))
        self.home_adv = float(p[2 * n])
        self.rho = float(p[2 * n + 1])
        self.weight = dict(zip(teams, np.bincount(h, w, n) + np.bincount(a, w, n)))
        self.ref_date = ref_date
        self._fit_nll = nll_grad  # testler için
        self._p = p
        return self

    # ---------- tahmin ----------
    def _team_params(self, team: str):
        if team in self.att:
            return self.att[team], self.dfn[team], False
        # Bilinmeyen takım (ör. ligden yeni çıkmış): en zayıf 3 takımın ortalaması.
        net = sorted(self.teams, key=lambda t: self.att[t] - self.dfn[t])[:3]
        return (float(np.mean([self.att[t] for t in net])),
                float(np.mean([self.dfn[t] for t in net])), True)

    def expected_goals(self, home: str, away: str):
        ah, dh, _ = self._team_params(home)
        aa, da, _ = self._team_params(away)
        lam = np.exp(self.home_adv + ah + da)
        mu = np.exp(aa + dh)
        return float(lam), float(mu)

    def score_matrix(self, home: str, away: str, sd: float = 0.10, n_draws: int = 400,
                     seed: int = 0) -> np.ndarray:
        """Skor olasılık matrisi: satır = ev sahibi golü, sütun = deplasman golü.

        sd > 0 ise beklenen gollere belirsizlik eklenir (sakatlık, kadro, şans vb.)
        ve sonuç bu senaryoların ortalamasıdır.
        """
        lam, mu = self.expected_goals(home, away)
        if sd > 0:
            rng = np.random.default_rng(seed)
            f = np.exp(rng.normal(-sd ** 2 / 2, sd, size=(n_draws, 2)))
            L, M = lam * f[:, 0], mu * f[:, 1]
        else:
            L, M = np.array([lam]), np.array([mu])
        g = np.arange(MAX_GOALS + 1)
        ph = poisson.pmf(g[None, :], L[:, None])
        pa = poisson.pmf(g[None, :], M[:, None])
        mats = ph[:, :, None] * pa[:, None, :]
        rho = self.rho
        mats[:, 0, 0] *= 1 - L * M * rho
        mats[:, 0, 1] *= 1 + L * rho
        mats[:, 1, 0] *= 1 + M * rho
        mats[:, 1, 1] *= 1 - rho
        mats = np.clip(mats, 0, None)
        mats /= mats.sum(axis=(1, 2), keepdims=True)
        return mats.mean(axis=0)

    def predict(self, home: str, away: str, sd: float = 0.10, n_sim: int = 10000,
                seed: int = 0) -> dict:
        mat = self.score_matrix(home, away, sd=sd, seed=seed)
        out = summarize(mat)
        lam, mu = self.expected_goals(home, away)
        out.update(home=home, away=away, xg_home=lam, xg_away=mu,
                   unknown=[t for t in (home, away) if t not in self.att])
        if n_sim:
            out["sim"] = simulate(mat, n_sim, seed)
        return out

    def ratings_table(self) -> pd.DataFrame:
        rows = []
        for t in self.teams:
            rows.append({"Takim": t, "Hucum": np.exp(self.att[t]),
                         "Savunma": np.exp(self.dfn[t] - np.mean(list(self.dfn.values()))),
                         "Net": self.att[t] - self.dfn[t]})
        return pd.DataFrame(rows).sort_values("Net", ascending=False).reset_index(drop=True)


def summarize(mat: np.ndarray) -> dict:
    g = np.arange(mat.shape[0])
    total = g[:, None] + g[None, :]
    flat = np.argsort(mat, axis=None)[::-1][:5]
    top = [(int(i // mat.shape[1]), int(i % mat.shape[1]), float(mat.flat[i])) for i in flat]
    return {
        "pH": float(np.tril(mat, -1).sum()),
        "pD": float(np.trace(mat)),
        "pA": float(np.triu(mat, 1).sum()),
        "over25": float(mat[total > 2].sum()),
        "btts": float(mat[1:, 1:].sum()),
        "top_scores": top,
        "matrix": mat,
    }


def simulate(mat: np.ndarray, n: int = 10000, seed: int = 0) -> dict:
    """Skor matrisinden n maç örnekler (simülasyon çıktısı)."""
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(mat.size, size=n, p=mat.ravel() / mat.sum())
    hg, ag = idx // mat.shape[1], idx % mat.shape[1]
    return {"n": n, "H": int((hg > ag).sum()), "D": int((hg == ag).sum()),
            "A": int((hg < ag).sum()), "avg_home": float(hg.mean()), "avg_away": float(ag.mean())}


# ---------------- Elo ----------------
def compute_elo(df: pd.DataFrame, k: float = 20, home_adv: float = 60, carry: float = 0.67):
    """Maç sırasıyla Elo hesaplar. Dönüş: (güncel puanlar, maç öncesi puanlar dizisi)."""
    ratings: dict[str, float] = {}
    pre = np.zeros((len(df), 2))
    prev_season = None
    for i, row in enumerate(df.itertuples(index=False)):
        if row.Season != prev_season:
            season_teams = set(df.loc[df["Season"] == row.Season, "HomeTeam"]) | set(
                df.loc[df["Season"] == row.Season, "AwayTeam"])
            if ratings:
                mean = float(np.mean(list(ratings.values())))
                ratings = {t: mean + carry * (r - mean) for t, r in ratings.items()}
                for t in season_teams - set(ratings):  # yeni çıkan takımlar
                    ratings[t] = mean - 75
            else:
                ratings = {t: 1500.0 for t in season_teams}
            prev_season = row.Season
        rh, ra = ratings[row.HomeTeam], ratings[row.AwayTeam]
        pre[i] = rh, ra
        e = 1 / (1 + 10 ** (-(rh + home_adv - ra) / 400))
        s = 1.0 if row.FTHG > row.FTAG else 0.5 if row.FTHG == row.FTAG else 0.0
        gd = abs(row.FTHG - row.FTAG)
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = k * mult * (s - e)
        ratings[row.HomeTeam] = rh + delta
        ratings[row.AwayTeam] = ra - delta
    return ratings, pre


def elo_expectation(r_home: float, r_away: float, home_adv: float = 60) -> float:
    return 1 / (1 + 10 ** (-(r_home + home_adv - r_away) / 400))


# ---------------- form ve geçmiş karşılaşmalar ----------------
def team_form(df: pd.DataFrame, team: str, n: int = 5) -> dict:
    m = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(n)
    letters, pts, gf, ga = [], 0, 0, 0
    for r in m.itertuples(index=False):
        home = r.HomeTeam == team
        f, a = (r.FTHG, r.FTAG) if home else (r.FTAG, r.FTHG)
        gf, ga = gf + f, ga + a
        if f > a:
            letters.append("G"); pts += 3
        elif f == a:
            letters.append("B"); pts += 1
        else:
            letters.append("M")
    return {"seq": " ".join(letters), "pts": pts, "gf": gf, "ga": ga, "n": len(m)}


def head_to_head(df: pd.DataFrame, a: str, b: str, n: int = 6) -> pd.DataFrame:
    m = df[((df["HomeTeam"] == a) & (df["AwayTeam"] == b)) |
           ((df["HomeTeam"] == b) & (df["AwayTeam"] == a))]
    return m.tail(n)
