"""Ek tahminler: ilk yarı, İY/MS, gol zamanlaması, sarı/kırmızı kart, korner, şut.

- İlk yarı ve İY/MS: ana modelin beklenen golleri, ligin gerçek "ilk yarı gol payına"
  göre iki yarıya bölünür. İY/MS tablosu, maç sonu olasılıkları ana 1/X/2 ile birebir
  aynı olacak şekilde ölçeklenir (çelişki olmasın diye).
- Gol zamanlaması: dakika verisi olmadığı için gollerin 15 dakikalık dilimlere GENEL
  dağılımı kullanılır, iki yarının payı ligin kendi verisine göre ayarlanır. Yaklaşıktır.
- Kart, korner, şut: her takım için "üretme" ve "rakibe yaptırma" eğilimi, ana modeldeki
  gibi zaman ağırlıklı öğrenilir. Toplamlar için aşırı yayılım (negatif binom) kullanılır,
  çünkü özellikle kartlar düz Poisson'dan daha değişkendir. Premier League'de hakemin
  kart eğilimi de hesaba katılır (diğer liglerde veri dosyasında hakem yok).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

# anahtar: (ev sütunu, deplasman sütunu, düzenleme gücü, toplam çizgileri)
STATS = {
    "sari_kart": ("HY", "AY", 2.0, [3.5, 4.5, 5.5]),
    "korner": ("HC", "AC", 1.0, [8.5, 9.5, 10.5]),
    "sut": ("HS", "AS", 1.0, []),
    "isabetli_sut": ("HST", "AST", 1.0, []),
    "kirmizi_kart": ("HR", "AR", 30.0, []),  # çok nadir: takımlar ortalamaya güçlü çekilir
}

DILIMLER = ["0-15", "16-30", "31-45+", "46-60", "61-75", "76-90+"]
# Gollerin 15 dakikalık dilimlere yaklaşık genel dağılımı (%; uzatmalar son dilimlerde).
GENEL_PAY = np.array([12.5, 14.5, 17.0, 16.0, 17.0, 23.0])
VARSAYILAN_ILK_YARI_PAYI = 0.445
HAKEM_BUZULME = 12.0  # hakem etkisi için ortalamaya çekme (kart cinsinden)


class CountModel:
    """Takım bazlı sayım modeli: E[ev] = exp(h + uretme_ev + yaptirma_dep)."""

    def __init__(self, reg: float = 1.0):
        self.reg = reg

    def fit(self, h, a, x, y, w, n):
        reg = self.reg

        def nll_grad(p):
            r, d, g = p[:n], p[n:2 * n], p[2 * n]
            att = r - r.mean()
            lam = np.exp(g + att[h] + d[a])
            mu = np.exp(att[a] + d[h])
            ll = w * (x * np.log(lam) - lam + y * np.log(mu) - mu)
            gl, gm = w * (x - lam), w * (y - mu)
            ga = np.bincount(h, gl, n) + np.bincount(a, gm, n)
            ga -= ga.mean()
            gd = np.bincount(a, gl, n) + np.bincount(h, gm, n)
            dc = d - d.mean()
            return (-ll.sum() + reg * (att @ att + dc @ dc),
                    np.concatenate([-ga + 2 * reg * att, -gd + 2 * reg * dc, [-gl.sum()]]))

        base = np.log(max(float(np.average(y, weights=w)), 0.01))
        home0 = np.log(max(float(np.average(x, weights=w)), 0.01)) - base
        p0 = np.concatenate([np.zeros(n), np.full(n, base), [home0]])
        p = minimize(nll_grad, p0, jac=True, method="L-BFGS-B", options={"maxiter": 2000}).x
        self.att = p[:n] - p[:n].mean()
        self.dfn = p[n:2 * n]
        self.home = float(p[2 * n])
        self._dfn_mean = float(self.dfn.mean())
        return self

    def means(self, hi, ai):
        """hi/ai: takım indeksi ya da None (bilinmeyen takım = lig ortalaması)."""
        ah = self.att[hi] if hi is not None else 0.0
        aa = self.att[ai] if ai is not None else 0.0
        dh = self.dfn[hi] if hi is not None else self._dfn_mean
        da = self.dfn[ai] if ai is not None else self._dfn_mean
        return float(np.exp(self.home + ah + da)), float(np.exp(aa + dh))


def _diff_dist(lam: float, mu: float, G: int = 12) -> np.ndarray:
    """Gol farkı dağılımı; indeks k, fark = k - (G - 1)."""
    g = np.arange(G)
    return np.convolve(poisson.pmf(g, lam), poisson.pmf(g, mu)[::-1])


def _over(mean: float, r: float, line: float) -> float:
    k = int(np.floor(line))
    if not np.isfinite(r):
        return float(poisson.sf(k, mean))
    return float(nbinom.sf(k, r, r / (r + mean)))


def _p_zero(mean: float, r: float) -> float:
    if not np.isfinite(r):
        return float(np.exp(-mean))
    return float(nbinom.pmf(0, r, r / (r + mean)))


class Extras:
    def fit(self, df: pd.DataFrame, ref_date=None, xi: float = 0.0019) -> "Extras":
        if ref_date is None:
            ref_date = df["Date"].max() + pd.Timedelta(days=1)
        d = df[df["Date"] < pd.Timestamp(ref_date)]
        w_all = np.exp(-xi * (pd.Timestamp(ref_date) - d["Date"]).dt.days.to_numpy(float))
        keep = w_all > 1e-3
        d, w_all = d[keep], w_all[keep]

        # İlk yarı gol payı (ligin kendi verisinden)
        self.f = VARSAYILAN_ILK_YARI_PAYI
        if {"HTHG", "HTAG"} <= set(d.columns):
            ht = d["HTHG"] + d["HTAG"]
            ft = d["FTHG"] + d["FTAG"]
            ok = ht.notna().to_numpy() & (ft > 0).to_numpy()
            if ok.sum() >= 100:
                f = float(np.sum(w_all[ok] * ht[ok]) / np.sum(w_all[ok] * ft[ok]))
                self.f = float(np.clip(f, 0.35, 0.55))

        self.teams = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
        ix = {t: i for i, t in enumerate(self.teams)}
        self.ix = ix
        self.models, self.disp, self.referee = {}, {}, {}
        for key, (hc, ac, reg, _) in STATS.items():
            if hc not in d.columns or ac not in d.columns:
                continue
            x = pd.to_numeric(d[hc], errors="coerce")
            y = pd.to_numeric(d[ac], errors="coerce")
            ok = (x.notna() & y.notna()).to_numpy()
            if ok.sum() < 100:
                continue
            sub = d[ok]
            h = sub["HomeTeam"].map(ix).to_numpy()
            a = sub["AwayTeam"].map(ix).to_numpy()
            xv, yv, w = x[ok].to_numpy(float), y[ok].to_numpy(float), w_all[ok]
            m = CountModel(reg).fit(h, a, xv, yv, w, len(self.teams))
            self.models[key] = m

            # aşırı yayılım (toplam için): var = m + m^2 / r
            mt = np.array([sum(m.means(hh, aa)) for hh, aa in zip(h, a)])
            yt = xv + yv
            excess = float(np.sum(w * ((yt - mt) ** 2 - mt)))
            self.disp[key] = float(np.sum(w * mt ** 2) / excess) if excess > 1e-6 * np.sum(w * mt) else np.inf

            # hakem etkisi (sadece sarı kart, hakem sütunu varsa)
            if key == "sari_kart" and "Referee" in sub.columns:
                ref = sub["Referee"].astype(str).str.strip().to_numpy()
                tab: dict[str, list[float]] = {}
                for rname, yy, mm, ww in zip(ref, yt, mt, w):
                    if rname and rname.lower() != "nan":
                        s = tab.setdefault(rname, [0.0, 0.0])
                        s[0] += ww * yy
                        s[1] += ww * mm
                self.referee = {r: (Y + HAKEM_BUZULME) / (M + HAKEM_BUZULME) for r, (Y, M) in tab.items()}
        return self

    # ---------------- tahmin ----------------
    def predict(self, home: str, away: str, lam: float, mu: float, p_main, p00: float,
                referee: str | None = None) -> dict:
        out = {}
        f = self.f

        # İlk yarı ve İY/MS
        G = 12
        d1 = _diff_dist(f * lam, f * mu, G)
        d2 = _diff_dist((1 - f) * lam, (1 - f) * mu, G)
        off = G - 1
        c2 = np.cumsum(d2)  # c2[k] = P(fark2 <= k - off)

        def p_d2_le(v):  # P(fark2 <= v)
            k = v + off
            return 0.0 if k < 0 else 1.0 if k >= len(c2) else float(c2[k])

        M = np.zeros((3, 3))  # satır: İY sonucu (1, X, 2), sütun: MS sonucu
        for k, p in enumerate(d1):
            v1 = k - off
            a = 0 if v1 > 0 else 1 if v1 == 0 else 2
            p_ft_win = 1 - p_d2_le(-v1)            # v1 + d2 > 0
            p_ft_draw = p_d2_le(-v1) - p_d2_le(-v1 - 1)
            p_ft_loss = p_d2_le(-v1 - 1)
            M[a] += p * np.array([p_ft_win, p_ft_draw, p_ft_loss])
        cols = M.sum(axis=0)
        M = M * (np.asarray(p_main) / np.where(cols > 0, cols, 1))  # MS ana 1/X/2 ile aynı olsun
        M /= M.sum()
        sec = ["1", "X", "2"]
        iy_ms = sorted(({"kod": f"{sec[i]}/{sec[j]}", "p": round(float(M[i, j]), 4)}
                        for i in range(3) for j in range(3)), key=lambda z: -z["p"])
        ht_total = f * (lam + mu)
        out["ilk_yari"] = {
            "p_ev": round(float(M[0].sum()), 4), "p_beraberlik": round(float(M[1].sum()), 4),
            "p_deplasman": round(float(M[2].sum()), 4),
            "xg_ev": round(f * lam, 2), "xg_deplasman": round(f * mu, 2),
            "ust_0_5": round(float(1 - np.exp(-ht_total)), 4),
            "ust_1_5": round(float(poisson.sf(1, ht_total)), 4),
            "ilk_yari_gol_payi": round(f, 3),
        }
        out["iy_ms"] = iy_ms

        # Gol zamanlaması
        pay = GENEL_PAY.copy()
        pay[:3] = pay[:3] / pay[:3].sum() * f
        pay[3:] = pay[3:] / pay[3:].sum() * (1 - f)
        tot = lam + mu
        out["gol_dakika"] = {
            "araliklar": [{"aralik": DILIMLER[i], "p_gol": round(float(1 - np.exp(-tot * pay[i])), 4)}
                          for i in range(6)],
            "ilk_gol_30_once": round(float(1 - np.exp(-tot * (pay[0] + pay[1]))), 4),
            "golsuz": round(float(p00), 4),
            "ilk_golu_ev": round(float(lam / tot * (1 - p00)), 4),
            "ilk_golu_deplasman": round(float(mu / tot * (1 - p00)), 4),
        }

        # Kart, korner, şut
        stats = {}
        hi, ai = self.ix.get(home), self.ix.get(away)
        for key, (_, _, _, lines) in STATS.items():
            m = self.models.get(key)
            if m is None:
                continue
            mh, ma = m.means(hi, ai)
            item = {}
            if key == "sari_kart" and referee and referee in self.referee:
                k = self.referee[referee]
                mh, ma = mh * k, ma * k
                item.update(hakem=referee, hakem_etkisi=round(k, 2))
            r = self.disp.get(key, np.inf)
            item.update(ev=round(mh, 2), deplasman=round(ma, 2), toplam=round(mh + ma, 2))
            if lines:
                item["cizgiler"] = [{"cizgi": ln, "ust": round(_over(mh + ma, r, ln), 4)} for ln in lines]
            if key == "kirmizi_kart":
                item["p_en_az_bir"] = round(1 - _p_zero(mh + ma, r), 4)
                item["dusuk_guven"] = True
            stats[key] = item
        out["istatistikler"] = stats
        return out
