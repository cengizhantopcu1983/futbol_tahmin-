"""Ek model özellikleri: topla oynama farkı ve eşleşme (H2H) etkisi.

Topla oynama (PosDiff):
  Her maç için, iki takımın o maçtan ÖNCEKİ son 8 maçındaki topla oynama ortalamaları
  arasındaki fark (puan / 10). Ör. ev sahibi ortalama %58, konuk %46 -> 1.2.
  Veri Sportmonks'tan gelir (HPOS / APOS); yoksa 0 (etkisiz).

Eşleşme etkisi (H2H):
  Ev sahibinin AYNI rakibi KENDİ SAHASINDA ağırladığı geçmiş maçlarda attığı/yediği goller,
  modelin bugünkü beklentisiyle karşılaştırılır. Maç sayısı az olduğu için sapma güçlü
  şekilde küçültülür:  çarpan = (gerçek gol + k × beklenti) / ((n + k) × beklenti),  k = 6.
  Ör. 3 maçta beklenti 1,5 iken 3 gol ortalaması -> ham ×2,0, modele giren ×1,33.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POS_WINDOW = 8
POS_MIN = 3
H2H_K = 6.0


def add_possession(df: pd.DataFrame) -> pd.DataFrame:
    """PosDiff sütununu ekler (maç öncesi bilgiyle, geleceğe bakmadan)."""
    df = df.copy()
    df["PosDiff"] = 0.0
    if not {"HPOS", "APOS"} <= set(df.columns) or not len(df):
        return df
    hist: dict[str, list[float]] = {}
    out = np.zeros(len(df))
    for i, r in enumerate(df[["HomeTeam", "AwayTeam", "HPOS", "APOS"]].itertuples(index=False)):
        h, a = hist.get(r.HomeTeam, []), hist.get(r.AwayTeam, [])
        if len(h) >= POS_MIN and len(a) >= POS_MIN:
            out[i] = (np.mean(h[-POS_WINDOW:]) - np.mean(a[-POS_WINDOW:])) / 10.0
        if pd.notna(r.HPOS) and pd.notna(r.APOS):
            hist.setdefault(r.HomeTeam, []).append(float(r.HPOS))
            hist.setdefault(r.AwayTeam, []).append(float(r.APOS))
    df["PosDiff"] = out
    return df


def possession_diff(df: pd.DataFrame, home: str, away: str, date=None) -> float:
    """Yaklaşan bir maç için PosDiff (verilen tarihten önceki maçlardan)."""
    if not {"HPOS", "APOS"} <= set(df.columns):
        return 0.0
    d = df if date is None else df[df["Date"] < pd.Timestamp(date)]

    def avg(t):
        v = pd.concat([d.loc[d.HomeTeam == t, ["Date", "HPOS"]].rename(columns={"HPOS": "P"}),
                       d.loc[d.AwayTeam == t, ["Date", "APOS"]].rename(columns={"APOS": "P"})])
        v = v.dropna().sort_values("Date")["P"].tail(POS_WINDOW)
        return float(v.mean()) if len(v) >= POS_MIN else None

    h, a = avg(home), avg(away)
    return 0.0 if h is None or a is None else (h - a) / 10.0


class H2HIndex:
    """Aynı sahadaki geçmiş karşılaşmalar: (ev, deplasman) -> [(tarih, ev golü, dep golü)]."""

    def __init__(self, df: pd.DataFrame):
        self.games: dict[tuple[str, str], list[tuple]] = {}
        for r in df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].itertuples(index=False):
            self.games.setdefault((r.HomeTeam, r.AwayTeam), []).append((r.Date, int(r.FTHG), int(r.FTAG)))

    def before(self, home: str, away: str, date=None) -> list[tuple]:
        g = self.games.get((home, away), [])
        if date is None:
            return g
        t = pd.Timestamp(date)
        return [x for x in g if x[0] < t]


def h2h_mult(index: H2HIndex, home: str, away: str, lam: float, mu: float, date=None,
             k: float = H2H_K) -> tuple[float, float, dict]:
    """Eşleşme çarpanları (ev, deplasman) ve gösterim için ayrıntı."""
    g = index.before(home, away, date)
    n = len(g)
    if n == 0 or lam <= 0 or mu <= 0:
        return 1.0, 1.0, {"mac": 0}
    gh, ga = sum(x[1] for x in g), sum(x[2] for x in g)
    fh = (gh + k * lam) / ((n + k) * lam)
    fa = (ga + k * mu) / ((n + k) * mu)
    info = {"mac": n, "ev_gol": gh, "dep_gol": ga,
            "beklenen_ev": round(n * lam, 2), "beklenen_dep": round(n * mu, 2),
            "carpan_ev": round(fh, 3), "carpan_dep": round(fa, 3),
            "son": [{"tarih": f"{pd.Timestamp(x[0]):%Y-%m-%d}", "skor": f"{x[1]}-{x[2]}"} for x in g[-5:]]}
    return float(fh), float(fa), info
