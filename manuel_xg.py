"""Uygulamadan elle girilen xG verisi (manuel_xg.json).

Dosya biçimi (uygulama yazar, burada sadece okunur):
{
  "maclar":   {"T1|2026-10-25|Galatasaray|Fenerbahce": {"ev_xg": 1.84, "dep_xg": 0.92}},
  "takimlar": {"T1": {"Galatasaray": {"ic_xg": 2.3, "ic_xga": 0.7, "dis_xg": 1.6, "dis_xga": 1.1}}}
}
- maclar: biten bir maçın iki takımının xG'si (anahtar: lig|tarih|ev|deplasman, adlar
  bitenler.json'daki gibi).
- takimlar: takımın bu sezon maç başına ortalama attığı (xg) ve yediği (xga) xG,
  iç saha (ic_) ve dış saha (dis_) için ayrı. Eski biçim ({"xg", "xga"}) de okunur;
  o durumda iç ve dış saha için aynı değer kullanılır.

Maç xG'si girilmemiş bu sezon maçları için, iki takımın ortalamalarından tahmini xG:
  ev xG  ≈ ev takımının İÇ SAHA xG'si × deplasmanın DIŞ SAHA xGA'sı / ligin ev sahibi xG ort.
  dep xG ≈ deplasmanın DIŞ SAHA xG'si × ev takımının İÇ SAHA xGA'sı / ligin deplasman xG ort.
Bu veriler "gölge model"e girer; ana tahmin değişmez, karnede ikisi karşılaştırılır.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

MANUEL_FILE = os.environ.get("MANUEL_XG_FILE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "manuel_xg.json")


def load_manual(path: str | None = None) -> dict:
    try:
        with open(path or MANUEL_FILE, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {"maclar": {}, "takimlar": {}}
    d.setdefault("maclar", {})
    d.setdefault("takimlar", {})
    return d


def match_key(code: str, date, home: str, away: str) -> str:
    return f"{code}|{pd.Timestamp(date):%Y-%m-%d}|{home}|{away}"


def _num(v):
    try:
        v = float(v)
        return v if np.isfinite(v) and 0 <= v <= 10 else None
    except (TypeError, ValueError):
        return None


def add_manual_xg(df: pd.DataFrame, code: str, manual: dict, season: str | None = None) -> pd.DataFrame:
    """HXG / AXG / XGSrc sütunlarını ekler (sadece verilen sezonun maçlarına)."""
    df = df.copy()
    df["HXG"], df["AXG"], df["XGSrc"] = np.nan, np.nan, ""
    if not len(df):
        return df
    season = season or df["Season"].iloc[-1]
    teams_avg = {}
    for t, v in (manual.get("takimlar", {}).get(code) or {}).items():
        sp = team_split(v)
        if sp:
            teams_avg[t] = sp
    # ligin ev sahibi ve deplasman xG ortalamaları (ev xG'si ~ deplasmanların dış saha xGA'sı)
    if len(teams_avg) >= 4:
        avg_home = float(np.mean([v["ic_xg"] for v in teams_avg.values()] + [v["dis_xga"] for v in teams_avg.values()]))
        avg_away = float(np.mean([v["dis_xg"] for v in teams_avg.values()] + [v["ic_xga"] for v in teams_avg.values()]))
    else:
        avg_home = avg_away = None
    games = manual.get("maclar", {})
    n = len(df)
    hx, ax = np.full(n, np.nan), np.full(n, np.nan)
    src = np.array([""] * n, dtype=object)
    for i, r in enumerate(df[["Date", "HomeTeam", "AwayTeam", "Season"]].itertuples(index=False)):
        if r.Season != season:
            continue
        g = games.get(match_key(code, r.Date, r.HomeTeam, r.AwayTeam))
        if isinstance(g, dict) and _num(g.get("ev_xg")) is not None and _num(g.get("dep_xg")) is not None:
            hx[i], ax[i], src[i] = _num(g["ev_xg"]), _num(g["dep_xg"]), "mac"
            continue
        h, a = teams_avg.get(r.HomeTeam), teams_avg.get(r.AwayTeam)
        if h and a and avg_home and avg_away:
            hx[i] = h["ic_xg"] * a["dis_xga"] / avg_home
            ax[i] = a["dis_xg"] * h["ic_xga"] / avg_away
            src[i] = "ortalama"
    df["HXG"], df["AXG"], df["XGSrc"] = hx, ax, src
    return df


def team_split(v) -> dict | None:
    """Takım ortalamalarını {ic_xg, ic_xga, dis_xg, dis_xga} biçimine getirir; eksikse None."""
    if not isinstance(v, dict):
        return None
    keys = ("ic_xg", "ic_xga", "dis_xg", "dis_xga")
    vals = {k: _num(v.get(k)) for k in keys}
    if all(x is not None for x in vals.values()):
        return vals
    xg, xga = _num(v.get("xg")), _num(v.get("xga"))  # eski biçim: iç = dış
    if xg is not None and xga is not None:
        return {"ic_xg": xg, "ic_xga": xga, "dis_xg": xg, "dis_xga": xga}
    return None


def shadow_weights(sot: float, bc: float) -> tuple[float, float, float]:
    """Gölge modelde xG ağırlığı; şut/şans ağırlıkları toplam 1'i geçmeyecek şekilde küçültülür."""
    wx = max(0.3, min(0.5, 1 - sot - bc))
    rest = sot + bc
    if rest + wx > 1 and rest > 0:
        k = (1 - wx) / rest
        sot, bc = sot * k, bc * k
    return round(sot, 4), round(bc, 4), wx


def summary(code: str, manual: dict) -> dict:
    games = [k for k in manual.get("maclar", {}) if k.startswith(code + "|")]
    teams = manual.get("takimlar", {}).get(code) or {}
    return {"mac": len(games), "takim": len(teams)}
