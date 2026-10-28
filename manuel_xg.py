"""Uygulamadan elle girilen xG verisi (manuel_xg.json).

Dosya biçimi (uygulama yazar, burada sadece okunur):
{
  "maclar":   {"T1|2026-10-25|Galatasaray|Fenerbahce": {"ev_xg": 1.84, "dep_xg": 0.92}},
  "takimlar": {"T1": {"Galatasaray": {"ic_xg": 2.3, "ic_xga": 0.7, "dis_xg": 1.6, "dis_xga": 1.1}}}
}
- maclar: biten bir maçın iki takımının xG'si (anahtar: lig|tarih|ev|deplasman, adlar
  bitenler.json'daki gibi). Aynı kayıtta isteğe bağlı maç istatistikleri de olabilir:
  "ev_is"/"dep_is" (isabetli şut), "ev_bs"/"dep_bs" (büyük şans), "ev_top"/"dep_top" (topla oynama %).
  Bunlar Sportmonks/football-data'da olmayan maçlarda ana modele girer (add_manual_stats).
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


# ---------------- bitiricilik: gerçekleşen gol / xG ----------------
SHRINK_XG = 40.0  # küçültme gücü (xG birimi). Simülasyonla seçildi: ham oran çok gürültülü, k=40 her senaryoda en iyiye yakın.


def _ratio(goals: float, xg: float, k: float = SHRINK_XG) -> float:
    """Küçültülmüş oran: (gol + k) / (xG + k). Az veride 1'e yakın, çok veride gol/xG'ye yaklaşır."""
    return (goals + k) / (xg + k)


def finishing_factors(df: pd.DataFrame, code: str, manual: dict, season: str | None = None,
                      k: float = SHRINK_XG) -> dict:
    """Bu sezonun iç/dış saha gerçek golleri ile elle girilen xG ortalamalarından takım çarpanları.

    Her takım için (iç saha 'ic', dış saha 'dis'):
      atak    = attığı gol / ürettiği xG       (> 1: xG'sinden fazla gol atıyor)
      savunma = yediği gol / rakibe verdiği xGA (> 1: xGA'sından fazla gol yiyor, ör. kaleci)
    'carpan' değerleri küçültülmüştür ve modele bunlar girer.
    """
    out: dict = {}
    if not len(df):
        return out
    season = season or df["Season"].iloc[-1]
    cur = df[df["Season"] == season]
    for t, v in (manual.get("takimlar", {}).get(code) or {}).items():
        sp = team_split(v)
        if not sp:
            continue
        home, away = cur[cur["HomeTeam"] == t], cur[cur["AwayTeam"] == t]
        res = {}
        for yer, games, gf_col, ga_col in (("ic", home, "FTHG", "FTAG"), ("dis", away, "FTAG", "FTHG")):
            n = len(games)
            if n == 0:
                continue
            gf, ga = float(games[gf_col].sum()), float(games[ga_col].sum())
            xgf, xga = sp[f"{yer}_xg"] * n, sp[f"{yer}_xga"] * n
            res[yer] = {"mac": n, "gol": int(gf), "xg": round(xgf, 2), "yenilen": int(ga), "xga": round(xga, 2),
                        "atak_oran": round(gf / xgf, 3) if xgf > 0 else None,
                        "savunma_oran": round(ga / xga, 3) if xga > 0 else None,
                        "atak_carpan": round(_ratio(gf, xgf, k), 3),
                        "savunma_carpan": round(_ratio(ga, xga, k), 3)}
        if res:
            out[t] = res
    return out


def match_multipliers(factors: dict, home: str, away: str) -> tuple[float, float]:
    """Gölge modelin beklenen gollerine uygulanacak çarpanlar (ev, deplasman)."""
    h, a = factors.get(home, {}), factors.get(away, {})
    fh = h.get("ic", {}).get("atak_carpan", 1.0) * a.get("dis", {}).get("savunma_carpan", 1.0)
    fa = a.get("dis", {}).get("atak_carpan", 1.0) * h.get("ic", {}).get("savunma_carpan", 1.0)
    return float(fh), float(fa)


# elle girilen maç istatistikleri: json alanı -> (ev sütunu, dep sütunu, en büyük geçerli değer)
STAT_FIELDS = {"is": ("HST", "AST", 40), "bs": ("HBC", "ABC", 15), "top": ("HPOS", "APOS", 100)}


def add_manual_stats(df: pd.DataFrame, code: str, manual: dict | None = None) -> pd.DataFrame:
    """Elle girilen isabetli şut / büyük şans / topla oynama değerlerini boş olan hücrelere yazar."""
    manual = manual if manual is not None else load_manual()
    games = {k: v for k, v in (manual.get("maclar") or {}).items() if k.startswith(code + "|") and isinstance(v, dict)}
    if not games or not len(df):
        return df
    df = df.copy()
    for hc, ac, _ in STAT_FIELDS.values():
        for c in (hc, ac):
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")
    index = {match_key(code, r.Date, r.HomeTeam, r.AwayTeam): i
             for i, r in enumerate(df[["Date", "HomeTeam", "AwayTeam"]].itertuples(index=False))}
    n = 0
    for key, g in games.items():
        i = index.get(key)
        if i is None:
            continue
        for f, (hc, ac, mx) in STAT_FIELDS.items():
            hv, av = g.get(f"ev_{f}"), g.get(f"dep_{f}")
            try:
                hv, av = float(hv), float(av)
            except (TypeError, ValueError):
                continue
            if not (0 <= hv <= mx and 0 <= av <= mx):
                continue
            if f == "top" and not 90 <= hv + av <= 110:
                continue  # topla oynama toplamı ~100 olmalı
            hi, ai = df.columns.get_loc(hc), df.columns.get_loc(ac)
            if pd.isna(df.iat[i, hi]) or pd.isna(df.iat[i, ai]):
                df.iat[i, hi], df.iat[i, ai] = hv, av
                n += 1
    df.attrs["manual_stats"] = n
    return df
