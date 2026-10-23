"""Geriye dönük test: model geçmiş bir sezonu, sadece o güne kadarki veriyle tahmin eder.

Örnekler:
  python backtest.py --lig premier --sezon 2526
  python backtest.py --hepsi --sezon 2526

Ölçütler (hepsinde küçük = iyi, isabet hariç):
  Log loss : olasılık tahminlerinin genel kalitesi
  RPS      : futbolda standart ölçüt (1-X-2 sırasını dikkate alır)
  Brier    : olasılık hatasının karesi
  İsabet   : en yüksek olasılık verilen sonucun tutma oranı
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from data import LEAGUES, current_season_start, load_league, resolve_league, season_codes
from models import DixonColes, summarize

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def scores(P: np.ndarray, o: np.ndarray) -> dict:
    P = np.clip(P, 1e-12, 1)
    P = P / P.sum(axis=1, keepdims=True)
    Y = np.eye(3)[o]
    cp, cy = np.cumsum(P, 1)[:, :2], np.cumsum(Y, 1)[:, :2]
    return {
        "Log loss": float(-np.mean(np.log(P[np.arange(len(o)), o]))),
        "RPS": float(np.mean(((cp - cy) ** 2).sum(1) / 2)),
        "Brier": float(np.mean(((P - Y) ** 2).sum(1))),
        "İsabet": float(np.mean(P.argmax(1) == o)),
    }


def backtest(code: str, season: str, history: int, refit_days: int, xi: float, sd: float):
    start_year = 2000 + int(season[:2])
    df = load_league(code, seasons=season_codes(history + 1, last_start=start_year))
    test = df[df["Season"] == season]
    if not len(test):
        raise RuntimeError(f"{season} sezonu için veri yok.")

    preds = np.zeros((len(test), 3))
    t = test["Date"].min().normalize()
    end = test["Date"].max()
    n_fit = 0
    while t <= end:
        win = test[(test["Date"] >= t) & (test["Date"] < t + pd.Timedelta(days=refit_days))]
        if len(win):
            model = DixonColes(xi=xi).fit(df, ref_date=t)
            n_fit += 1
            for i, m in zip(test.index.get_indexer(win.index), win.itertuples(index=False)):
                s = summarize(model.score_matrix(m.HomeTeam, m.AwayTeam, sd=sd, n_draws=200))
                preds[i] = s["pH"], s["pD"], s["pA"]
        t += pd.Timedelta(days=refit_days)

    o = np.where(test["FTHG"] > test["FTAG"], 0, np.where(test["FTHG"] == test["FTAG"], 1, 2))
    # Basit referans: önceki sezonlardaki ev/beraberlik/deplasman oranları
    past = df[df["Season"] < season]
    po = np.where(past["FTHG"] > past["FTAG"], 0, np.where(past["FTHG"] == past["FTAG"], 1, 2))
    base = np.bincount(po, minlength=3) / len(po)

    bk = test[["bk_H", "bk_D", "bk_A"]].to_numpy()
    has_bk = ~np.isnan(bk).any(1)

    print(f"\n=== {LEAGUES[code]} {season}: {len(test)} maç, model {n_fit} kez yeniden eğitildi ===")
    rows = {"Model": scores(preds[has_bk], o[has_bk]),
            "Bahis piyasası": scores(bk[has_bk], o[has_bk]) if has_bk.any() else None,
            "Referans (oranlar)": scores(np.tile(base, (has_bk.sum(), 1)), o[has_bk])}
    if not has_bk.all():
        print(f"  ({(~has_bk).sum()} maçta oran yok; kıyas oranı olan {has_bk.sum()} maçta yapıldı)")
    print(f"  {'':<20}{'Log loss':>10}{'RPS':>8}{'Brier':>8}{'İsabet':>9}")
    for name, s in rows.items():
        if s:
            print(f"  {name:<20}{s['Log loss']:>10.4f}{s['RPS']:>8.4f}{s['Brier']:>8.4f}{100 * s['İsabet']:>8.1f}%")

    # Kalibrasyon: "%60 dediğimiz maçların gerçekten %60'ı mı oldu?"
    print("\n  Kalibrasyon (ev sahibi galibiyeti):")
    print(f"  {'Tahmin aralığı':<16}{'Maç':>5}{'Ort. tahmin':>13}{'Gerçekleşen':>13}")
    edges = [0, .2, .35, .5, .65, 1.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (preds[:, 0] >= lo) & (preds[:, 0] < hi)
        if m.sum():
            print(f"  {f'%{lo * 100:.0f}-%{hi * 100:.0f}':<16}{m.sum():>5}"
                  f"{100 * preds[m, 0].mean():>12.1f}%{100 * (o[m] == 0).mean():>12.1f}%")
    return rows


def main():
    last = current_season_start() - 1
    ap = argparse.ArgumentParser(description="Modelin geriye dönük testi")
    ap.add_argument("--lig", help="super-lig, premier, laliga, seriea")
    ap.add_argument("--hepsi", action="store_true", help="4 ligin hepsi")
    ap.add_argument("--sezon", default=f"{last % 100:02d}{(last + 1) % 100:02d}",
                    help="Test sezonu, ör. 2526 (varsayılan: son tamamlanan sezon)")
    ap.add_argument("--gecmis", type=int, default=3, help="Eğitim için önceki sezon sayısı")
    ap.add_argument("--yenileme", type=int, default=7, help="Model kaç günde bir yeniden eğitilsin")
    ap.add_argument("--xi", type=float, default=0.0019)
    ap.add_argument("--belirsizlik", type=float, default=0.10)
    args = ap.parse_args()

    codes = list(LEAGUES) if args.hepsi else [resolve_league(args.lig or ap.error("--lig veya --hepsi gerekli"))]
    for c in codes:
        try:
            backtest(c, args.sezon, args.gecmis, args.yenileme, args.xi, args.belirsizlik)
        except RuntimeError as e:
            print(f"\n=== {LEAGUES[c]}: atlandı ({e})")


if __name__ == "__main__":
    main()
