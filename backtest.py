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
import json
import os
import sys

import numpy as np
import pandas as pd

from data import LEAGUES, current_season_start, load_league, resolve_league, season_codes
from models import DixonColes, summarize
from ozellikler import H2HIndex
from tune import predict_row

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


def league_settings(code: str, xi, reg, sot, sd, bc=None) -> dict:
    """Komut satırında verilmeyen ayarlar ayarlar.json'dan (tune.py çıktısı) ya da varsayılandan."""
    p = {"xi": 0.0019, "reg": 1.0, "sot": 0.0, "bc": 0.0, "sd": 0.10}
    flags = {"fatigue": False, "pos": False, "h2h": False}
    lineup = None
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ayarlar.json")
    try:
        with open(path, encoding="utf-8") as f:
            saved = json.load(f).get(code, {})
        p.update({k: float(saved[k]) for k in p if k in saved})
        flags = {k: bool(saved.get(k, False)) for k in flags}
        lineup = saved.get("lineup") if saved.get("lineup") in ("xi", "side") else None
    except (OSError, ValueError):
        pass
    for k, v in (("xi", xi), ("reg", reg), ("sot", sot), ("bc", bc), ("sd", sd)):
        if v is not None:
            p[k] = v
    p.update(flags)
    p["lineup"] = lineup
    return p


def backtest(code: str, season: str, history: int, refit_days: int, xi=None, sd=None, reg=None, sot=None,
             bc=None):
    cfg = league_settings(code, xi, reg, sot, sd, bc)
    xi, sd = cfg["xi"], cfg["sd"]
    start_year = 2000 + int(season[:2])
    df = load_league(code, seasons=season_codes(history + 1, last_start=start_year))
    test = df[df["Season"] == season]
    if not len(test):
        raise RuntimeError(f"{season} sezonu için veri yok.")

    preds = np.zeros((len(test), 3))
    t = test["Date"].min().normalize()
    end = test["Date"].max()
    n_fit = 0
    h2h_index = H2HIndex(df)
    while t <= end:
        win = test[(test["Date"] >= t) & (test["Date"] < t + pd.Timedelta(days=refit_days))]
        if len(win):
            model = DixonColes(xi=xi, reg=cfg["reg"], sot=cfg["sot"], bc=cfg["bc"],
                               fatigue=cfg["fatigue"], pos=cfg["pos"],
                               lineup=cfg["lineup"]).fit(df, ref_date=t)
            n_fit += 1
            for i, m in zip(test.index.get_indexer(win.index), win.itertuples(index=False)):
                s = summarize(predict_row(model, m, sd, cfg, h2h_index, n_draws=200))
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
    print(f"  ayarlar: xi={cfg['xi']}, reg={cfg['reg']}, isabetli şut={cfg['sot']}, büyük şans={cfg['bc']}, belirsizlik={cfg['sd']}, "
          f"yorgunluk={'açık' if cfg['fatigue'] else 'kapalı'}, topla oynama={'açık' if cfg['pos'] else 'kapalı'}, "
          f"eşleşme={'açık' if cfg['h2h'] else 'kapalı'}, "
          f"eksik oyuncu={ {'xi': 'ilk 11', 'side': 'sakat listesi'}.get(cfg['lineup'], 'kapalı') }")
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
    ap.add_argument("--xi", type=float)
    ap.add_argument("--reg", type=float)
    ap.add_argument("--sut", type=float, help="İsabetli şut ağırlığı (0-1)")
    ap.add_argument("--buyuk-sans", type=float, help="Büyük şans ağırlığı (0-1)")
    ap.add_argument("--belirsizlik", type=float)
    args = ap.parse_args()

    codes = list(LEAGUES) if args.hepsi else [resolve_league(args.lig or ap.error("--lig veya --hepsi gerekli"))]
    for c in codes:
        try:
            backtest(c, args.sezon, args.gecmis, args.yenileme, xi=args.xi, sd=args.belirsizlik,
                     reg=args.reg, sot=args.sut, bc=args.buyuk_sans)
        except RuntimeError as e:
            print(f"\n=== {LEAGUES[c]}: atlandı ({e})")


if __name__ == "__main__":
    main()
