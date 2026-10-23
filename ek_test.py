"""Ek tahminlerin geriye dönük testi (ilk yarı, kart, korner, şut, gol ortalaması).

Model sezonu hafta hafta yürür, her maçı sadece o güne kadarki veriyle tahmin eder.
Her tahmin, "lig ortalaması" referansıyla karşılaştırılır: model, ligin genel
ortalamasını söylemekten daha iyi değilse o tahmin işe yaramıyor demektir.

Ölçütler (hepsinde küçük = iyi):
  RPS / Brier : olasılık tahminlerinin hatası
  Ort. hata   : beklenen sayı ile gerçekleşen arasındaki ortalama mutlak fark

Örnek:  python ek_test.py --hepsi --sezon 2526
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from data import LEAGUES, current_season_start, load_league, resolve_league, season_code, season_codes
from ekstra import Extras
from models import DixonColes, summarize

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def rps3(P, o):
    Y = np.eye(3)[o]
    cp, cy = np.cumsum(P, 1)[:, :2], np.cumsum(Y, 1)[:, :2]
    return float(np.mean(((cp - cy) ** 2).sum(1) / 2))


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def run(code: str, season: str, history: int, refit_days: int):
    start = 2000 + int(season[:2])
    df = load_league(code, seasons=season_codes(history + 1, last_start=start))
    test = df[df["Season"] == season].reset_index(drop=True)
    past = df[df["Season"] < season]
    if not len(test) or not len(past):
        raise RuntimeError("veri yok")

    rows = []
    t, end = test["Date"].min().normalize(), test["Date"].max()
    while t <= end:
        win = test[(test["Date"] >= t) & (test["Date"] < t + pd.Timedelta(days=refit_days))]
        if len(win):
            dc = DixonColes().fit(df, ref_date=t)
            ex = Extras().fit(df, ref_date=t)
            for m in win.itertuples(index=False):
                mat = dc.score_matrix(m.HomeTeam, m.AwayTeam, n_draws=200)
                s = summarize(mat)
                lam, mu = dc.expected_goals(m.HomeTeam, m.AwayTeam)
                ref = m.Referee if isinstance(m.Referee, str) and m.Referee else None
                e = ex.predict(m.HomeTeam, m.AwayTeam, lam, mu, (s["pH"], s["pD"], s["pA"]),
                               float(mat[0, 0]), ref)
                rows.append((m, lam + mu, e))
        t += pd.Timedelta(days=refit_days)

    print(f"\n=== {LEAGUES[code]} {season}: {len(rows)} maç ===")
    print(f"  {'':<34}{'Model':>9}{'Referans':>10}")

    # Gol ortalaması (model gol sayısını düşük mü tahmin ediyor?)
    pred_goals = np.mean([g for _, g, _ in rows])
    real_goals = np.mean([m.FTHG + m.FTAG for m, _, _ in rows])
    print(f"  {'Maç başı gol (tahmin / gerçek)':<34}{pred_goals:>9.2f}{real_goals:>10.2f}")

    def line(name, model, base, fmt="{:.4f}"):
        better = "  ✓ model daha iyi" if model < base else "  ✗ referans daha iyi"
        print(f"  {name:<34}{fmt.format(model):>9}{fmt.format(base):>10}{better}")

    # İlk yarı sonucu
    ok = [(m, e) for m, _, e in rows if pd.notna(m.HTHG) and pd.notna(m.HTAG)]
    if ok:
        o = np.array([0 if m.HTHG > m.HTAG else 1 if m.HTHG == m.HTAG else 2 for m, _ in ok])
        P = np.array([[e["ilk_yari"]["p_ev"], e["ilk_yari"]["p_beraberlik"], e["ilk_yari"]["p_deplasman"]]
                      for _, e in ok])
        po = past.dropna(subset=["HTHG", "HTAG"])
        base = np.bincount(np.where(po.HTHG > po.HTAG, 0, np.where(po.HTHG == po.HTAG, 1, 2)), minlength=3)
        line("İlk yarı sonucu (RPS)", rps3(P, o), rps3(np.tile(base / base.sum(), (len(o), 1)), o))
        y05 = np.array([(m.HTHG + m.HTAG) > 0.5 for m, _ in ok], float)
        p05 = [e["ilk_yari"]["ust_0_5"] for _, e in ok]
        line("İlk yarı 0.5 üst (Brier)", brier(p05, y05),
             brier(np.full(len(y05), ((po.HTHG + po.HTAG) > 0.5).mean()), y05))

    # Kart, korner, şut
    specs = [("sari_kart", "HY", "AY", 4.5, "Sarı kart"), ("korner", "HC", "AC", 9.5, "Korner"),
             ("sut", "HS", "AS", None, "Şut"), ("isabetli_sut", "HST", "AST", None, "İsabetli şut")]
    for key, hc, ac, ln, name in specs:
        ok = [(m, e) for m, _, e in rows if key in e["istatistikler"]
              and pd.notna(getattr(m, hc)) and pd.notna(getattr(m, ac))]
        if not ok:
            continue
        real = np.array([getattr(m, hc) + getattr(m, ac) for m, _ in ok], float)
        pred = np.array([e["istatistikler"][key]["toplam"] for _, e in ok])
        base_tot = (past[hc] + past[ac]).dropna()
        line(f"{name} toplamı (ort. hata)", np.mean(np.abs(pred - real)),
             np.mean(np.abs(base_tot.mean() - real)), "{:.2f}")
        if ln is not None:
            p = [next(c["ust"] for c in e["istatistikler"][key]["cizgiler"] if c["cizgi"] == ln) for _, e in ok]
            y = (real > ln).astype(float)
            line(f"{name} {ln} üst (Brier)", brier(p, y), brier(np.full(len(y), (base_tot > ln).mean()), y))

    ok = [(m, e) for m, _, e in rows if "kirmizi_kart" in e["istatistikler"] and pd.notna(m.HR) and pd.notna(m.AR)]
    if ok:
        y = np.array([(m.HR + m.AR) > 0 for m, _ in ok], float)
        p = [e["istatistikler"]["kirmizi_kart"]["p_en_az_bir"] for _, e in ok]
        line("Kırmızı kart çıkar mı (Brier)", brier(p, y),
             brier(np.full(len(y), ((past.HR + past.AR) > 0).mean()), y))


def main():
    last = current_season_start() - 1
    ap = argparse.ArgumentParser(description="Ek tahminlerin geriye dönük testi")
    ap.add_argument("--lig")
    ap.add_argument("--hepsi", action="store_true")
    ap.add_argument("--sezon", default=season_code(last))
    ap.add_argument("--gecmis", type=int, default=3)
    ap.add_argument("--yenileme", type=int, default=14)
    args = ap.parse_args()
    if not args.hepsi and not args.lig:
        ap.error("--lig veya --hepsi gerekli")
    for code in (list(LEAGUES) if args.hepsi else [resolve_league(args.lig)]):
        try:
            run(code, args.sezon, args.gecmis, args.yenileme)
        except Exception as e:
            print(f"\n=== {LEAGUES[code]}: atlandı ({e})")


if __name__ == "__main__":
    main()
