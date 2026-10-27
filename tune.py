"""Her lig için model ayarlarını otomatik arar.

İki sezon kullanılır:
  1. Ayar sezonu (varsayılan 2024-25): tüm kombinasyonlar denenir; her "aile" için
     (sadece gol / +isabetli şut / +büyük şans / ikisi birden) en iyisi aday olur.
  2. Sınav sezonu (varsayılan 2025-26): adaylar ve varsayılan ayar burada da test edilir.
İki sezonun ORTALAMASINDA en iyi olan aday kullanılır; böylece tek sezonda şans eseri
iyi görünen bir ayar seçilmez. Sonuç ayarlar.json'a yazılır;
predict.py bu dosyayı otomatik okur.

Ölçüt: RPS (futbol tahmininde standart, küçük = iyi).

Örnek:
  python tune.py --hepsi
  python tune.py --lig laliga
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from data import LEAGUES, current_season_start, load_league, resolve_league, season_code, season_codes
from models import DixonColes

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

XI_GRID = [0.0010, 0.0019, 0.0030, 0.0045]   # form ne kadar hızlı eskisin
REG_GRID = [0.3, 1.0, 3.0]                   # az maçı olan takımları ne kadar ortalamaya çeksin
SD_GRID = [0.0, 0.10, 0.20]                  # beklenen gole eklenen belirsizlik
# (isabetli şut, büyük şans) ağırlıkları; (0, 0) = sadece goller
SHOT_GRID = [(0.0, 0.0), (0.3, 0.0), (0.5, 0.0), (0.7, 0.0), (0.0, 0.3), (0.0, 0.5),
             (0.0, 0.7), (0.3, 0.3), (0.2, 0.5)]
DEFAULT = {"xi": 0.0019, "reg": 1.0, "sot": 0.0, "bc": 0.0, "sd": 0.10}
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ayarlar.json")


def probs(mat: np.ndarray) -> tuple[float, float, float]:
    return float(np.tril(mat, -1).sum()), float(np.trace(mat)), float(np.triu(mat, 1).sum())


def outcome_index(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["FTHG"] > df["FTAG"], 0, np.where(df["FTHG"] == df["FTAG"], 1, 2))


def rps(P: np.ndarray, o: np.ndarray) -> float:
    Y = np.eye(3)[o]
    cp, cy = np.cumsum(P, 1)[:, :2], np.cumsum(Y, 1)[:, :2]
    return float(np.mean(((cp - cy) ** 2).sum(1) / 2))


def accuracy(P: np.ndarray, o: np.ndarray) -> float:
    return float(np.mean(P.argmax(1) == o))


def walk_forward(df, season, fit_configs, sds, refit_days, fatigue: bool = False):
    """Sezonu hafta hafta yürür; her (xi, reg, sot, bc) için modeli o güne kadarki veriyle eğitir."""
    test = df[df["Season"] == season]
    preds = {(*cfg, sd): np.zeros((len(test), 3)) for cfg in fit_configs for sd in sds}
    t, end = test["Date"].min().normalize(), test["Date"].max()
    while t <= end:
        win = test[(test["Date"] >= t) & (test["Date"] < t + pd.Timedelta(days=refit_days))]
        if len(win):
            idx = test.index.get_indexer(win.index)
            for xi, reg, sot, bc in fit_configs:
                model = DixonColes(xi=xi, reg=reg, sot=sot, bc=bc, fatigue=fatigue).fit(df, ref_date=t)
                for i, m in zip(idx, win.itertuples(index=False)):
                    ht = getattr(m, "HTired", 0) if fatigue else 0
                    at = getattr(m, "ATired", 0) if fatigue else 0
                    for sd in sds:
                        mat = model.score_matrix(m.HomeTeam, m.AwayTeam, sd=sd, n_draws=120,
                                                 h_tired=ht, a_tired=at)
                        preds[(xi, reg, sot, bc, sd)][i] = probs(mat)
        t += pd.Timedelta(days=refit_days)
    return test, preds


def tune_league(code: str, tune_season: str, test_season: str, refit_days: int, history: int):
    start = 2000 + int(test_season[:2])
    df = load_league(code, seasons=season_codes(history + 2, last_start=start))
    if not (df["Season"] == tune_season).any() or not (df["Season"] == test_season).any():
        raise RuntimeError("ayar veya sınav sezonu için veri yok")

    # 1) Ayar sezonunda tüm kombinasyonlar
    t0 = time.time()
    test, preds = walk_forward(df, tune_season, [(x, r, s, b) for x, r, (s, b) in itertools.product(XI_GRID, REG_GRID, SHOT_GRID)],
                               SD_GRID, refit_days)
    o = outcome_index(test)
    table = sorted(((rps(P, o), cfg) for cfg, P in preds.items()), key=lambda x: x[0])
    default_key = (DEFAULT["xi"], DEFAULT["reg"], DEFAULT["sot"], DEFAULT["bc"], DEFAULT["sd"])
    n1 = len(o)

    # 2) Aday listesi: her aileden en iyisi + varsayılan. Tek sezonda "şanslı" çıkan bir
    #    ayarın seçilmesini önlemek için adaylar İKİ sezonun ortalamasıyla karşılaştırılır.
    families = [
        ("Varsayılan (sadece gol)", lambda c: c == default_key),
        ("Sadece gol (ayarlı)", lambda c: c[2] == 0 and c[3] == 0),
        ("Gol + isabetli şut", lambda c: c[2] > 0 and c[3] == 0),
        ("Gol + büyük şans", lambda c: c[3] > 0 and c[2] == 0),
        ("Gol + şut + büyük şans", lambda c: c[2] > 0 and c[3] > 0),
    ]
    cands = []
    for label, cond in families:
        hits = [(r, c) for r, c in table if cond(c)]
        if hits:
            cands.append((label, hits[0][1], hits[0][0]))

    fit_cfgs = sorted({c[:4] for _, c, _ in cands})
    sds = sorted({c[4] for _, c, _ in cands})
    test2, preds2 = walk_forward(df, test_season, fit_cfgs, sds, refit_days)
    o2 = outcome_index(test2)
    n2 = len(o2)
    rows = []
    for label, c, r1 in cands:
        r2 = rps(preds2[c], o2)
        rows.append((label, c, r1, r2, (r1 * n1 + r2 * n2) / (n1 + n2)))
    label_best, cfg_best, _, exam_best, _ = min(rows, key=lambda z: z[4])
    chosen = dict(zip(["xi", "reg", "sot", "bc", "sd"], cfg_best))
    chosen["fatigue"] = False

    # 3) Yorgunluk: seçilen ayara yorgunluk etkisi eklenince iki sezonda da iyileşiyor mu?
    fat_row = None
    tired_share = float(df[["HTired", "ATired"]].to_numpy().mean()) if "HTired" in df.columns else 0.0
    if tired_share > 0.02:
        cfg4, sd_b = cfg_best[:4], cfg_best[4]
        _, pf1 = walk_forward(df, tune_season, [cfg4], [sd_b], refit_days, fatigue=True)
        _, pf2 = walk_forward(df, test_season, [cfg4], [sd_b], refit_days, fatigue=True)
        f1, f2 = rps(pf1[cfg_best], o), rps(pf2[cfg_best], o2)
        best_row = next(z for z in rows if z[1] == cfg_best)
        fat_row = ("  + yorgunluk etkisi", cfg_best, f1, f2, (f1 * n1 + f2 * n2) / (n1 + n2))
        if fat_row[4] < best_row[4]:
            chosen["fatigue"] = True
            label_best, exam_best = label_best + " + yorgunluk", f2
    exam_def = next(z[3] for z in rows if z[1] == default_key)

    bk = test2[["bk_H", "bk_D", "bk_A"]].to_numpy()
    has_bk = ~np.isnan(bk).any(1)
    bk_rps = rps(bk[has_bk] / bk[has_bk].sum(1, keepdims=True), o2[has_bk]) if has_bk.any() else None

    print(f"\n=== {LEAGUES[code]} ({time.time() - t0:.0f} sn) ===")
    bc_share = df.loc[df["Season"] == tune_season, "HBC"].notna().mean() if "HBC" in df.columns else 0
    print(f"  {len(table)} kombinasyon denendi (büyük şans verisi olan maç oranı, {tune_season}: "
          f"%{100 * bc_share:.0f}). Adaylar (RPS, küçük = iyi):")
    print(f"    {'':<26}{tune_season:>8}{test_season:>8}{'Ortalama':>10}   ayar")
    for label, c, r1, r2, avg in rows:
        mark = " <-" if c == cfg_best and not chosen["fatigue"] else ""
        print(f"    {label:<26}{r1:>8.4f}{r2:>8.4f}{avg:>10.4f}   "
              f"xi={c[0]} reg={c[1]} şut={c[2]} b.şans={c[3]} bel.={c[4]}{mark}")
    if fat_row:
        print(f"    {fat_row[0]:<26}{fat_row[2]:>8.4f}{fat_row[3]:>8.4f}{fat_row[4]:>10.4f}   "
              f"(maçların %{100 * tired_share:.0f}'inde bir takım ≤4 gün önce oynamış)"
              + (" <-" if chosen["fatigue"] else ""))
    else:
        print("    (yorgunluk testi yapılamadı: dinlenme verisi yetersiz)")
    if bk_rps is not None:
        print(f"    {'Bahis piyasası':<26}{'':>8}{bk_rps:>8.4f}")
    print(f"  Karar: {label_best}  ->  xi={chosen['xi']}, reg={chosen['reg']}, "
          f"isabetli şut={chosen['sot']}, büyük şans={chosen['bc']}, belirsizlik={chosen['sd']}, "
          f"yorgunluk={'açık' if chosen['fatigue'] else 'kapalı'}")
    return chosen, {"secilen_aile": label_best,
                    "adaylar": [{"aday": l, "ayar_rps": round(a, 5), "sinav_rps": round(b, 5),
                                 "ortalama": round(m, 5)} for l, _, a, b, m in rows],
                    "sinav_varsayilan_rps": round(exam_def, 5),
                    "sinav_secilen_rps": round(exam_best, 5),
                    "sinav_piyasa_rps": None if bk_rps is None else round(bk_rps, 5)}


def main():
    last = current_season_start() - 1
    ap = argparse.ArgumentParser(description="Lig başına model ayarı arama")
    ap.add_argument("--lig", help="super-lig, premier, laliga, seriea")
    ap.add_argument("--hepsi", action="store_true")
    ap.add_argument("--ayar-sezonu", default=season_code(last - 1), help="ör. 2425")
    ap.add_argument("--sinav-sezonu", default=season_code(last), help="ör. 2526")
    ap.add_argument("--yenileme", type=int, default=14, help="Model kaç günde bir yeniden eğitilsin")
    ap.add_argument("--gecmis", type=int, default=3)
    args = ap.parse_args()
    if not args.hepsi and not args.lig:
        ap.error("--lig veya --hepsi gerekli")
    codes = list(LEAGUES) if args.hepsi else [resolve_league(args.lig)]

    settings = {}
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            settings = json.load(f)
    for code in codes:
        try:
            chosen, report = tune_league(code, args.ayar_sezonu, args.sinav_sezonu, args.yenileme, args.gecmis)
        except RuntimeError as e:
            print(f"\n=== {LEAGUES[code]}: atlandı ({e})")
            continue
        settings[code] = {**chosen, "lig": LEAGUES[code], "rapor": report}
    settings["_guncelleme"] = dt.datetime.now().isoformat(timespec="minutes")
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    print(f"\nAyarlar {os.path.basename(SETTINGS_FILE)} dosyasına yazıldı.")


if __name__ == "__main__":
    main()
