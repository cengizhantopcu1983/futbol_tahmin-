"""Her lig için model ayarlarını otomatik arar.

Ezberi önlemek için iki sezon kullanılır:
  1. Ayar sezonu (varsayılan 2024-25): her ayar kombinasyonu denenir, en iyisi seçilir.
  2. Sınav sezonu (varsayılan 2025-26): seçilen ayar, varsayılan ayarla karşılaştırılır.
Yeni ayar ancak sınav sezonunda da daha iyiyse kullanılır. Sonuç ayarlar.json'a yazılır;
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
DEFAULT = {"xi": 0.0019, "reg": 1.0, "sd": 0.10}
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


def walk_forward(df, season, fit_configs, sds, refit_days):
    """Sezonu hafta hafta yürür; her (xi, reg) için modeli o güne kadarki veriyle eğitir."""
    test = df[df["Season"] == season]
    preds = {(xi, reg, sd): np.zeros((len(test), 3)) for xi, reg in fit_configs for sd in sds}
    t, end = test["Date"].min().normalize(), test["Date"].max()
    while t <= end:
        win = test[(test["Date"] >= t) & (test["Date"] < t + pd.Timedelta(days=refit_days))]
        if len(win):
            idx = test.index.get_indexer(win.index)
            for xi, reg in fit_configs:
                model = DixonColes(xi=xi, reg=reg).fit(df, ref_date=t)
                for i, m in zip(idx, win.itertuples(index=False)):
                    for sd in sds:
                        mat = model.score_matrix(m.HomeTeam, m.AwayTeam, sd=sd, n_draws=120)
                        preds[(xi, reg, sd)][i] = probs(mat)
        t += pd.Timedelta(days=refit_days)
    return test, preds


def tune_league(code: str, tune_season: str, test_season: str, refit_days: int, history: int):
    start = 2000 + int(test_season[:2])
    df = load_league(code, seasons=season_codes(history + 2, last_start=start))
    if not (df["Season"] == tune_season).any() or not (df["Season"] == test_season).any():
        raise RuntimeError("ayar veya sınav sezonu için veri yok")

    # 1) Ayar sezonunda tüm kombinasyonlar
    t0 = time.time()
    test, preds = walk_forward(df, tune_season, list(itertools.product(XI_GRID, REG_GRID)),
                               SD_GRID, refit_days)
    o = outcome_index(test)
    table = sorted(((rps(P, o), cfg) for cfg, P in preds.items()), key=lambda x: x[0])
    best_rps, (xi, reg, sd) = table[0]
    best = {"xi": xi, "reg": reg, "sd": sd}
    default_key = (DEFAULT["xi"], DEFAULT["reg"], DEFAULT["sd"])
    tune_default = rps(preds[default_key], o)

    # 2) Sınav sezonunda varsayılan vs seçilen
    fit_cfgs = sorted({(DEFAULT["xi"], DEFAULT["reg"]), (xi, reg)})
    test2, preds2 = walk_forward(df, test_season, fit_cfgs, sorted({DEFAULT["sd"], sd}), refit_days)
    o2 = outcome_index(test2)
    P_def, P_best = preds2[default_key], preds2[(xi, reg, sd)]
    exam_def, exam_best = rps(P_def, o2), rps(P_best, o2)
    chosen = best if exam_best < exam_def else dict(DEFAULT)

    bk = test2[["bk_H", "bk_D", "bk_A"]].to_numpy()
    has_bk = ~np.isnan(bk).any(1)
    bk_rps = rps(bk[has_bk] / bk[has_bk].sum(1, keepdims=True), o2[has_bk]) if has_bk.any() else None

    print(f"\n=== {LEAGUES[code]} ({time.time() - t0:.0f} sn) ===")
    print(f"  Ayar sezonu {tune_season}, {len(table)} kombinasyon denendi. En iyi 3 (RPS):")
    for r, (a, b, c) in table[:3]:
        print(f"    xi={a:.4f} reg={b:<4} belirsizlik={c:.2f}   RPS {r:.4f}")
    print(f"    varsayılan ayar                         RPS {tune_default:.4f}")
    print(f"  Sınav sezonu {test_season}:")
    print(f"    {'':<22}{'RPS':>8}{'İsabet':>9}")
    print(f"    {'Varsayılan ayar':<22}{exam_def:>8.4f}{100 * accuracy(P_def, o2):>8.1f}%")
    print(f"    {'Seçilen ayar':<22}{exam_best:>8.4f}{100 * accuracy(P_best, o2):>8.1f}%")
    if bk_rps is not None:
        print(f"    {'Bahis piyasası':<22}{bk_rps:>8.4f}{100 * accuracy(bk[has_bk], o2[has_bk]):>8.1f}%")
    verdict = "YENİ AYAR KULLANILACAK" if chosen == best and best != DEFAULT else "varsayılan ayar korunuyor"
    print(f"  Karar: {verdict}  ->  xi={chosen['xi']}, reg={chosen['reg']}, belirsizlik={chosen['sd']}")
    return chosen, {"ayar_rps": round(best_rps, 5), "sinav_varsayilan_rps": round(exam_def, 5),
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
