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
from models import MISS_COLS, DixonColes
from ozellikler import H2HIndex, h2h_mult

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# form ne kadar hızlı eskisin (yarılanma: ~1,9 yıl, 1 yıl, 7,6 ay, 5 ay, 3,8 ay, 2,5 ay)
XI_GRID = [0.0010, 0.0019, 0.0030, 0.0045, 0.0060, 0.0090]
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


def predict_row(model, m, sd, extras: dict, h2h_index=None, n_draws=120):
    """Bir maç için skor matrisi; extras: {"fatigue", "pos", "h2h"} açık/kapalı."""
    ht = getattr(m, "HTired", 0) if extras.get("fatigue") else 0
    at = getattr(m, "ATired", 0) if extras.get("fatigue") else 0
    pdv = getattr(m, "PosDiff", 0.0) if extras.get("pos") else 0.0
    pdv = 0.0 if pdv != pdv else pdv  # NaN -> 0
    miss = None
    if extras.get("lineup") in MISS_COLS:
        miss = tuple(float(getattr(m, c, 0.0) or 0.0) for c in MISS_COLS[extras["lineup"]])
        miss = tuple(0.0 if v != v else v for v in miss)
    mult = None
    if extras.get("h2h") and h2h_index is not None:
        lam, mu = model.expected_goals(m.HomeTeam, m.AwayTeam, ht, at, pdv, miss)
        fh, fa, _ = h2h_mult(h2h_index, m.HomeTeam, m.AwayTeam, lam, mu, date=m.Date)
        mult = (fh, fa)
    return model.score_matrix(m.HomeTeam, m.AwayTeam, sd=sd, n_draws=n_draws, h_tired=ht, a_tired=at,
                              mult=mult, pos_diff=pdv, miss=miss)


def walk_forward(df, season, fit_configs, sds, refit_days, extras: dict | None = None):
    """Sezonu hafta hafta yürür; her (xi, reg, sot, bc) için modeli o güne kadarki veriyle eğitir."""
    extras = extras or {}
    test = df[df["Season"] == season]
    preds = {(*cfg, sd): np.zeros((len(test), 3)) for cfg in fit_configs for sd in sds}
    h2h_index = H2HIndex(df) if extras.get("h2h") else None
    t, end = test["Date"].min().normalize(), test["Date"].max()
    while t <= end:
        win = test[(test["Date"] >= t) & (test["Date"] < t + pd.Timedelta(days=refit_days))]
        if len(win):
            idx = test.index.get_indexer(win.index)
            for xi, reg, sot, bc in fit_configs:
                model = DixonColes(xi=xi, reg=reg, sot=sot, bc=bc, fatigue=bool(extras.get("fatigue")),
                                   pos=bool(extras.get("pos")), lineup=extras.get("lineup")).fit(df, ref_date=t)
                for i, m in zip(idx, win.itertuples(index=False)):
                    for sd in sds:
                        preds[(xi, reg, sot, bc, sd)][i] = probs(predict_row(model, m, sd, extras, h2h_index))
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

    # 3) Ek etkiler, sırayla: yorgunluk -> topla oynama -> eşleşme (H2H).
    #    Her biri, o ana kadar seçilenlerin üstüne eklenince iki sezonun ortalamasında
    #    iyileştiriyorsa açık kalır.
    chosen.update(fatigue=False, pos=False, h2h=False, lineup=None)
    extra_rows = []
    cfg4, sd_b = cfg_best[:4], cfg_best[4]
    current = next(z for z in rows if z[1] == cfg_best)[4]
    tired_share = float(df[["HTired", "ATired"]].to_numpy().mean()) if "HTired" in df.columns else 0.0
    pos_share = float((df.loc[df["Season"].isin([tune_season, test_season]), "PosDiff"] != 0).mean()) \
        if "PosDiff" in df.columns else 0.0
    two = df["Season"].isin([tune_season, test_season])

    def miss_share(mode):
        cols = [c for c in MISS_COLS[mode] if c in df.columns]
        return float((df.loc[two, cols].abs().sum(1) > 0).mean()) if cols else 0.0

    xi_share, side_share = miss_share("xi"), miss_share("side")
    checks = [("fatigue", True, "yorgunluk", tired_share > 0.02,
               f"maçların %{100 * tired_share:.0f}'inde bir takım ≤4 gün önce oynamış"),
              ("pos", True, "topla oynama", pos_share > 0.3, f"maçların %{100 * pos_share:.0f}'inde veri var"),
              ("h2h", True, "eşleşme (H2H)", True, "aynı sahadaki geçmiş karşılaşmalar"),
              ("lineup", "xi", "eksik oyuncu (ilk 11)", xi_share > 0.5,
               f"maçların %{100 * xi_share:.0f}'inde ilk 11 verisi var"),
              ("lineup", "side", "eksik oyuncu (sakat l.)", side_share > 0.5,
               f"maçların %{100 * side_share:.0f}'inde sakat listesi var")]
    adi = {"fatigue": "yorgunluk", "pos": "topla oynama", "h2h": "eşleşme", "lineup": "eksik oyuncu"}
    for key, val, ad, ok, note in checks:
        if not ok:
            extra_rows.append((f"  + {ad}", None, None, None, f"test edilemedi: veri yetersiz ({note})", False))
            continue
        trial = {k: chosen[k] for k in ("fatigue", "pos", "h2h", "lineup")}
        trial[key] = val
        _, pf1 = walk_forward(df, tune_season, [cfg4], [sd_b], refit_days, extras=trial)
        _, pf2 = walk_forward(df, test_season, [cfg4], [sd_b], refit_days, extras=trial)
        f1, f2 = rps(pf1[cfg_best], o), rps(pf2[cfg_best], o2)
        avg = (f1 * n1 + f2 * n2) / (n1 + n2)
        better = avg < current
        extra_rows.append((f"  + {ad}", f1, f2, avg, note, better))
        if better:
            chosen[key] = val
            current = avg
            label_best = label_best.split(" + eksik oyuncu")[0] if key == "lineup" else label_best
            label_best, exam_best = label_best + f" + {adi[key]}" + (f" ({ad.split('(')[1]}" if key == "lineup" else ""), f2
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
        mark = " <-" if c == cfg_best else ""
        print(f"    {label:<26}{r1:>8.4f}{r2:>8.4f}{avg:>10.4f}   "
              f"xi={c[0]} reg={c[1]} şut={c[2]} b.şans={c[3]} bel.={c[4]}{mark}")
    print("    Ek etkiler (seçilen ayarın üstüne, sırayla):")
    for label, r1, r2, avg, note, better in extra_rows:
        if r1 is None:
            print(f"    {label:<26}   {note}")
        else:
            print(f"    {label:<26}{r1:>8.4f}{r2:>8.4f}{avg:>10.4f}   "
                  f"{'AÇIK  (iyileştirdi)' if better else 'kapalı (iyileştirmedi)'} · {note}")
    if bk_rps is not None:
        print(f"    {'Bahis piyasası':<26}{'':>8}{bk_rps:>8.4f}")
    print(f"  Karar: {label_best}  ->  xi={chosen['xi']}, reg={chosen['reg']}, "
          f"isabetli şut={chosen['sot']}, büyük şans={chosen['bc']}, belirsizlik={chosen['sd']}, "
          f"yorgunluk={'açık' if chosen['fatigue'] else 'kapalı'}, "
          f"topla oynama={'açık' if chosen['pos'] else 'kapalı'}, "
          f"eşleşme={'açık' if chosen['h2h'] else 'kapalı'}, "
          f"eksik oyuncu={ {'xi': 'ilk 11', 'side': 'sakat listesi'}.get(chosen['lineup'], 'kapalı') }")
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
