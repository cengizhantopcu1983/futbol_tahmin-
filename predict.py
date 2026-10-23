"""Maç tahmini.

Örnekler:
  python predict.py --lig super-lig --ev Galatasaray --deplasman Fenerbahce
  python predict.py --fikstur                     # 4 ligin yaklaşan tüm maçları
  python predict.py --fikstur --json tahminler.json   # Android uygulaması için çıktı
  python predict.py --lig premier --siralama      # takım güç sıralaması
  python predict.py --lig laliga --takimlar       # takım adları
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import numpy as np

from data import LEAGUES, find_team, load_fixtures, load_league, resolve_league
from models import DixonColes, compute_elo, elo_expectation, head_to_head, team_form

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def bar(p: float, width: int = 30) -> str:
    return "█" * int(round(p * width))


def pct(p: float) -> str:
    return f"{100 * p:5.1f}%"


class LeagueModel:
    """Bir ligin verisi + eğitilmiş modeli + Elo puanları."""

    def __init__(self, code: str, n_seasons: int, xi: float):
        self.code = code
        self.name = LEAGUES[code]
        self.df = load_league(code, n_seasons=n_seasons)
        self.model = DixonColes(xi=xi).fit(self.df)
        self.elo, _ = compute_elo(self.df)
        self.current_teams = sorted(set(self.df[self.df["Season"] == self.df["Season"].iloc[-1]]
                                        [["HomeTeam", "AwayTeam"]].to_numpy().ravel()))


def print_match(lm: LeagueModel, home: str, away: str, sd: float, n_sim: int, bk=None):
    r = lm.model.predict(home, away, sd=sd, n_sim=n_sim)
    line = "═" * 60
    print(line)
    print(f" {home}  -  {away}   ({lm.name})")
    print(line)
    if r["unknown"]:
        print(f"  ! Az verisi olan takım: {', '.join(r['unknown'])} (tahmin daha belirsiz)")
    print(f"Beklenen gol: {r['xg_home']:.2f} - {r['xg_away']:.2f}\n")
    print("Sonuç olasılıkları")
    for label, p in (("Ev sahibi kazanır", r["pH"]), ("Beraberlik", r["pD"]), ("Deplasman kazanır", r["pA"])):
        print(f"  {label:<18} {pct(p)}  {bar(p)}")
    print(f"  Adil oranlar:      {1 / r['pH']:.2f} / {1 / r['pD']:.2f} / {1 / r['pA']:.2f}")
    if bk is not None and not np.isnan(bk[0]):
        print(f"  Piyasa (oranlardan): {pct(bk[0])} / {pct(bk[1])} / {pct(bk[2])}")
    s = r.get("sim")
    if s:
        n = lambda v: f"{v:,}".replace(",", ".")
        print(f"\n{n(s['n'])} maç simülasyonu: {n(s['H'])} ev sahibi galibiyeti, {n(s['D'])} beraberlik, "
              f"{n(s['A'])} deplasman galibiyeti")
    print("\nEn olası skorlar: " + ", ".join(f"{h}-{a} ({100 * p:.1f}%)" for h, a, p in r["top_scores"]))
    print(f"2.5 üst: {pct(r['over25']).strip()}   |   Karşılıklı gol: {pct(r['btts']).strip()}")

    eh, ea = lm.elo.get(home), lm.elo.get(away)
    if eh and ea:
        print(f"\nElo: {home} {eh:.0f}  |  {away} {ea:.0f}  "
              f"(ev sahibi beklenen puan payı {100 * elo_expectation(eh, ea):.0f}%)")

    print("\nForm (son 5 lig maçı, en sağdaki en yeni):")
    for t in (home, away):
        f = team_form(lm.df, t)
        print(f"  {t:<20} {f['seq']:<10} {f['pts']} puan, gol {f['gf']}-{f['ga']}")

    h2h = head_to_head(lm.df, home, away)
    if len(h2h):
        print("\nSon karşılaşmalar (bilgi amaçlı, modele girmez):")
        for m in h2h.itertuples(index=False):
            print(f"  {m.Date:%Y-%m-%d}  {m.HomeTeam} {m.FTHG}-{m.FTAG} {m.AwayTeam}")
    print()


def run_fixtures(args):
    fx = load_fixtures()
    today = dt.date.today()
    fx = fx[(fx["Date"].dt.date >= today) | fx["Date"].isna()]
    if not len(fx):
        print("Yaklaşan maç bulunamadı.")
    output = []
    for code in [c for c in LEAGUES if c in set(fx["Div"])]:
        print(f"\n### {LEAGUES[code]}")
        try:
            lm = LeagueModel(code, args.sezon_sayisi, args.xi)
        except Exception as e:  # bir lig hata verirse diğerleri yine çalışsın
            print(f"  atlandı: {e}")
            continue
        for m in fx[fx["Div"] == code].itertuples(index=False):
            try:
                home = find_team(m.HomeTeam, lm.model.teams + lm.current_teams)
                away = find_team(m.AwayTeam, lm.model.teams + lm.current_teams)
            except ValueError as e:
                print(f"  atlandı: {e}")
                continue
            r = lm.model.predict(home, away, sd=args.belirsizlik, n_sim=0)
            h, a, p = r["top_scores"][0]
            date = f"{m.Date:%d.%m}" if m.Date == m.Date else "??"
            print(f"  {date} {m.Time!s:>5}  {home[:16]:>16} - {away[:16]:<16} "
                  f"1:{pct(r['pH'])} X:{pct(r['pD'])} 2:{pct(r['pA'])}  "
                  f"xG {r['xg_home']:.1f}-{r['xg_away']:.1f}  olası {h}-{a}")
            output.append({
                "lig": LEAGUES[code], "lig_kodu": code,
                "tarih": m.Date.strftime("%Y-%m-%d") if m.Date == m.Date else None,
                "saat": str(m.Time) if m.Time == m.Time else None,
                "ev": home, "deplasman": away,
                "p_ev": round(r["pH"], 4), "p_beraberlik": round(r["pD"], 4), "p_deplasman": round(r["pA"], 4),
                "xg_ev": round(r["xg_home"], 2), "xg_deplasman": round(r["xg_away"], 2),
                "ust_2_5": round(r["over25"], 4), "kg_var": round(r["btts"], 4),
                "olasi_skorlar": [{"skor": f"{h}-{a}", "p": round(p, 4)} for h, a, p in r["top_scores"]],
                "form_ev": team_form(lm.df, home)["seq"], "form_deplasman": team_form(lm.df, away)["seq"],
                "elo_ev": round(lm.elo[home]) if home in lm.elo else None,
                "elo_deplasman": round(lm.elo[away]) if away in lm.elo else None,
                "piyasa": None if np.isnan(m.bk_H) else {
                    "ev": round(float(m.bk_H), 4), "beraberlik": round(float(m.bk_D), 4),
                    "deplasman": round(float(m.bk_A), 4)},
                # satır = ev sahibi golü (0-5), sütun = deplasman golü (0-5)
                "skor_matrisi": np.round(r["matrix"][:6, :6], 4).tolist(),
                "az_veri": r["unknown"],
            })
    if args.json:
        istanbul = dt.timezone(dt.timedelta(hours=3))
        payload = {"olusturulma": dt.datetime.now(istanbul).isoformat(timespec="minutes"),
                   "model": "dixon-coles-v1", "maclar": output}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n{len(output)} maç {args.json} dosyasına yazıldı.")


def main():
    ap = argparse.ArgumentParser(description="Futbol maç tahmin modeli (Dixon-Coles + Elo)")
    ap.add_argument("--lig", help="super-lig, premier, laliga, seriea")
    ap.add_argument("--ev", help="Ev sahibi takım")
    ap.add_argument("--deplasman", help="Deplasman takımı")
    ap.add_argument("--fikstur", action="store_true", help="Yaklaşan tüm maçları tahmin et")
    ap.add_argument("--json", help="Fikstür tahminlerini bu JSON dosyasına yaz")
    ap.add_argument("--siralama", action="store_true", help="Takım güç sıralaması")
    ap.add_argument("--takimlar", action="store_true", help="Ligdeki takım adlarını listele")
    ap.add_argument("--sezon-sayisi", type=int, default=4, help="Kaç sezonluk veri (varsayılan 4)")
    ap.add_argument("--xi", type=float, default=0.0019, help="Form ağırlığı (büyük = yeni maçlar daha önemli)")
    ap.add_argument("--belirsizlik", type=float, default=0.10, help="Beklenen gole eklenen belirsizlik")
    ap.add_argument("--sim", type=int, default=10000, help="Simülasyon sayısı")
    args = ap.parse_args()

    if args.fikstur:
        run_fixtures(args)
        return
    if not args.lig:
        ap.error("--lig gerekli (veya --fikstur kullan)")
    lm = LeagueModel(resolve_league(args.lig), args.sezon_sayisi, args.xi)

    if args.takimlar:
        print(f"{lm.name} - bu sezonun takımları:")
        for t in lm.current_teams:
            print("  " + t)
        return
    if args.siralama:
        tab = lm.model.ratings_table()
        tab = tab[tab["Takim"].isin(lm.current_teams)].reset_index(drop=True)
        print(f"{lm.name} güç sıralaması (Hücum >1 iyi, Savunma <1 iyi)\n")
        print(f"{'#':>2}  {'Takım':<22}{'Hücum':>7}{'Savunma':>9}{'Elo':>7}")
        for i, r in tab.iterrows():
            print(f"{i + 1:>2}  {r['Takim']:<22}{r['Hucum']:>7.2f}{r['Savunma']:>9.2f}"
                  f"{lm.elo.get(r['Takim'], float('nan')):>7.0f}")
        return
    if not (args.ev and args.deplasman):
        ap.error("--ev ve --deplasman gerekli")
    teams = sorted(set(lm.model.teams) | set(lm.current_teams))
    home, away = find_team(args.ev, teams), find_team(args.deplasman, teams)
    print_match(lm, home, away, args.belirsizlik, args.sim)


if __name__ == "__main__":
    main()
