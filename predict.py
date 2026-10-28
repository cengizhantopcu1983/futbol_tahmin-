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
import os
import sys

import numpy as np
import pandas as pd

from data import (FIXTURE_COLS, LEAGUES, TZ_TR, current_season_start, find_team, load_fixtures,
                  load_fixtures_api, load_league, resolve_league, resolve_team, season_code)
from yorgunluk import TIRED_DAYS
from ekstra import Extras
from ozellikler import H2HIndex, h2h_mult, possession_diff
from manuel_xg import add_manual_xg, finishing_factors, load_manual, match_multipliers, shadow_weights
from yorgunluk import RestCalendar
from models import DixonColes, compute_elo, elo_expectation, head_to_head, team_form

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def bar(p: float, width: int = 30) -> str:
    return "█" * int(round(p * width))


def pct(p: float) -> str:
    return f"{100 * p:5.1f}%"


SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ayarlar.json")
DEFAULTS = {"xi": 0.0019, "reg": 1.0, "sot": 0.0, "bc": 0.0, "sd": 0.10,
            "fatigue": False, "pos": False, "h2h": False}
FLAGS = ("fatigue", "pos", "h2h")


def league_params(code: str, args) -> dict:
    """Ayarlar: varsayılan < ayarlar.json (tune.py çıktısı) < komut satırı."""
    p = dict(DEFAULTS)
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            saved = json.load(f).get(code, {})
        p.update({k: float(saved[k]) for k in DEFAULTS if k in saved and k not in FLAGS})
        p.update({k: bool(saved.get(k, False)) for k in FLAGS})
    except (OSError, ValueError):
        pass
    for key, arg in (("xi", "xi"), ("reg", "reg"), ("sot", "sut"), ("bc", "buyuk_sans"), ("sd", "belirsizlik")):
        if getattr(args, arg, None) is not None:
            p[key] = getattr(args, arg)
    return p


class LeagueModel:
    """Bir ligin verisi + eğitilmiş modeli + Elo puanları."""

    def __init__(self, code: str, args):
        self.code = code
        self.name = LEAGUES[code]
        self.params = league_params(code, args)
        self.sd = self.params["sd"]
        self.df = load_league(code, n_seasons=args.sezon_sayisi)
        self.model = DixonColes(xi=self.params["xi"], reg=self.params["reg"],
                                sot=self.params["sot"], bc=self.params["bc"],
                                fatigue=self.params["fatigue"], pos=self.params["pos"]).fit(self.df)
        self.h2h_index = H2HIndex(self.df)
        self.extras = Extras().fit(self.df, xi=self.params["xi"])
        self._fit_shadow()
        self.elo, _ = compute_elo(self.df)
        self.current_teams = sorted(set(self.df[self.df["Season"] == self.df["Season"].iloc[-1]]
                                        [["HomeTeam", "AwayTeam"]].to_numpy().ravel()))

    def _fit_shadow(self):
        """Elle girilen xG ile 'gölge model' (ana tahmini değiştirmez, karnede kıyaslanır)."""
        self.shadow, self.shadow_n, self.shadow_w = None, 0, 0.0
        self.finish = {}
        manual = load_manual()
        if not manual["maclar"] and not manual["takimlar"].get(self.code):
            return
        dfx = add_manual_xg(self.df, self.code, manual)
        n = int(dfx["HXG"].notna().sum())
        if n < 20:
            return
        sot, bc, wx = shadow_weights(self.params["sot"], self.params["bc"])
        try:
            m = DixonColes(xi=self.params["xi"], reg=self.params["reg"], sot=sot, bc=bc, xg=wx,
                           fatigue=self.params["fatigue"], pos=self.params["pos"]).fit(dfx)
        except Exception as e:
            print(f"  (xG'li gölge model kurulamadı: {e})")
            return
        if m.xg_used > 0:
            self.shadow, self.shadow_n, self.shadow_w = m, n, wx
            # bitiricilik: bu sezonun gerçek golleri / elle girilen xG (küçültülmüş çarpanlar)
            self.finish = finishing_factors(self.df, self.code, manual)
            self.shadow_detail = {"mac": int((dfx["XGSrc"] == "mac").sum()),
                                  "ortalama": int((dfx["XGSrc"] == "ortalama").sum())}

    def extra(self, r: dict, referee: str | None = None) -> dict:
        """İlk yarı, İY/MS, gol zamanlaması ve kart/korner/şut tahminleri."""
        return self.extras.predict(r["home"], r["away"], r["xg_home"], r["xg_away"],
                                   (r["pH"], r["pD"], r["pA"]), float(r["matrix"][0, 0]), referee)


def consistent_score(r: dict) -> dict:
    """En olası sonucu (1/X/2) ve SADECE o sonucu veren skorlar içinde en olasısını döner.

    Tek tek en olası skor çoğu zaman 1-1'dir (beraberlik az sayıda skora yığılır),
    ev sahibi %44 favori olsa bile. Bu fonksiyon tahminle çelişmeyen skoru verir.
    """
    mat = r["matrix"]
    pick = int(np.argmax([r["pH"], r["pD"], r["pA"]]))
    i, j = np.indices(mat.shape)
    mask = (i > j) if pick == 0 else (i == j) if pick == 1 else (i < j)
    flat = int(np.argmax(np.where(mask, mat, -1.0)))
    h, a = divmod(flat, mat.shape[1])
    return {"sonuc": "1X2"[pick], "skor": f"{h}-{a}", "p": round(float(mat[h, a]), 4)}


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
    ts = consistent_score(r)
    sonuc_adi = {"1": "ev sahibi galibiyeti", "X": "beraberlik", "2": "deplasman galibiyeti"}[ts["sonuc"]]
    print(f"\nTahmin: {sonuc_adi}, bu sonuçta en olası skor {ts['skor']}")
    print("En olası skorlar (tek tek): " + ", ".join(f"{h}-{a} ({100 * p:.1f}%)" for h, a, p in r["top_scores"]))
    print(f"2.5 üst: {pct(r['over25']).strip()}   |   Karşılıklı gol: {pct(r['btts']).strip()}")

    ek = lm.extra(r)
    iy = ek["ilk_yari"]
    print(f"\nİlk yarı: 1 {pct(iy['p_ev'])} | X {pct(iy['p_beraberlik'])} | 2 {pct(iy['p_deplasman'])}   "
          f"İY 0.5 üst {pct(iy['ust_0_5']).strip()}, 1.5 üst {pct(iy['ust_1_5']).strip()}")
    print("İY/MS (en olası 4): " + ", ".join(f"{z['kod']} {100 * z['p']:.1f}%" for z in ek["iy_ms"][:4]))
    gd = ek["gol_dakika"]
    print("Gol olma ihtimali (dakika): " + ", ".join(f"{z['aralik']} {100 * z['p_gol']:.0f}%" for z in gd["araliklar"]))
    print(f"İlk gol 30. dk'dan önce: {pct(gd['ilk_gol_30_once']).strip()}   İlk golü atan: "
          f"{home} {pct(gd['ilk_golu_ev']).strip()}, {away} {pct(gd['ilk_golu_deplasman']).strip()}, "
          f"golsüz {pct(gd['golsuz']).strip()}")
    adlar = {"sari_kart": "Sarı kart", "korner": "Korner", "sut": "Şut", "isabetli_sut": "İsabetli şut",
             "kirmizi_kart": "Kırmızı kart"}
    for key, v in ek["istatistikler"].items():
        cz = "  ".join(f"{c['cizgi']} üst {100 * c['ust']:.0f}%" for c in v.get("cizgiler", []))
        extra = f"  (en az bir: {100 * v['p_en_az_bir']:.0f}%, düşük güven)" if key == "kirmizi_kart" else ""
        print(f"{adlar[key]:<13} {v['ev']:.1f} - {v['deplasman']:.1f}  toplam {v['toplam']:.1f}  {cz}{extra}")

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


def collect_fixtures() -> pd.DataFrame:
    """İki kaynaktan yaklaşan maçları toplar. Önce football-data.co.uk (bahis oranları var)."""
    frames = []
    print("Fikstür kaynakları:")
    try:
        uk = load_fixtures()
        print(f"  football-data.co.uk: {len(uk)} maç (4 lig, tarih süzülmeden önce)")
        frames.append(uk)
    except Exception as e:
        print(f"  ! football-data.co.uk fikstürü alınamadı: {e}")
    try:
        frames.append(load_fixtures_api())
    except Exception as e:
        print(f"  ! football-data.org fikstürü alınamadı: {e}")
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=FIXTURE_COLS)
    fx = pd.concat(frames, ignore_index=True)
    today = dt.datetime.now(TZ_TR).date()
    fx["Date"] = pd.to_datetime(fx["Date"])
    return fx[fx["Date"].isna() | (fx["Date"].dt.date >= today)].reset_index(drop=True)


def run_fixtures(args):
    fx = collect_fixtures()
    if not len(fx):
        print("Yaklaşan maç bulunamadı.")
    output = []
    for code in [c for c in LEAGUES if c in set(fx["Div"])]:
        print(f"\n### {LEAGUES[code]}")
        try:
            lm = LeagueModel(code, args)
            p = lm.params
            print(f"  ayarlar: xi={p['xi']}, reg={p['reg']}, isabetli şut={p['sot']}, büyük şans={p['bc']}, "
                  f"belirsizlik={p['sd']}, yorgunluk={'açık' if p['fatigue'] else 'kapalı'}, "
                  f"topla oynama={'açık' if p['pos'] else 'kapalı'}, eşleşme={'açık' if p['h2h'] else 'kapalı'}")
            if lm.shadow is not None:
                print(f"  xG'li gölge model: {lm.shadow_n} maçta xG (elle: {lm.shadow_detail['mac']}, "
                      f"ortalamadan: {lm.shadow_detail['ortalama']}), ağırlık {lm.shadow_w}")
        except Exception as e:  # bir lig hata verirse diğerleri yine çalışsın
            print(f"  atlandı: {e}")
            continue
        teams = sorted(set(lm.model.teams) | set(lm.current_teams))
        seen, items = set(), []
        for m in fx[fx["Div"] == code].itertuples(index=False):
            try:
                home = resolve_team(m.HomeNames, teams)
                away = resolve_team(m.AwayNames, teams)
            except ValueError as e:
                print(f"  atlandı ({m.Kaynak}): {m.HomeTeam} - {m.AwayTeam}: {e}")
                continue
            key = (m.Date.strftime("%Y-%m-%d") if m.Date == m.Date else None, home, away)
            if key in seen:
                continue  # aynı maç iki kaynakta da var
            seen.add(key)
            items.append((m, home, away))
        # dinlenme takvimi: geçmiş lig maçları + Avrupa maçları + bu listedeki yaklaşan maçlar
        upcoming = pd.DataFrame([{"Date": m.Date, "HomeTeam": h, "AwayTeam": a} for m, h, a in items
                                 if m.Date == m.Date], columns=["Date", "HomeTeam", "AwayTeam"])
        try:
            cal = RestCalendar(lm.df, extra=upcoming)
        except Exception as e:
            print(f"  (dinlenme takvimi kurulamadı: {e})")
            cal = None
        for m, home, away in items:
            hr, he = cal.rest(home, m.Date) if cal and m.Date == m.Date else (None, False)
            ar, ae = cal.rest(away, m.Date) if cal and m.Date == m.Date else (None, False)
            ht = int(hr is not None and hr <= TIRED_DAYS)
            at = int(ar is not None and ar <= TIRED_DAYS)
            use_fat = lm.params["fatigue"]
            fh_t, fa_t = (ht, at) if use_fat else (0, 0)
            pdv = possession_diff(lm.df, home, away) if lm.params["pos"] else 0.0
            # eşleşme (H2H): aynı sahadaki geçmiş karşılaşmalar; etki sadece ayar açıksa uygulanır
            lam0, mu0 = lm.model.expected_goals(home, away, fh_t, fa_t, pdv)
            e_h, e_a, eslesme = h2h_mult(lm.h2h_index, home, away, lam0, mu0)
            eslesme["uygulandi"] = bool(lm.params["h2h"] and eslesme.get("mac", 0) > 0)
            mult_main = (e_h, e_a) if eslesme["uygulandi"] else None
            r = lm.model.predict(home, away, sd=lm.sd, n_sim=0, h_tired=fh_t, a_tired=fa_t,
                                 pos_diff=pdv, mult=mult_main)
            shadow = None
            if lm.shadow is not None:
                mult = match_multipliers(lm.finish, home, away)
                if mult_main:
                    mult = (mult[0] * mult_main[0], mult[1] * mult_main[1])
                rs = lm.shadow.predict(home, away, sd=lm.sd, n_sim=0, h_tired=fh_t, a_tired=fa_t,
                                       mult=mult, pos_diff=pdv)
                shadow = {"p_ev": round(rs["pH"], 4), "p_beraberlik": round(rs["pD"], 4),
                          "p_deplasman": round(rs["pA"], 4), "xg_ev": round(rs["xg_home"], 2),
                          "xg_deplasman": round(rs["xg_away"], 2), "mac_sayisi": lm.shadow_n,
                          "agirlik": lm.shadow_w,
                          # ev sahibinin iç saha, konuğun dış saha bitiricilik/savunma verileri
                          "bitiricilik": {"ev": lm.finish.get(home, {}).get("ic"),
                                          "dep": lm.finish.get(away, {}).get("dis"),
                                          "carpan_ev": round(mult[0], 3), "carpan_dep": round(mult[1], 3)}}
            ts = consistent_score(r)
            date = f"{m.Date:%d.%m}" if m.Date == m.Date else "??"
            print(f"  {date} {m.Time!s:>5}  {home[:16]:>16} - {away[:16]:<16} "
                  f"1:{pct(r['pH'])} X:{pct(r['pD'])} 2:{pct(r['pA'])}  "
                  f"xG {r['xg_home']:.1f}-{r['xg_away']:.1f}  tahmin {ts['sonuc']} ({ts['skor']})")
            output.append({
                "lig": LEAGUES[code], "lig_kodu": code,
                "tarih": m.Date.strftime("%Y-%m-%d") if m.Date == m.Date else None,
                "saat": str(m.Time) if m.Time == m.Time else None,
                "ev": home, "deplasman": away,
                "p_ev": round(r["pH"], 4), "p_beraberlik": round(r["pD"], 4), "p_deplasman": round(r["pA"], 4),
                "xg_ev": round(r["xg_home"], 2), "xg_deplasman": round(r["xg_away"], 2),
                "ust_2_5": round(r["over25"], 4), "kg_var": round(r["btts"], 4),
                "olasi_skorlar": [{"skor": f"{h}-{a}", "p": round(p, 4)} for h, a, p in r["top_scores"]],
                # en olası SONUÇ ve o sonuç içindeki en olası skor (ör. "1" ve "2-1")
                "tutarli_skor": ts,
                "form_ev": team_form(lm.df, home)["seq"], "form_deplasman": team_form(lm.df, away)["seq"],
                "elo_ev": round(lm.elo[home]) if home in lm.elo else None,
                "elo_deplasman": round(lm.elo[away]) if away in lm.elo else None,
                "piyasa": None if np.isnan(m.bk_H) else {
                    "ev": round(float(m.bk_H), 4), "beraberlik": round(float(m.bk_D), 4),
                    "deplasman": round(float(m.bk_A), 4)},
                # satır = ev sahibi golü (0-5), sütun = deplasman golü (0-5)
                "skor_matrisi": np.round(r["matrix"][:6, :6], 4).tolist(),
                "az_veri": r["unknown"],
                # ilk_yari, iy_ms, gol_dakika, istatistikler
                **lm.extra(r, referee=(m.Referee or None) if isinstance(m.Referee, str) else None),
                "yorgunluk": {"ev_dinlenme": hr, "dep_dinlenme": ar, "ev_avrupa": bool(he and ht),
                              "dep_avrupa": bool(ae and at), "ev_yorgun": bool(ht), "dep_yorgun": bool(at),
                              "etki_uygulandi": bool(use_fat and (ht or at))},
                "xg_modeli": shadow,
                "eslesme": eslesme,
                "topla_oynama_farki": round(pdv, 2) if lm.params["pos"] else None,
            })
    output.sort(key=lambda o: (o["tarih"] or "9999", o["saat"] or "", o["lig_kodu"]))
    if args.json:
        payload = {"olusturulma": dt.datetime.now(TZ_TR).isoformat(timespec="minutes"),
                   "model": "dixon-coles-v1", "maclar": output}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n{len(output)} maç {args.json} dosyasına yazıldı.")
        write_finished(os.path.join(os.path.dirname(os.path.abspath(args.json)), "bitenler.json"))


def write_finished(path: str) -> None:
    """Bu sezonun biten maçları ve takım listeleri (uygulamadaki 'Biten maçlar' ekranı için)."""
    sc = season_code(current_season_start())
    out = {"olusturulma": dt.datetime.now(TZ_TR).isoformat(timespec="minutes"), "sezon": sc, "ligler": []}
    for code, name in LEAGUES.items():
        try:
            df = load_league(code, seasons=[sc], use_sm=False)
        except Exception as e:
            print(f"  (biten maçlar: {name} atlandı: {e})")
            continue
        df = df.sort_values("Date", ascending=False)
        out["ligler"].append({
            "lig": name, "lig_kodu": code,
            "takimlar": sorted(set(df["HomeTeam"]) | set(df["AwayTeam"])),
            "maclar": [{"tarih": f"{r.Date:%Y-%m-%d}", "ev": r.HomeTeam, "deplasman": r.AwayTeam,
                        "skor": f"{int(r.FTHG)}-{int(r.FTAG)}"} for r in df.itertuples(index=False)],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Biten maçlar: {sum(len(l['maclar']) for l in out['ligler'])} maç {os.path.basename(path)} dosyasına yazıldı.")


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
    ap.add_argument("--xi", type=float, help="Form ağırlığı (büyük = yeni maçlar daha önemli); "
                                              "verilmezse ayarlar.json veya varsayılan")
    ap.add_argument("--reg", type=float, help="Az maçı olan takımları ortalamaya çekme gücü")
    ap.add_argument("--sut", type=float, help="İsabetli şut ağırlığı (0-1)")
    ap.add_argument("--buyuk-sans", type=float, help="Büyük şans ağırlığı (0-1)")
    ap.add_argument("--belirsizlik", type=float, help="Beklenen gole eklenen belirsizlik")
    ap.add_argument("--sim", type=int, default=10000, help="Simülasyon sayısı")
    args = ap.parse_args()

    if args.fikstur:
        run_fixtures(args)
        return
    if not args.lig:
        ap.error("--lig gerekli (veya --fikstur kullan)")
    lm = LeagueModel(resolve_league(args.lig), args)

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
    print_match(lm, home, away, lm.sd, args.sim)


if __name__ == "__main__":
    main()
