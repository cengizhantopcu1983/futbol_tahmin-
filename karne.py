"""Performans karnesi: tahminleri arşivler, maçlar oynanınca notlar.

Her çalıştırmada:
  1. tahminler.json'daki henüz BAŞLAMAMIŞ maçların tahminleri arşive yazılır
     (arsiv/tahmin_arsivi.json). Maç başladıktan sonra tahmin bir daha değişmez.
  2. Oynanmış maçların skorları football-data.co.uk'den bulunur ve arşive işlenir.
  3. karne.json üretilir: genel isabet, lig ve hafta bazında sonuçlar, son maçlar,
     ve aynı maçlarda bahis piyasasının isabeti (adil kıyas için).

Örnek:  python karne.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

from data import LEAGUES, TZ_TR, load_league, season_codes

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
SECIM = ["1", "X", "2"]


def load_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def kickoff(rec: dict) -> dt.datetime:
    """Maçın başlama zamanı (İstanbul). Saat bilinmiyorsa günün başı."""
    d = dt.date.fromisoformat(rec["tarih"])
    try:
        h, m = (int(x) for x in str(rec.get("saat") or "").split(":")[:2])
    except ValueError:
        h, m = 0, 0
    return dt.datetime(d.year, d.month, d.day, h, m, tzinfo=TZ_TR)


def match_key(m: dict) -> str:
    return f"{m['lig_kodu']}|{m['tarih']}|{m['ev']}|{m['deplasman']}"


# ---------------- 1) arşivleme ----------------
def archive_predictions(tahminler: dict, arsiv: dict, now: dt.datetime) -> int:
    n = 0
    for m in tahminler.get("maclar", []):
        if not m.get("tarih") or kickoff(m) <= now:
            continue  # başlamış maçın tahmini dondurulmuştur
        key = match_key(m)
        if "sonuc" in arsiv["maclar"].get(key, {}):
            continue
        arsiv["maclar"][key] = {
            "lig": m["lig"], "lig_kodu": m["lig_kodu"], "tarih": m["tarih"], "saat": m.get("saat"),
            "ev": m["ev"], "deplasman": m["deplasman"],
            "p_ev": m["p_ev"], "p_beraberlik": m["p_beraberlik"], "p_deplasman": m["p_deplasman"],
            "xg_ev": m.get("xg_ev"), "xg_deplasman": m.get("xg_deplasman"),
            # sonuca göre skor: tahmin edilen sonuç içindeki en olası skor
            "olasi_skor": (m.get("tutarli_skor") or {}).get("skor")
                          or (m.get("olasi_skorlar") or [{}])[0].get("skor"),
            # tek tek en olası skor (sonuçtan bağımsız; çoğu zaman 1-1)
            "en_olasi_skor": (m.get("olasi_skorlar") or [{}])[0].get("skor"),
            "piyasa": m.get("piyasa"),
            "kayit": now.isoformat(timespec="minutes"),
        }
        n += 1
    return n


# ---------------- 2) notlama ----------------
def grade(arsiv: dict, now: dt.datetime, loader=None) -> int:
    loader = loader or (lambda code: load_league(code, seasons=season_codes(2)))
    pending = [r for r in arsiv["maclar"].values()
               if "sonuc" not in r and kickoff(r) + dt.timedelta(hours=2, minutes=30) < now]
    n = 0
    for code in sorted({r["lig_kodu"] for r in pending}):
        try:
            df = loader(code)
        except Exception as e:
            print(f"  ! {LEAGUES.get(code, code)} sonuçları alınamadı: {e}")
            continue
        for r in (r for r in pending if r["lig_kodu"] == code):
            c = df[(df["HomeTeam"] == r["ev"]) & (df["AwayTeam"] == r["deplasman"])]
            target = pd.Timestamp(r["tarih"])
            c = c[(c["Date"] - target).abs() <= pd.Timedelta(days=1)]  # saat farkı gün değiştirebilir
            if len(c):
                row = c.iloc[-1]
                r["sonuc"] = {"ev": int(row["FTHG"]), "deplasman": int(row["FTAG"])}
                n += 1
    return n


# ---------------- 3) karne ----------------
def evaluate(r: dict) -> dict:
    P = np.array([r["p_ev"], r["p_beraberlik"], r["p_deplasman"]], float)
    h, a = r["sonuc"]["ev"], r["sonuc"]["deplasman"]
    o = 0 if h > a else 1 if h == a else 2
    pick = int(P.argmax())
    cp, cy = np.cumsum(P)[:2], np.cumsum(np.eye(3)[o])[:2]
    out = {"o": o, "pick": pick, "tuttu": pick == o, "rps": float(((cp - cy) ** 2).sum() / 2),
           "skor_tuttu": r.get("olasi_skor") == f"{h}-{a}",
           "en_olasi_skor_tuttu": r.get("en_olasi_skor") == f"{h}-{a}",
           "p_tahmin": float(P[pick])}
    pz = r.get("piyasa")
    if pz:
        bp = int(np.argmax([pz["ev"], pz["beraberlik"], pz["deplasman"]]))
        out.update(piyasa_pick=bp, piyasa_tuttu=bp == o)
    return out


def stats(items: list[tuple[dict, dict]]) -> dict:
    n = len(items)
    dogru = sum(e["tuttu"] for _, e in items)
    pz = [(r, e) for r, e in items if "piyasa_tuttu" in e]
    ratio = lambda a, b: round(a / b, 4) if b else None
    return {
        "mac": n, "dogru": dogru, "isabet": ratio(dogru, n),
        "rps": round(float(np.mean([e["rps"] for _, e in items])), 4) if n else None,
        "skor_tuttu": sum(e["skor_tuttu"] for _, e in items),
        "en_olasi_skor_tuttu": sum(e["en_olasi_skor_tuttu"] for _, e in items),
        # piyasa kıyası: sadece oranı olan maçlarda, ikisi aynı maçlar üzerinden
        "piyasa_mac": len(pz),
        "piyasa_isabet": ratio(sum(e["piyasa_tuttu"] for _, e in pz), len(pz)),
        "model_isabet_ayni_maclar": ratio(sum(e["tuttu"] for _, e in pz), len(pz)),
    }


def build_karne(arsiv: dict, now: dt.datetime) -> dict:
    graded = [(r, evaluate(r)) for r in arsiv["maclar"].values() if "sonuc" in r]
    graded.sort(key=lambda x: (x[0]["tarih"], x[0].get("saat") or ""), reverse=True)

    ligler = [{"lig": LEAGUES[c], "lig_kodu": c, **stats([g for g in graded if g[0]["lig_kodu"] == c])}
              for c in LEAGUES if any(g[0]["lig_kodu"] == c for g in graded)]

    weeks: dict[str, list] = {}
    for g in graded:
        d = dt.date.fromisoformat(g[0]["tarih"])
        weeks.setdefault((d - dt.timedelta(days=d.weekday())).isoformat(), []).append(g)
    haftalar = [{"hafta_baslangic": w, "hafta_bitis": (dt.date.fromisoformat(w) + dt.timedelta(days=6)).isoformat(),
                 **stats(items)} for w, items in sorted(weeks.items(), reverse=True)[:12]]

    son = []
    for r, e in graded[:60]:
        son.append({
            "lig": r["lig"], "lig_kodu": r["lig_kodu"], "tarih": r["tarih"], "saat": r.get("saat"),
            "ev": r["ev"], "deplasman": r["deplasman"],
            "skor": f"{r['sonuc']['ev']}-{r['sonuc']['deplasman']}",
            "tahmin": SECIM[e["pick"]], "p_tahmin": round(e["p_tahmin"], 4),
            "p_ev": r["p_ev"], "p_beraberlik": r["p_beraberlik"], "p_deplasman": r["p_deplasman"],
            "tuttu": e["tuttu"], "olasi_skor": r.get("olasi_skor"), "skor_tuttu": e["skor_tuttu"],
            "en_olasi_skor": r.get("en_olasi_skor"), "en_olasi_skor_tuttu": e["en_olasi_skor_tuttu"],
            "piyasa_tahmin": SECIM[e["piyasa_pick"]] if "piyasa_pick" in e else None,
            "piyasa_tuttu": e.get("piyasa_tuttu"),
        })
    bekleyen = sum(1 for r in arsiv["maclar"].values() if "sonuc" not in r)
    return {"olusturulma": now.isoformat(timespec="minutes"), "ozet": stats(graded),
            "bekleyen": bekleyen, "ligler": ligler, "haftalar": haftalar, "son_maclar": son}


def main():
    ap = argparse.ArgumentParser(description="Tahmin arşivi ve performans karnesi")
    ap.add_argument("--tahminler", default=os.path.join(BASE, "tahminler.json"))
    ap.add_argument("--arsiv", default=os.path.join(BASE, "arsiv", "tahmin_arsivi.json"))
    ap.add_argument("--karne", default=os.path.join(BASE, "karne.json"))
    args = ap.parse_args()

    now = dt.datetime.now(TZ_TR)
    arsiv = load_json(args.arsiv, {"maclar": {}})
    yeni = archive_predictions(load_json(args.tahminler, {"maclar": []}), arsiv, now)
    notlanan = grade(arsiv, now)
    save_json(args.arsiv, arsiv)
    karne = build_karne(arsiv, now)
    save_json(args.karne, karne)
    oz = karne["ozet"]
    print(f"Arşive yazılan/güncellenen tahmin: {yeni}")
    print(f"Yeni notlanan maç: {notlanan}  |  sonucu beklenen: {karne['bekleyen']}")
    if oz["mac"]:
        print(f"Toplam: {oz['mac']} maç, {oz['dogru']} doğru (%{100 * oz['isabet']:.1f}), "
              f"skor tutturma: sonuca göre {oz['skor_tuttu']}, en olası skor {oz['en_olasi_skor_tuttu']}")
        if oz["piyasa_mac"]:
            print(f"Aynı {oz['piyasa_mac']} maçta: model %{100 * oz['model_isabet_ayni_maclar']:.1f}, "
                  f"piyasa %{100 * oz['piyasa_isabet']:.1f}")


if __name__ == "__main__":
    main()
