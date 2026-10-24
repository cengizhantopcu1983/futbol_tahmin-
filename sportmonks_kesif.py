"""Sportmonks keşif testi: hesabımız hangi veriyi, hangi sezonlar için veriyor?

Her lig için son sezonlardan örnek maçlar çekilir ve şunların varlığı sayılır:
  xG, isabetli şut, büyük şans, topla oynama, ilk 11, sakatlık/ceza, hakem.
Ayrıca abonelikteki planlar/ek paketler ve kalan istek hakkı gösterilir.

Token, SPORTMONKS_TOKEN ortam değişkeninden okunur ve hiçbir yere yazdırılmaz.
Örnek:  python sportmonks_kesif.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = "https://api.sportmonks.com/v3/football"
LEAGUES = {600: "Süper Lig", 8: "Premier League", 564: "La Liga", 384: "Serie A"}
INCLUDES = "participants;scores;statistics;xGFixture;lineups;sidelined;referees"

XG = 5304            # beklenen gol
SOT = 86             # isabetli şut
BIG_CHANCES = 580    # büyük şans
POSSESSION = 45      # topla oynama
N_SEASONS = 5
TOKEN = os.environ.get("SPORTMONKS_TOKEN", "").strip()


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def get(path: str, params: dict | None = None, tries: int = 3) -> dict:
    """Sportmonks'a istek atar. Token başlıkta gider, URL'de değil (loglara düşmesin)."""
    url = BASE + path + ("?" + urllib.parse.urlencode(params, safe=";:,") if params else "")
    req = urllib.request.Request(url, headers={"Authorization": TOKEN, "Accept": "application/json",
                                               "User-Agent": "futbol-tahmin/1.0"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read())
                msg = body.get("message") or body.get("error") or str(body)[:200]
            except Exception:
                msg = e.reason
            if e.code == 429 and attempt < tries - 1:
                time.sleep(20 * (attempt + 1))
                continue
            raise ApiError(e.code, str(msg)[:200]) from None
        except urllib.error.URLError as e:
            if attempt < tries - 1:
                time.sleep(5)
                continue
            raise ApiError(0, f"bağlantı hatası: {e.reason}") from None
    raise ApiError(0, "bilinmeyen hata")


# ---------------- yardımcılar ----------------
def _items(obj, *keys):
    """Bir include'un listesini, olası anahtar adlarından hangisi varsa onu döndürür."""
    for k in keys:
        v = obj.get(k)
        if isinstance(v, list):
            return v
        if isinstance(v, dict) and isinstance(v.get("data"), list):  # eski biçim
            return v["data"]
    return []


def _value(item):
    d = item.get("data")
    if isinstance(d, dict):
        return d.get("value")
    return item.get("value")


def _date(s):
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def describe_subscription(resp: dict) -> list[str]:
    lines = []
    sub = resp.get("subscription")
    if isinstance(sub, list) and sub:
        sub = sub[0]
    if isinstance(sub, dict):
        for key, label in (("plans", "Plan"), ("add_ons", "Ek paket"), ("widgets", "Widget")):
            names = [p.get("plan") or p.get("name") or p.get("add_on") or str(p)
                     for p in (sub.get(key) or []) if isinstance(p, dict)]
            if names:
                lines.append(f"  {label}: {', '.join(names)}")
        meta = sub.get("meta")
        if isinstance(meta, dict):
            if meta.get("trial_ends_at"):
                lines.append(f"  Deneme bitişi: {meta['trial_ends_at']}")
            if meta.get("ends_at"):
                lines.append(f"  Abonelik bitişi: {meta['ends_at']}")
    rl = resp.get("rate_limit")
    if isinstance(rl, dict) and "remaining" in rl:
        lines.append(f"  Kalan istek hakkı (bu saat, bu veri türü): {rl.get('remaining')}")
    return lines


def pick_window(season: dict, today: dt.date) -> tuple[dt.date, dt.date] | None:
    start, end = _date(season.get("starting_at")), _date(season.get("ending_at"))
    if not start:
        return None
    end = min(end or today, today)
    if season.get("is_current") or end >= today - dt.timedelta(days=3):
        a = max(start, today - dt.timedelta(days=40))
        b = today
    else:  # sezonun ortasından bir ay
        a = start + dt.timedelta(days=60)
        b = min(a + dt.timedelta(days=30), end)
    return (a, b) if a < b else None


def analyse_fixtures(fixtures: list[dict]) -> dict:
    played = [f for f in fixtures if _items(f, "scores")]
    out = {"mac": len(played), "xg": 0, "isabetli_sut": 0, "buyuk_sans": 0, "topla_oynama": 0,
           "ilk_11": 0, "sakatlik": 0, "hakem": 0, "ornek_xg": None}
    for f in played:
        xg = [x for x in _items(f, "xgfixture", "xGFixture", "expected") if x.get("type_id") == XG]
        if xg:
            out["xg"] += 1
            if out["ornek_xg"] is None:
                names = {p.get("id"): p.get("name") for p in _items(f, "participants")}
                vals = {x.get("location"): (_value(x), names.get(x.get("participant_id"))) for x in xg}
                h, a = vals.get("home"), vals.get("away")
                if h and a:
                    out["ornek_xg"] = f"{f.get('name') or ''} -> xG {h[0]} - {a[0]}"
        types = {s.get("type_id") for s in _items(f, "statistics")}
        out["isabetli_sut"] += SOT in types
        out["buyuk_sans"] += BIG_CHANCES in types
        out["topla_oynama"] += POSSESSION in types
        out["ilk_11"] += len(_items(f, "lineups")) >= 22
        out["sakatlik"] += len(_items(f, "sidelined")) > 0
        out["hakem"] += len(_items(f, "referees")) > 0
    return out


# tablo sütunu -> hangi include'dan geliyor
INCLUDE_OF = {"xg": "xGFixture", "isabetli_sut": "statistics", "buyuk_sans": "statistics",
              "topla_oynama": "statistics", "ilk_11": "lineups", "sakatlik": "sidelined", "hakem": "referees"}


def fetch_fixtures(lid: int, win, denied: set[str]):
    """Maçları çeker. Bir include için yetki yoksa onu çıkarıp yeniden dener."""
    while True:
        incs = [i for i in INCLUDES.split(";") if i not in denied]
        try:
            return get(f"/fixtures/between/{win[0]}/{win[1]}",
                       {"filters": f"fixtureLeagues:{lid}", "include": ";".join(incs), "per_page": 25}), None
        except ApiError as e:
            msg = str(e).lower()
            bad = next((i for i in incs if e.status == 403 and f"'{i.lower()}'" in msg), None)
            if bad is None:
                return None, e
            denied.add(bad)  # bu veri aboneliğe dahil değil; onsuz devam et


# ---------------- ana akış ----------------
def main() -> int:
    if not TOKEN:
        print("HATA: SPORTMONKS_TOKEN tanımlı değil. GitHub'da Settings > Secrets and variables > "
              "Actions altına SPORTMONKS_TOKEN adıyla ekle.")
        return 1
    today = dt.date.today()
    print(f"Sportmonks keşif testi ({today})\n")
    sub_printed = False
    rows = []
    denied: set[str] = set()  # aboneliğin izin vermediği include'lar
    for lid, lname in LEAGUES.items():
        print(f"=== {lname} (lig no {lid}) ===")
        try:
            resp = get(f"/leagues/{lid}", {"include": "seasons"})
        except ApiError as e:
            print(f"  ✗ Lige erişilemedi: {e}")
            if e.status in (401, 403):
                print("    (Token yanlış olabilir ya da bu lig aboneliğe seçilmemiş.)")
            rows.append((lname, None))
            print()
            continue
        if not sub_printed:
            info = describe_subscription(resp)
            if info:
                print("Abonelik bilgisi:")
                print("\n".join(info))
            sub_printed = True
        seasons = _items(resp.get("data") or {}, "seasons")
        seasons = sorted((s for s in seasons if _date(s.get("starting_at")) and _date(s["starting_at"]) <= today),
                         key=lambda s: s["starting_at"], reverse=True)[:N_SEASONS]
        if not seasons:
            print("  ✗ Sezon bilgisi gelmedi.\n")
            rows.append((lname, None))
            continue
        print(f"  {'Sezon':<11}{'Maç':>5}{'xG':>6}{'İs.şut':>8}{'B.şans':>8}{'Topla oyn.':>11}"
              f"{'İlk 11':>8}{'Sakatlık':>10}{'Hakem':>7}")
        league_rows = []
        for s in seasons:
            win = pick_window(s, today)
            name = str(s.get("name") or s.get("id"))
            if not win:
                print(f"  {name:<11}  (uygun tarih aralığı yok)")
                continue
            fx, err = fetch_fixtures(lid, win, denied)
            if fx is None:
                print(f"  {name:<11}  ✗ {err}")
                league_rows.append((name, None))
                continue
            r = analyse_fixtures(fx.get("data") or [])
            league_rows.append((name, r))
            n = r["mac"]

            def cell(k):
                if INCLUDE_OF.get(k) in denied:
                    return "yetki"
                return f"{r[k]}/{n}" if n else "-"
            print(f"  {name:<11}{n:>5}{cell('xg'):>6}{cell('isabetli_sut'):>8}{cell('buyuk_sans'):>8}"
                  f"{cell('topla_oynama'):>11}{cell('ilk_11'):>8}{cell('sakatlik'):>10}{cell('hakem'):>7}")
            time.sleep(0.5)
        examples = [r["ornek_xg"] for _, r in league_rows if r and r["ornek_xg"]]
        if examples:
            print(f"  Örnek: {examples[0]}")
        rows.append((lname, league_rows))
        print()

    # ---- özet ----
    print("=== ÖZET ===")
    if denied:
        adlar = {"xGFixture": "xG", "statistics": "maç istatistikleri", "lineups": "ilk 11",
                 "sidelined": "sakatlık/ceza", "referees": "hakem", "scores": "skorlar",
                 "participants": "takımlar"}
        print("  Aboneliğe dahil OLMAYAN veriler ('yetki'): "
              + ", ".join(adlar.get(d, d) for d in sorted(denied)))
    for lname, league_rows in rows:
        if not league_rows:
            print(f"  {lname}: erişim yok")
            continue
        with_xg = [n for n, r in league_rows if r and r["mac"] and r["xg"] >= 0.8 * r["mac"]]
        print(f"  {lname}: xG olan sezonlar: {', '.join(with_xg) if with_xg else 'YOK'}")
    print("\n(Sayılar her sezondan örneklenen yaklaşık bir aylık maçlara aittir. "
          "Ör. '9/10' = 10 maçın 9'unda bu veri var.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
