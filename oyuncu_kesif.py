"""Oyuncu verisi keşfi: Sportmonks deneme paketi kadro, ilk 11 ve oyuncu istatistiği veriyor mu?

Her lig için:
  1) Bu sezonun takımları ve bir takımın kadrosu (oyuncu adı, mevki, forma no)
  2) Bu sezon ve 2 sezon önceki örnek maçlarda ilk 11 / yedekler / oyuncu istatistikleri
     (oynadığı dakika, gol, asist, puan) ve sakat-cezalı oyuncu listesi
Token SPORTMONKS_TOKEN ortam değişkeninden okunur, hiçbir yere yazdırılmaz.
Örnek:  python oyuncu_kesif.py
"""
from __future__ import annotations

import datetime as dt
import sys
from collections import Counter

from sportmonks import ApiError, LEAGUE_IDS, _list, fixtures_between, get, league_seasons

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NAMES = {"T1": "Süper Lig", "E0": "Premier League", "SP1": "La Liga", "I1": "Serie A"}
STARTER, BENCH = 11, 12
DETAIL_TYPES = {119: "dakika", 52: "gol", 79: "asist", 118: "puan", 86: "isabetli şut", 580: "büyük şans"}
LINEUP_INC = "participants;lineups.details;sidelined.sideline"


def pct(a, b):
    return f"{a}/{b}" + (f" (%{100 * a / b:.0f})" if b else "")


def squad_check(season: dict) -> None:
    try:
        teams = get(f"/teams/seasons/{season['id']}").get("data") or []
    except ApiError as e:
        print(f"    takım listesi: ALINAMADI ({e})")
        return
    print(f"    takım listesi: {len(teams)} takım")
    if not teams:
        return
    t = teams[0]
    try:
        sq = get(f"/squads/seasons/{season['id']}/teams/{t['id']}", {"include": "player"}).get("data") or []
    except ApiError as e:
        print(f"    kadro ({t.get('name')}): ALINAMADI ({e})")
        return
    named = sum(1 for p in sq if (p.get("player") or {}).get("display_name") or (p.get("player") or {}).get("name"))
    pos = sum(1 for p in sq if p.get("position_id") or (p.get("player") or {}).get("position_id"))
    print(f"    kadro ({t.get('name')}): {len(sq)} oyuncu, adı olan {pct(named, len(sq))}, mevkisi olan {pct(pos, len(sq))}")
    ex = [(p.get("player") or {}).get("display_name") for p in sq[:4]]
    print(f"      örnek: {', '.join(x for x in ex if x)}")


def lineup_check(lid: int, a: dt.date, b: dt.date, label: str) -> None:
    try:
        fx = fixtures_between(lid, a, b, includes=LINEUP_INC)
    except ApiError as e:
        print(f"    {label}: ALINAMADI ({e})")
        return
    fx = [f for f in fx if f.get("state_id") in (5, 7, 8)][:12]
    if not fx:
        print(f"    {label}: bitmiş maç bulunamadı")
        return
    with_lu = starters_ok = side_n = side_player = 0
    det = Counter()
    n_players = 0
    for f in fx:
        lu = _list(f, "lineups")
        if lu:
            with_lu += 1
        st = Counter(x.get("team_id") for x in lu if x.get("type_id") == STARTER)
        if len(st) == 2 and all(v == 11 for v in st.values()):
            starters_ok += 1
        for x in lu:
            n_players += 1
            for d in _list(x, "details"):
                if d.get("type_id") in DETAIL_TYPES:
                    det[DETAIL_TYPES[d["type_id"]]] += 1
        sl = _list(f, "sidelined")
        side_n += len(sl)
        side_player += sum(1 for s in sl if s.get("player_id") or (s.get("sideline") or {}).get("player_id"))
    n = len(fx)
    print(f"    {label}: {n} maç | kadrosu olan {pct(with_lu, n)} | tam 11+11 ilk 11 {pct(starters_ok, n)}")
    print(f"      oyuncu istatistikleri ({n_players} oyuncu satırı): "
          + (", ".join(f"{k} {pct(v, n_players)}" for k, v in det.items()) or "YOK"))
    print(f"      sakat/cezalı kaydı: {side_n} (oyuncu no'su olan {side_player})")


def main():
    try:
        for code, lid in LEAGUE_IDS.items():
            print(f"\n=== {NAMES[code]} ===")
            seasons = sorted([s for s in league_seasons(lid) if s.get("starting_at")],
                             key=lambda s: s["starting_at"], reverse=True)
            cur = next((s for s in seasons if s.get("is_current")), seasons[0] if seasons else None)
            if not cur:
                print("    sezon bulunamadı")
                continue
            squad_check(cur)
            today = dt.date.today()
            lineup_check(lid, today - dt.timedelta(days=21), today, f"bu sezon ({cur.get('name')})")
            old = seasons[2] if len(seasons) > 2 else None
            if old:
                s0 = dt.date.fromisoformat(str(old["starting_at"])[:10]) + dt.timedelta(days=60)
                lineup_check(lid, s0, s0 + dt.timedelta(days=14), f"eski sezon ({old.get('name')})")
    except ApiError as e:
        print(f"\nHATA: {e}")
        if e.status in (401, 403):
            print("Token geçersiz ya da deneme süresi bitmiş olabilir.")


if __name__ == "__main__":
    main()
