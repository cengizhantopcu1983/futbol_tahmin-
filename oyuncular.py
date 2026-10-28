"""Oyuncu verisi (Sportmonks): maç kadroları, oynanan dakikalar, sakat/cezalı listeleri ve güncel kadrolar.

Kaydedilen dosyalar (depoya eklenir, deneme süresi bitince de kalır):
  data_players/<lig>_<sezon>.csv : her maçta her oyuncu için bir satır
      sm_id, Date, Team, Side (home/away), player_id, player, pos (G/D/M/F),
      status (XI = ilk 11, SUB = yedek, OUT = sakat/cezalı), minutes
  kadrolar.json : {"T1": {"Galatasaray": [{"id": "123", "ad": "...", "mevki": "F", "no": 9}, ...]}}
      Takım adları Sportmonks'taki gibidir; eşleştirme eksik.py'de yapılır.

Geçmiş sezonlar bir kez indirilir; bu sezon her çalıştırmada güncellenir, kadrolar haftada bir.
Token SPORTMONKS_TOKEN ortam değişkeninden okunur, hiçbir yere yazdırılmaz.
Örnek:  python oyuncular.py            (eksikleri indir + bu sezonu güncelle)
        python oyuncular.py --sezon 4  (son 4 sezon)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import pandas as pd

from sportmonks import ApiError, LEAGUE_IDS, _list, _token, fixtures_between, get, league_seasons, season_code

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PLAYERS_DIR = os.environ.get("PLAYERS_DATA_DIR") or os.path.join(HERE, "data_players")
SQUADS_FILE = os.environ.get("KADRO_FILE") or os.path.join(HERE, "kadrolar.json")
COLS = ["sm_id", "Date", "Team", "Side", "player_id", "player", "pos", "status", "minutes"]
POS = {24: "G", 25: "D", 26: "M", 27: "F"}
STARTER, BENCH = 11, 12
MINUTES = 119
FINISHED = {5, 7, 8}
INCLUDES = "participants;lineups.details;sidelined.sideline"


def _pos(x: dict) -> str:
    for k in ("position_id", "detailed_position_id"):
        p = POS.get(x.get(k))
        if p:
            return p
    pl = x.get("player") if isinstance(x.get("player"), dict) else {}
    return POS.get(pl.get("position_id"), "")


def parse_players(f: dict) -> list[dict]:
    """Bir maçtan oyuncu satırları (bitmemiş maçta boş)."""
    if f.get("state_id") not in FINISHED:
        return []
    side_of, name_of = {}, {}
    for p in _list(f, "participants"):
        loc = (p.get("meta") or {}).get("location")
        if loc in ("home", "away"):
            side_of[p.get("id")] = loc
            name_of[p.get("id")] = p.get("name")
    if len(side_of) != 2:
        return []
    base = {"sm_id": f.get("id"), "Date": str(f.get("starting_at") or "")[:10]}
    rows = []
    for x in _list(f, "lineups"):
        tid = x.get("team_id") or x.get("participant_id")
        if tid not in side_of or not x.get("player_id"):
            continue
        st = "XI" if x.get("type_id") == STARTER else "SUB" if x.get("type_id") == BENCH else None
        if not st:
            continue
        mins = None
        for d in _list(x, "details"):
            if d.get("type_id") == MINUTES:
                v = d.get("data")
                mins = v.get("value") if isinstance(v, dict) else d.get("value")
        if mins is None:
            mins = 90 if st == "XI" else None
        rows.append({**base, "Team": name_of[tid], "Side": side_of[tid], "player_id": str(x["player_id"]),
                     "player": x.get("player_name") or "", "pos": _pos(x), "status": st, "minutes": mins})
    for s in _list(f, "sidelined"):
        tid = s.get("participant_id")
        sl = s.get("sideline") if isinstance(s.get("sideline"), dict) else {}
        pid = sl.get("player_id") or s.get("player_id")
        if tid not in side_of or not pid:
            continue
        rows.append({**base, "Team": name_of[tid], "Side": side_of[tid], "player_id": str(pid),
                     "player": "", "pos": "", "status": "OUT", "minutes": 0})
    return rows


def download_season(lid: int, season: dict) -> pd.DataFrame:
    start = dt.date.fromisoformat(str(season["starting_at"])[:10])
    end = min(dt.date.fromisoformat(str(season.get("ending_at") or dt.date.today())[:10]), dt.date.today())
    rows, a = [], start
    while a <= end:
        b = min(a + dt.timedelta(days=20), end)
        for f in fixtures_between(lid, a, b, includes=INCLUDES):
            rows.extend(parse_players(f))
        a = b + dt.timedelta(days=1)
        time.sleep(0.3)
    return pd.DataFrame(rows, columns=COLS).drop_duplicates(subset=["sm_id", "Team", "player_id", "status"])


def update(max_seasons: int = 3, force: bool = False) -> int:
    """Eksik sezonları indirir, bu sezonu yeniler. Dönüş: hata sayısı."""
    os.makedirs(PLAYERS_DIR, exist_ok=True)
    errors = 0
    for code, lid in LEAGUE_IDS.items():
        try:
            seasons = [s for s in league_seasons(lid) if s.get("starting_at")
                       and str(s["starting_at"])[:10] <= str(dt.date.today())]
        except ApiError as e:
            print(f"  {code}: sezon listesi alınamadı ({e})")
            errors += 1
            continue
        seasons = sorted(seasons, key=lambda s: s["starting_at"], reverse=True)[:max_seasons + 1]
        for i, s in enumerate(seasons):
            sc = season_code(s.get("name"))
            if not sc:
                continue
            path = os.path.join(PLAYERS_DIR, f"{code}_{sc}.csv")
            if os.path.exists(path) and i > 0 and not force:
                continue  # geçmiş sezon zaten var
            try:
                df = download_season(lid, s)
            except ApiError as e:
                print(f"  {code} {sc}: indirilemedi ({e})")
                errors += 1
                continue
            if len(df):
                df.to_csv(path, index=False)
            n_m = df["sm_id"].nunique() if len(df) else 0
            n_xi = (df["status"] == "XI").sum() if len(df) else 0
            n_out = (df["status"] == "OUT").sum() if len(df) else 0
            print(f"  {code} {sc}: {n_m} maç, {n_xi} ilk 11 kaydı, {n_out} sakat/cezalı kaydı")
    return errors


def update_squads(max_age_days: int = 7, force: bool = False) -> int:
    try:
        with open(SQUADS_FILE, encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, ValueError):
        old = {}
    if not force and old.get("_guncelleme"):
        age = (dt.date.today() - dt.date.fromisoformat(old["_guncelleme"])).days
        if age < max_age_days:
            print(f"  kadrolar {age} gün önce güncellenmiş, atlandı")
            return 0
    out, errors = {"_guncelleme": str(dt.date.today())}, 0
    for code, lid in LEAGUE_IDS.items():
        try:
            seasons = league_seasons(lid)
            cur = next((s for s in seasons if s.get("is_current")), None) or max(
                (s for s in seasons if s.get("starting_at")), key=lambda s: s["starting_at"])
            teams = get(f"/teams/seasons/{cur['id']}").get("data") or []
        except (ApiError, ValueError) as e:
            print(f"  {code}: takımlar alınamadı ({e})")
            out[code] = old.get(code, {})
            errors += 1
            continue
        league = {}
        for t in teams:
            try:
                sq = get(f"/squads/seasons/{cur['id']}/teams/{t['id']}", {"include": "player"}).get("data") or []
            except ApiError as e:
                print(f"  {code} {t.get('name')}: kadro alınamadı ({e})")
                errors += 1
                continue
            players = []
            for p in sq:
                pl = p.get("player") if isinstance(p.get("player"), dict) else {}
                pid = p.get("player_id") or pl.get("id")
                if not pid:
                    continue
                players.append({"id": str(pid), "ad": pl.get("display_name") or pl.get("common_name") or pl.get("name") or "",
                                "mevki": _pos(p) or POS.get(pl.get("position_id"), ""),
                                "no": p.get("jersey_number")})
            league[t.get("name")] = players
            time.sleep(0.2)
        out[code] = league or old.get(code, {})
        print(f"  {code}: {len(league)} takımın kadrosu, toplam {sum(len(v) for v in league.values())} oyuncu")
    with open(SQUADS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return errors


def main():
    ap = argparse.ArgumentParser(description="Sportmonks oyuncu verisini indir")
    ap.add_argument("--sezon", type=int, default=3, help="Bu sezona ek olarak kaç geçmiş sezon")
    ap.add_argument("--hepsi", action="store_true", help="Var olan sezonları da yeniden indir")
    args = ap.parse_args()
    if not _token():
        print("SPORTMONKS_TOKEN tanımlı değil; oyuncu verisi güncellenmedi (eldeki dosyalar kullanılır).")
        return
    print("Maç kadroları ve sakat listeleri:")
    e1 = update(args.sezon, args.hepsi)
    print("Güncel kadrolar:")
    e2 = update_squads(force=args.hepsi)
    if e1 + e2:
        print(f"{e1 + e2} hata oldu (yukarıda). Eldeki veriyle devam edilir.")


if __name__ == "__main__":
    main()
