"""Sportmonks verisi: indirme, saklama ve lig verisiyle eşleştirme.

İndirilen her maç için: skor, şut, isabetli şut, BÜYÜK ŞANS, topla oynama, korner,
kartlar, hakem ve sakatlık/cezalı oyuncu sayısı. Veriler data_sm/ klasöründe CSV
olarak saklanır ve depoya kaydedilir; abonelik bitse bile geçmiş veri kalır.

  python sportmonks.py            # eksik sezonları indir, güncel sezonu yenile
  python sportmonks.py --hepsi    # her şeyi baştan indir

Token SPORTMONKS_TOKEN ortam değişkeninden okunur, hiçbir yere yazdırılmaz.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_URL = "https://api.sportmonks.com/v3/football"
SM_DIR = os.environ.get("SPORTMONKS_DATA_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data_sm")
LEAGUE_IDS = {"T1": 600, "E0": 8, "SP1": 564, "I1": 384}
INCLUDES = "participants;scores;statistics;referees;sidelined"
FINISHED_STATES = {5, 7, 8}  # normal süre, uzatma, penaltılar sonrası bitti
MAIN_REFEREE_TYPE = 6

# istatistik tür no -> (ev sütunu, deplasman sütunu)
STAT_TYPES = {
    42: ("HS", "AS"),      # şut
    86: ("HST", "AST"),    # isabetli şut
    580: ("HBC", "ABC"),   # büyük şans
    45: ("HPOS", "APOS"),  # topla oynama %
    34: ("HC", "AC"),      # korner
    84: ("HY", "AY"),      # sarı kart
    83: ("HR", "AR"),      # kırmızı kart
}
CSV_COLS = ["sm_id", "Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG",
            *[c for pair in STAT_TYPES.values() for c in pair], "RefereeId", "HSide", "ASide"]
# lig verisine eklenen sütunlar (football-data.co.uk'de olmayanlar)
MERGE_COLS = ["HBC", "ABC", "HPOS", "APOS", "HSide", "ASide", "SM_Referee"]


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def _token() -> str:
    return os.environ.get("SPORTMONKS_TOKEN", "").strip()


def get(path: str, params: dict | None = None, tries: int = 4) -> dict:
    """Token başlıkta gider (URL'de değil), böylece loglara düşmez."""
    # yoldaki boşluk vb. karakterler kodlanır (ör. "/leagues/search/Europa League")
    url = (BASE_URL + urllib.parse.quote(path, safe="/:,;")
           + ("?" + urllib.parse.urlencode(params, safe=";:,") if params else ""))
    req = urllib.request.Request(url, headers={"Authorization": _token(), "Accept": "application/json",
                                               "User-Agent": "futbol-tahmin/1.0"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("message", "")
            except Exception:
                msg = str(e.reason)
            if e.code in (429, 500, 502, 503) and attempt < tries - 1:
                time.sleep(15 * (attempt + 1))
                continue
            raise ApiError(e.code, str(msg)[:200]) from None
        except urllib.error.URLError as e:
            if attempt < tries - 1:
                time.sleep(5)
                continue
            raise ApiError(0, f"bağlantı hatası: {e.reason}") from None
    raise ApiError(0, "bilinmeyen hata")


def _list(obj: dict, key: str) -> list:
    v = obj.get(key)
    if isinstance(v, dict) and isinstance(v.get("data"), list):
        return v["data"]
    return v if isinstance(v, list) else []


def season_code(name: str) -> str | None:
    """'2024/2025' -> '2425'."""
    try:
        a, b = str(name).split("/")[:2]
        return f"{int(a) % 100:02d}{int(b) % 100:02d}"
    except ValueError:
        return None


# ---------------- maç ayrıştırma ----------------
def parse_fixture(f: dict) -> dict | None:
    """Bir Sportmonks maçını düz satıra çevirir; bitmemiş maçlarda None."""
    scores = _list(f, "scores")
    cur = {s.get("score", {}).get("participant"): s for s in scores if s.get("description") == "CURRENT"}
    if f.get("state_id") not in FINISHED_STATES and not f.get("result_info"):
        return None
    if "home" not in cur or "away" not in cur:
        return None
    loc_of = {s.get("participant_id"): s["score"]["participant"] for s in cur.values()}
    for p in _list(f, "participants"):  # yedek: participants.meta.location
        loc = (p.get("meta") or {}).get("location")
        if loc in ("home", "away"):
            loc_of.setdefault(p.get("id"), loc)
    names = {loc_of.get(p.get("id")): p.get("name") for p in _list(f, "participants")}
    if not names.get("home") or not names.get("away"):
        return None

    row = {c: np.nan for c in CSV_COLS}
    row.update(sm_id=f.get("id"), Date=str(f.get("starting_at") or "")[:10],
               HomeTeam=names["home"], AwayTeam=names["away"],
               FTHG=cur["home"]["score"].get("goals"), FTAG=cur["away"]["score"].get("goals"))
    half = {s.get("score", {}).get("participant"): s for s in scores if s.get("description") == "1ST_HALF"}
    if "home" in half and "away" in half:
        row.update(HTHG=half["home"]["score"].get("goals"), HTAG=half["away"]["score"].get("goals"))

    for s in _list(f, "statistics"):
        cols = STAT_TYPES.get(s.get("type_id"))
        loc = s.get("location") or loc_of.get(s.get("participant_id"))
        if not cols or loc not in ("home", "away"):
            continue
        d = s.get("data")
        val = d.get("value") if isinstance(d, dict) else s.get("value")
        row[cols[0] if loc == "home" else cols[1]] = val

    refs = _list(f, "referees")
    main = next((r for r in refs if r.get("type_id") == MAIN_REFEREE_TYPE), refs[0] if refs else None)
    if main:
        row["RefereeId"] = main.get("referee_id")

    side = {"home": 0, "away": 0}
    for s in _list(f, "sidelined"):
        loc = loc_of.get(s.get("participant_id"))
        if loc in side:
            side[loc] += 1
    if _list(f, "sidelined") or "sidelined" in f:
        row.update(HSide=side["home"], ASide=side["away"])
    return row


# ---------------- indirme ----------------
def fixtures_between(lid: int, a: dt.date, b: dt.date, includes: str = INCLUDES) -> list[dict]:
    out, page = [], 1
    while True:
        resp = get(f"/fixtures/between/{a}/{b}", {"filters": f"fixtureLeagues:{lid}",
                                                   "include": includes, "per_page": 50, "page": page})
        out.extend(resp.get("data") or [])
        if not (resp.get("pagination") or {}).get("has_more"):
            return out
        page += 1
        time.sleep(0.3)


def league_seasons(lid: int) -> list[dict]:
    resp = get(f"/leagues/{lid}", {"include": "seasons"})
    return _list(resp.get("data") or {}, "seasons")


def download_season(code: str, season: dict) -> pd.DataFrame:
    lid = LEAGUE_IDS[code]
    start = dt.date.fromisoformat(str(season["starting_at"])[:10])
    end = min(dt.date.fromisoformat(str(season.get("ending_at") or dt.date.today())[:10]), dt.date.today())
    rows, a = [], start
    while a <= end:
        b = min(a + dt.timedelta(days=99), end)  # API en fazla 100 günlük aralık veriyor
        for f in fixtures_between(lid, a, b):
            r = parse_fixture(f)
            if r:
                rows.append(r)
        a = b + dt.timedelta(days=1)
    df = pd.DataFrame(rows, columns=CSV_COLS)
    df["Season"] = season_code(season.get("name"))
    return df.drop_duplicates("sm_id").sort_values("Date")


def update(force: bool = False, max_seasons: int = 3) -> int:
    """Eksik sezonları indirir, güncel sezonu her seferinde yeniler. Dönüş: hata sayısı."""
    if not _token():
        print("Sportmonks: SPORTMONKS_TOKEN yok, atlandı (model eski verilerle çalışır).")
        return 0
    os.makedirs(SM_DIR, exist_ok=True)
    today, errors = dt.date.today(), 0
    for code, lid in LEAGUE_IDS.items():
        try:
            seasons = [s for s in league_seasons(lid) if s.get("starting_at")
                       and dt.date.fromisoformat(str(s["starting_at"])[:10]) <= today]
        except ApiError as e:
            print(f"  ✗ {code}: sezon listesi alınamadı ({e})")
            errors += 1
            continue
        seasons = sorted(seasons, key=lambda s: s["starting_at"], reverse=True)[:max_seasons]
        for s in seasons:
            sc = season_code(s.get("name"))
            if not sc:
                continue
            path = os.path.join(SM_DIR, f"{code}_{sc}.csv")
            ended = s.get("ending_at") and dt.date.fromisoformat(str(s["ending_at"])[:10]) < today - dt.timedelta(days=3)
            if os.path.exists(path) and ended and not force:
                continue  # biten sezon zaten indirilmiş
            try:
                df = download_season(code, s)
            except ApiError as e:
                print(f"  ✗ {code} {s.get('name')}: {e}")
                errors += 1
                continue
            df.to_csv(path, index=False)
            bc = df["HBC"].notna().mean() if len(df) else 0
            print(f"  ✓ {code} {s.get('name')}: {len(df)} maç (büyük şans verisi %{100 * bc:.0f})")
    return errors


# ---------------- lig verisiyle eşleştirme ----------------
def load_sm(code: str) -> pd.DataFrame:
    if not os.path.isdir(SM_DIR):
        return pd.DataFrame(columns=CSV_COLS)
    frames = [pd.read_csv(os.path.join(SM_DIR, f)) for f in sorted(os.listdir(SM_DIR))
              if f.startswith(code + "_") and f.endswith(".csv")]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=CSV_COLS)
    df = pd.concat(frames, ignore_index=True).reindex(columns=CSV_COLS)  # eksik sütun = boş
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])


def merge_into(df: pd.DataFrame, code: str, verbose: bool = False) -> pd.DataFrame:
    """Lig verisine (football-data.co.uk) Sportmonks sütunlarını ekler.

    Takım adları eşleştirilir (ör. 'Fenerbahçe' -> 'Fenerbahce', 'Manchester City' -> 'Man City'),
    maçlar tarih (±1 gün) + ev + deplasman ile bulunur.
    """
    from data import resolve_team  # döngüsel içe aktarmayı önlemek için burada

    df = df.copy()
    for c in MERGE_COLS:
        if c not in df.columns:
            df[c] = np.nan if c != "SM_Referee" else ""
    sm = load_sm(code)
    if not len(sm) or not len(df):
        return df

    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    mapping, unmatched = {}, set()
    for name in sorted(set(sm["HomeTeam"]) | set(sm["AwayTeam"])):
        try:
            mapping[name] = resolve_team([name], teams)
        except ValueError:
            unmatched.add(name)

    index = {}
    for i, r in enumerate(df[["Date", "HomeTeam", "AwayTeam"]].itertuples(index=False)):
        index[(r.Date.normalize(), r.HomeTeam, r.AwayTeam)] = i
    pos = {c: df.columns.get_loc(c) for c in MERGE_COLS}
    hit = 0
    for r in sm.itertuples(index=False):
        h, a = mapping.get(r.HomeTeam), mapping.get(r.AwayTeam)
        if not h or not a:
            continue
        d = r.Date.normalize()
        i = next((index[k] for k in ((d, h, a), (d - pd.Timedelta(days=1), h, a), (d + pd.Timedelta(days=1), h, a))
                  if k in index), None)
        if i is None:
            continue
        hit += 1
        for c in ("HBC", "ABC", "HPOS", "APOS", "HSide", "ASide"):
            v = getattr(r, c)
            if pd.notna(v):
                df.iat[i, pos[c]] = float(v)
        if pd.notna(r.RefereeId):
            df.iat[i, pos["SM_Referee"]] = f"sm{int(r.RefereeId)}"
    # football-data.co.uk'de hakem yoksa (Süper Lig, La Liga, Serie A) Sportmonks hakemini kullan
    if "Referee" in df.columns:
        empty = df["Referee"].fillna("").astype(str).str.strip() == ""
        df.loc[empty, "Referee"] = df.loc[empty, "SM_Referee"]
    if verbose:
        print(f"  Sportmonks eşleşmesi: {hit}/{len(sm)} maç"
              + (f"; eşleşmeyen takım adları: {', '.join(sorted(unmatched))}" if unmatched else ""))
    return df


def main():
    ap = argparse.ArgumentParser(description="Sportmonks verisini indir/güncelle")
    ap.add_argument("--hepsi", action="store_true", help="Biten sezonları da yeniden indir")
    ap.add_argument("--sezon", type=int, default=3, help="Kaç sezon geriye gidilsin")
    args = ap.parse_args()
    print("Sportmonks verisi güncelleniyor:")
    errors = update(force=args.hepsi, max_seasons=args.sezon)
    # eşleşme raporu
    try:
        from data import LEAGUES, load_league
        for code in LEAGUE_IDS:
            if len(load_sm(code)):
                print(f"{LEAGUES[code]}:")
                merge_into(load_league(code, n_seasons=args.sezon + 1, use_sm=False), code, verbose=True)
    except Exception as e:  # rapor başarısız olsa da indirme geçerli
        print(f"  (eşleşme raporu oluşturulamadı: {e})")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
