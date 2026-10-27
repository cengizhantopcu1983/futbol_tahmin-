"""Dinlenme süresi ve Avrupa kupası yorgunluğu.

Avrupa maçları:
  - Şampiyonlar Ligi: football-data.org (FOOTBALL_DATA_ORG_KEY)
  - Avrupa Ligi: Sportmonks (SPORTMONKS_TOKEN; ligin aboneliğe seçilmiş olması gerekir)
data_euro/ klasöründe saklanır. Biten sezonlar bir kez, güncel sezon her çalıştırmada indirilir.

Her lig maçı için iki takımın da bir önceki resmi maçından (lig + Avrupa) bu yana geçen gün
sayısı hesaplanır. 4 gün veya daha az dinlenen takım "yorgun" sayılır (HTired / ATired).
Not: Türkiye Kupası, FA Cup gibi yerel kupa maçları verilerde yok; bu maçlar sayılmaz.

  python yorgunluk.py      # Avrupa maçlarını indir/güncelle
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EURO_DIR = os.environ.get("EURO_DATA_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data_euro")
TIRED_DAYS = 4
EURO_COLS = ["Date", "Comp", "HomeTeam", "AwayTeam", "HomeAlt", "AwayAlt"]
# eşleştirmede adın başından/sonundan atılan ekler
_AFFIX = ["fc", "afc", "cf", "sk", "as", "ac", "ssc", "ss", "us", "uc", "rc", "rcd", "cd", "ca", "sc",
          "fk", "jk", "acf", "bk", "if", "club", "calcio", "1909", "1907", "1899", "1913"]


# ---------------- indirme: Şampiyonlar Ligi (football-data.org) ----------------
def _fd_org(season: int) -> pd.DataFrame | None:
    from data import _download
    key = os.environ.get("FOOTBALL_DATA_ORG_KEY", "").strip()
    if not key:
        return None
    b = _download(f"https://api.football-data.org/v4/competitions/CL/matches?season={season}",
                  headers={"X-Auth-Token": key})
    if not b:
        return None
    try:
        matches = json.loads(b).get("matches", [])
    except ValueError:
        return None
    rows = []
    for m in matches:
        h, a = m.get("homeTeam") or {}, m.get("awayTeam") or {}
        if not m.get("utcDate") or not (h.get("name") or h.get("shortName")):
            continue
        rows.append({"Date": m["utcDate"][:10], "Comp": "CL",
                     "HomeTeam": h.get("shortName") or h.get("name"), "AwayTeam": a.get("shortName") or a.get("name"),
                     "HomeAlt": h.get("name") or "", "AwayAlt": a.get("name") or ""})
    return pd.DataFrame(rows, columns=EURO_COLS)


# ---------------- indirme: Avrupa Ligi (Sportmonks) ----------------
def _europa_league_id() -> int | None:
    import sportmonks as sm
    try:
        resp = sm.get("/leagues/search/Europa League")
        for lg in resp.get("data") or []:
            if str(lg.get("name", "")).strip().lower() == "europa league":
                return int(lg["id"])
    except sm.ApiError as e:
        print(f"  ✗ Avrupa Ligi aranamadı: {e}")
    return None


def _sportmonks_el(max_seasons: int, today: dt.date) -> list[tuple[str, pd.DataFrame, bool]]:
    import sportmonks as sm
    if not sm._token():
        return []
    lid = _europa_league_id()
    if lid is None:
        return []
    try:
        seasons = [s for s in sm.league_seasons(lid) if s.get("starting_at")]
    except sm.ApiError as e:
        print(f"  ✗ Avrupa Ligi sezonlarına erişilemedi ({e}). Sportmonks panelinde Europa League "
              "aboneliğe seçili mi?")
        return []
    seasons = sorted(seasons, key=lambda s: s["starting_at"], reverse=True)[:max_seasons]
    out = []
    for s in seasons:
        sc = sm.season_code(s.get("name"))
        start = dt.date.fromisoformat(str(s["starting_at"])[:10])
        end = dt.date.fromisoformat(str(s.get("ending_at") or today)[:10])
        ended = end < today - dt.timedelta(days=3)
        path = os.path.join(EURO_DIR, f"EL_{sc}.csv")
        if ended and os.path.exists(path):
            continue
        rows, a = [], start
        try:
            while a <= end:
                b = min(a + dt.timedelta(days=99), end)
                for f in sm.fixtures_between(lid, a, b, includes="participants"):
                    names = {}
                    for p in f.get("participants") or []:
                        loc = (p.get("meta") or {}).get("location")
                        if loc in ("home", "away"):
                            names[loc] = p.get("name")
                    if names.get("home") and names.get("away") and f.get("starting_at"):
                        rows.append({"Date": str(f["starting_at"])[:10], "Comp": "EL",
                                     "HomeTeam": names["home"], "AwayTeam": names["away"],
                                     "HomeAlt": "", "AwayAlt": ""})
                a = b + dt.timedelta(days=1)
        except sm.ApiError as e:
            print(f"  ✗ Avrupa Ligi {s.get('name')}: {e}")
            continue
        out.append((f"EL_{sc}", pd.DataFrame(rows, columns=EURO_COLS), ended))
    return out


def update(max_seasons: int = 3) -> None:
    os.makedirs(EURO_DIR, exist_ok=True)
    today = dt.date.today()
    y0 = today.year if today.month >= 7 else today.year - 1
    for season in range(y0 - max_seasons + 1, y0 + 1):
        sc = f"{season % 100:02d}{(season + 1) % 100:02d}"
        path = os.path.join(EURO_DIR, f"CL_{sc}.csv")
        if season < y0 and os.path.exists(path):
            continue
        df = _fd_org(season)
        if df is None or not len(df):
            print(f"  - Şampiyonlar Ligi {season}/{season + 1}: alınamadı (ücretsiz plan bu sezonu vermiyor olabilir)")
            continue
        df.to_csv(path, index=False)
        print(f"  ✓ Şampiyonlar Ligi {season}/{season + 1}: {len(df)} maç")
        time.sleep(7)  # football-data.org ücretsiz plan: dakikada 10 istek
    try:
        for name, df, _ in _sportmonks_el(max_seasons, today):
            df.to_csv(os.path.join(EURO_DIR, f"{name}.csv"), index=False)
            print(f"  ✓ Avrupa Ligi {name[3:]}: {len(df)} maç")
    except Exception as e:  # Avrupa Ligi alınamasa da Şampiyonlar Ligi verisi geçerli kalır
        print(f"  ✗ Avrupa Ligi alınamadı: {e}")


# ---------------- takım adı eşleştirme (katı) ----------------
def _strip(n: str) -> str:
    from data import norm
    import re
    words = re.split(r"[\s\-\.&']+", str(n).lower())
    words = [w for w in words if w and norm(w) not in _AFFIX]
    return norm(" ".join(words))


def match_team(name: str, alt: str, teams: list[str]) -> str | None:
    """Avrupa maçındaki takım adını lig takımlarından biriyle eşleştirir; emin değilse None.

    Yanlış eşleşme (ör. başka ülkeden bir takımı bizim takım sanmak) yorgunluğu bozacağı için
    benzerlik tahmini yapılmaz: tam ad, bilinen takma ad ya da en az 6 harfli tekil parça.
    """
    from data import TEAM_ALIASES, norm
    table = {norm(t): t for t in teams}
    stripped = {_strip(t): t for t in teams}
    for cand in (name, alt):
        if not cand:
            continue
        q = norm(cand)
        if q in table:
            return table[q]
        al = TEAM_ALIASES.get(q)
        if al in teams:
            return al
        qs = _strip(cand)
        if qs in stripped:
            return stripped[qs]
        al = TEAM_ALIASES.get(qs)
        if al in teams:
            return al
    for cand in (name, alt):
        if not cand:
            continue
        qs = _strip(cand)
        # parça eşleşmesi: fark en fazla 4 harf olabilir (ör. 'galatasarayas' ~ 'galatasaray'),
        # 'samsunsporgenclik' gibi başka bir takımın adı eşleşmesin
        subs = {t for k, t in stripped.items() if len(k) >= 6 and
                ((k in qs and len(qs) - len(k) <= 4) or (len(qs) >= 6 and qs in k and len(k) - len(qs) <= 4))}
        if len(subs) == 1:
            return subs.pop()
    return None


def load_euro() -> pd.DataFrame:
    if not os.path.isdir(EURO_DIR):
        return pd.DataFrame(columns=EURO_COLS)
    frames = [pd.read_csv(os.path.join(EURO_DIR, f), keep_default_na=False) for f in sorted(os.listdir(EURO_DIR))
              if f.endswith(".csv")]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=EURO_COLS)
    df = pd.concat(frames, ignore_index=True).reindex(columns=EURO_COLS).fillna("")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"]).drop_duplicates(["Date", "HomeTeam", "AwayTeam"])


def euro_dates(teams: list[str], euro: pd.DataFrame | None = None) -> dict[str, list[pd.Timestamp]]:
    """Lig takımlarının Avrupa maç tarihleri."""
    euro = load_euro() if euro is None else euro
    out: dict[str, list] = {}
    cache: dict[tuple, str | None] = {}
    for r in euro.itertuples(index=False):
        for nm, alt in ((r.HomeTeam, r.HomeAlt), (r.AwayTeam, r.AwayAlt)):
            k = (nm, alt)
            if k not in cache:
                cache[k] = match_team(nm, alt, teams)
            t = cache[k]
            if t:
                out.setdefault(t, []).append(pd.Timestamp(r.Date).normalize())
    return out


class RestCalendar:
    """Bir ligin takımları için tüm resmi maç tarihleri (lig + Avrupa)."""

    def __init__(self, league_df: pd.DataFrame, extra: pd.DataFrame | None = None,
                 euro: pd.DataFrame | None = None):
        teams = sorted(set(league_df["HomeTeam"]) | set(league_df["AwayTeam"]))
        self.dates: dict[str, np.ndarray] = {}
        self.euro: dict[str, set] = {}
        e = euro_dates(teams, euro)
        frames = [league_df[["Date", "HomeTeam", "AwayTeam"]]]
        if extra is not None and len(extra):
            frames.append(extra[["Date", "HomeTeam", "AwayTeam"]])
        allm = pd.concat(frames)
        for t in teams:
            d = pd.to_datetime(pd.concat([allm.loc[allm.HomeTeam == t, "Date"], allm.loc[allm.AwayTeam == t, "Date"]]))
            ed = e.get(t, [])
            self.euro[t] = set(ed)
            arr = np.array(sorted(set(d.dt.normalize().tolist()) | set(ed)), dtype="datetime64[ns]")
            self.dates[t] = arr

    def rest(self, team: str, date) -> tuple[int | None, bool]:
        """(önceki maçtan bu yana gün, önceki maç Avrupa maçı mı)."""
        arr = self.dates.get(team)
        if arr is None or not len(arr):
            return None, False
        d = np.datetime64(pd.Timestamp(date).normalize(), "ns")
        i = np.searchsorted(arr, d) - 1
        if i < 0:
            return None, False
        prev = pd.Timestamp(arr[i])
        days = int((pd.Timestamp(d) - prev).days)
        return days, prev in self.euro.get(team, set())


def add_fatigue(df: pd.DataFrame, euro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Lig verisine HRest/ARest (gün), HEuro/AEuro (önceki maç Avrupa'da mıydı), HTired/ATired ekler."""
    df = df.copy()
    if not len(df):
        return df
    cal = RestCalendar(df, euro=euro)
    cols = {c: [] for c in ("HRest", "ARest", "HEuro", "AEuro")}
    for r in df[["Date", "HomeTeam", "AwayTeam"]].itertuples(index=False):
        hr, he = cal.rest(r.HomeTeam, r.Date)
        ar, ae = cal.rest(r.AwayTeam, r.Date)
        cols["HRest"].append(np.nan if hr is None else hr)
        cols["ARest"].append(np.nan if ar is None else ar)
        cols["HEuro"].append(int(he))
        cols["AEuro"].append(int(ae))
    for c, v in cols.items():
        df[c] = v
    df["HTired"] = (df["HRest"] <= TIRED_DAYS).astype(int)
    df["ATired"] = (df["ARest"] <= TIRED_DAYS).astype(int)
    return df


def main():
    print("Avrupa kupası maçları güncelleniyor:")
    update()
    eu = load_euro()
    print(f"Toplam {len(eu)} Avrupa maçı kayıtlı.")
    try:
        from data import LEAGUES, load_league
        for code, name in LEAGUES.items():
            df = load_league(code, n_seasons=3, use_sm=False)
            teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
            found = euro_dates(teams, eu)
            if found:
                print(f"  {name}: Avrupa'da oynayan takımlar: "
                      + ", ".join(f"{t} ({len(v)})" for t, v in sorted(found.items())))
    except Exception as e:
        print(f"  (eşleşme raporu oluşturulamadı: {e})")


if __name__ == "__main__":
    main()
