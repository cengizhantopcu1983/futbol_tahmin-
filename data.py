"""Veri katmanı: football-data.co.uk'den maç sonuçları, bahis oranları ve fikstür.

Dosyalar data/ klasörüne önbelleklenir. Geçmiş sezonlar bir kez indirilir,
devam eden sezon 6 saatte bir yenilenir.
"""
from __future__ import annotations

import datetime as dt
import difflib
import os
import time
import unicodedata
import urllib.request

import numpy as np
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
DATA_DIR = os.environ.get("FUTBOL_DATA_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)

LEAGUES = {
    "T1": "Süper Lig",
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
}

ALIASES = {
    "t1": "T1", "superlig": "T1", "tr": "T1", "turkiye": "T1",
    "e0": "E0", "premier": "E0", "premierlig": "E0", "premierleague": "E0",
    "pl": "E0", "epl": "E0", "ingiltere": "E0",
    "sp1": "SP1", "laliga": "SP1", "ispanya": "SP1",
    "i1": "I1", "seriea": "I1", "italya": "I1",
}

# Sırayla denenen bahis oranı sütunları (Pinnacle en keskin piyasa).
ODDS_SETS = [
    ("PSH", "PSD", "PSA"),
    ("AvgH", "AvgD", "AvgA"),
    ("BbAvH", "BbAvD", "BbAvA"),
    ("B365H", "B365D", "B365A"),
]

_TR = str.maketrans({"ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
                     "ç": "c", "Ç": "c", "ö": "o", "Ö": "o", "ü": "u", "Ü": "u"})


def norm(s: str) -> str:
    """Karşılaştırma için metni sadeleştirir: küçük harf, Türkçe karakter yok, sadece harf/rakam."""
    s = str(s).translate(_TR).lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if ch.isalnum())


def resolve_league(name: str) -> str:
    key = norm(name)
    if key in ALIASES:
        return ALIASES[key]
    for code in LEAGUES:
        if norm(code) == key:
            return code
    raise ValueError(f"Bilinmeyen lig: {name}. Seçenekler: super-lig, premier, laliga, seriea")


def current_season_start(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_codes(n: int, last_start: int | None = None) -> list[str]:
    y0 = current_season_start() if last_start is None else last_start
    return [season_code(y) for y in range(y0 - n + 1, y0 + 1)]


def _download(url: str, timeout: int = 30) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (futbol-tahmin)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:  # ağ hatası, 404 vb.
        print(f"  ! İndirilemedi: {url} ({e})")
        return None


def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")


def _add_bookmaker_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Oranları marjı temizlenmiş olasılığa çevirir (bk_H, bk_D, bk_A)."""
    oh = pd.Series(np.nan, index=df.index)
    od, oa = oh.copy(), oh.copy()
    for h, d, a in ODDS_SETS:
        if all(c in df.columns for c in (h, d, a)):
            oh = oh.fillna(pd.to_numeric(df[h], errors="coerce"))
            od = od.fillna(pd.to_numeric(df[d], errors="coerce"))
            oa = oa.fillna(pd.to_numeric(df[a], errors="coerce"))
    inv = pd.concat([1 / oh, 1 / od, 1 / oa], axis=1)
    inv = inv.where(inv > 0)
    s = inv.sum(axis=1, min_count=3)
    df["bk_H"] = inv.iloc[:, 0] / s
    df["bk_D"] = inv.iloc[:, 1] / s
    df["bk_A"] = inv.iloc[:, 2] / s
    return df


def _clean(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    if not all(c in raw.columns for c in need):
        return pd.DataFrame()
    df = raw.dropna(subset=need).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["Date"])
    for c in ("HomeTeam", "AwayTeam"):
        df[c] = df[c].astype(str).str.strip()
    for c in ("FTHG", "FTAG"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df = _add_bookmaker_probs(df)
    df["Season"] = season
    return df[["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "bk_H", "bk_D", "bk_A"]]


def load_league(code: str, seasons: list[str] | None = None, n_seasons: int = 4,
                refresh: bool = True) -> pd.DataFrame:
    """Bir ligin son n sezonunu (veya verilen sezonları) yükler, tarihe göre sıralı döner."""
    os.makedirs(DATA_DIR, exist_ok=True)
    seasons = seasons or season_codes(n_seasons)
    current = season_code(current_season_start())
    frames = []
    for s in seasons:
        path = os.path.join(DATA_DIR, f"{code}_{s}.csv")
        missing = not os.path.exists(path)
        stale = (not missing and refresh and s == current
                 and time.time() - os.path.getmtime(path) > 6 * 3600)
        if missing or stale:
            b = _download(BASE_URL.format(season=s, code=code))
            if b and len(b) > 200:
                with open(path, "wb") as f:
                    f.write(b)
        if os.path.exists(path):
            part = _clean(_read_csv(path), s)
            if len(part):
                frames.append(part)
    if not frames:
        raise RuntimeError(f"{LEAGUES.get(code, code)} için veri yok. İnternet bağlantısını kontrol et.")
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def load_fixtures(refresh: bool = True) -> pd.DataFrame:
    """Yaklaşan maçlar (football-data.co.uk/fixtures.csv), sadece desteklenen ligler."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "fixtures.csv")
    if not os.path.exists(path) or (refresh and time.time() - os.path.getmtime(path) > 3 * 3600):
        b = _download(FIXTURES_URL)
        if b and len(b) > 50:
            with open(path, "wb") as f:
                f.write(b)
    if not os.path.exists(path):
        raise RuntimeError("Fikstür dosyası indirilemedi.")
    raw = _read_csv(path)
    raw = raw[raw["Div"].isin(LEAGUES)].copy()
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed", errors="coerce")
    for c in ("HomeTeam", "AwayTeam"):
        raw[c] = raw[c].astype(str).str.strip()
    raw = _add_bookmaker_probs(raw)
    if "Time" not in raw.columns:
        raw["Time"] = ""
    cols = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "bk_H", "bk_D", "bk_A"]
    return raw[cols].sort_values(["Date", "Time"]).reset_index(drop=True)


def find_team(query: str, teams) -> str:
    """Kullanıcının yazdığı adı veri setindeki takım adıyla eşleştirir ('Fener' -> 'Fenerbahce')."""
    teams = list(teams)
    table = {norm(t): t for t in teams}
    q = norm(query)
    if q in table:
        return table[q]
    subs = sorted({t for k, t in table.items() if q and (q in k or (len(k) >= 4 and k in q))})
    if len(subs) == 1:
        return subs[0]
    if len(subs) > 1:
        raise ValueError(f"'{query}' birden fazla takımla eşleşti: {', '.join(subs)}")
    close = difflib.get_close_matches(q, list(table), n=1, cutoff=0.6)
    if close:
        return table[close[0]]
    raise ValueError(f"'{query}' bulunamadı. Takım listesi için --takimlar kullan.")
