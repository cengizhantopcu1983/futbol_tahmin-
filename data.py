"""Veri katmanı.

- football-data.co.uk: maç sonuçları, bahis oranları ve fikstür (4 lig)
- football-data.org: Premier League, La Liga, Serie A fikstürü (ücretsiz anahtar,
  FOOTBALL_DATA_ORG_KEY ortam değişkeninden okunur; yoksa bu kaynak atlanır)

Dosyalar data/ klasörüne önbelleklenir. Geçmiş sezonlar bir kez indirilir,
devam eden sezon 6 saatte bir yenilenir. Fikstür saatleri İstanbul saatine çevrilir.
"""
from __future__ import annotations

import datetime as dt
import difflib
import json
import os
import time
import unicodedata
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
FD_ORG_URL = "https://api.football-data.org/v4/competitions/{comp}/matches?dateFrom={start}&dateTo={end}"
FD_ORG_COMPS = {"E0": "PL", "SP1": "PD", "I1": "SA"}  # Süper Lig ücretsiz planda yok

TZ_TR = ZoneInfo("Europe/Istanbul")
TZ_UK = ZoneInfo("Europe/London")

# İki kaynağın farklı yazdığı takım adları -> football-data.co.uk adı
TEAM_ALIASES = {
    "wolverhampton": "Wolves", "wolverhamptonwanderersfc": "Wolves",
    "nottingham": "Nott'm Forest", "nottinghamforest": "Nott'm Forest",
    "nottinghamforestfc": "Nott'm Forest",
    "manchestercityfc": "Man City", "manchesterunitedfc": "Man United",
    "brightonhove": "Brighton", "brightonhovealbionfc": "Brighton",
    "atleti": "Ath Madrid", "atleticomadrid": "Ath Madrid", "clubatleticodemadrid": "Ath Madrid",
    "athletic": "Ath Bilbao", "athleticclub": "Ath Bilbao", "athleticbilbao": "Ath Bilbao",
    "espanyol": "Espanol", "rcdespanyoldebarcelona": "Espanol",
    "realbetis": "Betis", "realbetisbalompie": "Betis",
    "realsociedad": "Sociedad", "realsociedaddefutbol": "Sociedad",
    "rayovallecano": "Vallecano", "rayovallecanodemadrid": "Vallecano",
    "celtavigo": "Celta", "rccelta": "Celta", "rcceltadevigo": "Celta",
    "fcinternazionalemilano": "Inter", "internazionale": "Inter",
    "acmilan": "Milan", "hellasverona": "Verona", "hellasveronafc": "Verona",
    # Sportmonks / uzun adlar
    "manchestercity": "Man City", "manchesterunited": "Man United", "newcastleunited": "Newcastle",
    "tottenhamhotspur": "Tottenham", "westhamunited": "West Ham", "brightonhovealbion": "Brighton",
    "wolverhamptonwanderers": "Wolves", "afcbournemouth": "Bournemouth", "leedsunited": "Leeds",
    "leicestercity": "Leicester", "ipswichtown": "Ipswich", "sheffieldunited": "Sheffield United",
    "lutontown": "Luton", "nottinghamforestfc2": "Nott'm Forest",
    "deportivoalaves": "Alaves", "rcdmallorca": "Mallorca", "realvalladolid": "Valladolid",
    "udlaspalmas": "Las Palmas", "caosasuna": "Osasuna", "realoviedo": "Oviedo", "cdleganes": "Leganes",
    "girona": "Girona", "elchecf": "Elche", "levanteud": "Levante",
    "asroma": "Roma", "sslazio": "Lazio", "sscnapoli": "Napoli", "acffiorentina": "Fiorentina",
    "torinofc": "Torino", "usssassuolo": "Sassuolo", "uscremonese": "Cremonese", "acpisa": "Pisa",
    "bodrumfk": "Bodrumspor", "istanbulbasaksehir": "Basaksehir", "caykurrizespor": "Rizespor",
    "mkeankaragucu": "Ankaragucu", "fatihkaragumruk": "Karagumruk",
}
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


def _download(url: str, timeout: int = 30, headers: dict | None = None) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (futbol-tahmin)", **(headers or {})})
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
    for c in STAT_COLS:  # devre skoru, şut, korner, kart (dosyada yoksa boş kalır)
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    df["Referee"] = df["Referee"].astype(str).str.strip().replace("nan", "") if "Referee" in df.columns else ""
    return df[["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "bk_H", "bk_D", "bk_A",
               *STAT_COLS, "Referee"]]


STAT_COLS = ["HTHG", "HTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]


def load_league(code: str, seasons: list[str] | None = None, n_seasons: int = 4,
                refresh: bool = True, use_sm: bool = True) -> pd.DataFrame:
    """Bir ligin son n sezonunu (veya verilen sezonları) yükler, tarihe göre sıralı döner.

    use_sm: data_sm/ klasöründe Sportmonks verisi varsa (büyük şans, topla oynama, hakem,
    sakatlık) maçlara eklenir.
    """
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
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    if use_sm:
        try:
            from sportmonks import merge_into
            df = merge_into(df, code)
        except ImportError:
            pass
    return df


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
    # Saatler İngiltere saatiyle verilir -> İstanbul saatine çevir
    conv = [_uk_to_tr(d, t) for d, t in zip(raw["Date"], raw["Time"])]
    raw["Date"] = [c[0] for c in conv]
    raw["Time"] = [c[1] for c in conv]
    raw["Kaynak"] = "football-data.co.uk"
    raw["Referee"] = (raw["Referee"].astype(str).str.strip().replace("nan", "")
                      if "Referee" in raw.columns else "")
    raw["HomeNames"] = [[n] for n in raw["HomeTeam"]]
    raw["AwayNames"] = [[n] for n in raw["AwayTeam"]]
    return raw[FIXTURE_COLS].sort_values(["Date", "Time"]).reset_index(drop=True)


FIXTURE_COLS = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "HomeNames", "AwayNames",
                "bk_H", "bk_D", "bk_A", "Kaynak", "Referee"]


def _uk_to_tr(date, time_str):
    """İngiltere saatindeki tarih+saati İstanbul saatine çevirir. Dönüş: (tarih, 'SS:DD')."""
    if pd.isna(date):
        return pd.NaT, ""
    try:
        h, mi = (int(x) for x in str(time_str).split(":")[:2])
        t = (pd.Timestamp(date).normalize() + pd.Timedelta(hours=h, minutes=mi))
        t = t.tz_localize(TZ_UK, ambiguous=False, nonexistent="shift_forward").tz_convert(TZ_TR)
        return t.tz_localize(None).normalize(), t.strftime("%H:%M")
    except (ValueError, TypeError):
        return pd.Timestamp(date).normalize(), ""


def load_fixtures_api(days: int = 8) -> pd.DataFrame:
    """football-data.org'dan önümüzdeki günlerin maçları (Premier League, La Liga, Serie A)."""
    key = os.environ.get("FOOTBALL_DATA_ORG_KEY", "").strip()
    empty = pd.DataFrame(columns=FIXTURE_COLS)
    if not key:
        print("  (football-data.org anahtarı tanımlı değil, bu kaynak atlandı)")
        return empty
    start = dt.datetime.now(TZ_TR).date()
    end = start + dt.timedelta(days=days)
    rows = []
    for code, comp in FD_ORG_COMPS.items():
        b = _download(FD_ORG_URL.format(comp=comp, start=start, end=end),
                      headers={"X-Auth-Token": key})
        if not b:
            continue
        try:
            matches = json.loads(b).get("matches", [])
        except ValueError:
            print(f"  ! football-data.org {comp}: yanıt okunamadı")
            continue
        n = 0
        for m in matches:
            if m.get("status") not in ("SCHEDULED", "TIMED"):
                continue
            h, a = m.get("homeTeam") or {}, m.get("awayTeam") or {}
            hn = [x for x in (h.get("shortName"), h.get("name"), h.get("tla")) if x]
            an = [x for x in (a.get("shortName"), a.get("name"), a.get("tla")) if x]
            if not hn or not an or not m.get("utcDate"):
                continue  # rakip henüz belli değil
            t = pd.Timestamp(m["utcDate"]).tz_convert(TZ_TR)
            rows.append({"Div": code, "Date": t.tz_localize(None).normalize(), "Time": t.strftime("%H:%M"),
                         "HomeTeam": hn[0], "AwayTeam": an[0], "HomeNames": hn, "AwayNames": an,
                         "bk_H": np.nan, "bk_D": np.nan, "bk_A": np.nan, "Kaynak": "football-data.org",
                         "Referee": ""})
            n += 1
        print(f"  football-data.org {LEAGUES[code]}: {n} maç")
        time.sleep(1)  # ücretsiz plan: dakikada 10 istek
    return pd.DataFrame(rows, columns=FIXTURE_COLS) if rows else empty


def resolve_team(names, teams) -> str:
    """Bir takımın olası adlarını (kısa ad, tam ad, kısaltma) sırayla dener."""
    last = None
    for n in names:
        try:
            return find_team(n, teams)
        except ValueError as e:
            last = e
    raise last or ValueError("takım adı yok")


def find_team(query: str, teams) -> str:
    """Kullanıcının yazdığı adı veri setindeki takım adıyla eşleştirir ('Fener' -> 'Fenerbahce')."""
    teams = list(teams)
    table = {norm(t): t for t in teams}
    q = norm(query)
    if q in table:
        return table[q]
    alias = TEAM_ALIASES.get(q) or TEAM_ALIASES.get(q.removesuffix("fc"))
    if alias in teams:
        return alias
    subs = sorted({t for k, t in table.items() if q and (q in k or (len(k) >= 4 and k in q))})
    if len(subs) == 1:
        return subs[0]
    if len(subs) > 1:
        raise ValueError(f"'{query}' birden fazla takımla eşleşti: {', '.join(subs)}")
    close = difflib.get_close_matches(q, list(table), n=1, cutoff=0.6)
    if close:
        return table[close[0]]
    raise ValueError(f"'{query}' bulunamadı. Takım listesi için --takimlar kullan.")
