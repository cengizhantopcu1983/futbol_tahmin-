"""Eksik oyuncu etkisi: oyuncu önemi ve maç başına "eksik hücum / savunma gücü".

Oyuncu önemi: takımın o maçtan ÖNCEKİ son 10 lig maçında oynadığı dakika / mümkün dakika.
  Her maç 90 dk oynayan oyuncu = 1.0; maçların yarısında oynayan = ~0.5.
Mevki ağırlıkları:
  hücum gücü  : forvet 1, orta saha 0.5, defans 0.15, kaleci 0
  savunma gücü: kaleci 1, defans 1, orta saha 0.5, forvet 0.1
Bir maç için:
  ilk 11'e göre ("xi")   : ilk 11'de OLMAYAN oyuncuların önem toplamı (rotasyon + sakat + ceza)
  sakat listesine göre ("side"): sakat/cezalı listesindeki oyuncuların önem toplamı
Değerler lig ortalamasına göre merkezlenir; bilgi yoksa 0 (= ortalama bir maç).
Model bu değerlerin gol beklentisine etkisini öğrenir (models.py, lineup=...).

Elle girilen kadrolar (uygulama yazar): manuel_kadro.json
{
  "maclar": {"T1|2026-11-01|Galatasaray|Fenerbahce":
               {"ev_11": ["123", ...], "dep_11": [...], "ev_eksik": ["456"], "dep_eksik": []}},
  "ekle":  {"T1": {"Galatasaray": [{"id": "m:Ali Veli", "ad": "Ali Veli", "mevki": "F", "no": 9}]}},
  "cikar": {"T1": {"Galatasaray": ["789"]}}
}
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PLAYERS_DIR = os.environ.get("PLAYERS_DATA_DIR") or os.path.join(HERE, "data_players")
SQUADS_FILE = os.environ.get("KADRO_FILE") or os.path.join(HERE, "kadrolar.json")
MANUEL_KADRO = os.environ.get("MANUEL_KADRO_FILE") or os.path.join(HERE, "manuel_kadro.json")

WINDOW = 10
MIN_GAMES = 3
W_ATT = {"F": 1.0, "M": 0.5, "D": 0.15, "G": 0.0}
W_DEF = {"G": 1.0, "D": 1.0, "M": 0.5, "F": 0.1}
COLS = {"xi": ("HMA", "HMD", "AMA", "AMD"), "side": ("HSA", "HSD", "ASA", "ASD")}


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load_manual_lineups(path: str | None = None) -> dict:
    d = _read_json(path or MANUEL_KADRO)
    for k in ("maclar", "ekle", "cikar"):
        d.setdefault(k, {})
    return d


def _team_map(names, teams) -> dict:
    from data import resolve_team
    out = {}
    for n in set(names):
        if n in teams:
            out[n] = n
            continue
        try:
            out[n] = resolve_team([n], teams)
        except ValueError:
            pass
    return out


def load_players(code: str, teams, manual: dict | None = None) -> pd.DataFrame:
    """Sportmonks oyuncu satırları + elle girilen kadrolar; takım adları lig verisindeki gibi."""
    frames = []
    if os.path.isdir(PLAYERS_DIR):
        for f in sorted(os.listdir(PLAYERS_DIR)):
            if f.startswith(code + "_") and f.endswith(".csv"):
                frames.append(pd.read_csv(os.path.join(PLAYERS_DIR, f), dtype={"player_id": str}))
    sm = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["Date", "Team", "player_id", "pos", "status", "minutes"])
    if len(sm):
        m = _team_map(sm["Team"].dropna().unique(), set(teams))
        sm["Team"] = sm["Team"].map(m)
        sm = sm.dropna(subset=["Team"])
        sm["Date"] = pd.to_datetime(sm["Date"], errors="coerce")
        sm = sm.dropna(subset=["Date"])
    rows = []
    have = set(zip(sm["Team"], sm["Date"])) if len(sm) else set()
    for key, v in ((manual or {}).get("maclar") or {}).items():
        try:
            c, date, home, away = key.split("|")
        except ValueError:
            continue
        if c != code or not isinstance(v, dict):
            continue
        d = pd.Timestamp(date)
        for team, side in ((home, "ev"), (away, "dep")):
            if (team, d) in have:
                continue  # Sportmonks verisi varsa o kullanılır
            xi = [str(x) for x in v.get(f"{side}_11") or []]
            for pid in xi:
                rows.append({"Date": d, "Team": team, "player_id": pid, "pos": "", "status": "XI", "minutes": 90})
            for pid in v.get(f"{side}_eksik") or []:
                rows.append({"Date": d, "Team": team, "player_id": str(pid), "pos": "", "status": "OUT", "minutes": 0})
    if rows:
        sm = pd.concat([sm, pd.DataFrame(rows)], ignore_index=True)
    sm["minutes"] = pd.to_numeric(sm.get("minutes"), errors="coerce").fillna(0).clip(0, 120)
    sm["pos"] = sm["pos"].fillna("").astype(str)
    return sm.sort_values("Date", kind="stable").reset_index(drop=True)


def _positions(pl: pd.DataFrame, squads: dict) -> dict:
    pos = {}
    for pid, p in pl.loc[pl["pos"] != "", ["player_id", "pos"]].itertuples(index=False):
        pos[pid] = p  # en son bilinen mevki
    for team_list in squads.values():
        for p in team_list:
            if p.get("mevki"):
                pos.setdefault(str(p["id"]), p["mevki"])
    return pos


class Strength:
    """Takım takım, maç maç oyuncu önemi ve eksik gücü."""

    def __init__(self, pl: pd.DataFrame, pos: dict):
        self.pos = pos
        self.removed: dict[str, set] = {}  # uygulamadan "kadrodan çıkar" denenler
        self.names = {pid: n for pid, n in pl.loc[pl.get("player", pd.Series(dtype=str)).fillna("").astype(str) != "",
                                                   ["player_id", "player"]].itertuples(index=False)} \
            if "player" in pl.columns else {}
        self.games: dict[str, list] = defaultdict(list)  # takım -> [(tarih, {oyuncu: dk}, ilk11 kümesi, eksik kümesi)]
        for (team, date), g in pl.groupby(["Team", "Date"], sort=True):
            mins = {r.player_id: r.minutes for r in g.itertuples(index=False) if r.status in ("XI", "SUB")}
            xi = {r.player_id for r in g.itertuples(index=False) if r.status == "XI"}
            out = {r.player_id for r in g.itertuples(index=False) if r.status == "OUT"}
            self.games[team].append((date, mins, xi, out))
        # sakat listesi verisinin olduğu dönemler (yakınında hiç kayıt yoksa "bilinmiyor" sayılır)
        self.out_dates = np.sort(pl.loc[pl["status"] == "OUT", "Date"].unique()) if len(pl) else np.array([])

    def side_known(self, date) -> bool:
        if not len(self.out_dates):
            return False
        t = np.datetime64(pd.Timestamp(date))
        i = np.searchsorted(self.out_dates, t)
        near = [abs(self.out_dates[j] - t) for j in (i - 1, i) if 0 <= j < len(self.out_dates)]
        return bool(near) and min(near) <= np.timedelta64(120, "D")

    def importance(self, team: str, date=None) -> dict:
        """Tarihten önceki son 10 maça göre oyuncu önemleri (0-1). Yeterli maç yoksa {}."""
        g = self.games.get(team, [])
        if date is not None:
            t = pd.Timestamp(date)
            g = [x for x in g if x[0] < t]
        g = [x for x in g if x[2]][-WINDOW:]  # sadece ilk 11'i bilinen maçlar
        if len(g) < MIN_GAMES:
            return {}
        tot = defaultdict(float)
        for _, mins, _, _ in g:
            for p, m in mins.items():
                tot[p] += m
        return {p: min(m / (90.0 * len(g)), 1.0) for p, m in tot.items() if m > 0}

    def weights(self, pid: str) -> tuple[float, float]:
        p = self.pos.get(pid, "")
        return W_ATT.get(p, 0.4), W_DEF.get(p, 0.4)

    def missing(self, imp: dict, absent) -> tuple[float, float]:
        a = d = 0.0
        for p in absent:
            v = imp.get(p, 0.0)
            wa, wd = self.weights(p)
            a += v * wa
            d += v * wd
        return a, d

    def best_xi(self, imp: dict, exclude=()) -> set:
        """Eksikler çıkarıldıktan sonra en önemli 11 (1 kaleci + 10 oyuncu)."""
        avail = sorted(((v, p) for p, v in imp.items() if p not in exclude), reverse=True)
        gk = next((p for v, p in avail if self.pos.get(p) == "G"), None)
        rest = [p for v, p in avail if p != gk and self.pos.get(p) != "G"][:10]
        return set(rest) | ({gk} if gk else set())

    def match_features(self, team: str, date) -> dict:
        """Tarihteki maç için (xi_att, xi_def, side_att, side_def); bilinmeyen NaN."""
        t = pd.Timestamp(date)
        rec = next((x for x in self.games.get(team, []) if abs((x[0] - t).days) <= 1), None)
        out = {"xi": (np.nan, np.nan), "side": (np.nan, np.nan)}
        if rec is None:
            return out
        imp = self.importance(team, rec[0])
        if not imp:
            return out
        if len(rec[2]) >= 10:
            out["xi"] = self.missing(imp, [p for p in imp if p not in rec[2]])
        if self.side_known(rec[0]):
            out["side"] = self.missing(imp, rec[3])
        return out


def build(df: pd.DataFrame, code: str, manual: dict | None = None):
    teams = set(df["HomeTeam"]) | set(df["AwayTeam"])
    manual = manual if manual is not None else load_manual_lineups()
    pl = load_players(code, teams, manual)
    squads = _squads_for(code, teams, manual)
    st = Strength(pl, _positions(pl, squads))
    st.removed = {t: {str(x) for x in ids} for t, ids in ((manual.get("cikar") or {}).get(code) or {}).items()}
    return st, squads


def add_missing(df: pd.DataFrame, code: str, manual: dict | None = None) -> pd.DataFrame:
    """HMA/HMD/AMA/AMD (ilk 11) ve HSA/HSD/ASA/ASD (sakat listesi) sütunlarını ekler (merkezlenmiş)."""
    df = df.copy()
    for cols in COLS.values():
        for c in cols:
            df[c] = 0.0
    df.attrs["miss_center"] = {"xi": (0.0, 0.0), "side": (0.0, 0.0)}
    if not len(df):
        return df
    try:
        st, _ = build(df, code, manual)
    except Exception as e:  # oyuncu verisi bozuksa model onsuz devam eder
        print(f"  (eksik oyuncu verisi okunamadı: {e})")
        return df
    if not st.games:
        return df
    vals = {k: np.full((len(df), 4), np.nan) for k in COLS}
    for i, r in enumerate(df[["Date", "HomeTeam", "AwayTeam"]].itertuples(index=False)):
        fh, fa = st.match_features(r.HomeTeam, r.Date), st.match_features(r.AwayTeam, r.Date)
        for k in COLS:
            vals[k][i] = (*fh[k], *fa[k])
    centers = {}
    for k, cols in COLS.items():
        v = vals[k]
        ca = np.nanmean(np.r_[v[:, 0], v[:, 2]]) if np.isfinite(v[:, [0, 2]]).any() else 0.0
        cd = np.nanmean(np.r_[v[:, 1], v[:, 3]]) if np.isfinite(v[:, [1, 3]]).any() else 0.0
        centers[k] = (float(ca), float(cd))
        cen = np.array([ca, cd, ca, cd])
        for j, c in enumerate(cols):
            df[c] = np.nan_to_num(v[:, j] - cen[j], nan=0.0)
    df.attrs["miss_center"] = centers
    return df


def _squads_for(code: str, teams, manual: dict) -> dict:
    """kadrolar.json + elle eklenen/çıkarılan oyuncular; takım adları lig verisindeki gibi."""
    raw = (_read_json(SQUADS_FILE).get(code) or {})
    m = _team_map(raw.keys(), set(teams))
    out = defaultdict(list)
    for name, players in raw.items():
        if name in m:
            out[m[name]].extend(players)
    for team, players in ((manual.get("ekle") or {}).get(code) or {}).items():
        out[team].extend(p for p in players if isinstance(p, dict) and p.get("id"))
    for team, ids in ((manual.get("cikar") or {}).get(code) or {}).items():
        drop = {str(x) for x in ids}
        out[team] = [p for p in out.get(team, []) if str(p.get("id")) not in drop]
    return dict(out)


def team_sheet(st: Strength, squads: dict, team: str, date=None) -> list[dict]:
    """Uygulama için takım listesi: kadro + son maçlarda oynayanlar, öneme göre sıralı."""
    imp = st.importance(team, date)
    seen, rows = set(), []
    for p in squads.get(team, []):
        pid = str(p.get("id"))
        if pid in seen:
            continue
        seen.add(pid)
        rows.append({"id": pid, "ad": p.get("ad") or pid, "mevki": p.get("mevki") or st.pos.get(pid, ""),
                     "no": p.get("no"), "onem": round(imp.get(pid, 0.0), 3)})
    gone = st.removed.get(team, set())
    for pid, v in imp.items():  # kadro listesinde olmayan ama oynayan (ör. kiralık dönüşü)
        if pid not in seen and pid not in gone and v >= 0.1:
            rows.append({"id": pid, "ad": st.names.get(pid) or pid, "mevki": st.pos.get(pid, ""), "no": None, "onem": round(v, 3)})
    order = {"G": 0, "D": 1, "M": 2, "F": 3}
    rows.sort(key=lambda r: (-r["onem"], order.get(r["mevki"], 4)))
    return rows


def upcoming_missing(st: Strength, team: str, entry_xi, entry_out, mode: str, center) -> tuple[float, float] | None:
    """Yaklaşan maç için elle girilen kadroya göre (merkezlenmiş) eksik gücü; giriş yoksa None."""
    gone = st.removed.get(team, set())
    imp = {p: v for p, v in st.importance(team).items() if p not in gone}
    if not imp or (not entry_xi and not entry_out):
        return None
    if mode == "xi":
        if entry_xi and len(entry_xi) >= 10:  # açıklanan ilk 11
            xi = set(map(str, entry_xi))
            a, d = st.missing(imp, [p for p in imp if p not in xi])
        else:
            # sadece eksikler girildi: tahmini ilk 11 ile tam kadro ilk 11'i arasındaki fark
            # (olağan rotasyon zaten ortalamada var, sadece eksiklerin etkisi eklenir)
            full = st.best_xi(imp)
            xi = st.best_xi(imp, set(map(str, entry_out or [])))
            a0, d0 = st.missing(imp, [p for p in imp if p not in full])
            a1, d1 = st.missing(imp, [p for p in imp if p not in xi])
            return a1 - a0, d1 - d0
    else:
        if not entry_out:
            return None
        a, d = st.missing(imp, list(map(str, entry_out)))
    return a - center[0], d - center[1]
