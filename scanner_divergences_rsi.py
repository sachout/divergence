#!/usr/bin/env python3
"""
Scanner de divergences RSI - S&P 500 + Nasdaq - timeframes DAILY et WEEKLY uniquement.
Ne garde QUE les divergences où le RSI est en surachat (>=70) ou en survente (<=30),
et élimine celles dont le RSI a déjà recroisé la barre des 50 depuis le 2e pivot.
Section « en formation » : la dernière bougie serait le 2e pivot (non encore confirmé).
Règles : 5 à 16 bougies entre les deux pivots ; le « bruit » (extrême du RSI entre les
deux pivots) ne doit pas dépasser 40 (haussière) ni descendre sous 60 (baissière).

Lancement :  python3 scanner_divergences_rsi.py
-> génère divergences_rsi.html (un graphique prix + RSI par divergence) et l'ouvre.

Installation (une fois) :
    pip3 install yfinance pandas numpy scipy lxml requests matplotlib

Options :
    --market us | tsx            us = S&P 500 + Nasdaq (défaut) ; tsx = S&P/TSX Composite (Canada)
    --tf daily | weekly | both   (défaut : both)
    --keep-crossed               garder aussi les divergences dont le RSI a déjà recroisé 50
                                 (par défaut elles sont ÉLIMINÉES)
    --nasdaq-all                 tout le Nasdaq (~3000 titres) au lieu du Nasdaq-100
    --diagnose SCHW              explique règle par règle pourquoi SCHW est (ou n'est pas) dans la liste
"""
import argparse
import base64
import html
import io
import os
import sys
import warnings
import webbrowser
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.signal import argrelextrema

warnings.filterwarnings("ignore")
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Réglages par timeframe (modifiables)
#   order   : nb de barres de chaque côté pour valider un sommet/creux
#   min/max_gap : écart (en bougies) autorisé entre les deux pivots : GAP_MIN à GAP_MAX, pas plus pas moins
#   max_age : le 2e pivot doit dater de moins de X barres
GAP_MIN, GAP_MAX = 5, 16      # nombre de bougies entre les deux pivots (bornes incluses)
TF_CONFIG = {
    "Daily":  dict(interval="1d",  period="1y", order=4, min_gap=GAP_MIN, max_gap=GAP_MAX, max_age=10),
    "Weekly": dict(interval="1wk", period="5y", order=3, min_gap=GAP_MIN, max_gap=GAP_MAX, max_age=4),
}

MARKETS = {"us": "S&P 500 + Nasdaq", "tsx": "TSX · S&P/TSX Composite"}

# « Bruit » = extrême du RSI entre les deux pivots de la divergence
NOISE_BEAR_MIN = 60   # baissière : le RSI entre les 2 sommets ne doit pas descendre sous 60
NOISE_BULL_MAX = 40   # haussière : le RSI entre les 2 creux ne doit pas dépasser 40


# ----------------------------- Univers -----------------------------
def _wiki_tables(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


def get_sp500():
    t = _wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return t["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()


def get_nasdaq100():
    for t in _wiki_tables("https://en.wikipedia.org/wiki/Nasdaq-100"):
        if "Ticker" in t.columns:
            return t["Ticker"].astype(str).str.replace(".", "-", regex=False).tolist()
    return []


def get_nasdaq_all():
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep="|")
    df = df[df["Symbol"].notna() & ~df["Symbol"].astype(str).str.startswith("File Creation")]
    df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N")]
    return df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()


def get_tsx():
    """S&P/TSX Composite (~220 actions). Tickers Yahoo : suffixe .TO et '.' -> '-'
    (ex. BBD.B -> BBD-B.TO, REI.UN -> REI-UN.TO)."""
    best = None
    for t in _wiki_tables("https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"):
        if "Ticker" in t.columns and (best is None or len(t) > len(best)):
            best = t
    if best is None:
        return []
    syms = (best["Ticker"].dropna().astype(str).str.replace(r"\[.*?\]", "", regex=True)
            .str.strip().str.replace(".", "-", regex=False))
    return [x + ".TO" for x in syms if x and x.lower() != "nan"]


def tv_symbol(t):
    """Symbole TradingView : SHOP.TO -> TSX:SHOP ; BBD-B.TO -> TSX:BBD.B"""
    if t.endswith(".TO"):
        return "TSX:" + t[:-3].replace("-", ".")
    return t


def safe(fn, name):
    try:
        return fn()
    except Exception as e:
        print(f"[!] Impossible de charger {name}: {e}", file=sys.stderr)
        return []


def get_universe(market="us", nasdaq_all=False):
    if market == "tsx":
        t = set(safe(get_tsx, "S&P/TSX Composite"))
    else:
        t = set(safe(get_sp500, "S&P 500"))
        t |= set(safe(get_nasdaq_all if nasdaq_all else get_nasdaq100, "Nasdaq"))
    return sorted(x for x in t if isinstance(x, str) and x and x != "nan")


# ----------------------------- RSI -----------------------------
def rsi_wilder(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss          # perte nulle -> rs infini -> RSI = 100 (et non NaN)
    return 100 - 100 / (1 + rs)


# ----------------------------- Détection -----------------------------
def evaluate(kind, i1, i2, px, rv, n, cfg, ob, os_):
    """Teste toutes les règles sur la paire de pivots (i1, i2).
    Retourne (liste de (nom, ok, texte), bruit, rsi_a_recroise_50). La dernière règle est « RSI et 50 »."""
    bear = kind == "BAISSIERE"
    nom = "sommet" if bear else "creux"
    r1, r2 = rv[i1], rv[i2]
    seg = rv[i1 + 1:i2]                                  # le « bruit » : RSI entre les deux pivots
    seg_ok = len(seg) > 0 and not np.isnan(seg).any()
    noise = float(seg.min() if bear else seg.max()) if seg_ok else float("nan")
    crossed = bool(np.any(rv[i2:] < 50) if bear else np.any(rv[i2:] > 50))
    if bear:
        zone_ok = bool(max(r1, r2) >= ob)
        zone_txt = f"RSI en surachat ≥ {ob} sur un des pivots (max {max(r1, r2):.1f})"
        noise_ok = seg_ok and noise >= NOISE_BEAR_MIN
        noise_txt = f"RSI minimum entre les pivots = {noise:.1f} (limite ≥ {NOISE_BEAR_MIN})"
    else:
        zone_ok = bool(min(r1, r2) <= os_)
        zone_txt = f"RSI en survente ≤ {os_} sur un des pivots (min {min(r1, r2):.1f})"
        noise_ok = seg_ok and noise <= NOISE_BULL_MAX
        noise_txt = f"RSI maximum entre les pivots = {noise:.1f} (limite ≤ {NOISE_BULL_MAX})"
    checks = [
        ("Prix", bool(px[i2] > px[i1]) if bear else bool(px[i2] < px[i1]),
         f"{nom} 2 {'plus haut' if bear else 'plus bas'} que le 1 ({px[i1]:.2f} → {px[i2]:.2f})"),
        ("RSI", bool(r2 < r1) if bear else bool(r2 > r1),
         f"{'plus bas' if bear else 'plus haut'} au pivot 2 ({r1:.1f} → {r2:.1f})"),
        ("Zone", zone_ok, zone_txt),
        ("Écart", bool(cfg["min_gap"] <= i2 - i1 <= cfg["max_gap"]),
         f"{i2 - i1} bougies entre les pivots (autorisé {cfg['min_gap']} à {cfg['max_gap']})"),
        ("Bruit", bool(noise_ok), noise_txt if seg_ok else "aucun point entre les pivots"),
        ("Fraîcheur", bool(n - 1 - i2 <= cfg["max_age"]),
         f"2e pivot il y a {n - 1 - i2} bougies (maximum {cfg['max_age']})"),
        ("RSI et 50", not crossed,
         "a déjà recroisé 50 depuis le pivot 2" if crossed else "n’a pas recroisé 50 depuis le pivot 2"),
    ]
    return checks, noise, crossed


def detect(df, order=4, min_gap=5, max_gap=16, ob=70, os_=30, max_age=10):
    """Divergences régulières, RSI en surachat/survente uniquement.
    Chaque ligne : (type, date1, date2, prix1, prix2, rsi1, rsi2, age, croise50, bruit, en_formation)."""
    close = df["Close"].dropna()
    if len(close) < 60:
        return []
    rsi = rsi_wilder(close)
    px, rv, n = close.values, rsi.values, len(close)
    cfg = dict(order=order, min_gap=min_gap, max_gap=max_gap, max_age=max_age)
    highs = argrelextrema(px, np.greater, order=order)[0]
    lows = argrelextrema(px, np.less, order=order)[0]
    out = []

    # 1) Divergences confirmées : les deux derniers pivots (« RSI et 50 » est filtré plus tard)
    for kind, piv in (("BAISSIERE", highs), ("HAUSSIERE", lows)):
        if len(piv) >= 2:
            i1, i2 = piv[-2], piv[-1]
            checks, noise, crossed = evaluate(kind, i1, i2, px, rv, n, cfg, ob, os_)
            if all(ok for _, ok, _ in checks[:-1]):
                out.append((kind, close.index[i1], close.index[i2], px[i1], px[i2],
                            rv[i1], rv[i2], n - 1 - i2, crossed, noise, False))

    # 2) Divergences « en formation » : la DERNIÈRE bougie serait le 2e pivot (pas encore confirmé)
    last = n - 1
    if last > order:
        left = px[last - order:last]
        for kind, piv, new_extreme in (("BAISSIERE", highs, px[last] > left.max()),
                                       ("HAUSSIERE", lows, px[last] < left.min())):
            if new_extreme and len(piv) >= 1:
                i1 = piv[-1]
                checks, noise, crossed = evaluate(kind, i1, last, px, rv, n, cfg, ob, os_)
                if all(ok for _, ok, _ in checks[:-1]):
                    out.append((kind, close.index[i1], close.index[last], px[i1], px[last],
                                rv[i1], rv[last], 0, crossed, noise, True))
    return out


def _frame(data, t):
    """Extrait le DataFrame d'un ticker, quel que soit le format renvoyé par yfinance."""
    cols = data.columns
    if isinstance(cols, pd.MultiIndex):
        for lvl in range(cols.nlevels):
            if t in cols.get_level_values(lvl):
                return data.xs(t, axis=1, level=lvl)
        return pd.DataFrame()
    return data


# ----------------------------- Scan multi-timeframe -----------------------------
COLS = ["Ticker", "TF", "Type", "Date1", "Date2", "Prix1", "Prix2",
        "RSI1", "RSI2", "Age", "Croise50", "Bruit", "Formation"]


def run_scan(tickers, tfs, ob=70, os_=30, batch=100, keep_crossed=False):
    rows, frames = [], {}
    for tf in tfs:
        cfg = TF_CONFIG[tf]
        print(f"\n== {tf} ==", flush=True)
        for k in range(0, len(tickers), batch):
            chunk = tickers[k:k + batch]
            print(f"  {k + 1}-{k + len(chunk)} / {len(tickers)}", flush=True)
            try:
                data = yf.download(chunk, period=cfg["period"], interval=cfg["interval"],
                                   group_by="ticker", auto_adjust=True,
                                   threads=True, progress=False)
            except Exception as e:
                print(f"  [!] lot ignoré: {e}", file=sys.stderr)
                continue
            for t in chunk:
                try:
                    df = _frame(data, t)
                    for r in detect(df, cfg["order"], cfg["min_gap"], cfg["max_gap"],
                                    ob, os_, cfg["max_age"]):
                        rows.append((t, tf) + r)
                        frames[(t, tf)] = df
                except Exception:
                    continue
    res = pd.DataFrame(rows, columns=COLS)
    if not keep_crossed:
        # on élimine les divergences dont le RSI a déjà recroisé 50 depuis le 2e pivot
        res = res[~res["Croise50"]].reset_index(drop=True)
    # même action, même sens, détectée en Daily ET en Weekly = signal fort
    res["Les2"] = False
    if not res.empty:
        res["Les2"] = res.groupby(["Ticker", "Type"])["TF"].transform("nunique") > 1
    return res, frames


# ----------------------------- Diagnostic -----------------------------
def _diag_png(ticker, tf, close, rsi, highs, lows, W=90):
    start = max(0, len(close) - W)
    c, r = close.iloc[start:], rsi.iloc[start:]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})
    ax1.set_title(f"{ticker} · {tf}  (▼ sommets / ▲ creux détectés ; traits = deux derniers)", fontsize=9, loc="left")
    ax1.plot(c.index, c.values, lw=1.2, color="#333")
    ax2.plot(r.index, r.values, lw=1.1, color="#6a3d9a")
    for piv, col, mk in ((highs, "#c0392b", "v"), (lows, "#1e8449", "^")):
        sel = [i for i in piv if i >= start]
        ax1.plot([close.index[i] for i in sel], [close.values[i] for i in sel], mk, color=col, ms=5, ls="")
        if len(piv) >= 2 and piv[-2] >= start:
            a_, b_ = piv[-2], piv[-1]
            ax1.plot([close.index[a_], close.index[b_]], [close.values[a_], close.values[b_]], color=col, lw=2)
            ax2.plot([close.index[a_], close.index[b_]], [rsi.values[a_], rsi.values[b_]], color=col, lw=2, marker="o", ms=4)
    for y, ls, col in ((70, "--", "#c0392b"), (30, "--", "#1e8449"), (60, ":", "#e67e22"),
                       (40, ":", "#e67e22"), (50, ":", "#999")):
        ax2.axhline(y, ls=ls, lw=0.8, color=col)
    ax2.set_ylim(0, 100); ax2.set_yticks([30, 40, 50, 60, 70]); ax2.set_ylabel("RSI 14", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y" if tf == "Weekly" else "%d/%m"))
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def diagnose(ticker, tfs=("Daily", "Weekly"), ob=70, os_=30):
    """Explique règle par règle pourquoi une action est ou n'est pas dans la liste.
    Retourne une liste de (texte, png_bytes) : un élément par timeframe."""
    results = []
    for tf in tfs:
        cfg = TF_CONFIG[tf]
        L = [f"===== {ticker} · {tf} ====="]
        data = yf.download(ticker, period=cfg["period"], interval=cfg["interval"],
                           auto_adjust=True, progress=False)
        df = _frame(data, ticker) if len(data) else pd.DataFrame()
        close = df["Close"].dropna() if "Close" in df else pd.Series(dtype=float)
        if len(close) < 60:
            L.append("Pas assez de données Yahoo pour ce ticker (orthographe ? TSX : SHOP.TO, BBD-B.TO).")
            results.append(("\n".join(L), None))
            continue
        rsi = rsi_wilder(close)
        px, rv, n = close.values, rsi.values, len(close)
        highs = argrelextrema(px, np.greater, order=cfg["order"])[0]
        lows = argrelextrema(px, np.less, order=cfg["order"])[0]
        L.append(f"Dernière bougie : {close.index[-1]:%d/%m/%Y} · prix {px[-1]:.2f} · RSI {rv[-1]:.1f}")
        L.append(f"Règles : {cfg['min_gap']} à {cfg['max_gap']} bougies entre pivots · "
                 f"2e pivot de moins de {cfg['max_age']} bougies · "
                 f"bruit ≥ {NOISE_BEAR_MIN} (baissière) / ≤ {NOISE_BULL_MAX} (haussière)")
        last = n - 1
        left = px[max(0, last - cfg["order"]):last]
        for kind, piv, nom in (("BAISSIERE", highs, "sommets"), ("HAUSSIERE", lows, "creux")):
            bear = kind == "BAISSIERE"
            sing = "sommet" if bear else "creux"
            L.append(f"\n-- Divergence {kind.lower()} --")
            last4 = ", ".join(f"{close.index[i]:%d/%m/%y} (prix {px[i]:.2f}, RSI {rv[i]:.1f})" for i in piv[-4:])
            L.append(f"Derniers {nom} détectés : {last4 or 'aucun'}")
            # -- confirmée
            if len(piv) < 2:
                L.append("✘ Moins de deux pivots : pas de divergence confirmée possible.")
            else:
                checks, _, _ = evaluate(kind, piv[-2], piv[-1], px, rv, n, cfg, ob, os_)
                for name, ok, txt in checks:
                    L.append(("  ✔ " if ok else "  ✘ ") + f"{name} : {txt}")
                failed = [name for name, ok, _ in checks if not ok]
                L.append("➜ CONFIRMÉE, DANS LA LISTE ✔" if not failed
                         else "➜ EXCLUE (confirmée) à cause de : " + ", ".join(failed))
            # -- en formation
            is_ext = bool(px[last] > left.max()) if bear else bool(px[last] < left.min())
            if not is_ext or len(piv) < 1:
                L.append(f"  En formation : non (la dernière bougie n’est pas un nouveau {sing} "
                         f"sur les {cfg['order']} dernières bougies)")
            else:
                checks, _, _ = evaluate(kind, piv[-1], last, px, rv, n, cfg, ob, os_)
                L.append(f"  En formation : la dernière bougie ({close.index[-1]:%d/%m}) est un nouveau {sing} ; test :")
                for name, ok, txt in checks:
                    L.append(("    ✔ " if ok else "    ✘ ") + f"{name} : {txt}")
                failed = [name for name, ok, _ in checks if not ok]
                L.append("  ➜ EN FORMATION ✔ (section « en formation » de la page)" if not failed
                         else "  ➜ non retenue en formation : " + ", ".join(failed))
        L.append("\nNote : un sommet/creux est compté dès que la bougie suivante le confirme (même une seule) ; "
                 "il peut donc disparaître si le prix refait un nouvel extrême. "
                 "La toute dernière bougie n'est jamais un pivot confirmé : elle peut seulement être « en formation ».")
        results.append(("\n".join(L), _diag_png(ticker, tf, close, rsi, highs, lows)))
    return results


# ----------------------------- Graphiques -----------------------------
def make_chart(df, r, ob, os_):
    """Prix (haut) + RSI (bas) avec la divergence tracée. Retourne du PNG base64."""
    weekly = r["TF"] == "Weekly"
    close = df["Close"].dropna()
    rsi = rsi_wilder(close)
    i1 = close.index.get_loc(pd.Timestamp(r["Date1"]))
    i2 = close.index.get_loc(pd.Timestamp(r["Date2"]))
    a_ = max(0, i1 - (20 if weekly else 30))
    b_ = min(len(close), i2 + (10 if weekly else 20))
    c, rr = close.iloc[a_:b_], rsi.iloc[a_:b_]
    d1, d2 = close.index[i1], close.index[i2]
    col = "#c0392b" if r["Type"] == "BAISSIERE" else "#1e8449"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4.4), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})
    ax1.set_title(f"{r['Ticker']} · {r['TF']}", fontsize=9, loc="left")
    ax1.plot(c.index, c.values, lw=1.3, color="#333")
    ax1.plot([d1, d2], [r["Prix1"], r["Prix2"]], color=col, lw=2.2, marker="o", ms=5)
    ax1.set_ylabel("Prix", fontsize=8)
    ax2.plot(rr.index, rr.values, lw=1.2, color="#6a3d9a")
    ax2.axhline(ob, ls="--", lw=0.8, color="#c0392b")
    ax2.axhline(os_, ls="--", lw=0.8, color="#1e8449")
    ax2.axhline(50, ls=":", lw=0.6, color="#999")
    ax2.axhspan(ob, 100, color="#c0392b", alpha=0.07)
    ax2.axhspan(0, os_, color="#1e8449", alpha=0.07)
    ax2.plot([d1, d2], [r["RSI1"], r["RSI2"]], color=col, lw=2.2, marker="o", ms=5)
    bear = r["Type"] == "BAISSIERE"
    seg = rsi.iloc[i1 + 1:i2]
    if len(seg):
        ni = seg.idxmin() if bear else seg.idxmax()          # le « bruit »
        ax2.plot([ni], [seg[ni]], marker="s", ms=5, color="#e67e22", zorder=5)
        ax2.annotate("bruit", (ni, seg[ni]), textcoords="offset points",
                     xytext=(0, -12 if bear else 7), ha="center", fontsize=7, color="#e67e22")
    lim = NOISE_BEAR_MIN if bear else NOISE_BULL_MAX
    ax2.axhline(lim, ls=":", lw=0.9, color="#e67e22")
    ax2.set_ylim(0, 100)
    ax2.set_yticks(sorted({os_, 50, ob, lim}))
    ax2.set_ylabel("RSI 14", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y" if weekly else "%d/%m"))
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def make_charts(res, frames, ob, os_):
    charts = {}
    for _, r in res.iterrows():
        try:
            charts[(r["Ticker"], r["TF"], r["Type"], bool(r["Formation"]))] = make_chart(
                frames[(r["Ticker"], r["TF"])], r, ob, os_)
        except Exception as e:
            print(f"  [!] graphique {r['Ticker']} {r['TF']}: {e}", file=sys.stderr)
    return charts


# ----------------------------- Rapport HTML -----------------------------
CSS = """
:root{--bg:#f5f6f8;--fg:#1a1a1a;--muted:#666;--card:#fff;--border:#bbb;--info:#555;--tfbg:#eef1f5;
--link:#0b5ed7;--shadow:rgba(0,0,0,.12);--bear:#c0392b;--bull:#1e8449;--pend:#b9770e}
html[data-theme="dark"]{--bg:#0f1115;--fg:#e8e8ea;--muted:#9aa0aa;--card:#1a1d23;--border:#3a3f4a;--info:#b4b9c2;
--tfbg:#2a2f38;--link:#6ea8fe;--shadow:rgba(0,0,0,.6);--bear:#ff7b6e;--bull:#4cd08a;--pend:#e0a040;color-scheme:dark}
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:16px;background:var(--bg);color:var(--fg)}
h1{font-size:20px;margin:0 0 4px;padding-right:48px} .sub{color:var(--muted);font-size:13px;margin-bottom:10px}
h2{font-size:17px;margin:26px 0 10px} .bear h2{color:var(--bear)} .bull h2{color:var(--bull)}
.theme{position:fixed;top:12px;right:12px;z-index:10;width:40px;height:40px;border-radius:50%;
border:1px solid var(--border);background:var(--card);color:var(--fg);font-size:18px;cursor:pointer;box-shadow:0 1px 4px var(--shadow)}
.nav{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.nav a{padding:6px 14px;border-radius:999px;border:1px solid var(--border);background:var(--card);color:var(--fg);text-decoration:none;font-size:14px}
.nav a.on{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.filters{display:flex;gap:8px;margin:10px 0 4px;flex-wrap:wrap}
.filters button{border:1px solid var(--border);background:var(--card);color:var(--fg);border-radius:999px;padding:7px 14px;font-size:14px;cursor:pointer}
.filters button.on{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.card{background:var(--card);color:var(--fg);border-radius:10px;box-shadow:0 1px 3px var(--shadow);padding:12px}
.card img{width:100%;height:auto;display:block;margin-top:6px}
html[data-theme="dark"] .card img{filter:invert(1) hue-rotate(180deg)}
.top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.top a{font-size:17px;font-weight:700;color:var(--link);text-decoration:none}
.badges{display:flex;gap:6px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
.tf{background:var(--tfbg);border-radius:6px;padding:2px 7px;font-size:12px;font-weight:600}
.both{background:#f1c40f;color:#222;border-radius:6px;padding:2px 7px;font-size:12px;font-weight:700}
.info{font-size:12.5px;color:var(--info);margin-top:4px;line-height:1.5}
.ok{color:var(--bull);font-weight:700;font-size:12px} .pend{color:var(--pend);font-size:12px}
.none{padding:16px;color:var(--muted);background:var(--card);border-radius:10px}
.form h2{color:var(--pend)}
.typ{border:1px solid currentColor;border-radius:6px;padding:1px 7px;font-size:12px;font-weight:700}
.typ.b{color:var(--bear)} .typ.h{color:var(--bull)}
.hint{font-size:12.5px;color:var(--muted);margin:-4px 0 10px}
"""

THEME_HEAD = """<script>(function(){var t=null;try{t=localStorage.getItem('theme');}catch(e){}
if(!t){t=(window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches)?'dark':'light';}
document.documentElement.setAttribute('data-theme',t);})();</script>"""

JS = """
function setF(f){
  document.querySelectorAll('.filters button').forEach(function(b){b.classList.toggle('on', b.dataset.f===f);});
  document.querySelectorAll('.sec').forEach(function(sec){
    var n=0;
    sec.querySelectorAll('.card').forEach(function(c){
      var show=(f==='all'||c.dataset.tf===f); c.style.display=show?'':'none'; if(show)n++;
    });
    sec.querySelector('.n').textContent=n;
    sec.querySelector('.none').style.display=n?'none':'';
  });
}
document.querySelectorAll('.filters button').forEach(function(b){b.onclick=function(){setF(b.dataset.f);};});

function applyTheme(t){
  document.documentElement.setAttribute('data-theme',t);
  var b=document.getElementById('theme');
  if(b){b.textContent=(t==='dark')?'☀️':'🌙'; b.title=(t==='dark')?'Passer en mode clair':'Passer en mode sombre';}
}
var tb=document.getElementById('theme');
if(tb){tb.onclick=function(){
  var t=(document.documentElement.getAttribute('data-theme')==='dark')?'light':'dark';
  applyTheme(t); try{localStorage.setItem('theme',t);}catch(e){}
};}
applyTheme(document.documentElement.getAttribute('data-theme')||'light');
"""


def section(df, title, cls, charts, show_type=False, hint=""):
    df = df.assign(_w=(df["TF"] == "Weekly")).sort_values(
        ["Les2", "_w", "Age"], ascending=[False, False, True])
    h = [f'<div class="sec {cls}"><h2>{title} (<span class="n">{len(df)}</span>)</h2>',
         f'<div class="none" style="display:{"" if df.empty else "none"}">'
         "Aucune divergence pour ce filtre.</div>"]
    if hint:
        h.append(f'<div class="hint">{hint}</div>')
    h.append('<div class="grid">')
    for _, r in df.iterrows():
        t = html.escape(r["Ticker"])
        sym = tv_symbol(r["Ticker"]).replace(":", "%3A")
        img = charts.get((r["Ticker"], r["TF"], r["Type"], bool(r["Formation"])), "")
        typ = ""
        if show_type:
            typ = ('<span class="typ b">Baissière</span>' if r["Type"] == "BAISSIERE"
                   else '<span class="typ h">Haussière</span>')
        age_txt = "dernière bougie · à confirmer" if r["Formation"] else f'il y a {r["Age"]} barres'
        conf = ('<span class="pend">RSI déjà repassé 50</span>' if r["Croise50"] else "")
        both = '<span class="both">Daily + Weekly</span>' if r["Les2"] else ""
        h.append(
            f'<div class="card" data-tf="{r["TF"]}"><div class="top">'
            f'<a href="https://www.tradingview.com/chart/?symbol={sym}" target="_blank">{t}</a>'
            f'<div class="badges">{typ}{both}<span class="tf">{r["TF"]}</span>{conf}</div></div>'
            f'<div class="info">Prix {r["Prix1"]:.2f} → {r["Prix2"]:.2f} · '
            f'RSI {r["RSI1"]:.1f} → {r["RSI2"]:.1f} · bruit {r["Bruit"]:.1f}<br>'
            f'{pd.Timestamp(r["Date1"]):%d/%m/%Y} → {pd.Timestamp(r["Date2"]):%d/%m/%Y} · '
            f'{age_txt}</div>'
            + (f'<img alt="{t}" src="data:image/png;base64,{img}">' if img else "") + "</div>")
    h.append("</div></div>")
    return "".join(h)


def build_html(res, n_tickers, a, charts):
    conf_df = res[~res["Formation"]]
    form_df = res[res["Formation"]]
    bear = conf_df[conf_df["Type"] == "BAISSIERE"]
    bull = conf_df[conf_df["Type"] == "HAUSSIERE"]
    tfs = " + ".join(a.tfs)
    c = TF_CONFIG[a.tfs[0]]
    gap_txt = f"{c['min_gap']} à {c['max_gap']}"
    market = getattr(a, "market", "us")
    label = html.escape(MARKETS[market])
    suffix = " — TSX" if market == "tsx" else ""
    nav = ""
    if getattr(a, "nav", False):
        nav = ('<div class="nav"><a href="index.html"' + (' class="on"' if market == "us" else "")
               + '>S&amp;P 500 + Nasdaq</a><a href="tsx.html"' + (' class="on"' if market == "tsx" else "")
               + '>TSX</a></div>')
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Divergences RSI{suffix}</title>{THEME_HEAD}<style>{CSS}</style></head><body>
<button id="theme" class="theme" title="Passer en mode sombre">🌙</button>
{nav}<h1>Divergences RSI — surachat / survente{suffix}</h1>
<div class="sub">Scan du {datetime.now():%d/%m/%Y %H:%M} · {n_tickers} actions ({label}) ·
{tfs} · RSI(14) · surachat ≥ {a.overbought:g} / survente ≤ {a.oversold:g}<br>
Sur chaque graphique, la ligne colorée relie les deux pivots sur le prix (haut) et sur le RSI (bas).
Le badge « Daily + Weekly » signale la même divergence sur les deux timeframes.<br>
Les divergences dont le RSI a déjà recroisé 50 depuis le 2e pivot sont éliminées.<br>
Filtres : {gap_txt} bougies entre les deux pivots · bruit (RSI entre les pivots) ≥ {NOISE_BEAR_MIN} pour une baissière, ≤ {NOISE_BULL_MAX} pour une haussière.</div>
<div class="filters"><button data-f="all" class="on">Daily + Weekly</button>
<button data-f="Daily">Daily</button><button data-f="Weekly">Weekly</button></div>
{section(bear, "Divergences baissières (RSI en surachat)", "bear", charts)}
{section(bull, "Divergences haussières (RSI en survente)", "bull", charts)}
{section(form_df, "Divergences en formation (dernière bougie, à confirmer)", "form", charts, True,
         "La dernière bougie serait le 2e pivot : la divergence n'est confirmée que si la bougie suivante "
         "s'éloigne de l'extrême. Elle peut disparaître si le prix refait un nouveau sommet / creux.")}
<script>{JS}</script></body></html>"""


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnose", metavar="TICKER", help="explique pourquoi ce ticker est ou n'est pas dans la liste")
    ap.add_argument("--market", choices=["us", "tsx"], default="us")
    ap.add_argument("--tf", choices=["daily", "weekly", "both"], default="both")
    ap.add_argument("--overbought", type=float, default=70)
    ap.add_argument("--oversold", type=float, default=30)
    ap.add_argument("--keep-crossed", action="store_true",
                    help="garder les divergences dont le RSI a déjà recroisé 50")
    ap.add_argument("--nasdaq-all", action="store_true")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    a.tfs = {"daily": ["Daily"], "weekly": ["Weekly"], "both": ["Daily", "Weekly"]}[a.tf]
    if a.diagnose:
        for (txt, png), tf in zip(diagnose(a.diagnose.upper(), a.tfs, a.overbought, a.oversold), a.tfs):
            print(txt, "\n")
            if png:
                fn = f"diagnostic_{a.diagnose.upper()}_{tf}.png"
                open(fn, "wb").write(png)
                print(f"(graphique : {fn})\n")
        return
    a.nav = True
    a.out = a.out or ("tsx.html" if a.market == "tsx" else "divergences_rsi.html")

    print(f"Chargement de la liste des actions ({MARKETS[a.market]})...")
    tickers = get_universe(a.market, a.nasdaq_all)
    if not tickers:
        sys.exit("Aucun ticker chargé (vérifie ta connexion Internet).")
    print(f"{len(tickers)} actions à scanner.")

    res, frames = run_scan(tickers, a.tfs, a.overbought, a.oversold,
                           a.batch, a.keep_crossed)
    print(f"\n{len(res)} divergences. Création des graphiques...")
    charts = make_charts(res, frames, a.overbought, a.oversold)

    path = os.path.abspath(a.out)
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_html(res, len(tickers), a, charts))
    res.to_csv(os.path.splitext(path)[0] + ".csv", index=False)
    print(f"-> {path}")
    if not a.no_open:
        webbrowser.open("file://" + path)


if __name__ == "__main__":
    main()
