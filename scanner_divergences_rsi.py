#!/usr/bin/env python3
"""
Scanner de divergences RSI - S&P 500 + Nasdaq - timeframes DAILY et WEEKLY uniquement.
Ne garde QUE les divergences où le RSI est en surachat (>=70) ou en survente (<=30).

Lancement :  python3 scanner_divergences_rsi.py
-> génère divergences_rsi.html (un graphique prix + RSI par divergence) et l'ouvre.

Installation (une fois) :
    pip3 install yfinance pandas numpy scipy lxml requests matplotlib

Options :
    --tf daily | weekly | both   (défaut : both)
    --confirmed-only             seulement si le RSI a recroisé 50 après le 2e pivot
    --nasdaq-all                 tout le Nasdaq (~3000 titres) au lieu du Nasdaq-100
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
#   min/max_gap : écart (en barres) autorisé entre les deux pivots
#   max_age : le 2e pivot doit dater de moins de X barres
TF_CONFIG = {
    "Daily":  dict(interval="1d",  period="1y", order=5, min_gap=5, max_gap=60, max_age=10),
    "Weekly": dict(interval="1wk", period="5y", order=3, min_gap=4, max_gap=40, max_age=4),
}


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


def safe(fn, name):
    try:
        return fn()
    except Exception as e:
        print(f"[!] Impossible de charger {name}: {e}", file=sys.stderr)
        return []


# ----------------------------- RSI -----------------------------
def rsi_wilder(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ----------------------------- Détection -----------------------------
def detect(df, order=5, min_gap=5, max_gap=60, ob=70, os_=30, max_age=10):
    """Divergences régulières, RSI en surachat/survente uniquement."""
    close = df["Close"].dropna()
    if len(close) < 60:
        return []
    rsi = rsi_wilder(close)
    px, rv, n = close.values, rsi.values, len(close)
    out = []

    def confirmed(i2, bearish):
        after = rv[i2:]
        return bool(np.any(after < 50) if bearish else np.any(after > 50))

    highs = argrelextrema(px, np.greater, order=order)[0]
    lows = argrelextrema(px, np.less, order=order)[0]

    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if (min_gap <= i2 - i1 <= max_gap and n - 1 - i2 <= max_age
                and not np.isnan(rv[i1]) and not np.isnan(rv[i2])
                and px[i2] > px[i1] and rv[i2] < rv[i1]
                and max(rv[i1], rv[i2]) >= ob):
            out.append(("BAISSIERE", close.index[i1], close.index[i2],
                        px[i1], px[i2], rv[i1], rv[i2], n - 1 - i2, confirmed(i2, True)))

    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if (min_gap <= i2 - i1 <= max_gap and n - 1 - i2 <= max_age
                and not np.isnan(rv[i1]) and not np.isnan(rv[i2])
                and px[i2] < px[i1] and rv[i2] > rv[i1]
                and min(rv[i1], rv[i2]) <= os_):
            out.append(("HAUSSIERE", close.index[i1], close.index[i2],
                        px[i1], px[i2], rv[i1], rv[i2], n - 1 - i2, confirmed(i2, False)))
    return out


# ----------------------------- Scan multi-timeframe -----------------------------
COLS = ["Ticker", "TF", "Type", "Date1", "Date2", "Prix1", "Prix2",
        "RSI1", "RSI2", "Age", "Confirmee"]


def run_scan(tickers, tfs, ob=70, os_=30, batch=100, confirmed_only=False):
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
                    df = data[t] if len(chunk) > 1 else data
                    for r in detect(df, cfg["order"], cfg["min_gap"], cfg["max_gap"],
                                    ob, os_, cfg["max_age"]):
                        rows.append((t, tf) + r)
                        frames[(t, tf)] = df
                except Exception:
                    continue
    res = pd.DataFrame(rows, columns=COLS)
    if confirmed_only:
        res = res[res["Confirmee"]]
    # même action, même sens, détectée en Daily ET en Weekly = signal fort
    res["Les2"] = False
    if not res.empty:
        res["Les2"] = res.groupby(["Ticker", "Type"])["TF"].transform("nunique") > 1
    return res, frames


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
    ax2.set_ylim(0, 100)
    ax2.set_yticks([os_, 50, ob])
    ax2.set_ylabel("RSI 14", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y" if weekly else "%d/%m"))
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def make_charts(res, frames, ob, os_):
    charts = {}
    for _, r in res.iterrows():
        try:
            charts[(r["Ticker"], r["TF"], r["Type"])] = make_chart(
                frames[(r["Ticker"], r["TF"])], r, ob, os_)
        except Exception as e:
            print(f"  [!] graphique {r['Ticker']} {r['TF']}: {e}", file=sys.stderr)
    return charts


# ----------------------------- Rapport HTML -----------------------------
CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:16px;background:#f5f6f8;color:#1a1a1a}
h1{font-size:20px;margin:0 0 4px} .sub{color:#666;font-size:13px;margin-bottom:10px}
h2{font-size:17px;margin:26px 0 10px} .bear h2{color:#c0392b} .bull h2{color:#1e8449}
.filters{display:flex;gap:8px;margin:10px 0 4px;flex-wrap:wrap}
.filters button{border:1px solid #bbb;background:#fff;color:#1a1a1a;border-radius:999px;padding:7px 14px;font-size:14px;cursor:pointer}
.filters button.on{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.card{background:#fff;color:#1a1a1a;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.12);padding:12px}
.card img{width:100%;height:auto;display:block;margin-top:6px}
.top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.top a{font-size:17px;font-weight:700;color:#0b5ed7;text-decoration:none}
.badges{display:flex;gap:6px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
.tf{background:#eef1f5;border-radius:6px;padding:2px 7px;font-size:12px;font-weight:600}
.both{background:#f1c40f;color:#222;border-radius:6px;padding:2px 7px;font-size:12px;font-weight:700}
.info{font-size:12.5px;color:#555;margin-top:4px;line-height:1.5}
.ok{color:#1e8449;font-weight:700;font-size:12px} .pend{color:#b9770e;font-size:12px}
.none{padding:16px;color:#888;background:#fff;border-radius:10px}
"""

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
"""


def section(df, title, cls, charts):
    df = df.assign(_w=(df["TF"] == "Weekly")).sort_values(
        ["Les2", "_w", "Age"], ascending=[False, False, True])
    h = [f'<div class="sec {cls}"><h2>{title} (<span class="n">{len(df)}</span>)</h2>',
         f'<div class="none" style="display:{"" if df.empty else "none"}">'
         "Aucune divergence pour ce filtre.</div>", '<div class="grid">']
    for _, r in df.iterrows():
        t = html.escape(r["Ticker"])
        img = charts.get((r["Ticker"], r["TF"], r["Type"]), "")
        conf = ('<span class="ok">✔ confirmée</span>' if r["Confirmee"]
                else '<span class="pend">non confirmée</span>')
        both = '<span class="both">Daily + Weekly</span>' if r["Les2"] else ""
        h.append(
            f'<div class="card" data-tf="{r["TF"]}"><div class="top">'
            f'<a href="https://www.tradingview.com/chart/?symbol={t}" target="_blank">{t}</a>'
            f'<div class="badges">{both}<span class="tf">{r["TF"]}</span>{conf}</div></div>'
            f'<div class="info">Prix {r["Prix1"]:.2f} → {r["Prix2"]:.2f} · '
            f'RSI {r["RSI1"]:.1f} → {r["RSI2"]:.1f}<br>'
            f'{pd.Timestamp(r["Date1"]):%d/%m/%Y} → {pd.Timestamp(r["Date2"]):%d/%m/%Y} · '
            f'il y a {r["Age"]} barres</div>'
            + (f'<img alt="{t}" src="data:image/png;base64,{img}">' if img else "") + "</div>")
    h.append("</div></div>")
    return "".join(h)


def build_html(res, n_tickers, a, charts):
    bear = res[res["Type"] == "BAISSIERE"]
    bull = res[res["Type"] == "HAUSSIERE"]
    tfs = " + ".join(a.tfs)
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Divergences RSI</title><style>{CSS}</style></head><body>
<h1>Divergences RSI — surachat / survente</h1>
<div class="sub">Scan du {datetime.now():%d/%m/%Y %H:%M} · {n_tickers} actions (S&amp;P 500 + Nasdaq) ·
{tfs} · RSI(14) · surachat ≥ {a.overbought:g} / survente ≤ {a.oversold:g}<br>
Sur chaque graphique, la ligne colorée relie les deux pivots sur le prix (haut) et sur le RSI (bas).
Le badge « Daily + Weekly » signale la même divergence sur les deux timeframes.</div>
<div class="filters"><button data-f="all" class="on">Daily + Weekly</button>
<button data-f="Daily">Daily</button><button data-f="Weekly">Weekly</button></div>
{section(bear, "Divergences baissières (RSI en surachat)", "bear", charts)}
{section(bull, "Divergences haussières (RSI en survente)", "bull", charts)}
<script>{JS}</script></body></html>"""


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=["daily", "weekly", "both"], default="both")
    ap.add_argument("--overbought", type=float, default=70)
    ap.add_argument("--oversold", type=float, default=30)
    ap.add_argument("--confirmed-only", action="store_true")
    ap.add_argument("--nasdaq-all", action="store_true")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--out", default="divergences_rsi.html")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    a.tfs = {"daily": ["Daily"], "weekly": ["Weekly"], "both": ["Daily", "Weekly"]}[a.tf]

    print("Chargement des listes S&P 500 / Nasdaq...")
    tickers = set(safe(get_sp500, "S&P 500"))
    tickers |= set(safe(get_nasdaq_all if a.nasdaq_all else get_nasdaq100, "Nasdaq"))
    tickers = sorted(t for t in tickers if t and t != "nan")
    if not tickers:
        sys.exit("Aucun ticker chargé (vérifie ta connexion Internet).")
    print(f"{len(tickers)} actions à scanner.")

    res, frames = run_scan(tickers, a.tfs, a.overbought, a.oversold,
                           a.batch, a.confirmed_only)
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
