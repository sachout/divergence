#!/usr/bin/env python3
"""
Scanner 0DTE - divergences RSI et CDV sur le S&P 500, le Nasdaq-100 et le Dow Jones
Timeframes : 5 minutes, 15 minutes, 1 heure.

Instruments (ETF liquides, avec un vrai volume intraday) :
    SPY = S&P 500 · QQQ = Nasdaq-100 · DIA = Dow Jones

- RSI : mêmes règles que le scanner d'actions (zone 70/30, bruit 60/40, 5 à 16 bougies,
        RSI pas recroisé 50), appliquées sur des bougies COMPLÈTES.
- CDV : « Cumulative Delta Volume » ESTIMÉ à partir des bougies
        (volume x position du corps dans la bougie). Ce n'est pas le delta réel
        tick par tick : il peut différer de celui de TradingView.
- Divergence « en formation » : la dernière bougie complète serait le 2e pivot.
- Une notification (ntfy.sh) est envoyée UNE fois par nouveau signal si NTFY_TOPIC est défini.

Lancement local :  python3 scanner_0dte.py --force
-> écrit 0dte.html (et state_0dte.json). Le marché doit être ouvert, sinon ajoute --force
   (--force ignore l'heure du marché, montre aussi des signaux plus anciens, sans notification : utile pour tester).
"""
import argparse
import base64
import html
import io
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.signal import argrelextrema

import scanner_divergences_rsi as base   # RSI, règles de divergence, CSS / thème de la page

warnings.filterwarnings("ignore")
ET = ZoneInfo("America/New_York")

INSTRUMENTS = {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq-100 (QQQ)", "DIA": "Dow Jones (DIA)"}

# Réglages par timeframe (modifiables)
#   order   : nb de bougies de chaque côté pour valider un sommet/creux
#   min/max_gap : écart (en bougies) entre les deux pivots
#   max_age : le 2e pivot doit dater de moins de X bougies (signal « en temps utile »)
TF0 = {
    "5m":  dict(label="5 min",   interval="5m",  period="5d",  minutes=5,  order=3, min_gap=5, max_gap=16, max_age=6),
    "15m": dict(label="15 min",  interval="15m", period="10d", minutes=15, order=3, min_gap=5, max_gap=16, max_age=4),
    "1h":  dict(label="1 heure", interval="1h",  period="1mo", minutes=60, order=3, min_gap=5, max_gap=16, max_age=3),
}
OB, OS = 70, 30            # surachat / survente (RSI)
ELIMINATE_CROSSED = True   # éliminer si le RSI a déjà recroisé 50 depuis le 2e pivot
KIND_FR = {"BAISSIERE": "Baissière", "HAUSSIERE": "Haussière"}


# ----------------------------- Données -----------------------------
def to_et(df):
    if df.empty:
        return df
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df = df.copy()
    df.index = idx.tz_convert(ET)
    return df


def fetch(tf):
    """Télécharge les bougies des 3 ETF pour un timeframe. Retourne {symbole: DataFrame}."""
    cfg = TF0[tf]
    data = yf.download(list(INSTRUMENTS), period=cfg["period"], interval=cfg["interval"],
                       group_by="ticker", auto_adjust=True, prepost=False,
                       threads=True, progress=False)
    out = {}
    for sym in INSTRUMENTS:
        try:
            df = base._frame(data, sym)
            df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
            df["Volume"] = df["Volume"].fillna(0.0)
            out[sym] = to_et(df)
        except Exception as e:
            print(f"  [!] {sym} {tf}: {e}", file=sys.stderr)
            out[sym] = pd.DataFrame()
    return out


def drop_incomplete(df, minutes, now):
    """Retire la bougie en cours : on ne travaille que sur des bougies terminées."""
    if df.empty:
        return df
    ends = pd.Series(df.index + pd.Timedelta(minutes=minutes), index=df.index)
    closes = pd.Series(df.index.normalize() + pd.Timedelta(hours=16), index=df.index)
    ends = ends.where(ends <= closes, closes)       # la dernière bougie du jour s'arrête à 16:00
    return df[ends <= now]


def est_delta(df):
    """Delta estimé par bougie : volume x (clôture - ouverture) / (haut - bas)."""
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    return (df["Volume"] * (df["Close"] - df["Open"]) / rng).fillna(0.0)


# ----------------------------- Analyse -----------------------------
def analyze(df, tf, now, sym="", age_factor=1):
    """Analyse un instrument sur un timeframe. Retourne None si pas assez de données.
    age_factor > 1 (mode --force) : montre aussi des signaux plus anciens, pour tester."""
    cfg = dict(TF0[tf], max_age=TF0[tf]["max_age"] * age_factor)
    df = drop_incomplete(df, cfg["minutes"], now)
    if len(df) < 60:
        return None
    close = df["Close"]
    rsi = base.rsi_wilder(close)
    cdv = est_delta(df).cumsum()
    px, rv, cv, n = close.values, rsi.values, cdv.values, len(close)
    last_end = df.index[-1] + timedelta(minutes=cfg["minutes"])
    fresh = (now - last_end) <= timedelta(minutes=3 * cfg["minutes"] + 5)

    highs = argrelextrema(px, np.greater, order=cfg["order"])[0]
    lows = argrelextrema(px, np.less, order=cfg["order"])[0]
    last = n - 1
    left = px[max(0, last - cfg["order"]):last]
    signals = []
    for kind, piv in (("BAISSIERE", highs), ("HAUSSIERE", lows)):
        bear = kind == "BAISSIERE"
        pairs = []
        if len(piv) >= 2:
            pairs.append((int(piv[-2]), int(piv[-1]), False))                  # divergence confirmée
        new_ext = bool(px[last] > left.max()) if bear else bool(px[last] < left.min())
        if new_ext and len(piv) >= 1:
            pairs.append((int(piv[-1]), last, True))                            # en formation
        for i1, i2, formation in pairs:
            checks, noise, crossed = base.evaluate(kind, i1, i2, px, rv, n, cfg, OB, OS)
            rsi_ok = all(ok for _, ok, _ in checks[:-1]) and not (crossed and ELIMINATE_CROSSED)
            gap_ok = cfg["min_gap"] <= i2 - i1 <= cfg["max_gap"]
            age_ok = (n - 1 - i2) <= cfg["max_age"]
            price_ok = bool(px[i2] > px[i1]) if bear else bool(px[i2] < px[i1])
            cdv_move = bool(cv[i2] < cv[i1]) if bear else bool(cv[i2] > cv[i1])
            cdv_ok = gap_ok and age_ok and price_ok and cdv_move
            if not (rsi_ok or cdv_ok):
                continue
            signals.append(dict(
                sym=None, tf=tf, kind=kind, formation=formation, rsi=bool(rsi_ok), cdv=bool(cdv_ok),
                t1=df.index[i1], t2=df.index[i2], age=n - 1 - i2,
                px1=float(px[i1]), px2=float(px[i2]), r1=float(rv[i1]), r2=float(rv[i2]),
                c1=float(cv[i1]), c2=float(cv[i2]), noise=float(noise),
                old=bool((n - 1 - i2) > TF0[tf]["max_age"]),
                png=chart_png(sym, tf, df.index, px, rv, cv, i1, i2, kind, rsi_ok, cdv_ok)))
    return dict(last_bar=df.index[-1], last_end=last_end, price=float(px[-1]), rsi=float(rv[-1]),
                fresh=bool(fresh), signals=signals)


# ----------------------------- Graphique (même style que la page des actions) -----------------------------
def chart_png(sym, tf, times, px, rv, cv, i1, i2, kind, rsi_ok, cdv_ok):
    """Prix (haut), RSI avec bruit (milieu), CDV estimé (bas). PNG transparent en base64."""
    n = len(px)
    a = max(0, i1 - 20)
    xs = np.arange(a, n)
    bear = kind == "BAISSIERE"
    col = "#c0392b" if bear else "#1e8449"
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 5.8), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 2, 2], "hspace": 0.1})
    ax1.set_title(f"{sym} · {TF0[tf]['label']}", fontsize=9, loc="left")
    ax1.plot(xs, px[a:], lw=1.3, color="#333")
    ax1.plot([i1, i2], [px[i1], px[i2]], color=col, lw=2.2, marker="o", ms=5)
    ax1.set_ylabel("Prix", fontsize=8)

    # RSI + bruit
    ax2.plot(xs, rv[a:], lw=1.2, color="#6a3d9a")
    ax2.axhline(OB, ls="--", lw=0.8, color="#c0392b")
    ax2.axhline(OS, ls="--", lw=0.8, color="#1e8449")
    ax2.axhline(50, ls=":", lw=0.6, color="#999")
    ax2.axhspan(OB, 100, color="#c0392b", alpha=0.07)
    ax2.axhspan(0, OS, color="#1e8449", alpha=0.07)
    ax2.plot([i1, i2], [rv[i1], rv[i2]], color=col if rsi_ok else "#999", lw=2.2, marker="o", ms=5)
    seg = rv[i1 + 1:i2]
    if len(seg):
        k = int(np.argmin(seg) if bear else np.argmax(seg))
        ax2.plot([i1 + 1 + k], [seg[k]], marker="s", ms=5, color="#e67e22", zorder=5)
        ax2.annotate("bruit", (i1 + 1 + k, seg[k]), textcoords="offset points",
                     xytext=(0, -12 if bear else 7), ha="center", fontsize=7, color="#e67e22")
    lim = base.NOISE_BEAR_MIN if bear else base.NOISE_BULL_MAX
    ax2.axhline(lim, ls=":", lw=0.9, color="#e67e22")
    ax2.set_ylim(0, 100)
    ax2.set_yticks(sorted({OS, 50, OB, lim}))
    ax2.set_ylabel("RSI 14", fontsize=8)

    # CDV estimé
    ax3.plot(xs, cv[a:], lw=1.2, color="#2e9fb3")
    ax3.plot([i1, i2], [cv[i1], cv[i2]], color=col if cdv_ok else "#999", lw=2.2, marker="o", ms=5)
    ax3.set_ylabel("CDV (est.)", fontsize=8)
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(-3, 5))

    # axe du temps : positions = numéro de bougie, étiquettes = heure
    multi = times[a].date() != times[-1].date()
    ticks = sorted(set(np.linspace(a, n - 1, 5).astype(int)))
    ax3.set_xticks(ticks)
    ax3.set_xticklabels([times[i].strftime("%d/%m %H:%M" if multi else "%H:%M") for i in ticks])
    for ax in (ax1, ax2, ax3):
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------- Page HTML -----------------------------
EXTRA_CSS = """
.tbl{width:100%;border-collapse:collapse;background:var(--card);border-radius:10px;overflow:hidden;box-shadow:0 1px 3px var(--shadow)}
.tbl th,.tbl td{padding:8px 10px;text-align:center;font-size:13px;border-bottom:1px solid var(--border)}
.tbl th:first-child,.tbl td:first-child{text-align:left;font-weight:700}
.tbl small{display:block;color:var(--muted);font-size:11px}
.sig{display:inline-block;margin:1px 2px;padding:1px 6px;border-radius:6px;font-size:11px;font-weight:700;border:1px solid currentColor}
.sig.b{color:var(--bear)} .sig.h{color:var(--bull)}
.warn{background:var(--card);border-left:4px solid var(--pend);padding:8px 12px;border-radius:6px;font-size:12.5px;color:var(--info);margin:10px 0}
.stale{color:var(--bear);font-weight:700}
.nav a.on{background:var(--fg);color:var(--bg)}
"""

AGE_JS = """
(function(){
  var el=document.getElementById('age'); var ts=el?Date.parse(el.dataset.ts):NaN;
  function upd(){ if(!el||isNaN(ts))return; var m=Math.round((Date.now()-ts)/60000);
    el.textContent='il y a '+m+' min'; el.className=(m>12)?'stale':''; }
  upd(); setInterval(upd,30000);
})();
"""
RELOAD_JS = """
setTimeout(function(){ location.replace(location.pathname+'?t='+Date.now()); }, 60000);
"""


def _fmt_cdv(v):
    a = abs(v)
    return f"{v / 1e6:.2f} M" if a >= 1e6 else (f"{v / 1e3:.0f} k" if a >= 1e3 else f"{v:.0f}")


def build_page(results, now, embed=False):
    rows = []
    for sym, name in INSTRUMENTS.items():
        tds = [f"<td>{html.escape(name)}</td>"]
        for tf in TF0:
            r = results.get((sym, tf))
            if not r:
                tds.append("<td>—<small>pas de données</small></td>")
                continue
            badges = ""
            for s in r["signals"]:
                kind = "b" if s["kind"] == "BAISSIERE" else "h"
                arrow = "▼" if kind == "b" else "▲"
                what = "RSI+CDV" if s["rsi"] and s["cdv"] else ("RSI" if s["rsi"] else "CDV")
                badges += f'<span class="sig {kind}">{arrow} {what}{" ◌" if s["formation"] else ""}</span>'
            state = "" if r["fresh"] else '<span class="stale">données anciennes</span> · '
            tds.append(f"<td>{badges or '·'}<small>{state}{r['last_bar']:%H:%M} · {r['price']:.2f} · RSI {r['rsi']:.0f}</small></td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    head = "<tr><th></th>" + "".join(f"<th>{TF0[t]['label']}</th>" for t in TF0) + "</tr>"

    cards = []
    allsig = []
    for (sym, tf), r in results.items():
        if r and r["fresh"]:
            for s in r["signals"]:
                allsig.append(dict(s, sym=sym))
    allsig.sort(key=lambda s: (not (s["rsi"] and s["cdv"]), s["age"], list(TF0).index(s["tf"])))
    for s in allsig:
        kind = "b" if s["kind"] == "BAISSIERE" else "h"
        both = '<span class="both">RSI + CDV</span>' if s["rsi"] and s["cdv"] else ""
        which = "" if both else f'<span class="tf">{"RSI" if s["rsi"] else "CDV"}</span>'
        status = '<span class="pend">en formation · à confirmer</span>' if s["formation"] else '<span class="ok">confirmée</span>'
        age_txt = "dernière bougie" if s["formation"] else "il y a %d bougies" % s["age"]
        oldb = '<span class="pend">ancien · test</span>' if s.get("old") else ""
        img = '<img alt="%s" src="data:image/png;base64,%s">' % (s["sym"], s["png"])
        cards.append(
            f'<div class="card"><div class="top"><a href="https://www.tradingview.com/chart/?symbol={s["sym"]}" '
            f'target="_blank">{s["sym"]} · {TF0[s["tf"]]["label"]}</a>'
            f'<div class="badges"><span class="typ {kind}">{KIND_FR[s["kind"]]}</span>{both}{which}{status}{oldb}</div></div>'
            f'<div class="info">Prix {s["px1"]:.2f} → {s["px2"]:.2f} · RSI {s["r1"]:.1f} → {s["r2"]:.1f} · '
            f'CDV {_fmt_cdv(s["c1"])} → {_fmt_cdv(s["c2"])}<br>'
            f'{s["t1"]:%d/%m %H:%M} → {s["t2"]:%d/%m %H:%M} ET · '
            f'{age_txt}</div>{img}</div>')
    cards_html = ('<div class="grid">' + "".join(cards) + "</div>") if cards else \
        '<div class="none">Aucune divergence active pour le moment.</div>'

    iso = now.isoformat()
    nav = "" if embed else ('<div class="nav"><a href="index.html">S&amp;P 500 + Nasdaq</a><a href="tsx.html">TSX</a>'
                            '<a href="0dte.html" class="on">0DTE</a></div>')
    how = ("actualisation automatique toutes les 60 s tant que cette page est ouverte" if embed
           else "scan GitHub planifié toutes les 5 min (sans garantie d'horaire)")
    js = AGE_JS if embed else AGE_JS + RELOAD_JS
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Divergences 0DTE</title>{base.THEME_HEAD}<style>{base.CSS}{EXTRA_CSS}</style></head><body>
<button id="theme" class="theme" title="Passer en mode sombre">🌙</button>
{nav}<h1>Divergences 0DTE — RSI &amp; CDV</h1>
<div class="sub">Dernier scan : {now:%d/%m/%Y %H:%M:%S} ET · <span id="age" data-ts="{iso}">…</span> ·
SPY (S&amp;P 500), QQQ (Nasdaq-100), DIA (Dow Jones) · bougies complètes uniquement</div>
<div class="warn">⚠ Données Yahoo gratuites, éventuellement retardées ; {how}.
CDV <b>estimé</b> depuis les bougies, pas le delta réel tick par tick.
Vérifie toujours sur ton graphique avant d'agir. Ceci n'est pas un conseil financier.</div>
<table class="tbl">{head}{"".join(rows)}</table>
<h2>Signaux actifs ({len(cards)})</h2>
<div class="hint">▲ haussière · ▼ baissière · ◌ en formation (dernière bougie, à confirmer).
« RSI + CDV » = les deux divergences au même endroit.</div>
{cards_html}
<script>{base.JS}</script><script>{js}</script></body></html>"""


# ----------------------------- Notifications (ntfy.sh) + état -----------------------------
def load_state(path, now):
    day = f"{now:%Y-%m-%d}"
    try:
        st = json.load(open(path, encoding="utf-8"))
        if st.get("day") == day:
            return st
    except Exception:
        pass
    return {"day": day, "sent": []}


def signal_key(s):
    what = ("RSI" if s["rsi"] else "") + ("CDV" if s["cdv"] else "")
    return f'{s["sym"]}|{s["tf"]}|{s["kind"]}|{s["t2"].isoformat()}|{"F" if s["formation"] else "C"}|{what}'


def notify(topic, s, page_url):
    arrow = "🔻" if s["kind"] == "BAISSIERE" else "🔼"
    what = "RSI + CDV" if s["rsi"] and s["cdv"] else ("RSI" if s["rsi"] else "CDV")
    title = f'{arrow} {s["sym"]} {TF0[s["tf"]]["label"]} · {KIND_FR[s["kind"]]} {what}'
    status = "en formation (à confirmer)" if s["formation"] else "confirmée"
    msg = (f'Divergence {status} · prix {s["px1"]:.2f} → {s["px2"]:.2f} · RSI {s["r1"]:.0f} → {s["r2"]:.0f} · '
           f'bougie {s["t2"]:%H:%M} ET')
    body = {"topic": topic, "title": title, "message": msg, "priority": 4,
            "tags": ["chart_with_downwards_trend" if s["kind"] == "BAISSIERE" else "chart_with_upwards_trend"]}
    if page_url:
        body["click"] = page_url
    requests.post("https://ntfy.sh/", data=json.dumps(body).encode("utf-8"), timeout=15)


# ----------------------------- Main -----------------------------
def market_open(now):
    hm = now.hour * 100 + now.minute
    return now.weekday() < 5 and 930 <= hm <= 1610


def scan_once(now, force=False, state=None, topic="", page_url="", embed=False):
    """Un scan complet des 3 instruments x 3 timeframes.
    Retourne (page_html, nombre_de_signaux_actifs, nouveaux_signaux) ; page_html = None si aucune donnée.
    state : dict {"day", "sent"} pour ne notifier qu'une fois chaque signal (None = pas de notification)."""
    results = {}
    for tf in TF0:
        try:
            dfs = fetch(tf)
        except Exception as e:
            print(f"[!] téléchargement {tf} impossible: {e}", file=sys.stderr)
            dfs = {}
        for sym in INSTRUMENTS:
            df = dfs.get(sym, pd.DataFrame())
            try:
                r = analyze(df, tf, now, sym, 10 if force else 1) if len(df) else None
                if r and force:
                    r["fresh"] = True
                results[(sym, tf)] = r
            except Exception as e:
                print(f"[!] analyse {sym} {tf}: {e}", file=sys.stderr)
                results[(sym, tf)] = None
    if not any(results.values()):
        return None, 0, []

    new = []
    for (sym, tf), r in results.items():
        if not r:
            continue
        for sg in r["signals"]:
            sg["sym"] = sym
        if state is None or not r["fresh"]:
            continue
        for sg in r["signals"]:
            if sg["old"]:
                continue                      # signal ancien (mode test) : affiché mais pas notifié
            key = signal_key(sg)
            if key not in state["sent"]:
                state["sent"].append(key)
                new.append(sg)
    if new and topic:
        for sg in new:
            try:
                notify(topic, sg, page_url)
            except Exception as e:
                print(f"[!] notification: {e}", file=sys.stderr)
    n_active = sum(len(r["signals"]) for r in results.values() if r)
    return build_page(results, now, embed), n_active, new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="0dte.html")
    ap.add_argument("--state", default="state_0dte.json")
    ap.add_argument("--force", action="store_true", help="mode test : ignore l'heure du marché et la fraîcheur, montre aussi les signaux plus anciens (x10), sans notification")
    ap.add_argument("--no-notify", action="store_true")
    a = ap.parse_args()

    now = datetime.now(ET)
    if not a.force and not market_open(now):
        print(f"Marché fermé ({now:%a %H:%M} ET) : rien à faire.")
        return
    state = load_state(a.state, now)
    topic = "" if a.no_notify else os.environ.get("NTFY_TOPIC", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    page_url = f"https://{repo.split('/')[0]}.github.io/{repo.split('/')[1]}/0dte.html" if "/" in repo else ""
    page, n_active, new = scan_once(now, a.force, state, topic, page_url)
    if page is None:
        print("Aucune donnée disponible : page non modifiée.")
        return
    print(f"{n_active} signaux actifs, {len(new)} nouveau(x).")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(page)
    with open(a.state, "w", encoding="utf-8") as f:
        json.dump(state, f)
    print(f"-> {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
