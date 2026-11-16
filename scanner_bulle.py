#!/usr/bin/env python3
"""
Stratégie « bulle » (bandes de Bollinger) pour le 0DTE.

SIGNAL (sur une bougie VALIDÉE, c'est-à-dire terminée) :
    1) la bougie sort COMPLÈTEMENT des bandes de Bollinger :
         - au-dessus : son plus bas (mèche basse incluse) est au-dessus de la bande haute
         - en dessous : son plus haut (mèche haute incluse) est sous la bande basse
       (variante « corps seul » : seul le corps doit être hors de la bande ; la mèche peut y rentrer)
    2) sa mèche ou son corps est le PLUS HAUT (ou le plus bas) du mouvement :
         le plus haut / plus bas de la bougie dépasse tous ceux des N bougies précédentes.
    Seulement dans ces cas-là, un signal est émis :
         mode « retournement » (par défaut) : sortie par le haut -> VENTE ; sortie par le bas -> ACHAT
         mode « continuation »                : sortie par le haut -> ACHAT ; sortie par le bas -> VENTE

Option first_only : un seul signal par séquence de bulles consécutives du même côté.
Réglages par défaut (modifiables) : Bollinger 20 périodes, écart-type 2, extrême sur 20 bougies, bougies de 5 min.
"""
import base64
import html
import io
import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import scanner_0dte as z
import scanner_divergences_rsi as base
import scanner_opr as zo           # bougies, noms, styles communs

warnings.filterwarnings("ignore")
ET = z.ET

CFG = dict(candle_minutes=5, bb_n=20, bb_k=2.0, lookback=20, wick_rule="full", mode="reversal", first_only=False)
INTERVAL = {1: ("1m", "5d"), 5: ("5m", "5d"), 15: ("15m", "5d"), 60: ("1h", "1mo")}


# ----------------------------- Bollinger + règles -----------------------------
def bollinger(close, n, k):
    """Bandes de Bollinger classiques : moyenne mobile simple et écart-type de population (comme TradingView)."""
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def analyze_symbol(sym, df, now, cfg):
    cm = cfg["candle_minutes"]
    df = z.drop_incomplete(df, cm, now) if df is not None and len(df) else df
    if df is None or len(df) < cfg["bb_n"] + cfg["lookback"] + 5:
        return None
    mid, up, lo = (b.values for b in bollinger(df["Close"], cfg["bb_n"], cfg["bb_k"]))
    O, H, L, C = (df[c].values for c in ("Open", "High", "Low", "Close"))
    n = len(df)
    day = df.index[-1].date()
    first = int(np.argmax(np.array(df.index.date == day)))        # première bougie de la dernière séance
    lb = cfg["lookback"]
    signals, outs = [], []
    prev_sig = (None, None)                                         # (bougie, côté) du dernier signal
    for i in range(max(first, cfg["bb_n"], lb), n):
        if np.isnan(up[i]):
            continue
        if cfg["wick_rule"] == "full":                              # bougie ENTIÈRE hors des bandes
            out_up, out_dn = L[i] > up[i], H[i] < lo[i]
        else:                                                       # corps seul hors des bandes
            out_up, out_dn = min(O[i], C[i]) > up[i], max(O[i], C[i]) < lo[i]
        if not (out_up or out_dn):
            continue
        extreme = (H[i] >= H[i - lb:i].max()) if out_up else (L[i] <= L[i - lb:i].min())
        width = float(up[i] - lo[i]) or 1.0
        reach = float((H[i] - up[i]) if out_up else (lo[i] - L[i]))   # de combien le point extrême dépasse la bande
        item = dict(i=i - first, t=df.index[i], side="UP" if out_up else "DOWN", price=float(C[i]),
                    band=float(up[i] if out_up else lo[i]), reach=reach, reach_pct=100.0 * reach / width)
        if not extreme:
            outs.append(item)                                       # sortie complète MAIS pas à l'extrême : pas de signal
            continue
        in_sequence = prev_sig == (i - 1, item["side"])                # bulle qui fait suite à une bulle du même côté
        prev_sig = (i, item["side"])
        if cfg.get("first_only") and in_sequence:
            continue                                                # option : un seul signal par séquence de bulles
        reversal = cfg["mode"] == "reversal"
        item["dir"] = ("SELL" if out_up else "BUY") if reversal else ("BUY" if out_up else "SELL")
        signals.append(item)
    sess = df.iloc[first:]
    bands = (mid[first:], up[first:], lo[first:])
    last = float(C[-1])
    res = dict(sym=sym, day=day, sess=sess, bands=bands, signals=signals, outs=outs,
               last_close=last, last_bar=df.index[-1],
               status=("Clôture au-dessus de la bande haute" if last > up[-1]
                       else "Clôture sous la bande basse" if last < lo[-1] else "Dans les bandes"))
    res["png"] = chart_png(sym, res, cfg)
    return res


# ----------------------------- Graphique -----------------------------
def chart_png(sym, r, cfg):
    sess = r["sess"]
    n = len(sess)
    o, h, l, c = (sess[k].values for k in ("Open", "High", "Low", "Close"))
    mid, up, lo = r["bands"]
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    rngp = float(np.nanmax(h) - np.nanmin(l)) or 1.0
    ax.fill_between(x, lo, up, color="#4a78d4", alpha=0.10)
    ax.plot(x, up, color="#4a78d4", lw=1.1)
    ax.plot(x, lo, color="#4a78d4", lw=1.1)
    ax.plot(x, mid, color="#4a78d4", lw=0.8, ls="--")
    for i in range(n):
        col = "#1e8449" if c[i] >= o[i] else "#c0392b"
        ax.vlines(i, l[i], h[i], color=col, lw=0.8)
        ax.add_patch(Rectangle((i - 0.32, min(o[i], c[i])), 0.64, max(abs(c[i] - o[i]), rngp * 0.002), color=col))
    for e in r["outs"]:                                    # sortie complète sans extrême : point gris
        ax.plot(e["i"], (h[e["i"]] + rngp * 0.015) if e["side"] == "UP" else (l[e["i"]] - rngp * 0.015),
                "o", color="#999", ms=4)
    for s in r["signals"]:
        i = s["i"]
        if s["dir"] == "BUY":
            ax.annotate("ACHAT", (i, l[i]), textcoords="offset points", xytext=(0, -24), ha="center", fontsize=8,
                        fontweight="bold", color="#1e8449", arrowprops=dict(arrowstyle="-|>", color="#1e8449", lw=1.6))
        else:
            ax.annotate("VENTE", (i, h[i]), textcoords="offset points", xytext=(0, 24), ha="center", fontsize=8,
                        fontweight="bold", color="#c0392b", arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.6))
    ax.set_title(f"{sym} · {sess.index[0]:%d/%m} · bandes de Bollinger {cfg['bb_n']} / {cfg['bb_k']:g}", fontsize=9, loc="left")
    ticks = sorted(set(np.linspace(0, n - 1, 6).astype(int)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([sess.index[k].strftime("%H:%M") for k in ticks])
    ax.set_xlim(-1, n + 0.5)
    ax.margins(y=0.14)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------- Page HTML -----------------------------
def build_page(results, now, cfg, embed=True):
    rows, cards, allsig = [], [], []
    for sym, r in results.items():
        name = html.escape(zo.NAMES.get(sym, sym))
        if not r:
            rows.append("<tr><td>%s</td><td colspan='3'>pas assez de données</td></tr>" % name)
            continue
        for s in r["signals"]:
            allsig.append(dict(s, sym=sym, r=r))
        last = r["signals"][-1] if r["signals"] else None
        last_txt = "—"
        if last:
            cls = "buy" if last["dir"] == "BUY" else "sell"
            last_txt = '<span class="chip %s">%s</span> %s' % (cls, "ACHAT" if last["dir"] == "BUY" else "VENTE",
                                                               last["t"].strftime("%H:%M"))
        rows.append("<tr><td>%s</td><td>%.2f</td><td>%s</td><td>%s</td></tr>"
                    % (name, r["last_close"], html.escape(r["status"]), last_txt))
        tl = []
        for s in sorted(r["signals"] + [dict(o, dir=None) for o in r["outs"]], key=lambda x: x["t"]):
            above = s["side"] == "UP"
            what = "au-dessus de la bande haute" if above else "sous la bande basse"
            core = "Bougie %s %s (dépasse de %.2f, soit %.0f %% de la largeur des bandes)" % (
                "entièrement" if cfg["wick_rule"] == "full" else "(corps)", what, s["reach"], s["reach_pct"])
            if s["dir"]:
                cls = "s-buy" if s["dir"] == "BUY" else "s-sell"
                act = "ACHAT" if s["dir"] == "BUY" else "VENTE"
                tl.append('<li class="%s">%s · %s, plus %s des %d dernières bougies → <b>%s</b> (clôture %.2f)</li>'
                          % (cls, s["t"].strftime("%H:%M"), core, "haut" if above else "bas", cfg["lookback"], act, s["price"]))
            else:
                tl.append('<li>%s · %s, mais pas à l\'extrême du mouvement → pas de signal</li>'
                          % (s["t"].strftime("%H:%M"), core))
        tl_html = '<ul class="tl">%s</ul>' % "".join(tl) if tl else '<div class="info">Aucune sortie complète des bandes pour l\'instant.</div>'
        img = '<img alt="%s" src="data:image/png;base64,%s">' % (html.escape(sym), r["png"])
        cards.append(
            '<div class="card"><div class="top"><a href="https://www.tradingview.com/chart/?symbol=%s" target="_blank">%s</a>'
            '<div class="badges"><span class="tf">%d signal(aux)</span></div></div>'
            '<div class="info">Séance du %s · dernière bougie %s · %s</div>%s%s</div>'
            % (html.escape(zo.TV.get(sym, sym)), name, len(r["signals"]), r["day"].strftime("%d/%m/%Y"),
               r["last_bar"].strftime("%H:%M"), html.escape(r["status"]), img, tl_html))
    allsig.sort(key=lambda s: s["t"], reverse=True)
    sig_rows = []
    for s in allsig[:15]:
        cls = "buy" if s["dir"] == "BUY" else "sell"
        age = (len(s["r"]["sess"]) - 1) - s["i"]
        new = '<span class="chip new">NOUVEAU</span> ' if age <= 1 and s["r"]["day"] == now.date() else ""
        sig_rows.append('<tr><td>%s<span class="chip %s">%s</span></td><td>%s</td><td>%s</td><td>%s</td><td>%.2f</td><td>%.2f</td></tr>'
                        % (new, cls, "ACHAT" if s["dir"] == "BUY" else "VENTE",
                           html.escape(zo.NAMES.get(s["sym"], s["sym"])), s["t"].strftime("%H:%M"),
                           "sortie haute" if s["side"] == "UP" else "sortie basse", s["price"], s["band"]))
    sig_table = ('<table class="tbl"><tr><th>Signal</th><th>Instrument</th><th>Bougie</th><th>Bulle</th>'
                 '<th>Clôture</th><th>Bande</th></tr>%s</table>' % "".join(sig_rows)) if sig_rows else \
        '<div class="none">Aucun signal pour le moment.</div>'
    iso = now.isoformat()
    nav = "" if embed else ('<div class="nav"><a href="index.html">S&amp;P 500 + Nasdaq</a><a href="tsx.html">TSX</a>'
                            '<a href="0dte.html">0DTE</a><a href="bulle.html" class="on">Bulle</a></div>')
    sens = "retournement (sortie haute = VENTE, sortie basse = ACHAT)" if cfg["mode"] == "reversal" \
        else "continuation (sortie haute = ACHAT, sortie basse = VENTE)"
    sortie = "bougie entière (mèche incluse)" if cfg["wick_rule"] == "full" else "corps seul"
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stratégie bulle</title>{base.THEME_HEAD}<style>{base.CSS}{z.EXTRA_CSS}{zo.OPR_CSS}</style></head><body>
<button id="theme" class="theme" title="Passer en mode sombre">🌙</button>
{nav}<h1>Stratégie bulle — bandes de Bollinger</h1>
<div class="sub">Dernier scan : {now:%d/%m/%Y %H:%M:%S} ET · <span id="age" data-ts="{iso}">…</span> ·
bougies {cfg['candle_minutes']} min validées · Bollinger {cfg['bb_n']} / {cfg['bb_k']:g} ·
sortie : {sortie} · extrême sur {cfg['lookback']} bougies · sens : {sens}</div>
<div class="warn">⚠ Données Yahoo gratuites, éventuellement retardées. Un signal est émis sur une bougie terminée :
il arrive à la clôture de cette bougie. Vérifie toujours sur ton graphique avant d'agir.
Ceci n'est pas un conseil financier.</div>
<table class="tbl"><tr><th></th><th>Dernier prix</th><th>Position</th><th>Dernier signal</th></tr>{"".join(rows)}</table>
<h2>Derniers signaux ({len(allsig)})</h2>{sig_table}
<h2>Détail par instrument</h2><div class="grid">{"".join(cards)}</div>
<script>{base.JS}</script><script>{z.AGE_JS}</script></body></html>"""


# ----------------------------- Scan -----------------------------
def scan_once(now, symbols=None, cfg=None):
    """Retourne (page_html, nombre_de_signaux, résultats) ; page_html = None si aucune donnée."""
    cfg = dict(CFG, **(cfg or {}))
    symbols = list(symbols or zo.SYMBOLS_DEFAULT)
    interval, period = INTERVAL.get(cfg["candle_minutes"], ("5m", "5d"))
    try:
        data = zo.get_bars(symbols, interval, period)
    except Exception as e:
        print(f"[!] téléchargement impossible: {e}", file=sys.stderr)
        return None, 0, {}
    results = {}
    for sym in symbols:
        try:
            results[sym] = analyze_symbol(sym, data.get(sym, pd.DataFrame()), now, cfg)
        except Exception as e:
            print(f"[!] analyse {sym}: {e}", file=sys.stderr)
            results[sym] = None
    if not any(results.values()):
        return None, 0, results
    n = sum(len(r["signals"]) for r in results.values() if r)
    return build_page(results, now, cfg, embed=True), n, results


if __name__ == "__main__":
    page, n, _ = scan_once(datetime.now(ET))
    if page is None:
        sys.exit("Aucune donnée reçue.")
    open("bulle.html", "w", encoding="utf-8").write(page)
    print(f"{n} signal(aux) -> bulle.html")
