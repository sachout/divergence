#!/usr/bin/env python3
"""
Indicateur OPR (Open Price Range) pour le 0DTE.

OPR = plus haut / plus bas des N premières minutes de la séance (par défaut 15 min : 9h30-9h45 ET).

FLUX (tendance) = moyenne mobile 20 vs moyenne mobile 50 sur les bougies 1 heure complètes :
    MM20 > MM50 -> flux haussier      MM20 < MM50 -> flux baissier

ÉVÉNEMENTS (sur des bougies VALIDÉES, c'est-à-dire terminées) :
    - Sortie de l'OPR : le corps de la bougie sort de l'OPR (clôture au-dessus du haut / sous le bas).
    - Réintégration : une bougie clôture de nouveau à l'intérieur de l'OPR après une sortie.
    - Pullback : une bougie suivante vient retester le bord de l'OPR et tient (clôture du bon côté).

SIGNAUX (toujours dans le sens du flux) :
    ACHAT  (flux haussier) :
        1) sortie par le haut  -> pullback sur le HAUT de l'OPR, la bougie tient au-dessus   (continuation)
        2) sortie par le bas -> réintégration -> pullback sur le BAS de l'OPR, la bougie tient au-dessus
    VENTE  (flux baissier) :
        3) sortie par le bas   -> pullback sur le BAS de l'OPR, la bougie tient en dessous    (continuation)
        4) sortie par le haut -> réintégration -> pullback sur le HAUT de l'OPR, la bougie reste en dessous

Hypothèses à confirmer (réglables dans CFG) :
    opr_minutes=15 · candle_minutes=5 · body_rule="close" (clôture hors de l'OPR ; "full" = corps entier hors de l'OPR)
    tol_pct=0.10 (le pullback compte s'il arrive à moins de 10 % de la largeur de l'OPR du bord)
"""
import base64
import html
import io
import sys
import warnings
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.patches import Rectangle

import scanner_0dte as z          # outils communs : fuseau, bougies complètes, heures de marché
import scanner_divergences_rsi as base

warnings.filterwarnings("ignore")
ET = z.ET

SYMBOLS_DEFAULT = ["SPY", "QQQ", "DIA"]
NAMES = {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq-100 (QQQ)", "DIA": "Dow Jones (DIA)",
         "^GSPC": "S&P 500 (indice)", "^NDX": "Nasdaq-100 (indice)", "^DJI": "Dow Jones (indice)"}
CFG = dict(opr_minutes=15, candle_minutes=5, body_rule="close", tol_pct=0.10, ma_fast=20, ma_slow=50)

OPEN_MIN = 9 * 60 + 30   # 9h30 ET en minutes
TV = {"^GSPC": "SPX", "^NDX": "NDX", "^DJI": "DJI"}

EV_LABEL = {
    "BREAK_UP": "Sortie de l'OPR par le haut",
    "BREAK_DOWN": "Sortie de l'OPR par le bas",
    "REENTRY_UP": "Réintégration de l'OPR (venant du haut)",
    "REENTRY_DOWN": "Réintégration de l'OPR (venant du bas)",
    "PULL_UP": "Pullback sur le HAUT de l'OPR",
    "PULL_DOWN": "Pullback sur le BAS de l'OPR",
    "PULL_RE_UP": "Pullback sur le HAUT de l'OPR (après réintégration)",
    "PULL_RE_DOWN": "Pullback sur le BAS de l'OPR (après réintégration)",
}


# ----------------------------- Données -----------------------------
def get_bars(symbols, interval, period):
    data = yf.download(list(symbols), period=period, interval=interval, group_by="ticker",
                       auto_adjust=True, prepost=False, threads=True, progress=False)
    out = {}
    for sym in symbols:
        try:
            df = base._frame(data, sym).dropna(subset=["Open", "High", "Low", "Close"]).copy()
            out[sym] = z.to_et(df)
        except Exception as e:
            print(f"  [!] {sym} {interval}: {e}", file=sys.stderr)
            out[sym] = pd.DataFrame()
    return out


# ----------------------------- Flux 1 h -----------------------------
def flow_series(df1h, cfg):
    """+1 flux haussier, -1 flux baissier, 0 inconnu ; connu à partir de la FIN de chaque bougie 1 h."""
    if df1h is None or df1h.empty:
        return pd.Series(dtype=float)
    c = df1h["Close"]
    fast = c.rolling(cfg["ma_fast"]).mean()
    slow = c.rolling(cfg["ma_slow"]).mean()
    flow = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
    flow = np.where(fast.isna() | slow.isna(), 0, flow)
    ends = pd.Series(df1h.index + pd.Timedelta(minutes=60), index=df1h.index)
    closes = pd.Series(df1h.index.normalize() + pd.Timedelta(hours=16), index=df1h.index)
    ends = ends.where(ends <= closes, closes)
    return pd.Series(flow, index=pd.DatetimeIndex(ends))


def flow_at(flow, t):
    if flow is None or len(flow) == 0:
        return 0
    s = flow[flow.index <= t]
    return int(s.iloc[-1]) if len(s) else 0


# ----------------------------- Règles OPR -----------------------------
def zone(o, c, hi, lo, rule):
    """Retourne (sortie_haut, sortie_bas, clôture_dans_l'OPR) pour une bougie validée."""
    if rule == "full":                                   # le corps ENTIER est hors de l'OPR
        out_up, out_dn = min(o, c) > hi, max(o, c) < lo
    else:                                                # le corps CLÔTURE hors de l'OPR
        out_up, out_dn = c > hi, c < lo
    return out_up, out_dn, (lo <= c <= hi)


def run_rules(cands, hi, lo, cfg, flow):
    """Parcourt les bougies validées après l'OPR et retourne (événements, signaux)."""
    tol = cfg["tol_pct"] * (hi - lo)
    rule = cfg["body_rule"]
    cm = timedelta(minutes=cfg["candle_minutes"])
    events, signals = [], []
    phase, signalled = "IN", False

    def ev(i, t, kind, price, note=""):
        events.append(dict(i=i, t=t, kind=kind, price=float(price), note=note))

    def sig(i, t, direction, typ, price, level, fl):
        signals.append(dict(i=i, t=t, dir=direction, type=typ, price=float(price), level=float(level), flow=fl))

    for i, (t, row) in enumerate(cands.iterrows()):
        o, h, l, c = float(row.Open), float(row.High), float(row.Low), float(row.Close)
        up, dn, inr = zone(o, c, hi, lo, rule)
        fl = flow_at(flow, t + cm)                       # flux connu à la fermeture de la bougie
        if phase in ("IN", "RE_UP", "RE_DN") and (up or dn):
            ev(i, t, "BREAK_UP" if up else "BREAK_DOWN", c)
            phase, signalled = ("OUT_UP" if up else "OUT_DN"), False
            continue
        if phase == "OUT_UP":
            if inr:
                ev(i, t, "REENTRY_UP", c); phase, signalled = "RE_UP", False
            elif dn:
                ev(i, t, "BREAK_DOWN", c); phase, signalled = "OUT_DN", False
            elif not signalled and l <= hi + tol and c > hi:           # retest du haut qui tient
                if fl > 0:
                    ev(i, t, "PULL_UP", c, "ACHAT (flux haussier)"); sig(i, t, "BUY", "Continuation", c, hi, fl)
                else:
                    ev(i, t, "PULL_UP", c, "pas de signal : flux non haussier")
                signalled = True
        elif phase == "OUT_DN":
            if inr:
                ev(i, t, "REENTRY_DOWN", c); phase, signalled = "RE_DN", False
            elif up:
                ev(i, t, "BREAK_UP", c); phase, signalled = "OUT_UP", False
            elif not signalled and h >= lo - tol and c < lo:           # retest du bas qui tient
                if fl < 0:
                    ev(i, t, "PULL_DOWN", c, "VENTE (flux baissier)"); sig(i, t, "SELL", "Continuation", c, lo, fl)
                else:
                    ev(i, t, "PULL_DOWN", c, "pas de signal : flux non baissier")
                signalled = True
        elif phase == "RE_UP" and not signalled and h >= hi - tol and c < hi:      # échec de la sortie haute
            if fl < 0:
                ev(i, t, "PULL_RE_UP", c, "VENTE (flux baissier)"); sig(i, t, "SELL", "Réintégration", c, hi, fl)
            else:
                ev(i, t, "PULL_RE_UP", c, "pas de signal : flux non baissier")
            signalled = True
        elif phase == "RE_DN" and not signalled and l <= lo + tol and c > lo:      # échec de la sortie basse
            if fl > 0:
                ev(i, t, "PULL_RE_DOWN", c, "ACHAT (flux haussier)"); sig(i, t, "BUY", "Réintégration", c, lo, fl)
            else:
                ev(i, t, "PULL_RE_DOWN", c, "pas de signal : flux non haussier")
            signalled = True
    return events, signals


# ----------------------------- Analyse d'un instrument -----------------------------
def analyze_symbol(sym, df_c, df_1h, now, cfg):
    cm = cfg["candle_minutes"]
    df_c = z.drop_incomplete(df_c, cm, now) if len(df_c) else df_c
    df_1h = z.drop_incomplete(df_1h, 60, now) if df_1h is not None and len(df_1h) else df_1h
    if df_c is None or df_c.empty:
        return None
    day = df_c.index[-1].date()
    sess = df_c[df_c.index.date == day]
    flow = flow_series(df_1h, cfg)
    fl_now = flow_at(flow, sess.index[-1] + timedelta(minutes=cm))
    minutes = sess.index.hour * 60 + sess.index.minute
    opr_bars = sess[(minutes >= OPEN_MIN) & (minutes < OPEN_MIN + cfg["opr_minutes"])]
    need = max(1, cfg["opr_minutes"] // cm)
    complete = len(opr_bars) >= need
    res = dict(sym=sym, day=day, sess=sess, flow=fl_now, opr=None, complete=complete,
               events=[], signals=[], last_close=float(sess["Close"].iloc[-1]), last_bar=sess.index[-1],
               n_opr=len(opr_bars))
    if not complete:
        res["status"] = "OPR en construction"
        res["png"] = chart_png(sym, sess, None, [], [], fl_now, cfg)
        return res
    hi, lo = float(opr_bars["High"].max()), float(opr_bars["Low"].min())
    res["opr"] = (hi, lo)
    after = sess[minutes >= OPEN_MIN + cfg["opr_minutes"]]
    off = len(sess) - len(after)                       # décalage pour positionner les événements sur le graphique
    ev, sg = run_rules(after, hi, lo, cfg, flow)
    for e in ev + sg:
        e["i"] += off
    res["events"], res["signals"] = ev, sg
    last = res["last_close"]
    res["status"] = ("Prix au-dessus de l'OPR" if last > hi else "Prix sous l'OPR" if last < lo else "Prix dans l'OPR")
    res["png"] = chart_png(sym, sess, (hi, lo), ev, sg, fl_now, cfg)
    return res


# ----------------------------- Graphique -----------------------------
def chart_png(sym, sess, opr, events, signals, fl, cfg):
    n = len(sess)
    o, h, l, c = (sess[k].values for k in ("Open", "High", "Low", "Close"))
    fig, ax = plt.subplots(figsize=(7, 4.3))
    rngp = float(h.max() - l.min()) or 1.0
    for i in range(n):
        col = "#1e8449" if c[i] >= o[i] else "#c0392b"
        ax.vlines(i, l[i], h[i], color=col, lw=0.8)
        ax.add_patch(Rectangle((i - 0.32, min(o[i], c[i])), 0.64, max(abs(c[i] - o[i]), rngp * 0.002), color=col))
    nopr = max(1, cfg["opr_minutes"] // cfg["candle_minutes"])
    ax.axvspan(-0.5, nopr - 0.5, color="#999", alpha=0.12)
    if opr:
        hi, lo = opr
        ax.axhspan(lo, hi, color="#4a78d4", alpha=0.13)
        ax.axhline(hi, color="#4a78d4", lw=1.2)
        ax.axhline(lo, color="#4a78d4", lw=1.2)
        ax.text(n - 0.4, hi, f" OPR haut {hi:.2f}", va="bottom", ha="right", fontsize=7, color="#4a78d4")
        ax.text(n - 0.4, lo, f" OPR bas {lo:.2f}", va="top", ha="right", fontsize=7, color="#4a78d4")
    for e in events:
        i, k = e["i"], e["kind"]
        if k == "BREAK_UP":
            ax.plot(i, h[i] + rngp * 0.02, "^", color="#4a78d4", ms=6)
        elif k == "BREAK_DOWN":
            ax.plot(i, l[i] - rngp * 0.02, "v", color="#4a78d4", ms=6)
        elif k.startswith("REENTRY"):
            ax.plot(i, c[i], "o", mfc="none", mec="#e67e22", ms=8, mew=1.6)
        elif k.startswith("PULL"):
            ax.plot(i, (l[i] if k in ("PULL_UP", "PULL_RE_DOWN") else h[i]), "x", color="#e67e22", ms=6, mew=1.6)
    for s in signals:
        i = s["i"]
        if s["dir"] == "BUY":
            ax.annotate("ACHAT", (i, l[i]), textcoords="offset points", xytext=(0, -22), ha="center", fontsize=8,
                        fontweight="bold", color="#1e8449", arrowprops=dict(arrowstyle="-|>", color="#1e8449", lw=1.6))
        else:
            ax.annotate("VENTE", (i, h[i]), textcoords="offset points", xytext=(0, 22), ha="center", fontsize=8,
                        fontweight="bold", color="#c0392b", arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.6))
    flow_txt = "flux haussier ▲ (MM20 > MM50 en 1 h)" if fl > 0 else "flux baissier ▼ (MM20 < MM50 en 1 h)" if fl < 0 else "flux indéterminé"
    ax.set_title(f"{sym} · {sess.index[0]:%d/%m} · {flow_txt}", fontsize=9, loc="left")
    ticks = sorted(set(np.linspace(0, n - 1, 6).astype(int)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([sess.index[k].strftime("%H:%M") for k in ticks])
    ax.set_xlim(-1, n + 0.5)
    ax.margins(y=0.12)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------- Page HTML -----------------------------
OPR_CSS = """
.chip{display:inline-block;padding:2px 8px;border-radius:6px;font-size:12px;font-weight:700;color:#fff}
.chip.buy{background:#1e8449} .chip.sell{background:#c0392b} .chip.new{background:#f1c40f;color:#222}
.flow-up{color:var(--bull);font-weight:700} .flow-dn{color:var(--bear);font-weight:700}
.tl{margin:6px 0 0;padding:0;list-style:none;font-size:12.5px;line-height:1.55}
.tl li{border-left:3px solid var(--border);padding-left:8px;margin:2px 0}
.tl li.s-buy{border-color:var(--bull)} .tl li.s-sell{border-color:var(--bear)}
"""


def _flow_html(fl):
    return ('<span class="flow-up">▲ haussier</span>' if fl > 0 else
            '<span class="flow-dn">▼ baissier</span>' if fl < 0 else "—")


def build_page(results, now, cfg, embed=True):
    cm = cfg["candle_minutes"]
    rows, cards, allsig = [], [], []
    for sym, r in results.items():
        name = html.escape(NAMES.get(sym, sym))
        if not r:
            rows.append(f"<tr><td>{name}</td><td colspan='4'>pas de données</td></tr>")
            continue
        sigs = r["signals"]
        for s in sigs:
            allsig.append(dict(s, sym=sym, r=r))
        last = sigs[-1] if sigs else None
        last_txt = "—"
        if last:
            cls = "buy" if last["dir"] == "BUY" else "sell"
            last_txt = f'<span class="chip {cls}">{"ACHAT" if last["dir"] == "BUY" else "VENTE"}</span> {last["t"]:%H:%M}'
        opr_txt = f'{r["opr"][1]:.2f} – {r["opr"][0]:.2f}' if r["opr"] else "en construction"
        rows.append(f"<tr><td>{name}</td><td>{_flow_html(r['flow'])}</td><td>{opr_txt}</td>"
                    f"<td>{html.escape(r['status'])}</td><td>{last_txt}</td></tr>")
        tl = []
        for e in sorted(r["events"], key=lambda x: x["t"]):
            is_sig = e["note"].startswith(("ACHAT", "VENTE"))
            cls = (" s-buy" if e["note"].startswith("ACHAT") else " s-sell") if is_sig else ""
            note = f" → <b>{html.escape(e['note'])}</b>" if e["note"] else ""
            tl.append(f'<li class="{cls.strip()}">{e["t"]:%H:%M} · {html.escape(EV_LABEL[e["kind"]])} '
                      f'(clôture {e["price"]:.2f}){note}</li>')
        tl_html = f'<ul class="tl">{"".join(tl)}</ul>' if tl else '<div class="info">Aucun événement pour l\'instant.</div>'
        img = '<img alt="%s" src="data:image/png;base64,%s">' % (html.escape(sym), r["png"])
        opr_info = (" · OPR %.2f – %.2f" % (r["opr"][1], r["opr"][0])) if r["opr"] else ""
        cards.append(
            f'<div class="card"><div class="top"><a href="https://www.tradingview.com/chart/?symbol={html.escape(TV.get(sym, sym))}" '
            f'target="_blank">{name}</a><div class="badges">{_flow_html(r["flow"])}</div></div>'
            f'<div class="info">Séance du {r["day"]:%d/%m/%Y} · dernière bougie {r["last_bar"]:%H:%M} · {html.escape(r["status"])}'
            f'{opr_info}</div>{img}{tl_html}</div>')
    allsig.sort(key=lambda s: s["t"], reverse=True)
    sig_rows = []
    for s in allsig[:12]:
        cls = "buy" if s["dir"] == "BUY" else "sell"
        age = (len(s["r"]["sess"]) - 1) - s["i"]
        new = '<span class="chip new">NOUVEAU</span> ' if age <= 1 and s["r"]["day"] == now.date() else ""
        sig_rows.append(
            f'<tr><td>{new}<span class="chip {cls}">{"ACHAT" if s["dir"] == "BUY" else "VENTE"}</span></td>'
            f'<td>{html.escape(NAMES.get(s["sym"], s["sym"]))}</td><td>{s["t"]:%H:%M}</td>'
            f'<td>{s["type"]}</td><td>{s["price"]:.2f}</td><td>{s["level"]:.2f}</td></tr>')
    sig_table = ('<table class="tbl"><tr><th>Signal</th><th>Instrument</th><th>Bougie</th><th>Type</th>'
                 '<th>Clôture</th><th>Niveau OPR</th></tr>' + "".join(sig_rows) + "</table>") if sig_rows else \
        '<div class="none">Aucun signal pour le moment.</div>'
    iso = now.isoformat()
    nav = "" if embed else ('<div class="nav"><a href="index.html">S&amp;P 500 + Nasdaq</a><a href="tsx.html">TSX</a>'
                            '<a href="0dte.html">0DTE</a><a href="opr.html" class="on">OPR</a></div>')
    js = z.AGE_JS
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Signaux OPR</title>{base.THEME_HEAD}<style>{base.CSS}{z.EXTRA_CSS}{OPR_CSS}</style></head><body>
<button id="theme" class="theme" title="Passer en mode sombre">🌙</button>
{nav}<h1>Signaux OPR — Open Price Range</h1>
<div class="sub">Dernier scan : {now:%d/%m/%Y %H:%M:%S} ET · <span id="age" data-ts="{iso}">…</span> ·
OPR = {cfg['opr_minutes']} premières minutes (9h30-{(OPEN_MIN + cfg['opr_minutes']) // 60}h{(OPEN_MIN + cfg['opr_minutes']) % 60:02d}) ·
bougies {cm} min validées · flux = MM{cfg['ma_fast']} vs MM{cfg['ma_slow']} en 1 h ·
corps {'entier hors de l’OPR' if cfg['body_rule'] == 'full' else 'clôturant hors de l’OPR'}</div>
<div class="warn">⚠ Données Yahoo gratuites, éventuellement retardées. Un signal est émis sur une bougie terminée :
il arrive donc à la clôture de cette bougie. Vérifie toujours sur ton graphique avant d'agir.
Ceci n'est pas un conseil financier.</div>
<table class="tbl"><tr><th></th><th>Flux 1 h</th><th>OPR (bas – haut)</th><th>Statut</th><th>Dernier signal</th></tr>{"".join(rows)}</table>
<h2>Derniers signaux ({len(allsig)})</h2>{sig_table}
<h2>Détail par instrument</h2><div class="grid">{"".join(cards)}</div>
<script>{base.JS}</script><script>{js}</script></body></html>"""


# ----------------------------- Scan -----------------------------
def scan_once(now, symbols=None, cfg=None):
    """Un scan complet. Retourne (page_html, nombre_de_signaux, résultats) ; page_html = None si aucune donnée."""
    cfg = dict(CFG, **(cfg or {}))
    cfg["opr_minutes"] = max(cfg["opr_minutes"], cfg["candle_minutes"])
    symbols = list(symbols or SYMBOLS_DEFAULT)
    try:
        d_c = get_bars(symbols, f"{cfg['candle_minutes']}m", "5d")
        d_h = get_bars(symbols, "1h", "1mo")
    except Exception as e:
        print(f"[!] téléchargement impossible: {e}", file=sys.stderr)
        return None, 0, {}
    results = {}
    for sym in symbols:
        try:
            results[sym] = analyze_symbol(sym, d_c.get(sym, pd.DataFrame()), d_h.get(sym, pd.DataFrame()), now, cfg)
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
    open("opr.html", "w", encoding="utf-8").write(page)
    print(f"{n} signal(aux) -> opr.html")
