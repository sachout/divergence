"""
Page web (Streamlit) pour le 0DTE, avec trois onglets :
  1) Divergences RSI + CDV estimé sur SPY, QQQ, DIA (5 min, 15 min, 1 h)
  2) OPR (Open Price Range) : sorties, réintégrations, pullbacks et signaux achat / vente
  3) Bulle : bougie entièrement hors des bandes de Bollinger, à l'extrême du mouvement

Elle scanne quand tu l'ouvres, puis toutes les 60 secondes.
Déploiement gratuit : share.streamlit.io -> dépôt GitHub, fichier principal : streamlit_app.py
(scanner_bulle.py, scanner_opr.py, scanner_0dte.py et scanner_divergences_rsi.py doivent être dans le même dépôt).
"""
from datetime import datetime
from math import ceil

import streamlit as st
import streamlit.components.v1 as components

import scanner_0dte as z
import scanner_opr as zo
import scanner_bulle as zb

st.set_page_config(page_title="0DTE · Divergences & OPR", page_icon="📈", layout="wide")

now0 = datetime.now(z.ET)
market_is_open = z.market_open(now0)

with st.sidebar:
    st.header("Réglages")
    auto = st.checkbox("Actualisation automatique (60 s)", value=True)
    test = st.checkbox(
        "Mode test (onglet Divergences)",
        value=not market_is_open,
        help="Marché fermé ou envie de revoir les signaux des dernières heures : "
             "affiche aussi des signaux plus anciens, marqués « ancien · test ».",
    )
    if st.button("Scanner maintenant"):
        st.cache_data.clear()
    st.caption("Données Yahoo gratuites (retard possible). CDV estimé depuis les bougies. "
               "Pas un conseil financier.")

st.caption(
    ("🟢 Marché ouvert" if market_is_open else "🔴 Marché fermé (9h30-16h00 ET)")
    + f" · {now0:%a %d/%m %H:%M} ET"
)

tab_div, tab_opr, tab_bulle = st.tabs(["Divergences RSI / CDV", "OPR", "Bulle (Bollinger)"])


# ----------------------------- Onglet 1 : divergences RSI / CDV -----------------------------
@st.cache_data(ttl=45, show_spinner=False)
def run_scan(force: bool):
    """Un scan complet ; le résultat est partagé 45 s entre tous les visiteurs."""
    now = datetime.now(z.ET)
    page, n_active, _ = z.scan_once(now, force, None, "", "", True)
    return page, n_active


@st.fragment(run_every=60 if auto else None)
def live_div():
    page, n_active = run_scan(test)
    if page is None:
        st.error("Aucune donnée reçue de Yahoo pour le moment. Nouvel essai dans une minute.")
        return
    n_cards = page.count('class="card"')
    components.html(page, height=640 + 560 * ceil(n_cards / 2), scrolling=True)


with tab_div:
    live_div()


# ----------------------------- Onglet 2 : OPR -----------------------------
@st.cache_data(ttl=45, show_spinner=False)
def run_opr(symbols: tuple, opr_minutes: int, candle_minutes: int, body_rule: str, tol_pct: float):
    now = datetime.now(z.ET)
    cfg = dict(opr_minutes=opr_minutes, candle_minutes=candle_minutes, body_rule=body_rule, tol_pct=tol_pct)
    page, n, _ = zo.scan_once(now, list(symbols), cfg)
    return page, n


with tab_opr:
    c1, c2, c3, c4 = st.columns(4)
    opr_minutes = c1.selectbox("Durée de l'OPR", [5, 15, 30, 60], index=1,
                               format_func=lambda m: f"{m} premières minutes")
    candle_minutes = c2.selectbox("Bougies de signal", [1, 5, 15], index=1, format_func=lambda m: f"{m} min")
    body_rule = c3.selectbox("Sortie de l'OPR", ["close", "full"], index=0,
                             format_func=lambda r: "clôture hors de l'OPR" if r == "close"
                             else "corps entier hors de l'OPR")
    tol_pct = c4.selectbox("Tolérance du pullback", [0.0, 0.1, 0.2], index=1,
                           format_func=lambda t: f"{int(t * 100)} % de la largeur de l'OPR")
    symbols = st.multiselect("Instruments", ["SPY", "QQQ", "DIA", "^GSPC", "^NDX", "^DJI"],
                             default=["SPY", "QQQ", "DIA"],
                             format_func=lambda s: zo.NAMES.get(s, s))

    @st.fragment(run_every=60 if auto else None)
    def live_opr():
        if not symbols:
            st.info("Choisis au moins un instrument.")
            return
        page, n = run_opr(tuple(symbols), int(opr_minutes), int(candle_minutes), body_rule, float(tol_pct))
        if page is None:
            st.error("Aucune donnée reçue de Yahoo pour le moment. Nouvel essai dans une minute.")
            return
        components.html(page, height=980 + 620 * ceil(len(symbols) / 2), scrolling=True)

    live_opr()


# ----------------------------- Onglet 3 : bulle (Bollinger) -----------------------------
@st.cache_data(ttl=45, show_spinner=False)
def run_bulle(symbols: tuple, candle_minutes: int, bb_n: int, bb_k: float, lookback: int,
              wick_rule: str, mode: str, first_only: bool):
    now = datetime.now(z.ET)
    cfg = dict(candle_minutes=candle_minutes, bb_n=bb_n, bb_k=bb_k, lookback=lookback,
               wick_rule=wick_rule, mode=mode, first_only=first_only)
    page, n, _ = zb.scan_once(now, list(symbols), cfg)
    return page, n


with tab_bulle:
    b1, b2, b3, b4 = st.columns(4)
    b_candle = b1.selectbox("Bougies", [1, 5, 15, 60], index=1, format_func=lambda m: f"{m} min", key="b_candle")
    b_n = b2.number_input("Période Bollinger", min_value=5, max_value=100, value=20, step=1, key="b_n")
    b_k = b3.number_input("Écart-type", min_value=1.0, max_value=4.0, value=2.0, step=0.5, key="b_k")
    b_look = b4.number_input("Extrême sur N bougies", min_value=3, max_value=100, value=20, step=1, key="b_look")
    b5, b6, b7 = st.columns(3)
    b_wick = b5.selectbox("Sortie des bandes", ["full", "body"], index=0, key="b_wick",
                          format_func=lambda r: "bougie entière (mèche incluse)" if r == "full" else "corps seul")
    b_mode = b6.selectbox("Sens du signal", ["reversal", "continuation"], index=0, key="b_mode",
                          format_func=lambda m: "retournement (haut = vente, bas = achat)" if m == "reversal"
                          else "continuation (haut = achat, bas = vente)")
    b_first = b7.checkbox("Un seul signal par séquence de bulles", value=False, key="b_first")
    b_symbols = st.multiselect("Instruments ", ["SPY", "QQQ", "DIA", "^GSPC", "^NDX", "^DJI"],
                               default=["SPY", "QQQ", "DIA"], key="b_symbols",
                               format_func=lambda s: zo.NAMES.get(s, s))

    @st.fragment(run_every=60 if auto else None)
    def live_bulle():
        if not b_symbols:
            st.info("Choisis au moins un instrument.")
            return
        page, n = run_bulle(tuple(b_symbols), int(b_candle), int(b_n), float(b_k), int(b_look),
                            b_wick, b_mode, bool(b_first))
        if page is None:
            st.error("Aucune donnée reçue de Yahoo pour le moment. Nouvel essai dans une minute.")
            return
        components.html(page, height=980 + 620 * ceil(len(b_symbols) / 2), scrolling=True)

    live_bulle()
