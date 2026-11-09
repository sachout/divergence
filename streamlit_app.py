"""
Page web « Divergences 0DTE » (Streamlit) : scanne quand tu l'ouvres, puis toutes les 60 secondes.

RSI + CDV estimé sur SPY (S&P 500), QQQ (Nasdaq-100) et DIA (Dow Jones),
en 5 minutes, 15 minutes et 1 heure, sur des bougies complètes.

Déploiement gratuit : share.streamlit.io -> dépôt GitHub, fichier principal : streamlit_app.py
(le fichier scanner_0dte.py et scanner_divergences_rsi.py doivent être dans le même dépôt).
"""
from datetime import datetime
from math import ceil

import streamlit as st
import streamlit.components.v1 as components

import scanner_0dte as z

st.set_page_config(page_title="Divergences 0DTE", page_icon="📈", layout="wide")

now0 = datetime.now(z.ET)
market_is_open = z.market_open(now0)

with st.sidebar:
    st.header("Réglages")
    auto = st.checkbox("Actualisation automatique (60 s)", value=True)
    test = st.checkbox(
        "Mode test",
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


@st.cache_data(ttl=45, show_spinner=False)
def run_scan(force: bool):
    """Un scan complet ; le résultat est partagé 45 s entre tous les visiteurs."""
    now = datetime.now(z.ET)
    page, n_active, _ = z.scan_once(now, force, None, "", "", True)
    return page, n_active


@st.fragment(run_every=60 if auto else None)
def live():
    page, n_active = run_scan(test)
    if page is None:
        st.error("Aucune donnée reçue de Yahoo pour le moment. Nouvel essai dans une minute.")
        return
    n_cards = page.count('class="card"')
    components.html(page, height=640 + 560 * ceil(n_cards / 2), scrolling=True)


live()
