"""
Interface Streamlit pour le ChatBot RH
---------------------------------------
Application web interactive pour poser des questions RH
et obtenir des réponses basées sur les documents d'entreprise.
"""

import sys
from pathlib import Path

import streamlit as st

# Ajouter la racine du projet au path AVANT tous les imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.utils.rag import generer_reponse

# ----------------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Assistant RH — AVL Maroc",
    page_icon="📁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Design tokens & styles
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink:        #12253E;
    --ink-soft:   #4C5A6E;
    --paper:      #F2F4F5;
    --surface:    #FFFFFF;
    --line:       #DCE1E4;
    --accent:     #2E6E62;
    --accent-soft:#E4EEEC;
    --amber:      #B4802C;
}

#MainMenu, footer, header { visibility: hidden; }

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--paper);
}

.block-container {
    max-width: 760px;
    padding-top: 1rem;
    padding-bottom: 6rem;
}

/* ---------- Header ---------- */
.brand-bar {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid var(--line);
    padding-bottom: 1.1rem;
    margin-bottom: 1.6rem;
}
.brand-mark {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    color: var(--ink-soft);
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.brand-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.65rem;
    font-weight: 600;
    color: var(--ink);
    margin: 0;
    line-height: 1.15;
}
.brand-sub {
    font-size: 0.85rem;
    color: var(--ink-soft);
    margin-top: 0.25rem;
}

/* ---------- Empty state ---------- */
.empty-state {
    padding: 1.4rem 0 0.4rem 0;
    color: var(--ink-soft);
    font-size: 0.92rem;
    line-height: 1.6;
}
.empty-state strong { color: var(--ink); }

/* ---------- Chat bubbles ---------- */
[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0.15rem 0;
}

.msg-user {
    text-align: right;
    font-size: 0.95rem;
    color: var(--ink);
    padding: 0.2rem 0 0.6rem 0;
}
.msg-user .msg-label {
    display: block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: var(--ink-soft);
    text-transform: uppercase;
    margin-bottom: 0.15rem;
}

.answer-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 3px solid var(--accent);
    border-radius: 3px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.5rem;
}
.answer-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}
.answer-text {
    font-family: 'Source Serif 4', serif;
    font-size: 1.04rem;
    line-height: 1.65;
    color: var(--ink);
    white-space: pre-wrap;
}

/* ---------- Sources ---------- */
.sources-block {
    margin-top: 0.9rem;
    padding-top: 0.9rem;
    border-top: 1px solid var(--line);
}
.sources-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: var(--ink-soft);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.source-row {
    margin-bottom: 0.55rem;
}
.source-name {
    font-size: 0.82rem;
    color: var(--ink);
    margin-bottom: 0.22rem;
}
.source-ref {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--ink-soft);
}
.relevance-track {
    height: 3px;
    background: var(--line);
    border-radius: 2px;
    margin-top: 0.3rem;
    overflow: hidden;
}
.relevance-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
}

/* ---------- Formulaires ---------- */
.forms-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: var(--amber);
    text-transform: uppercase;
    margin: 0.9rem 0 0.5rem 0;
}
div[data-testid="stDownloadButton"] button {
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 3px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    padding: 0.3rem 0.8rem;
}
div[data-testid="stDownloadButton"] button:hover {
    background: var(--accent);
    color: white;
}

/* ---------- Example chips ---------- */
div[data-testid="column"] .stButton button {
    background: var(--surface);
    border: 1px solid var(--line);
    color: var(--ink-soft);
    border-radius: 3px;
    font-size: 0.82rem;
    padding: 0.5rem 0.7rem;
    text-align: left;
}
div[data-testid="column"] .stButton button:hover {
    border-color: var(--accent);
    color: var(--accent);
}

/* ---------- Reset button ---------- */
.reset-btn button {
    background: transparent;
    border: 1px solid var(--line);
    color: var(--ink-soft);
    font-size: 0.78rem;
    padding: 0.3rem 0.7rem;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# État de session
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # liste de dicts: role, content, sources?, formulaires?

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

EXEMPLES = [
    "Quelle est la politique de télétravail ?",
    "Comment demander un congé ?",
    "Comment déclarer que mon domicile est conforme pour le télétravail ?",
]

# ----------------------------------------------------------------------------
# En-tête
# ----------------------------------------------------------------------------
col_a, col_b = st.columns([5, 1])
with col_a:
    st.markdown("""
    <div class="brand-bar">
        <div>
            <div class="brand-mark">AVL MAROC · RESSOURCES HUMAINES</div>
            <h1 class="brand-title">Assistant RH</h1>
            <div class="brand-sub">Politiques, procédures et formulaires internes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_b:
    st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    if st.button("Effacer", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# État vide + suggestions
# ----------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        Posez une question sur une politique ou une procédure AVL Maroc.
        L'assistant répond <strong>uniquement</strong> à partir des documents internes
        et indique ses sources.
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(EXEMPLES))
    for c, exemple in zip(cols, EXEMPLES):
        with c:
            if st.button(exemple, use_container_width=True, key=f"ex_{exemple}"):
                st.session_state.pending_question = exemple
                st.rerun()

# ----------------------------------------------------------------------------
# Historique de conversation
# ----------------------------------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user"><span class="msg-label">Vous</span>{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        # On affiche d'abord le texte de la réponse
        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-label">Assistant RH</div>
            <div class="answer-text">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)

        # On affiche les sources SEULEMENT s'il y en a et APRES la carte de réponse
        if msg.get("sources"):
            rows = ""
            for s in msg["sources"]:
                pct = max(0, min(100, round(s["score"] * 100)))
                rows += f"""
                <div class="source-row">
                    <div class="source-name">📄 {s['source']}</div>
                    <div class="source-ref">Section #{s['chunk_id']} · {pct}% pertinence</div>
                </div>"""
            
            st.markdown(f'<div class="sources-block"><div class="sources-title">Sources</div>{rows}</div>', unsafe_allow_html=True)

        if msg.get("formulaires"):
            st.markdown('<div class="forms-title">Formulaires associés</div>', unsafe_allow_html=True)
            for f in msg["formulaires"]:
                chemin = Path(f["chemin_complet"])
                if chemin.exists():
                    st.download_button(
                        label=f"⬇ {f['source']}",
                        data=chemin.read_bytes(),
                        file_name=chemin.name,
                        key=f"dl_{chemin.name}_{f['chunk_id']}_{f['score']}",
                    )

# ----------------------------------------------------------------------------
# Saisie de la question
# ----------------------------------------------------------------------------
question_input = st.chat_input("Posez votre question RH…")

question = st.session_state.pending_question or question_input
st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Recherche dans la base documentaire…"):
        try:
            result = generer_reponse(
                question=question,
                top_k=6,
                similarity_threshold=0.3,
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["reponse"],
                "sources": result.get("sources", []),
                "formulaires": result.get("formulaires", []),
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Une erreur est survenue lors de la génération de la réponse : {e}",
                "sources": [],
                "formulaires": [],
            })

    st.rerun()