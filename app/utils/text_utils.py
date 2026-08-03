"""
Normalisation de texte pour la recherche sémantique.
"""

import unicodedata


def normaliser_texte(texte: str) -> str:
    """
    Normalise un texte AVANT embedding (recherche insensible aux accents/casse) :
    - minuscules
    - suppression des accents (é -> e, à -> a, etc.)

    Doit être appliquée à l'IDENTIQUE à l'indexation et à la recherche.
    Le texte original (avec accents) reste inchangé pour l'affichage
    et pour le contexte envoyé au LLM.
    """
    texte = texte.strip().lower()
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return texte