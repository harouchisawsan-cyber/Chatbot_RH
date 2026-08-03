"""
Étape 4 : Nettoyage des données
Étape 5 : Découpage du texte (Chunking)
----------------------------------------
Ce script prend les documents extraits (extraction.py) et :
  1. Nettoie le texte (espaces, lignes vides, caractères parasites)
  2. Découpe chaque document en chunks de taille raisonnable
     pour préparer la création des embeddings.

Chaque chunk final est un dictionnaire :
    {
        "source": "nom_du_fichier.docx",
        "chunk_id": 0,
        "text": "morceau de texte nettoyé...",
        "type": "document" ou "formulaire",
        "chemin_complet": "/chemin/vers/le/fichier"
    }

Cas particulier des formulaires : on ne les découpe PAS en plusieurs
chunks. Un formulaire (ex: Attestation sur l'honneur) est généralement
court et ses champs n'ont de sens que pris ensemble : le découper
casserait le contexte. Il reste donc en un seul chunk, quelle que soit
sa longueur.
"""

import re
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Permet d'importer extraction.py qui est dans le même dossier
sys.path.append(str(Path(__file__).resolve().parent))
from extraction import extraire_tous_les_documents


# Paramètres de chunking recommandés dans le roadmap
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def nettoyer_texte(texte: str) -> str:
    """
    Nettoie un texte brut :
    - supprime les espaces multiples
    - supprime les lignes vides superflues
    - normalise les retours à la ligne
    - encode proprement en UTF-8
    """
    # Normaliser les retours à la ligne (Windows -> Unix)
    texte = texte.replace("\r\n", "\n").replace("\r", "\n")

    # Supprimer les espaces multiples (mais garder les retours à la ligne)
    texte = re.sub(r"[ \t]+", " ", texte)

    # Supprimer les lignes vides répétées (2+ retours à la ligne -> 1 seul)
    texte = re.sub(r"\n{2,}", "\n", texte)

    # Supprimer les espaces en début/fin de ligne
    lignes = [ligne.strip() for ligne in texte.split("\n")]
    texte = "\n".join(ligne for ligne in lignes if ligne)

    # S'assurer d'un encodage propre
    texte = texte.encode("utf-8", errors="ignore").decode("utf-8")

    return texte.strip()

STOPWORDS_DEBUT = {
    "le", "la", "les", "un", "une", "des", "ce", "cette", "ces",
    "il", "elle", "ils", "elles", "on", "dans", "pour", "après",
    "avant", "chaque", "aucun", "aucune", "tout", "toute", "tous",
    "afin", "sans", "avec", "sont", "doit", "doivent",
}


def _est_titre_majuscules(ligne: str) -> str | None:
    """
    Détecte les titres tout en majuscules, avec ou sans numérotation
    (ex: "ARTICLE 1 DEFINITION", "PREAMBULE"). Exclut les cellules de
    tableau/grille tarifaire (ex: "J+1", "110€ / J", "800 MAD/J") qui
    sont courtes, majuscules par accident, mais sans contenu textuel réel.
    """
    ligne = ligne.strip()
    if not ligne or len(ligne) < 3:
        return None

    lettres = [c for c in ligne if c.isalpha()]
    chiffres = [c for c in ligne if c.isdigit()]

    # Trop peu de lettres au total -> probablement une valeur/donnée, pas un titre
    if len(lettres) < 4:
        return None

    # Plus de chiffres que de lettres -> probablement une valeur numérique/tarif
    if len(chiffres) > len(lettres):
        return None

    part_majuscules = sum(1 for c in lettres if c.isupper()) / len(lettres)
    if part_majuscules < 0.9:
        return None

    mots = ligne.split()
    if not (1 <= len(mots) <= 8):
        return None

    # Exige au moins un mot purement alphabétique de 3+ lettres
    # (élimine les tokens collés type "MAD/J", "110€", "J+1")
    a_un_mot_textuel = any(
        re.fullmatch(r"[A-ZÀ-Üa-zà-ü'’\-]{3,}", mot) for mot in mots
    )
    if not a_un_mot_textuel:
        return None

    return ligne.strip(" :")


def extraire_titres_sections(texte: str) -> list[str]:
    """
    Détecte les lignes qui ressemblent à des titres de section, selon deux
    styles possibles :
    1. Titre classique se terminant par ':' (style "Objectif :")
    2. Titre tout en majuscules, avec ou sans numérotation
       (style "ARTICLE 1 DEFINITION")
    """
    titres = []
    for ligne in texte.split("\n"):
        ligne = ligne.strip()
        if not ligne:
            continue

        # --- Style 2 : tout en majuscules ---
        titre_maj = _est_titre_majuscules(ligne)
        if titre_maj:
            titres.append(titre_maj)
            continue

        # --- Style 1 : se termine par ':' (logique existante, inchangée) ---
        if not ligne.endswith(":"):
            continue

        ligne_sans_num = re.sub(r'^\d+(?:[-\.]\d+)*\.?\s+', '', ligne)
        contenu = ligne_sans_num.rstrip(":").strip()
        if not contenu or not contenu[0].isupper():
            continue

        mots = contenu.split()
        if not (1 <= len(mots) <= 6):
            continue
        if any(c.isdigit() for c in contenu):
            continue

        premier_mot = mots[0].lower().strip("’'")
        if premier_mot in STOPWORDS_DEBUT:
            continue

        titres.append(contenu)
    return titres


def construire_chunk_sommaire(nom_source: str, texte_propre: str) -> dict | None:
    """Construit un chunk résumé listant les sections du document, s'il y en a assez."""
    titres = extraire_titres_sections(texte_propre)
    if len(titres) < 3:  # pas assez de structure détectée, pas utile
        return None

    contenu = (
        f"Document: {nom_source}\n"
        f"Type: document\n"
        f"Ce document couvre les sections suivantes :\n"
        + "\n".join(f"- {t}" for t in titres)
    )
    return {
        "source": nom_source,
        "chunk_id": "sommaire",
        "text": contenu,
        "type": "document",
        "chemin_complet": "",
    }

def decouper_en_chunks(documents: list[dict]) -> list[dict]:
    """
    Nettoie puis découpe chaque document en chunks.
    Retourne une liste plate de chunks prêts pour les embeddings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    tous_les_chunks = []

    for document in documents:
        texte_propre = nettoyer_texte(document["text"])
        type_contenu = document.get("type", "document")
        nom_source = document["source"]

        if type_contenu == "document":
            chunk_sommaire = construire_chunk_sommaire(nom_source, texte_propre)
            if chunk_sommaire:
                chunk_sommaire["chemin_complet"] = document.get("chemin_complet", "")
                tous_les_chunks.append(chunk_sommaire)

        contexte = f"Document: {nom_source}\nType: {type_contenu}\nContenu:\n"
        texte_avec_contexte = contexte + texte_propre

        if type_contenu == "formulaire" and len(texte_avec_contexte) < 1000:
            morceaux = [texte_avec_contexte]
        else:
            morceaux = splitter.split_text(texte_avec_contexte)

        for i, morceau in enumerate(morceaux):
            if i > 0 and nom_source not in morceau:
                morceau = f"Suite de {nom_source} :\n" + morceau

            tous_les_chunks.append({
                "source": nom_source,
                "chunk_id": i,
                "text": morceau,
                "type": type_contenu,
                "chemin_complet": document.get("chemin_complet", ""),
            })

    return tous_les_chunks


if __name__ == "__main__":
    # 1. Extraction
    documents = extraire_tous_les_documents()
    print(f"\n {len(documents)} document(s) extrait(s).")

    # 2. Nettoyage + Chunking
    chunks = decouper_en_chunks(documents)
    print(f"  {len(chunks)} chunk(s) généré(s) au total.\n")

    # 3. Aperçu des chunks générés
    for chunk in chunks:
        print("=" * 60)
        print(f"Source : {chunk['source']} | Chunk #{chunk['chunk_id']}")
        print("-" * 60)
        print(chunk["text"])
        print()