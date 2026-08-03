"""
Étape 7 : Index vectoriel FAISS
-------------------------------
Ce module construit un index FAISS à partir des embeddings sauvegardés
et permet d'effectuer des recherches par similarité sémantique.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import os
import re

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache(maxsize=2)
def _charger_modele(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def _charger_index_et_metadata(output_dir_str: str):
    index_path = Path(output_dir_str) / "faiss_index.index"
    metadata_path = Path(output_dir_str) / "faiss_metadata.json"
    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return index, metadata

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if __package__ in {None, ""}:
    from app.utils.embeddings import EMBEDDINGS_DIR, MODEL_NAME
    from app.utils.text_utils import normaliser_texte
else:
    from .embeddings import EMBEDDINGS_DIR, MODEL_NAME
    from .text_utils import normaliser_texte


VECTOR_DB_DIR = BASE_DIR / "vector_db"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "chunks_embeddings.jsonl"
INDEX_PATH = VECTOR_DB_DIR / "faiss_index.index"
METADATA_PATH = VECTOR_DB_DIR / "faiss_metadata.json"


def charger_embeddings(embeddings_path: Path = EMBEDDINGS_PATH) -> Tuple[np.ndarray, List[Dict]]:
    """Charge les embeddings depuis le fichier JSONL généré à l'étape 6."""
    rows = []
    with embeddings_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"Aucun embedding trouvé dans {embeddings_path}")

    vectors = np.asarray([item["embedding"] for item in rows], dtype="float32")

    # IMPORTANT : on conserve "type" et "chemin_complet" pour pouvoir
    # filtrer proprement les formulaires ET proposer leur téléchargement
    # plus tard dans l'interface, au lieu de filtrer par nom en dur.
    metadata = [
        {
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "type": item.get("type", "document"),
            "chemin_complet": item.get("chemin_complet", ""),
        }
        for item in rows
    ]
    return vectors, metadata


def construire_index_vectoriel(
    embeddings_path: Path = EMBEDDINGS_PATH,
    output_dir: Path = VECTOR_DB_DIR,
) -> Dict[str, Path]:
    """Construit et sauvegarde un index FAISS à partir des embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    vectors, metadata = charger_embeddings(embeddings_path)
    vectors = vectors.astype("float32")

    faiss.normalize_L2(vectors)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    index_path = output_dir / "faiss_index.index"
    metadata_path = output_dir / "faiss_metadata.json"

    faiss.write_index(index, str(index_path))
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"index_path": index_path, "metadata_path": metadata_path}


def _tokens_pertinents(texte: str) -> set:
    """Tokenise et normalise, en ignorant la ponctuation collée aux mots
    (virgules, points d'interrogation...) et les mots trop courts."""
    texte_normalise = normaliser_texte(texte)
    return set(re.findall(r"[a-z0-9]{4,}", texte_normalise))


def _boost_lexical(question: str, texte_chunk: str, score: float) -> float:
    """Booste le score si des mots significatifs de la question
    apparaissent tels quels (recouvrement exact) dans le chunk.
    Corrige les cas où un chunk pertinent a un score cosine bas
    (embedding dilué par un préfixe générique ou un contexte thématique
    voisin) alors qu'il contient littéralement le mot-clé cherché."""
    tokens_q = _tokens_pertinents(question)
    tokens_c = _tokens_pertinents(texte_chunk)
    recouvrement = tokens_q & tokens_c
    if recouvrement:
        return min(score + 0.05 * len(recouvrement), 1.0)
    return score


def rechercher_similaires(
    question: str,
    top_k: int = 5,
    model_name: str = MODEL_NAME,
    output_dir: Path = VECTOR_DB_DIR,
    similarity_threshold: float = 0.2,
    inclure_formulaires: bool = True,
    debug: bool = False,
) -> List[Dict]:
    """
    Recherche les chunks les plus similaires à une question donnée.

    Params:
    - question: la requête utilisateur
    - top_k: nombre de résultats à retourner
    - similarity_threshold: score minimum de similarité pour accepter un résultat
      (appliqué APRÈS le boost lexical)
    - inclure_formulaires: si True, les formulaires peuvent être retournés
      (utile pour répondre "voici le formulaire à remplir"). Mis à False,
      seuls les documents informatifs sont utilisés pour répondre.
    - debug: si True, affiche tous les scores trouvés (avant filtrage)
      pour t'aider à calibrer le threshold.
    """
    model = _charger_modele(model_name)
    query_vector = model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    index_path = output_dir / "faiss_index.index"
    metadata_path = output_dir / "faiss_metadata.json"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("L'index vectoriel n'a pas encore été construit.")

    index, metadata = _charger_index_et_metadata(str(output_dir))

    # On récupère plus de résultats initialement pour filtrer/booster après.
    # Important : ce nombre doit rester assez large pour que des chunks
    # pertinents mais mal classés par le score cosine pur (score bas)
    # aient une chance d'être repêchés par le boost lexical.
    k_candidats = min(max(top_k * 5, 50), len(metadata))
    distances, indices = index.search(query_vector, k_candidats)

    if debug:
        print(f"\n DEBUG scores bruts pour '{question}' :")
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            item = metadata[int(idx)]
            print(f"   score={score:.3f} | {item['source']} #{item['chunk_id']} | {item['text'][:60]}")

    # Étape 1 : on rassemble tous les candidats avec leur score boosté,
    # SANS filtrer ni tronquer tout de suite (pour pouvoir retrier après boost).
    candidats = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue

        item = metadata[int(idx)]

        if not inclure_formulaires and item.get("type") == "formulaire":
            continue

        score_boostee = _boost_lexical(question, item["text"], float(score))

        candidats.append({
            "score": score_boostee,
            "score_original": float(score),
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "type": item.get("type", "document"),
            "chemin_complet": item.get("chemin_complet", ""),
        })

    # Étape 2 : retri par score après boost (l'ordre FAISS initial n'est
    # plus garanti valide une fois le boost lexical appliqué).
    candidats.sort(key=lambda c: c["score"], reverse=True)

    if debug:
        print(f"\n DEBUG après boost lexical (top {top_k * 2}) :")
        for c in candidats[: top_k * 2]:
            marqueur = " ⬆" if c["score"] > c["score_original"] else ""
            print(f"   score={c['score']:.3f} (brut={c['score_original']:.3f}){marqueur} | {c['source']} #{c['chunk_id']}")

    # Étape 3 : filtrage par seuil + troncature à top_k, sur les scores boostés
    resultats = []
    for c in candidats:
        if c["score"] < similarity_threshold:
            continue
        c.pop("score_original", None)
        resultats.append(c)
        if len(resultats) >= top_k:
            break

    return resultats

# Mots-clés associés à chaque formulaire, en complément du nom de fichier.
# À mettre à jour si un nouveau formulaire est ajouté.
MOTS_CLES_FORMULAIRES = {
    "Copy of QES24022 Note de frais AVL Maroc.xlsx": [
        "frais", "remboursement", "rembourser", "hotel", "hôtel",
        "hebergement", "hébergement", "repas", "restaurant",
        "transport", "billet", "taxi", "deplacement", "déplacement",
    ],
    "Engagement sur le télétravail.pdf": [
        "teletravail", "télétravail", "remote", "domicile", "distance",
    ],
    "QES23012 Remote work request form AVL Maroc.docx": [
        "teletravail", "télétravail", "remote", "domicile", "distance",
    ],
    "attestation sur l'honneur.pdf": [
        "attestation", "honneur",
    ],
}

def rechercher_formulaires(
    question: str,
    top_k: int = 6,
    model_name: str = MODEL_NAME,
    output_dir: Path = VECTOR_DB_DIR,
    similarity_threshold: float = 0.35,
    exiger_recouvrement_lexical: bool = True,
    debug: bool = False,
) -> List[Dict]:
    """Recherche uniquement parmi les formulaires (type == 'formulaire')."""
    tous_resultats = rechercher_similaires(
        question=question,
        top_k=top_k * 20,
        model_name=model_name,
        output_dir=output_dir,
        similarity_threshold=similarity_threshold,
        inclure_formulaires=True,
        debug=debug,
    )
    formulaires = [r for r in tous_resultats if r.get("type") == "formulaire"]

    if exiger_recouvrement_lexical:
        tokens_question = _tokens_pertinents(question)
        formulaires_filtres = []
        for f in formulaires:
            tokens_source = _tokens_pertinents(Path(f["source"]).stem)
            tokens_source |= set(MOTS_CLES_FORMULAIRES.get(f["source"], []))
            if tokens_question & tokens_source:
                 formulaires_filtres.append(f)
            elif debug:
                 print(f"    --rejeté (aucun recouvrement lexical) : {f['source']} score={f['score']:.3f}")
        formulaires = formulaires_filtres

    # Déduplication : un même formulaire ne doit apparaître qu'une fois,
    # on garde le chunk avec le meilleur score.
    meilleurs_par_source: Dict[str, Dict] = {}
    for f in formulaires:
        source = f["source"]
        if source not in meilleurs_par_source or f["score"] > meilleurs_par_source[source]["score"]:
            meilleurs_par_source[source] = f
    formulaires_dedupliques = sorted(meilleurs_par_source.values(), key=lambda r: r["score"], reverse=True)

    return formulaires_dedupliques[:top_k]

if __name__ == "__main__":
    print(" Construction de l'index vectoriel FAISS...")
    result = construire_index_vectoriel()
    print(f"Index créé : {result['index_path']}")
    print(f"Métadonnées sauvegardées : {result['metadata_path']}")

    def tester(question: str):
        print(f"\n--- Test : {question} ---")
        resultats = rechercher_formulaires(question, debug=True)
        if not resultats:
            print(" Aucun formulaire trouvé.")
        for res in resultats:
            print(f" Trouvé : {res['source']} | Score: {res['score']:.3f}")
        return resultats

    tester("Combien de jours de congés ai-je par an ?")
    tester("Je veux me faire rembourser mon hôtel")
    tester("Je veux me faire rembourser mes frais de repas")
