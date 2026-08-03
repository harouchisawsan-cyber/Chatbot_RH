"""
Étape 6 : Création des embeddings
--------------------------------
Ce module transforme chaque chunk de texte en vecteur numérique
avec sentence-transformers, puis sauvegarde les embeddings
et les métadonnées associées dans un dossier dédié.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if __package__ in {None, ""}:
    from app.utils.chunking import decouper_en_chunks
    from app.utils.extraction import extraire_tous_les_documents
    from app.utils.text_utils import normaliser_texte
else:
    from .chunking import decouper_en_chunks
    from .extraction import extraire_tous_les_documents
    from .text_utils import normaliser_texte
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def initialiser_modele(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Charge le modèle sentence-transformers utilisé pour générer les embeddings."""
    return SentenceTransformer(model_name)


def creer_embeddings(chunks: List[Dict], model: SentenceTransformer | None = None) -> List[Dict]:
    """
    Génère des embeddings pour chaque chunk et retourne une liste
    de dictionnaires contenant le texte, le chunk_id, la source, le type,
    le chemin complet et le vecteur.
    """
    if model is None:
        model = initialiser_modele()

    textes = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(textes, convert_to_numpy=True, normalize_embeddings=True)

    resultats = []
    for chunk, embedding in zip(chunks, embeddings):
        resultats.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "type": chunk.get("type", "document"),
            "chemin_complet": chunk.get("chemin_complet", ""),
            "embedding": embedding,
        })

    return resultats


def sauvegarder_embeddings(embeddings_data: List[Dict], output_dir: Path = EMBEDDINGS_DIR) -> Path:
    """Sauvegarde les embeddings au format JSONL et les métadonnées associées."""
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "chunks_embeddings.jsonl"
    metadata_path = output_dir / "chunks_metadata.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in embeddings_data:
            payload = {
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "type": item.get("type", "document"),
                "chemin_complet": item.get("chemin_complet", ""),
                "embedding": item["embedding"].tolist(),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    metadata = [
        {
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "type": item.get("type", "document"),
            "chemin_complet": item.get("chemin_complet", ""),
        }
        for item in embeddings_data
    ]
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_dir


def generer_embeddings(output_dir: Path = EMBEDDINGS_DIR) -> List[Dict]:
    """Pipeline complète : extraction -> chunking -> embedding -> sauvegarde."""
    documents = extraire_tous_les_documents()
    chunks = decouper_en_chunks(documents)
    embeddings_data = creer_embeddings(chunks)
    sauvegarder_embeddings(embeddings_data, output_dir)
    return embeddings_data


if __name__ == "__main__":
    print(" Génération des embeddings en cours...")
    data = generer_embeddings()
    print(f" {len(data)} embedding(s) généré(s) dans {EMBEDDINGS_DIR}")