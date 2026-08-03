"""
Étape 9 : Mémoire conversationnelle
------------------------------------
Stocke un historique court par session pour permettre la gestion
des questions de suivi elliptiques (ex: "juste ça ?", "et pour X ?").
En mémoire uniquement : suffisant pour un déploiement mono-instance,
l'historique est perdu au redémarrage du serveur.
"""

import time
import uuid
from collections import defaultdict
from typing import Dict, List, TypedDict

MAX_TOURS_HISTORIQUE = 3          # nb d'échanges précédents conservés
DUREE_EXPIRATION_SECONDES = 30 * 60  # session oubliée après 30 min d'inactivité


class Tour(TypedDict):
    question: str
    reponse: str
    timestamp: float


_sessions: Dict[str, List[Tour]] = defaultdict(list)


def nouvelle_session() -> str:
    return str(uuid.uuid4())


def ajouter_tour(session_id: str, question: str, reponse: str) -> None:
    _nettoyer_sessions_expirees()
    _sessions[session_id].append({
        "question": question,
        "reponse": reponse,
        "timestamp": time.time(),
    })
    _sessions[session_id] = _sessions[session_id][-MAX_TOURS_HISTORIQUE:]


def obtenir_historique(session_id: str) -> List[Tour]:
    return _sessions.get(session_id, [])


def _nettoyer_sessions_expirees() -> None:
    maintenant = time.time()
    expirees = [
        sid for sid, tours in _sessions.items()
        if tours and maintenant - tours[-1]["timestamp"] > DUREE_EXPIRATION_SECONDES
    ]
    for sid in expirees:
        del _sessions[sid]