"""
API HTTP exposant le chatbot RH pour le front-end : API rest
"""

#pour connaitre @IP: hostname -I
#pour se connecter : http://172.18.199.53:8000

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

from app.utils.rag import generer_reponse
from app.utils.memoire_conversation import nouvelle_session, ajouter_tour, obtenir_historique


app = FastAPI(title="Assistant RH API")

# CORS : autorise le front-end (autre port/domaine) à appeler cette API.
# En dev, "*" suffit ; à restreindre plus tard à l'URL exacte du front.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    session_id: str | None = None
    top_k: int = 12
    similarity_threshold: float = 0.3


@app.get("/health")
def health():
    return {"status": "ok"}

#donne : reponse + sources + forms si dispo
@app.post("/chat")
def chat(payload: QuestionRequest):
    session_id = payload.session_id or nouvelle_session()
    historique = obtenir_historique(session_id)

    resultat = generer_reponse(
        question=payload.question,
        historique=historique,
        top_k=payload.top_k,
        similarity_threshold=payload.similarity_threshold,
    )

    ajouter_tour(session_id, payload.question, resultat["reponse"])
    resultat["session_id"] = session_id   # le frontend doit le renvoyer au prochain appel

    resultat["formulaires"] = [
        {
            "source": f["source"],
            "score": f["score"],
            "url_telechargement": f"/formulaires/{f['source']}",
        }
        for f in resultat.get("formulaires", [])
    ]
    return resultat

# sert a telecharger le form par le collaborateur
@app.get("/formulaires/{nom_fichier}")
def telecharger_formulaire(nom_fichier: str):
    """Sert le fichier formulaire pour téléchargement côté front-end."""
    chemin = Path("data/formulaires") / nom_fichier
    if not chemin.exists():
        return {"error": "Fichier introuvable"}
    return FileResponse(chemin, filename=chemin.name)