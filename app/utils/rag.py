"""
Étape 8 : RAG (Retrieval-Augmented Generation)
----------------------------------------------
Ce module récupère les chunks pertinents et les envoie à Ollama
pour générer une réponse naturelle basée sur le contexte RH.

Il propose aussi, indépendamment de la réponse textuelle, les
formulaires pertinents à télécharger (ex: attestation sur l'honneur,
demande de télétravail) via une recherche dédiée.
"""

import sys
from pathlib import Path
from typing import List, Dict

import ollama

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if __package__ in {None, ""}:
    from app.utils.vector_db import rechercher_similaires ,rechercher_formulaires
else:
    from .vector_db import rechercher_similaires, rechercher_formulaires

MODEL_OLLAMA = "llama3.2"


def construire_contexte(chunks: List[Dict]) -> str:
    """Construit un contexte à partir des chunks récupérés."""
    contexte_lines = []
    for i, chunk in enumerate(chunks, 1):
        contexte_lines.append(f"Document {i} (Source: {chunk['source']}):")
        contexte_lines.append(chunk["text"])
        contexte_lines.append("")
    return "\n".join(contexte_lines)

def reformuler_question(question: str, model: str = MODEL_OLLAMA) -> str:
    """
    Corrige l'orthographe et les accents de la question utilisateur
    avant la recherche sémantique, pour maximiser le score de similarité
    avec les documents (rédigés en français correct).

    Ne reformule PAS le sens de la question, uniquement la forme.
    En cas d'échec (Ollama indisponible, etc.), retourne la question originale.
    """
    prompt = f"""Corrige uniquement l'orthographe, la grammaire et les accents manquants de cette question en français. 
Ne change pas le sens. Ne réponds pas à la question. Renvoie UNIQUEMENT la question corrigée, sans commentaire ni guillemets.

Question: {question}

Question corrigée:"""

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options={"temperature": 0, "num_predict": 60},
            keep_alive="30m",

        )
        corrigee = response.get("response", "").strip().strip('"')
        # Filet de sécurité : si le LLM renvoie quelque chose d'absurde (vide, trop long), on garde l'original
        if not corrigee or len(corrigee) > len(question) * 3:
            return question
        return corrigee
    except Exception:
        return question

def contextualiser_question(
    question: str,
    historique: List[Dict],
    model: str = MODEL_OLLAMA,
) -> str:
    """
    Si la question dépend du contexte conversationnel (pronom, référence
    implicite, question de suivi du type "juste ça ?", "et pour X ?"),
    la reformule en question autonome à l'aide de l'historique récent.
    Sinon la retourne telle quelle. En cas d'échec, retourne l'originale.
    """
    if not historique:
        return question

    historique_texte = "\n".join(
        f"Q: {tour['question']}\nR: {tour['reponse']}" for tour in historique
    )

    prompt = f"""Voici l'historique récent d'une conversation entre un employé et un assistant RH, suivi d'une nouvelle question.

Historique :
{historique_texte}

Nouvelle question : {question}

Si la nouvelle question dépend du contexte (pronom, référence implicite, question de suivi comme "et pour X ?", "juste ça ?", "et le formulaire ?"), reformule-la en question autonome et complète, compréhensible sans le contexte.
Ne reprends PAS une unité de mesure (jours, semaines, mois...) de la question précédente si le nouveau sujet peut avoir une unité différente : reste neutre sur l'unité, ne demande pas explicitement une unité précise sauf si l'employé l'a fait lui-même dans sa nouvelle question.
Si elle est déjà autonome, renvoie-la telle quelle.
Renvoie UNIQUEMENT la question finale, sans commentaire ni guillemets.

Question finale:"""

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options={"temperature": 0, "num_predict": 80},
            keep_alive="30m",

        )
        contextualisee = response.get("response", "").strip().strip('"')
        if not contextualisee:
            return question
        return contextualisee
    except Exception:
        return question

def generer_reponse(
    question: str,
    historique: List[Dict] = None,          # ajouté
    model: str = MODEL_OLLAMA,
    top_k: int = 6,
    similarity_threshold: float = 0.3,
    top_k_formulaires: int = 2,
    seuil_formulaires: float = 0.3,
) -> Dict[str, object]:
    """
    Génère une réponse basée sur les documents RH pertinents.

    Process:
    1. Récupère les chunks similaires à la question (pour le texte de réponse)
    2. Récupère séparément les formulaires pertinents (pour proposer un
       téléchargement), indépendamment de leur score dans la réponse texte
    3. Construit un prompt avec le contexte
    4. Envoie à Ollama pour génération
    5. Retourne la réponse, les sources ET les formulaires suggérés
    """

    question = contextualiser_question(question, historique or [], model=model)  # ajouté, avant reformuler_question
    question = reformuler_question(question, model=model)


    # Étape 1 : Récupération des chunks pertinents pour la réponse texte
    chunks = rechercher_similaires(
        question=question,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )


    # Priorise les chunks sommaire en tête du contexte, s'ils sont présents
    chunks_sommaire = [c for c in chunks if c["chunk_id"] == "sommaire"]
    autres_chunks = [c for c in chunks if c["chunk_id"] != "sommaire"]
    chunks=chunks_sommaire+autres_chunks


    # Étape 2 : Recherche dédiée aux formulaires (indépendante du top_k
    # ci-dessus, pour ne pas rater un formulaire pertinent noyé par des
    # chunks de politique mieux classés sur le même sujet)
    formulaires_suggeres = rechercher_formulaires(
        question=question,
        top_k=top_k_formulaires,
        similarity_threshold=seuil_formulaires,
    )

    if not chunks:
        return {
            "reponse": "Désolé, je n'ai pas trouvé de documents pertinents pour répondre à votre question.",
            "sources": [],
            "formulaires": formulaires_suggeres,
            "question": question,
        }

    # Étape 3 : Construction du contexte et du prompt
    contexte = construire_contexte(chunks)
    print("=== CHUNKS ENVOYÉS AU LLM ===")
    for c in chunks:
         print(f"  {c['source']} #{c['chunk_id']} (score={c['score']:.3f})")
    print("==============================")


    prompt = f"""Tu es un assistant RH chez AVL Maroc. Réponds à la question de l'employé en te basant UNIQUEMENT sur les documents fournis ci-dessous.

RÈGLES :
1. Réponds directement, sans salutation ni préambule.
2. Rédige en phrases naturelles. N'utilise des puces que si plusieurs éléments distincts.
3. Cite la référence du document naturellement dans la phrase si pertinent.
4. N'INFÈRE JAMAIS une information qui n'est pas écrite explicitement. Si les documents décrivent un délai de préavis pour déposer une demande, ne transforme JAMAIS cela en durée de congé accordée, ou en toute autre règle non explicitement énoncée.
5. Si AUCUNE information pertinente à la question n'est présente dans les documents, réponds uniquement : "Je ne trouve pas cette information dans les documents RH."
6. Si une partie de la question a une réponse dans les documents (même si ce n'est pas une réponse complète ou exacte au sens strict), donne cette information partielle clairement, en précisant qu'elle ne couvre peut-être pas tout le sujet. Ne réponds "je ne trouve pas" que si vraiment rien d'utile n'est disponible.Exemple de bonne réponse en cas d'info partielle : "Les documents ne précisent pas le nombre total de jours de congé annuel. Ils mentionnent en revanche un congé de maternité de 14 semaines et un congé exceptionnel de 3 jours par an."
7. Ne mélange JAMAIS une information (chiffre, durée, condition) associée à un type de congé avec un autre type de congé. Chaque durée ou règle doit rester strictement rattachée au sujet auquel le document l'associe. Reprends TOUJOURS l'unité exacte utilisée dans le document (jours, semaines, ou mois) — ne convertis jamais une unité en une autre, et ne remplace jamais "semaines" par "jours" même si le chiffre te semble similaire.
8. Quand une durée est donnée avec une conversion entre parenthèses (ex: "14 semaines (98 jours)"), ces deux valeurs désignent EXACTEMENT la même durée. Ne prends jamais le chiffre de l'une avec l'unité de l'autre (ex: ne dis jamais "14 jours"). Utilise l'unité principale (avant la parenthèse) sauf si la question porte explicitement sur l'autre unité.
9. Quand un extrait contient une règle générale ET une ou plusieurs exceptions liées à des critères spécifiques (âge, statut cadre/non-cadre, ancienneté...), réponds par défaut avec la règle générale. N'utilise l'exception que si la question mentionne explicitement le critère correspondant (ex: "moins de 18 ans", "cadre").
10. Si un document contient une section "Ce document couvre les sections suivantes", utilise-la en priorité pour répondre aux questions demandant une liste ou une énumération complète (ex: "quels sont les types de..."). Cite bien les éléments qui y figurent, même si les autres extraits ne détaillent pas chacun d'eux.
11. Avant de répondre avec un chiffre, une durée, ou un taux, identifie précisément à quelle ligne, condition, ou catégorie de la question il correspond (ex: pays, type de transport, jour de la semaine, statut cadre/non-cadre, ancienneté). Si un document contient plusieurs valeurs proches pour des conditions différentes (ex: un tableau ou une liste), ne prends QUE la valeur correspondant exactement à la situation décrite dans la question — ignore les autres valeurs même si elles sont proches dans le texte.
12. Ne jamais associer une fréquence, un délai ou une condition (ex: "une fois par an", "après 6 ans d'ancienneté") à un type de congé si cette information ne provient pas du même passage de texte que ce congé. Si un chiffre ou une fréquence n'est pas explicitement rattaché au congé mentionné dans la question, ne l'invente pas et ne l'emprunte pas à un autre congé du contexte.
13. Reprends TOUJOURS les termes exacts du document pour désigner qui bénéficie d'un droit (ex: "le salarié", "la salariée", "le conjoint", "l'enfant"). Ne substitue jamais un terme par un autre semblable, même si le sens te paraît proche — par exemple, ne confonds jamais "le salarié se marie" avec "le conjoint du salarié se marie".
14. Si deux documents différents semblent décrire le même droit ou avantage sous des noms ou sections différentes (ex: deux passages parlant tous deux d'un congé de 3 jours à la naissance d'un enfant), signale explicitement ce chevauchement au lieu d'inventer une distinction entre eux qui n'est écrite nulle part.

Documents:

{contexte}

Question de l'employé: {question}

Réponse:"""

    # Étape 4 : Appel à Ollama
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options={"temperature": 0},

        )
        reponse_texte = response.get("response", "").strip()
    except Exception as e:
        return {
            "reponse": f"Erreur lors de la génération: {e}",
            "sources": [],
            "formulaires": formulaires_suggeres,
            "question": question,
        }

    # Étape 5 : Retour structuré
    sources = [
        {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "score": chunk["score"],
        }
        for chunk in chunks
    ]

    return {
        "reponse": reponse_texte,
        "sources": sources,
        "formulaires": formulaires_suggeres,
        "question": question,
    }


if __name__ == "__main__":
    # Test simple
    questions_test = [
        "Quelle est la politique de télétravail ?",
        "Combien de jours de congés ai-je par an ?",
        "Comment demander un congé ?",
        "Comment déclarer que mon domicile est conforme pour le télétravail ?",
        "Je veux me faire rembourser mon hôtel",
    ]

    for q in questions_test:
        print("=" * 60)
        print(f"Question: {q}\n")
        resultat = generer_reponse(q)
        print(f"Réponse:\n{resultat['reponse']}\n")

        print("Sources:")
        for source in resultat["sources"]:
            print(f"  - {source['source']} (Chunk #{source['chunk_id']}, Score: {source['score']:.3f})")

        if resultat["formulaires"]:
            print("\n📎 Formulaire(s) suggéré(s) :")
            for f in resultat["formulaires"]:
                print(f"  - {f['source']} (Score: {f['score']:.3f}) -> {f['chemin_complet']}")
        print()