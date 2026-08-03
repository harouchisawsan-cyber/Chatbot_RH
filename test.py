"""
Script de test du chatbot RH
-----------------------------
Envoie une série de scénarios à l'API /chat et sauvegarde toutes les
réponses (+ métadonnées) dans un fichier JSON horodaté, pour diagnostic
manuel après coup.

Catégories de scénarios couvertes :
- factuel_simple           : une question, une réponse attendue claire
- memoire_conversationnelle : plusieurs tours chaînés via session_id,
                              pour tester les questions de suivi elliptiques
- formulaire                : vérifie qu'un formulaire pertinent (ou aucun)
                              est bien suggéré
- ambigu_ou_conflit          : questions où plusieurs documents se
                               recoupent ou semblent se contredire
- hors_perimetre             : questions sans réponse dans les documents

Usage :
    pip install requests   # si pas déjà installé
    python test_chatbot.py

Résultat :
    Un fichier test_results/resultats_AAAAMMJJ_HHMMSS.json est créé,
    contenant le détail complet de chaque échange, plus un résumé
    lisible affiché dans le terminal et sauvegardé en .md à côté.
"""

import json
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
OUTPUT_DIR = Path("test_results")


# ---------------------------------------------------------------------------
# Définition des scénarios de test
# ---------------------------------------------------------------------------
# Chaque scénario est une liste de "tours". Les tours d'un même scénario
# sont enchaînés avec le même session_id (pour tester la mémoire quand
# il y a plusieurs tours). "mots_cles_attendus" est une liste de mots/
# expressions dont on s'attend à trouver AU MOINS UN dans la réponse
# (vérification grossière, juste pour repérer vite les échecs évidents
# - une lecture manuelle reste nécessaire pour juger la qualité réelle).

SCENARIOS = [
    {
        "id": "teletravail_jours_max",
        "categorie": "factuel_simple",
        "tours": [
            {
                "question": "Combien de jours de télétravail par mois puis-je faire ?",
                "mots_cles_attendus": ["13 jours", "13"],
            }
        ],
    },
    {
        "id": "conges_mariage",
        "categorie": "factuel_simple",
        "tours": [
            {
                "question": "Combien de jours de congé pour un mariage ?",
                "mots_cles_attendus": ["4 jours"],
            }
        ],
    },
    {
        "id": "conges_deces_conjoint",
        "categorie": "factuel_simple",
        "tours": [
            {
                "question": "Combien de jours de congé en cas de décès du conjoint ?",
                "mots_cles_attendus": ["3 jours"],
            }
        ],
    },
    {
        "id": "deplacement_indemnite_km",
        "categorie": "factuel_simple",
        "tours": [
            {
                "question": "Quel est le montant de l'indemnité kilométrique pour un déplacement professionnel ?",
                "mots_cles_attendus": ["3 MAD", "3 dh", "3 dirhams"],
            }
        ],
    },
    {
        "id": "deplacement_repas_maroc",
        "categorie": "factuel_simple",
        "tours": [
            {
                "question": "Quel est le plafond de remboursement pour le déjeuner lors d'un déplacement au Maroc ?",
                "mots_cles_attendus": ["200 MAD", "200 dh"],
            }
        ],
    },
    {
        "id": "reglement_heures_sup",
        "categorie": "factuel_simple",
        "tours": [
            {
                "question": "Quelle est la majoration de salaire pour les heures supplémentaires effectuées la journée ?",
                "mots_cles_attendus": ["25%", "25 %"],
            }
        ],
    },
    {
        "id": "reglement_prime_anciennete",
        "categorie": "factuel_simple",
        "tours": [
            {
                "question": "Quel est le taux de la prime d'ancienneté après 5 ans dans l'entreprise ?",
                "mots_cles_attendus": ["10%", "10 %"],
            }
        ],
    },
    {
        "id": "reglement_preavis_cadre",
        "categorie": "factuel_simple",
        "tours": [
            {
                "question": "Quel est le préavis pour un cadre ayant 6 ans d'ancienneté qui démissionne ?",
                "mots_cles_attendus": ["trois mois", "3 mois"],
            }
        ],
    },
    {
        "id": "memoire_maternite_puis_indemnite",
        "categorie": "memoire_conversationnelle",
        "tours": [
            {
                "question": "Quelle est la durée du congé maternité ?",
                "mots_cles_attendus": ["14 semaines"],
            },
            {
                "question": "et l'indemnité versée pendant ce congé, comment elle est calculée ?",
                "mots_cles_attendus": ["CNSS", "6 000", "6000", "100%", "100 %"],
            },
            {
                "question": "et pour le père, il a droit à quoi ?",
                "mots_cles_attendus": ["3 jours", "paternité"],
            },
        ],
    },
    {
        "id": "memoire_conges_familiaux_puis_naissance",
        "categorie": "memoire_conversationnelle",
        "tours": [
            {
                "question": "Quels sont les congés possibles pour événements familiaux ?",
                "mots_cles_attendus": ["mariage", "décès", "naissance"],
            },
            {
                "question": "et juste pour une naissance ?",
                "mots_cles_attendus": ["3 jours"],
            },
        ],
    },
    {
        "id": "formulaire_demande_teletravail",
        "categorie": "formulaire",
        "tours": [
            {
                "question": "Je veux faire une demande de télétravail, quel formulaire dois-je remplir ?",
                "mots_cles_attendus": [],
                "verifier_formulaire_present": True,
            }
        ],
    },
    {
        "id": "formulaire_absent_sur_question_disciplinaire",
        "categorie": "formulaire",
        "tours": [
            {
                "question": "Quelles sont les sanctions disciplinaires possibles en cas de faute grave ?",
                "mots_cles_attendus": ["vol", "abus de confiance", "faute grave"],
                "verifier_formulaire_absent": True,
            }
        ],
    },
    {
        "id": "conflit_delai_declaration_maladie",
        "categorie": "ambigu_ou_conflit",
        "tours": [
            {
                "question": "Dans quel délai dois-je informer l'entreprise en cas de maladie ?",
                "mots_cles_attendus": ["24", "48"],
            }
        ],
    },
    {
        "id": "conflit_conges_naissance_vs_paternite",
        "categorie": "ambigu_ou_conflit",
        "tours": [
            {
                "question": "Quelle est la différence entre le congé de naissance et le congé de paternité ?",
                "mots_cles_attendus": ["3 jours"],
            }
        ],
    },
    {
        "id": "conges_annuel_payes_formule",
        "categorie": "ambigu_ou_conflit",
        "tours": [
            {
                "question": "Combien de jours de congé annuel payé ai-je droit par mois de travail ?",
                "mots_cles_attendus": ["jour et demi", "1,5", "1.5"],
            }
        ],
    },
    {
        "id": "hors_perimetre_salaire_marche",
        "categorie": "hors_perimetre",
        "tours": [
            {
                "question": "Quel est le salaire moyen d'un développeur web au Maroc ?",
                "mots_cles_attendus": ["ne trouve pas", "ne précise pas"],
            }
        ],
    },
    {
        "id": "hors_perimetre_meteo",
        "categorie": "hors_perimetre",
        "tours": [
            {
                "question": "Quel temps fait-il aujourd'hui à Casablanca ?",
                "mots_cles_attendus": ["ne trouve pas", "ne précise pas"],
            }
        ],
    },
    {
        "id": "ambigu_sans_contexte_prealable",
        "categorie": "ambigu_ou_conflit",
        "tours": [
            {
                "question": "et pour les enfants ?",
                "mots_cles_attendus": [],
            }
        ],
    },
    {
        "id": "complexe_deplacement_europe_longue_duree",
        "categorie": "factuel_simple",
        "tours": [
            {
                "question": "Je pars en mission de deux mois en Allemagne, quel forfait mensuel vais-je toucher et ai-je droit à une prime supplémentaire ?",
                "mots_cles_attendus": ["110", "3 000", "3000", "prime d'éloignement"],
            }
        ],
    },
    {
        "id": "memoire_deplacement_maroc_puis_europe",
        "categorie": "memoire_conversationnelle",
        "tours": [
            {
                "question": "Quel est le plafond hôtel pour un déplacement courte durée au Maroc ?",
                "mots_cles_attendus": ["800 MAD"],
            },
            {
                "question": "et si c'est en Europe ?",
                "mots_cles_attendus": ["100"],
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def _normaliser(texte: str) -> str:
    """Minuscules + suppression des accents, pour une comparaison plus tolérante."""
    texte = texte.lower()
    texte = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in texte if not unicodedata.combining(c))


def _contient_un_mot_cle(reponse: str, mots_cles: list) -> bool:
    if not mots_cles:
        return True  # rien à vérifier -> pas d'échec sur ce critère
    reponse_norm = _normaliser(reponse)
    return any(_normaliser(mot) in reponse_norm for mot in mots_cles)


def _appeler_chat(question: str, session_id: str | None) -> dict:
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id

    debut = time.time()
    reponse = requests.post(f"{BASE_URL}/chat", json=payload, timeout=500)
    duree = time.time() - debut

    reponse.raise_for_status()
    data = reponse.json()
    data["_latence_secondes"] = round(duree, 2)
    return data


# ---------------------------------------------------------------------------
# Exécution des scénarios
# ---------------------------------------------------------------------------

def executer_scenarios() -> list:
    resultats = []

    for scenario in SCENARIOS:
        print(f"\n{'=' * 70}")
        print(f"Scénario : {scenario['id']} [{scenario['categorie']}]")
        print("=" * 70)

        session_id = None
        tours_resultats = []

        for i, tour in enumerate(scenario["tours"], 1):
            question = tour["question"]
            print(f"\n  Tour {i} — Q: {question}")

            try:
                data = _appeler_chat(question, session_id)
            except Exception as e:
                print(f"  ❌ ERREUR D'APPEL : {e}")
                tours_resultats.append({
                    "question_posee": question,
                    "erreur": str(e),
                })
                break

            session_id = data.get("session_id", session_id)
            reponse_texte = data.get("reponse", "")

            mots_cles = tour.get("mots_cles_attendus", [])
            mots_cles_ok = _contient_un_mot_cle(reponse_texte, mots_cles)

            formulaires = data.get("formulaires", [])
            verif_form_presente = tour.get("verifier_formulaire_present", False)
            verif_form_absente = tour.get("verifier_formulaire_absent", False)

            formulaire_ok = True
            if verif_form_presente and not formulaires:
                formulaire_ok = False
            if verif_form_absente and formulaires:
                formulaire_ok = False

            statut = "✅" if (mots_cles_ok and formulaire_ok) else "⚠️ À VÉRIFIER"

            print(f"  R: {reponse_texte[:200]}{'...' if len(reponse_texte) > 200 else ''}")
            print(f"  Question reformulée (mémoire) : {data.get('question')}")
            if formulaires:
                print(f"  📎 Formulaires suggérés : {[f['source'] for f in formulaires]}")
            print(f"  Statut : {statut} (mots-clés {'ok' if mots_cles_ok else 'MANQUANTS'}, "
                  f"formulaire {'ok' if formulaire_ok else 'INATTENDU'})")

            tours_resultats.append({
                "question_posee": question,
                "question_reformulee": data.get("question"),
                "reponse": reponse_texte,
                "sources": data.get("sources", []),
                "formulaires": formulaires,
                "session_id": session_id,
                "latence_secondes": data.get("_latence_secondes"),
                "mots_cles_attendus": mots_cles,
                "mots_cles_trouves": mots_cles_ok,
                "formulaire_check_ok": formulaire_ok,
                "statut": statut,
            })

        resultats.append({
            "scenario_id": scenario["id"],
            "categorie": scenario["categorie"],
            "tours": tours_resultats,
        })

    return resultats


def sauvegarder_resultats(resultats: list) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_json = OUTPUT_DIR / f"resultats_{horodatage}.json"
    chemin_json.write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Résumé lisible en Markdown
    chemin_md = OUTPUT_DIR / f"resume_{horodatage}.md"
    lignes = ["# Résumé des tests\n"]
    total = 0
    a_verifier = 0
    for scenario in resultats:
        lignes.append(f"\n## {scenario['scenario_id']} ({scenario['categorie']})\n")
        for i, tour in enumerate(scenario["tours"], 1):
            total += 1
            if tour.get("statut") != "✅":
                a_verifier += 1
            lignes.append(f"- Tour {i} : {tour.get('statut', '❌ erreur')} — "
                           f"\"{tour.get('question_posee', '')}\"")
    lignes.insert(1, f"\n**{total} tour(s) testé(s), {a_verifier} à vérifier manuellement.**\n")
    chemin_md.write_text("\n".join(lignes), encoding="utf-8")

    return chemin_json, chemin_md


if __name__ == "__main__":
    print(f"🚀 Lancement des tests contre {BASE_URL} ...")
    resultats = executer_scenarios()
    chemin_json, chemin_md = sauvegarder_resultats(resultats)

    print(f"\n{'=' * 70}")
    print(f"✅ Résultats détaillés : {chemin_json}")
    print(f"✅ Résumé lisible      : {chemin_md}")
    print("=" * 70)