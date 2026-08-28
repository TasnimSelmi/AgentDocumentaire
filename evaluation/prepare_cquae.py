"""
Préparation du benchmark CQuAE pour l'évaluation externe de l'agent.

Ce script ne fait QUE préparer des données : télécharger, auditer, vérifier
les liens question <-> document, sélectionner un sous-ensemble stratifié,
écrire un corpus documentaire local et un fichier gold au format pivot du
harnais (`evaluation.common.Enregistrement`). Il ne modifie ni le RAG, ni les
outils, ni l'agent, et ne déclenche aucune ingestion.

Sources officielles (Hugging Face)
-----------------------------------
    LsTam/CQuAE            questions, réponses de référence, evidence gold.
    LsTam/CQuAE_documents  corpus documentaire source (1150 « documents »,
                            en réalité des sections d'articles éducatifs).

Les révisions sont figées ci-dessous (SHA du commit HF au moment de la
préparation) pour que deux exécutions de ce script, même après une mise à
jour amont du dataset, téléchargent exactement les mêmes données. Pour
rafraîchir volontairement vers la dernière version, passer --revision-latest.

Structure réelle constatée (voir le rapport imprimé par --verbose ou le JSON
écrit dans evaluation/reports/prepare_cquae/) :

    qa[split]   : question, title, documents (list[str]), documents_title
                  (list[str]), type, qid (int), output (str)
    docs[train] : title (str), did (int), data (list[{source_text,title,img}]),
                  collection (str), url (str)

Le lien question -> document est un appariement EXACT sur le champ `title` :
`qa[split][i]['title'] == docs['train'][j]['title']`. Chaque évidence de
`documents[k]` est le texte verbatim d'une section `data[m]['source_text']`
du document apparié, retrouvée via `documents_title[k] == data[m]['title']`.
Ce lien a été vérifié empiriquement (voir section « Vérification des liens »)
avant d'écrire une seule ligne de ce script.

Aucune question de CQuAE n'est marquée sans réponse : `answerable=True`
partout. Ce benchmark ne permet donc pas de tester le refus — c'est déjà le
rôle de finance_esg.jsonl pour ce projet, pas de CQuAE.

Exemples
--------
    # Aperçu sans rien écrire sur disque (audit + sélection, pas d'écriture)
    python -m evaluation.prepare_cquae --dry-run --verbose

    # Préparation complète (~240 questions, seed par défaut du harnais)
    python -m evaluation.prepare_cquae

    # Reproductibilité : deux runs avec le même seed donnent le même jeu
    python -m evaluation.prepare_cquae --seed 20240601
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.common import (
    DOSSIER_DONNEES,
    DOSSIER_RAPPORTS,
    RACINE_PROJET,
    SEED_PAR_DEFAUT,
    Enregistrement,
    configurer_logs,
    ecrire_jsonl,
    fixer_seed,
    horodatage,
)

logger = logging.getLogger("evaluation.prepare_cquae")

# ===========================================================================
# Sources officielles
# ===========================================================================

JEU_QUESTIONS = "LsTam/CQuAE"
REVISION_QUESTIONS = "63481c6431f699bd740846cf9b52f9316f4ef860"
JEU_DOCUMENTS = "LsTam/CQuAE_documents"
REVISION_DOCUMENTS = "81be8b8ea94364e4895444bbe8beda55ab63c8b7"

SPLIT_PAR_DEFAUT = "test"
N_CIBLE_PAR_DEFAUT = 240

DOSSIER_CORPUS_PAR_DEFAUT = RACINE_PROJET / "data" / "documents" / "cquae_eval"
FICHIER_GOLD_PAR_DEFAUT = DOSSIER_DONNEES / "cquae_agent_gold.jsonl"
DOSSIER_RAPPORT_PREPARATION = DOSSIER_RAPPORTS / "prepare_cquae"


# ===========================================================================
# 1. Téléchargement
# ===========================================================================


def telecharger(revision_latest: bool = False) -> tuple[Any, Any]:
    """
    Télécharge les deux ressources CQuAE via l'API `datasets`.

    Le cache local de la librairie (~/.cache/huggingface/datasets) rend les
    exécutions suivantes instantanées et hors-ligne. Les révisions sont
    figées par défaut pour la reproductibilité (voir module docstring).
    """
    from datasets import load_dataset

    rev_q = None if revision_latest else REVISION_QUESTIONS
    rev_d = None if revision_latest else REVISION_DOCUMENTS

    logger.info("Téléchargement de %s (révision=%s)…", JEU_QUESTIONS, rev_q or "latest")
    qa = load_dataset(JEU_QUESTIONS, revision=rev_q)

    logger.info("Téléchargement de %s (révision=%s)…", JEU_DOCUMENTS, rev_d or "latest")
    docs = load_dataset(JEU_DOCUMENTS, revision=rev_d)["train"]

    return qa, docs


# ===========================================================================
# 2. Audit de structure
# ===========================================================================


def auditer_structure(qa: Any, docs: Any) -> dict[str, Any]:
    """Inspecte la structure réelle des deux ressources, sans supposer la doc."""
    audit: dict[str, Any] = {
        "questions": {
            "champs": sorted(qa[list(qa.keys())[0]].features.keys()),
            "splits": {},
        },
        "documents": {
            "champs": sorted(docs.features.keys()),
            "n_documents": len(docs),
        },
    }

    for split in qa:
        types = Counter(qa[split]["type"])
        audit["questions"]["splits"][split] = {
            "n": len(qa[split]),
            "types": dict(sorted(types.items())),
        }

    return audit


# ===========================================================================
# 3. Index des documents (title -> enregistrement complet)
# ===========================================================================


def indexer_documents(docs: Any) -> dict[str, list[dict[str, Any]]]:
    """
    Construit l'index title -> liste de documents, et leur texte reconstruit.

    Une LISTE, pas un document unique par titre : `title` n'est PAS une clé
    globalement unique sur les 1150 documents. Quatre intitulés génériques de
    fin de chapitre (par ex. « Ce que j'ai appris ») sont réutilisés par 32
    documents différents, appartenant à des chapitres sans rapport. Un
    dict[title, doc] perdrait silencieusement 28 de ces documents (seul le
    dernier lu survivrait) et résoudrait des questions vers le mauvais
    contenu sans jamais lever d'erreur. La désambiguïsation réelle se fait
    dans `resoudre_document()`, sur le contenu de l'evidence, pas sur le
    titre seul.

    Le texte complet est la seule reconstruction fidèle du document source :
    titre + sections, dans l'ordre d'apparition, sans aucune donnée gold.
    """
    index: dict[str, list[dict[str, Any]]] = {}

    for enr in docs:
        morceaux: list[str] = [f"# {enr['title']}".strip()]
        for section in enr["data"]:
            titre_section = (section.get("title") or "").strip()
            texte_section = (section.get("source_text") or "").strip()
            if not texte_section:
                continue
            if titre_section:
                morceaux.append(f"## {titre_section}\n\n{texte_section}")
            else:
                morceaux.append(texte_section)

        index.setdefault(enr["title"], []).append(
            {
                "did": enr["did"],
                "title": enr["title"],
                "collection": enr["collection"],
                "url": enr["url"],
                # Liste, pas dict, pour la même raison qu'au niveau de
                # l'index ci-dessus : des sections d'un même document
                # partagent parfois le même titre (souvent vide).
                "textes_sections": [
                    (s.get("source_text") or "").strip() for s in enr["data"]
                ],
                "texte_complet": "\n\n".join(morceaux).strip(),
            }
        )

    return index


def resoudre_document(
    ex: dict[str, Any], index_titres: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, Any] | None, str]:
    """
    Résout la question vers SON document, en désambiguïsant les titres
    partagés par plusieurs documents grâce au contenu de l'evidence.

    Returns:
        (document, raison_echec) — document est None si non résolu, auquel
        cas raison_echec explique pourquoi (à usage de `verifier_liens`).
    """
    candidats = index_titres.get(ex["title"])
    if not candidats:
        return None, f"document introuvable pour title={ex['title']!r}"

    if len(candidats) == 1:
        return candidats[0], ""

    evidences = [t.strip() for t in ex["documents"]]
    correspondants = [
        c for c in candidats if all(e in c["textes_sections"] for e in evidences)
    ]
    if len(correspondants) == 1:
        return correspondants[0], ""
    if not correspondants:
        return None, (
            f"title={ex['title']!r} partagé par {len(candidats)} documents, "
            "aucun ne contient l'evidence attendue"
        )
    return None, (
        f"title={ex['title']!r} partagé par {len(candidats)} documents, "
        f"{len(correspondants)} contiennent l'evidence attendue (ambigu)"
    )


# ===========================================================================
# 4. Vérification des liens question <-> document <-> evidence
# ===========================================================================


def verifier_liens(
    pool: Any, index_titres: dict[str, list[dict[str, Any]]]
) -> tuple[list[int], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """
    Vérifie, pour chaque question du pool, que le document et les sections
    d'evidence attendus se résolvent réellement, sans ambiguïté, dans le
    corpus documentaire.

    Ne corrige rien : une question dont le lien est cassé ou ambigu est
    exclue et signalée avec la raison exacte. Le document résolu (unique,
    via `resoudre_document`) est conservé pour éviter toute nouvelle
    résolution en aval, qui rouvrirait le risque de collision de titre.

    Returns:
        (indices_valides, resolutions {index: document}, exclusions)
    """
    valides: list[int] = []
    resolutions: dict[int, dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []

    for i, ex in enumerate(pool):
        qid = ex["qid"]

        if len(ex["documents"]) != len(ex["documents_title"]):
            exclusions.append(
                {
                    "qid": qid,
                    "raison": (
                        "documents et documents_title de longueurs différentes "
                        f"({len(ex['documents'])} vs {len(ex['documents_title'])})"
                    ),
                }
            )
            continue

        doc, raison = resoudre_document(ex, index_titres)
        if doc is None:
            exclusions.append({"qid": qid, "raison": raison})
            continue

        if not ex["output"].strip():
            exclusions.append({"qid": qid, "raison": "gold_answer vide"})
            continue

        valides.append(i)
        resolutions[i] = doc

    return valides, resolutions, exclusions


# ===========================================================================
# 5. Sélection stratifiée, déterministe
# ===========================================================================


def allocation_stratifiee(compte_par_type: dict[str, int], n_cible: int) -> dict[str, int]:
    """
    Alloue n_cible questions entre les types, proportionnellement à la
    distribution réelle du pool (méthode du plus grand reste).

    Déterministe : à distribution égale, les egalités de reste sont
    tranchées par ordre alphabétique du type, jamais par ordre d'itération
    d'un dict (non garanti stable entre exécutions Python).
    """
    total = sum(compte_par_type.values())
    if total == 0:
        return {t: 0 for t in compte_par_type}

    quotas: dict[str, int] = {}
    restes: dict[str, float] = {}
    for t, n in compte_par_type.items():
        part = n_cible * n / total
        quotas[t] = int(part)
        restes[t] = part - quotas[t]

    manque = n_cible - sum(quotas.values())
    ordre = sorted(compte_par_type, key=lambda t: (-restes[t], t))
    for t in ordre[:manque]:
        quotas[t] += 1

    # Garde-fou : ne jamais promettre plus que ce que le type contient.
    for t in quotas:
        quotas[t] = min(quotas[t], compte_par_type[t])

    return quotas


def selectionner(
    pool: Any,
    indices_valides: list[int],
    *,
    n_cible: int,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    """
    Échantillonnage stratifié par `type`, seed fixe, ordre de traitement
    déterministe (types triés alphabétiquement, questions triées par qid
    avant tirage).
    """
    rng = fixer_seed(seed)

    par_type: dict[str, list[int]] = {}
    for i in indices_valides:
        par_type.setdefault(pool[i]["type"], []).append(i)
    for t in par_type:
        par_type[t].sort(key=lambda i: pool[i]["qid"])

    compte_par_type = {t: len(v) for t, v in par_type.items()}
    quotas = allocation_stratifiee(compte_par_type, n_cible)

    selection: list[int] = []
    for t in sorted(par_type):  # ordre alphabétique : déterministe
        tirage = rng.sample(par_type[t], quotas[t])
        selection.extend(tirage)

    selection.sort(key=lambda i: pool[i]["qid"])  # ordre de sortie stable

    rapport = {
        "n_cible": n_cible,
        "n_selectionne": len(selection),
        "distribution_pool": compte_par_type,
        "quotas_alloues": quotas,
        "distribution_obtenue": dict(
            sorted(Counter(pool[i]["type"] for i in selection).items())
        ),
    }
    return selection, rapport


# ===========================================================================
# 6. Corpus documentaire local
# ===========================================================================


def construire_corpus(
    docs_uniques: dict[int, dict[str, Any]],
    dossier_corpus: Path,
    *,
    ecrire: bool = True,
) -> dict[int, str]:
    """
    Écrit un fichier .txt par document unique référencé par la sélection.

    Le contenu est exactement `texte_complet` (titre + sections sources) —
    aucune réponse gold, aucune evidence marquée spécialement. Nom de
    fichier stable : `cquae_doc_{did}.txt`, où `did` est l'identifiant CQuAE
    d'origine (conservé aussi dans les métadonnées du benchmark gold). `did`
    est la seule clé fiable ici (voir `indexer_documents` : `title` ne l'est
    pas).
    """
    mapping: dict[int, str] = {}

    if ecrire:
        dossier_corpus.mkdir(parents=True, exist_ok=True)

    for did in sorted(docs_uniques):
        doc = docs_uniques[did]
        nom_fichier = f"cquae_doc_{did}.txt"
        mapping[did] = nom_fichier

        if ecrire:
            chemin = dossier_corpus / nom_fichier
            chemin.write_text(doc["texte_complet"] + "\n", encoding="utf-8")

    return mapping


# ===========================================================================
# 7. Benchmark gold (format pivot du harnais)
# ===========================================================================


def construire_enregistrements(
    pool: Any,
    selection: list[int],
    resolutions: dict[int, dict[str, Any]],
    mapping_fichiers: dict[int, str],
    *,
    split: str,
) -> list[Enregistrement]:
    enregistrements: list[Enregistrement] = []

    for i in selection:
        ex = pool[i]
        doc = resolutions[i]

        enregistrements.append(
            Enregistrement(
                id=f"cquae:{split}:{ex['qid']}",
                question=ex["question"],
                expected_answer=ex["output"].strip(),
                expected_document=mapping_fichiers[doc["did"]],
                evidence_text="\n\n".join(t.strip() for t in ex["documents"]),
                page=None,
                answerable=True,
                subset=f"cquae_{split}",
                question_type=ex["type"],
                answer_variants=[],
                metadata={
                    "source_dataset": "CQuAE",
                    "original_split": split,
                    "original_id": ex["qid"],
                    "cquae_did": doc["did"],
                    "cquae_title": doc["title"],
                    "cquae_collection": doc["collection"],
                    "cquae_url": doc["url"],
                    "cquae_documents_title": list(ex["documents_title"]),
                    "hf_revision_questions": REVISION_QUESTIONS,
                    "hf_revision_documents": REVISION_DOCUMENTS,
                },
            )
        )

    return enregistrements


# ===========================================================================
# 8. Contrôles qualité
# ===========================================================================


def controles_qualite(
    enregistrements: list[Enregistrement],
    dossier_corpus: Path,
    docs_uniques: dict[int, dict[str, Any]],
    *,
    pool: Any,
) -> dict[str, Any]:
    """Vérifie automatiquement les points listés dans la mission (section 12)."""
    resultats: dict[str, Any] = {}

    # 1. IDs uniques.
    ids = [e.id for e in enregistrements]
    resultats["ids_uniques"] = len(ids) == len(set(ids))

    # 2. gold_answer exploitable (non vide) pour chaque question.
    resultats["gold_answer_non_vide"] = all(
        e.expected_answer.strip() for e in enregistrements
    )

    # 3 + 6. expected_document présent dans le corpus local préparé.
    fichiers_attendus = {e.expected_document for e in enregistrements}
    fichiers_presents = {p.name for p in dossier_corpus.glob("*.txt")}
    manquants = sorted(fichiers_attendus - fichiers_presents)
    resultats["tous_documents_gold_presents"] = not manquants
    resultats["documents_gold_manquants"] = manquants

    # 4. Aucune référence documentaire cassée (expected_document non vide).
    resultats["aucune_reference_cassee"] = all(
        bool(e.expected_document) for e in enregistrements
    )

    # 5. Questions dupliquées (texte identique, qid différent) — signalé,
    #    pas exclu : ce sont de vraies questions distinctes du benchmark.
    textes = Counter(e.question.strip().lower() for e in enregistrements)
    doublons = {t: n for t, n in textes.items() if n > 1}
    resultats["questions_dupliquees_dans_selection"] = len(doublons)

    # 7. Aucun gold answer / evidence injecté dans le corpus : le contenu
    #    écrit doit être EXACTEMENT la reconstruction depuis index_titres,
    #    jamais le texte source + une réponse ajoutée après coup.
    fichiers_divergents: list[str] = []
    chevauchements_lexicaux: list[str] = []
    fichiers_verifies: set[str] = set()
    for e in enregistrements:
        if e.expected_document in fichiers_verifies:
            continue
        fichiers_verifies.add(e.expected_document)

        doc = docs_uniques[e.metadata["cquae_did"]]
        chemin = dossier_corpus / e.expected_document
        if not chemin.exists():
            continue
        contenu = chemin.read_text(encoding="utf-8")
        if contenu.strip() != doc["texte_complet"].strip():
            fichiers_divergents.append(e.expected_document)

    for e in enregistrements:
        # Signal informatif seulement : une réponse abstractive peut, par
        # coïncidence, être une sous-chaîne de son propre document source
        # (l'evidence en fait partie). Ce n'est pas une fuite tant que le
        # fichier reste la reconstruction exacte vérifiée ci-dessus.
        chemin = dossier_corpus / e.expected_document
        if chemin.exists() and e.expected_answer.strip():
            contenu = chemin.read_text(encoding="utf-8")
            if e.expected_answer.strip() in contenu:
                chevauchements_lexicaux.append(e.id)

    resultats["corpus_non_contamine"] = not fichiers_divergents
    resultats["fichiers_divergents_de_la_reconstruction"] = sorted(fichiers_divergents)
    resultats["chevauchements_lexicaux_reponse_dans_document"] = chevauchements_lexicaux

    # 8. Répartition par type.
    resultats["repartition_type"] = dict(
        sorted(Counter(e.question_type for e in enregistrements).items())
    )

    # 9. Reproductibilité : recalculer la sélection avec le même seed doit
    #    donner exactement les mêmes IDs (vérifié par l'appelant, voir main()).
    resultats["seed_reproductibilite_verifiee"] = None  # rempli par main()

    return resultats


# ===========================================================================
# 9. Rapport / inspection manuelle
# ===========================================================================


def exemples_inspection(enregistrements: list[Enregistrement], n: int = 7) -> list[dict[str, Any]]:
    """Échantillon fixe (premiers n par id trié) pour revue humaine du rapport."""
    tries = sorted(enregistrements, key=lambda e: e.id)
    pas = max(1, len(tries) // n)
    echantillon = tries[::pas][:n]

    return [
        {
            "id": e.id,
            "question": e.question,
            "gold_answer": e.expected_answer,
            "expected_document": e.expected_document,
            "expected_evidence": e.evidence_text[:400]
            + ("…" if len(e.evidence_text) > 400 else ""),
            "question_type": e.question_type,
        }
        for e in echantillon
    ]


# ===========================================================================
# CLI
# ===========================================================================


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Prépare le benchmark CQuAE (download -> audit -> select -> corpus -> gold)."
    )
    parseur.add_argument("--split", default=SPLIT_PAR_DEFAUT, help="split CQuAE source du pool")
    parseur.add_argument("--n", type=int, default=N_CIBLE_PAR_DEFAUT, help="taille cible du sous-ensemble")
    parseur.add_argument("--seed", type=int, default=SEED_PAR_DEFAUT)
    parseur.add_argument("--dossier-corpus", type=Path, default=DOSSIER_CORPUS_PAR_DEFAUT)
    parseur.add_argument("--fichier-gold", type=Path, default=FICHIER_GOLD_PAR_DEFAUT)
    parseur.add_argument(
        "--revision-latest",
        action="store_true",
        help="ignore les révisions figées, télécharge la dernière version des datasets",
    )
    parseur.add_argument(
        "--dry-run",
        action="store_true",
        help="télécharge, audite et sélectionne, mais n'écrit rien sur disque",
    )
    parseur.add_argument("--verbose", action="store_true")
    return parseur


def main(argv: list[str] | None = None) -> int:
    args = construire_parseur().parse_args(argv)
    configurer_logs(args.verbose)

    print("=" * 78)
    print("PRÉPARATION DU BENCHMARK CQuAE")
    print("=" * 78)

    # --- 1. Téléchargement ---------------------------------------------------
    qa, docs = telecharger(revision_latest=args.revision_latest)

    if args.split not in qa:
        logger.error("Split %r introuvable. Splits disponibles : %s", args.split, list(qa.keys()))
        return 1
    pool = qa[args.split]

    # --- 2. Audit --------------------------------------------------------------
    audit = auditer_structure(qa, docs)
    print("\n--- Audit de structure ---")
    print(f"Splits questions : {audit['questions']['splits']}")
    print(f"Champs questions : {audit['questions']['champs']}")
    print(f"Documents : {audit['documents']['n_documents']} (champs : {audit['documents']['champs']})")

    # --- 3. Index documentaire ---------------------------------------------
    index_titres = indexer_documents(docs)

    # --- 4. Vérification des liens (sur le pool entier avant sélection) -----
    indices_valides, resolutions, exclusions = verifier_liens(pool, index_titres)
    print("\n--- Vérification des liens question <-> document <-> evidence ---")
    print(f"Pool ({args.split}) : {len(pool)} questions, {len(indices_valides)} valides, "
          f"{len(exclusions)} exclue(s).")
    for exc in exclusions[:10]:
        print(f"  exclue qid={exc['qid']} : {exc['raison']}")

    # --- 5. Sélection stratifiée --------------------------------------------
    selection, rapport_selection = selectionner(
        pool, indices_valides, n_cible=args.n, seed=args.seed
    )
    # Contrôle de reproductibilité : même seed -> même sélection.
    selection_bis, _ = selectionner(pool, indices_valides, n_cible=args.n, seed=args.seed)
    seed_ok = selection == selection_bis

    print("\n--- Sélection stratifiée ---")
    print(f"Seed : {args.seed}")
    print(f"Distribution du pool valide par type : {rapport_selection['distribution_pool']}")
    print(f"Quotas alloués                       : {rapport_selection['quotas_alloues']}")
    print(f"Distribution obtenue                 : {rapport_selection['distribution_obtenue']}")
    print(f"Questions sélectionnées : {rapport_selection['n_selectionne']} / {args.n} ciblées")
    print(f"Reproductibilité (double tirage identique) : {seed_ok}")

    docs_uniques = {resolutions[i]["did"]: resolutions[i] for i in selection}
    print(f"Documents uniques nécessaires : {len(docs_uniques)}")

    # --- 6. Corpus documentaire local ---------------------------------------
    mapping_fichiers = construire_corpus(
        docs_uniques, args.dossier_corpus, ecrire=not args.dry_run
    )

    # --- 7. Benchmark gold ---------------------------------------------------
    enregistrements = construire_enregistrements(
        pool, selection, resolutions, mapping_fichiers, split=args.split
    )

    # --- 8. Contrôles qualité -------------------------------------------------
    if not args.dry_run:
        qc = controles_qualite(enregistrements, args.dossier_corpus, docs_uniques, pool=pool)
    else:
        qc = {"note": "contrôles fichiers ignorés en --dry-run (rien n'est écrit)"}
    qc["seed_reproductibilite_verifiee"] = seed_ok

    print("\n--- Contrôles qualité ---")
    for cle, valeur in qc.items():
        print(f"  {cle}: {valeur}")

    # --- 9. Inspection manuelle (rapport uniquement, jamais dans le corpus) -
    echantillon = exemples_inspection(enregistrements, n=7)
    print("\n--- Inspection manuelle (échantillon, rapport uniquement) ---")
    for ex in echantillon:
        print(f"\n  [{ex['id']}] ({ex['question_type']})")
        print(f"  Question          : {ex['question']}")
        print(f"  Gold answer       : {ex['gold_answer']}")
        print(f"  Expected document : {ex['expected_document']}")
        print(f"  Expected evidence : {ex['expected_evidence']}")

    # --- Écriture -------------------------------------------------------------
    if args.dry_run:
        print("\n[--dry-run] Aucun fichier écrit (ni corpus, ni gold, ni rapport).")
    else:
        n_ecrit = ecrire_jsonl(args.fichier_gold, enregistrements)

        DOSSIER_RAPPORT_PREPARATION.mkdir(parents=True, exist_ok=True)
        chemin_rapport = DOSSIER_RAPPORT_PREPARATION / f"prepare_cquae_{horodatage()}.json"
        chemin_rapport.write_text(
            json.dumps(
                {
                    "sources": {
                        "questions": JEU_QUESTIONS,
                        "revision_questions": REVISION_QUESTIONS,
                        "documents": JEU_DOCUMENTS,
                        "revision_documents": REVISION_DOCUMENTS,
                    },
                    "audit": audit,
                    "exclusions": exclusions,
                    "selection": rapport_selection,
                    "controles_qualite": qc,
                    "exemples_inspection": echantillon,
                    "n_documents_uniques": len(docs_uniques),
                    "dossier_corpus": str(args.dossier_corpus),
                    "fichier_gold": str(args.fichier_gold),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        print(f"\nCorpus écrit    : {args.dossier_corpus}  ({len(mapping_fichiers)} fichier(s))")
        print(f"Gold écrit      : {args.fichier_gold}  ({n_ecrit} ligne(s))")
        print(f"Rapport écrit   : {chemin_rapport}")

    print("\n" + "=" * 78)
    verdict_ok = (
        qc.get("ids_uniques", True)
        and qc.get("gold_answer_non_vide", True)
        and qc.get("tous_documents_gold_presents", True)
        and qc.get("aucune_reference_cassee", True)
        and qc.get("corpus_non_contamine", True)
        and seed_ok
    )
    if args.dry_run:
        print("VERDICT : --dry-run, aucun verdict (rien écrit).")
    elif verdict_ok:
        print("VERDICT : READY FOR INGESTION")
    else:
        print("VERDICT : REVIEW NEEDED")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
