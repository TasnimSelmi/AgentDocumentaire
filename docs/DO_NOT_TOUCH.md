# DO NOT TOUCH — socle RAG V1 gelé

> Ce document est **normatif**. Il fixe ce qui ne doit pas changer tant que le
> socle RAG V1 n'est pas explicitement déprécié et remplacé par une V2
> évaluée.

## Principe

> **Le benchmark mesure le système ; le système ne se déforme pas pour le
> benchmark.**

Aucune modification des modules listés ci-dessous n'est autorisée pour :

- faire monter un score d'évaluation ;
- faire passer un cas de test end-to-end ou un cas du smoke CQuAE ;
- accommoder les particularités d'un corpus (CQuAE, UDA, ESG ou autre) ;
- accommoder un modèle LLM particulier (qwen3 / Ollama ou autre).

Une modification de ces modules n'est légitime que dans le cadre d'un **cycle
d'évaluation complet** (ré-ingestion + retrieval doc + retrieval evidence +
end-to-end + ablation) débouchant sur une **nouvelle version de socle**
documentée dans [../CHANGELOG.md](../CHANGELOG.md).

---

## 1. Gelé — cœur RAG (`src/rag/`)

| Fichier | Raison du gel |
|---|---|
| `src/rag/ingestion.py` | pipeline d'ingestion générique validé ; toute modification invalide l'index et impose une ré-ingestion + re-mesure |
| `src/rag/chunking.py` | découpage structure-aware calibré (`config/default.yaml`) ; change la granularité de tout l'index |
| `src/rag/embeddings.py` | BGE-M3 + BGE-reranker-v2-m3, singletons ; changer de modèle = changer d'espace vectoriel |
| `src/rag/vectorstore.py` | schéma de collection (vecteurs nommés, RRF côté Qdrant, UUID v5 déterministe) |
| `src/rag/retrieval.py` | recherche hybride + résolution documentaire générique (catalogue IDF) ; **frontière** base ↔ génération |
| `src/rag/generation.py` | génération sourcée, cloisonnement documentaire, refus déterministe, réparation de citations (1 essai) |
| `src/rag/validation.py` | suffisance du contexte + validation de provenance des citations |
| `src/rag/normalization.py` | normalisation par type + résolution d'entités (seuils calibrés) |
| `src/rag/loaders.py` | extraction de texte + déclenchement OCR |

### Cas particulier — `src/rag/generation.py`, ligne ~624

`SourceCitee.extrait` est tronqué à 320 caractères. **C'est un artefact
d'affichage, à ne pas “corriger” dans le moteur.** Le texte complet des
passages est déjà disponible via `RapportRecherche.passages[i].texte`. Toute
mesure d'ancrage (groundedness) doit se faire côté harnais sur ce texte
complet — voir `scorecard_reference.md` §4 et
`evaluation/evaluate_end_to_end.py::calculer_groundedness`.

---

## 2. Gelé — outils de l'agent (`src/tools/`)

| Fichier | Raison du gel |
|---|---|
| `src/tools/base.py` | contrat commun (schéma dynamique, `ResultatOutil`, `rendu_agent`) |
| `src/tools/search.py` | façade `retrieval` ; schéma exposé au LLM dérivé du profil |
| `src/tools/summarize.py` | map-reduce borné ; garde-fou périmètre ; jamais de succès sans provenance |
| `src/tools/classify.py` | vote majoritaire absolu déterministe par lots |
| `src/tools/extract.py` | agrégation par **déduplication** (jamais un vote) ; toutes les valeurs distinctes sourcées |

> Ne PAS modifier `src/tools/extract.py` pour satisfaire des golds
> d'extraction : l'appariement des libellés de champs est un problème de
> **scoring**, traité dans `evaluation/cquae_multicapacite.py::_associer_champ`.

---

## 3. Gelé — couche agentique (`src/agent/`)

| Fichier | Raison du gel |
|---|---|
| `src/agent/graph.py` | topologie du graphe LangGraph ; routage 100 % déterministe |
| `src/agent/nodes.py` | classifieurs LLM bornés (prompts figés), évaluation pertinence/suffisance, boucle reformulation, refus. **Exception : le bloc de détection d'intention** (`_detecter_intention` et ses tables de vocabulaire) est la **surface sanctionnée de l'étape *routing* dédiée** — voir ci-dessous. |
| `src/agent/state.py` | `EtatAgent` : budget de tentatives, trace |
| `src/agent/session.py` | `SessionAgent` : assemblage, aucune décision, aucun prompt |
| `src/agent/graph_state.py` | `EtatGraphe` porté par LangGraph |

### Étape *routing* dédiée — `_detecter_intention` uniquement

Le bloc de **détection d'intention** de `src/agent/nodes.py` (`_detecter_intention`
et ses constantes de vocabulaire : `_JETONS_*`, `_BIGRAMMES_*`, `_EXPRESSIONS_*`,
`_MARQUEURS_*`) peut évoluer **dans ce cadre précis**, sous conditions :

- mesuré **avant / après** par `evaluation/evaluate_routing.py`
  (`deterministic_only` **et** `production_routing`, métriques jamais fusionnées) ;
- **SEARCH ne perd jamais un cas** (invariant vérifié : SEARCH 24/24) ;
- expressions multi-mots, aucun vocabulaire métier, aucune règle liée à un corpus ;
- **prompts** des désambiguïsateurs LLM bornés **inchangés** ;
- `graph.py`, `state.py`, `session.py`, `graph_state.py` **non touchés** ;
- tests ajoutés avec la correction ; re-run du smoke CQuAE pour non-régression ;
- consigné dans [../CHANGELOG.md](../CHANGELOG.md).

Tout le reste de `nodes.py` (prompts, boucle QA, refus) reste **gelé**.

### Défauts système — suivi

- ~~**SU-02**~~ — **corrigé (P1.2, 2026-08-29)**. « points essentiels »,
  « idées principales », « grands axes », « main takeaways », « TL;DR » routées
  vers SUMMARIZE via `_EXPRESSIONS_SUMMARIZE`. Smoke CQuAE : SU-02 WRONG → PASS,
  0 régression de routage. Routing `production_routing` après P1.2 :
  SEARCH 24/24, SUMMARIZE 10/10, CLASSIFY 8/8, EXTRACT 9/9.
  - **RT-040** (« s'agit-il d'un X ou d'un Y ? ») volontairement laissé en
    **zone grise** en `deterministic_only` (repli `search`) — une règle
    lexicale « X ou Y ? » risquerait un faux positif SEARCH — mais
    correctement résolu en **CLASSIFY** en `production_routing` par le
    désambiguïsateur borné.
- **EX-03** : résolution documentaire **contextuelle** d'EXTRACT (requête sans
  document nommé) plus faible que celle de CLASSIFY/SUMMARIZE. **Non corrigé**
  — seul WRONG restant au smoke CQuAE P1.2. Reste visible et assumé
  (`scorecard_reference.md`). Toute correction relève d'une étape dédiée, pas
  d'une retouche opportuniste du socle.

---

## 4. Gelé — configuration de référence

| Élément | Valeur gelée |
|---|---|
| LLM | Ollama `qwen3:8b`, `temperature=0.0`, `num_ctx=16384` |
| Embeddings | `BAAI/bge-m3` (dense 1024d + sparse) |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| `config/default.yaml` | chunking, `recherche.*`, `qdrant.*`, `agent.*` — **valeurs figées** |
| `config/schemas/generic.yaml` | profil technique de référence |

Changer une de ces valeurs = nouveau socle + re-mesure complète.

---

## 5. Modifiable SANS toucher au socle

- **`evaluation/`** — harnais de mesure. Consomme les points d'entrée publics
  de `src/`, n'en importe aucune logique interne. C'est **ici** que se
  corrigent les défauts de *scoring* (groundedness, appariement documentaire
  UUID↔nom de fichier, appariement de champs EXTRACT).
- **`tests/`** — couverture.
- **`docs/`**, **`README.md`**, **`CHANGELOG.md`**, **`.env.example`**.
- **`profiles/domains/*.yaml`** — ajout d'un profil de vocabulaire métier
  (n'affecte ni retrieval ni routage).
- **`scripts/`** — démos et utilitaires hors chemin de production.
- Nouveaux modules pour les chantiers post-gel (routing avancé,
  multi-document, service, API, UI) **tant qu'ils n'éditent pas** les fichiers
  des §1–§4 : ils doivent les appeler, pas les modifier.

---

## 6. Fichier à ne pas déplacer

`test_rag.py` (racine) — fournit `comparer_reponse` / `TOLERANCE_RELATIVE`,
importés par `evaluation/evaluate_end_to_end.py`,
`evaluation/evaluate_agent.py`, `evaluation/cquae_multicapacite.py`. Le
déplacer casse ces imports.
