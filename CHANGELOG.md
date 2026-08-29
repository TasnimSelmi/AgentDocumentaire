# Changelog

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).
Ce projet ne suit pas encore SemVer : les jalons sont nommés par **version de
socle** (`rag-v1`, …).

---

## [Non publié]

Fiabilisation du harnais d'évaluation et **clôture P0** (socle RAG V1 gelé,
tag [`rag-v1`](#rag-v1--2026-08-29)), puis **P1.1** (banc de routage) et
**P1.2** (correction des lacunes de routage SUMMARIZE / CLASSIFY / EXTRACT,
dont **SU-02**).

### Ajouté
- `README.md`, `docs/architecture.md`, `docs/DO_NOT_TOUCH.md`, `CHANGELOG.md`.
- `evaluation/cquae_multicapacite.py` : harnais d'évaluation agent **4
  capacités** (SEARCH / SUMMARIZE / CLASSIFY / EXTRACT) sur CQuAE, avec
  frontière anti-fuite gold et préconditions vérifiées avant tout appel agent.
- `evaluation/evaluate_agent.py`, `evaluation/prepare_cquae.py`.
- `tests/evaluation/test_corrections_scoring.py` : couverture ciblée des
  correctifs de scoring (déterministe, sans réseau).
- Livrable `scorecard_reference.md` (scorecard de référence, run
  `cquae_multicapacite_20260829-094620`).

### Résultat de référence (P0)
- Smoke CQuAE `cquae_multicapacite_20260829-094620` : **20 / 28 PASS**
  (2 ANSWER_ONLY, 2 RETRIEVAL_ONLY, 2 WRONG, 2 ABSTAIN_CORRECT), 0
  `TECHNICAL_ERROR`, **0 faux négatif du harnais**.
- Critères de sortie P0 satisfaits : verdicts cohérents avec la revue
  manuelle ; SEARCH `exactitude=True` sur 14/16 = 87,5 % (≥ 80 %, ANSWER_ONLY
  inclus car réponses exactes) ; plus aucun faux `DOCUMENT_RETRIEVAL_FAILURE`
  sur CLASSIFY / SUMMARIZE.
- Défauts système restants, mesurés et assumés, à traiter en P1 : **SU-02**
  (routing), **EX-03** (résolution documentaire contextuelle EXTRACT).

### Corrigé — harnais d'évaluation uniquement (aucun module `src/` touché)
- **Groundedness SEARCH** (`evaluate_end_to_end.calculer_groundedness`) :
  l'ancrage était mesuré contre `SourceCitee.extrait`, tronqué à 320
  caractères par la génération, ce qui produisait de faux
  `PROVENANCE_FAILURE`. Mesure désormais sur le **texte complet des passages
  récupérés** (`ReponseRAG.recherche.passages`), repli sûr sur l'extrait si
  aucun rapport de recherche. Seuil de décision (0.5) inchangé. Effet confirmé
  end-to-end : +6 PASS SEARCH entre `20260828-182551` et `20260829-094620`.
- **Appariement documentaire UUID ↔ nom de fichier**
  (`cquae_multicapacite._cle_document_resolu`) : la trace agent enregistre un
  UUID de version documentaire, le gold un nom de fichier. Traduction
  générique via `retrieval.catalogue().par_identifiant()` avant comparaison,
  repli sûr. Corrige de faux `DOCUMENT_RETRIEVAL_FAILURE` sur CLASSIFY /
  SUMMARIZE (CL-01, CL-02, SU-01, SU-03).
- **Scoring EXTRACT** (`cquae_multicapacite.noter_extract`) : lecture des
  champs sous `donnees["extractions"]` (et non à la racine — bug qui rendait
  tous les champs `trouve=False`) ; `_associer_champ` réécrit en 3 passes
  déterministes (égalité normalisée → inclusion → Jaccard ≥ 0.34) ;
  `_valeur_champ_extrait` gère les champs à valeurs multiples.
- **`.env.example`** : réécrit pour refléter `src/config.py::Settings` —
  Ollama uniquement (suppression des clés OpenAI trompeuses), toutes les clés
  réellement lues, avertissement explicite sur `QDRANT_PATH` absolu (`/data`).

### Modifié
- `wheels/` regroupe désormais tout le cache d'installation hors ligne ; la
  roue `torch` égarée à la racine y a été déplacée. Dossier ignoré par git,
  supprimable si installation en ligne.

### Connu / non corrigé (volontairement, hors périmètre du gel)
- ~~**SU-02**~~ : **corrigé en P1.2** (voir section dédiée ci-dessous).
- **EX-03** : résolution documentaire contextuelle d'EXTRACT insuffisante — P1
  (seul WRONG restant au smoke CQuAE P1.2).
- **SQ-11** : gold à 2 assertions, `exactitude` binaire stricte — revue gold.
- **SQ-16** : gold `cquae:test:11761` générique, à revoir (qualité dataset).
- **SQ-08 / SQ-09** : groundedness lexicale &lt; 0.5 malgré une réponse exacte
  (dérivation / reformulation) — limite documentée, seuil **non** assoupli
  (`scorecard_reference.md` §4, L7).
- `.env` local : `QDRANT_PATH` avec barre oblique de tête → 99 tests en
  `PermissionError` ; run de référence produit avec la surcharge
  `QDRANT_PATH=data/vectordb/qdrant_cquae_eval`. À corriger dans le `.env` de
  la machine (hors dépôt).

---

## P1.1 — Banc de mesure du routage

Instrument de mesure construit **avant** toute modification du routage.

### Ajouté
- `evaluation/data/routing_cases.jsonl` : **65 cas** de routage génériques
  (FR + EN, ≥ 30 % hors histoire/culture, noms de documents fictifs). Couvre
  SEARCH / SUMMARIZE / CLASSIFY / EXTRACT, les intentions futures
  COMPARE / SYNTHESIZE / CLARIFY, les anti-faux-positifs et les zones grises
  connues. Dossier `evaluation/data/` non suivi par git (cf. `.gitignore`).
- `evaluation/evaluate_routing.py` : banc de mesure du **routage seul** (pas
  la qualité de réponse). Deux modes, métriques **toujours séparées, jamais
  fusionnées** :
  - `deterministic_only` : `nodes._detecter_intention` seul, zones grises
    repliées sur `search`. Aucun appel LLM / réseau / Qdrant.
  - `production_routing` : zones grises résolues par **les désambiguïsateurs
    bornés existants** (`nodes._desambiguiser_intention_*`), prompts inchangés.
  Sorties : accuracy globale, accuracy par intention, matrice de confusion,
  liste des cas échoués.
- `tests/evaluation/test_evaluate_routing.py` : validité du dataset, unicité
  des id, taxonomie, couverture (intentions / langues / domaines),
  déterminisme du runner, absence d'appel LLM en `deterministic_only`.

### Baseline mesuré (avant P1.2)
- `deterministic_only` : **37 / 65 (56,9 %)**, SEARCH 24/24.
- `production_routing` : **42 / 65 (64,6 %)**, SEARCH 24/24, EXTRACT 9/9.
- 28 échecs classés : 14 « intention future non implémentée »
  (COMPARE / SYNTHESIZE / CLARIFY), 11 « vraie lacune de routing » (bande B),
  3 « ambiguïté légitime » — ces 3 (RT-042, RT-050, RT-051) **correctement
  résolues en `production_routing`** par les désambiguïsateurs bornés.

---

## P1.2 — Correction des lacunes de routage (bande B) — **terminé**

Étape « routing dédiée » prévue par [docs/DO_NOT_TOUCH.md](docs/DO_NOT_TOUCH.md)
§3. Périmètre **strictement limité** aux 11 cas de la bande B du banc P1.1.

### Corrigé — `src/agent/nodes.py`, bloc de détection d'intention uniquement
- **SU-02 corrigé.** « Donne-moi les points essentiels du document … » et
  variantes (« idées principales », « grands axes », « main takeaways »,
  « TL;DR ») sont désormais routées vers **SUMMARIZE**. Ajout de
  `_EXPRESSIONS_SUMMARIZE` (expressions multi-mots, jamais un jeton isolé).
- **CLASSIFY** : « type / nature de (ce) document », « determine(r) la
  nature / le type de », « is this document a … » routées vers **classify**
  (`_EXPRESSIONS_CLASSIFY_SURES`). « s'agit-il d'un(e) … » (RT-040) routée
  vers la **zone grise** `_AMBIGU_CLASSIFY` (`_EXPRESSIONS_AMBIGU_CLASSIFY`).
- **EXTRACT** : « récupère » n'est un déclencheur **que combiné à une
  énumération** (`_JETONS_EXTRACT_RECUPERATION`) ; « récupère les champs … »
  est un déclencheur sûr (`_EXPRESSIONS_EXTRACT_SURES`). Corrige RT-048,
  RT-049.
- Aucun vocabulaire métier, aucune règle liée à CQuAE. `graph.py` **non
  modifié**. `src/rag/` et `src/tools/` **inchangés depuis le tag `rag-v1`**
  (`git diff rag-v1 -- src/rag src/tools` vide).

### Tests — `tests/agent/test_nodes.py`
- Paramétrage des 11 cas bande B + 5 anti-faux-positifs explicites
  (RT-015/016/017/018/020 → `search`) + RT-019 / RT-040 confirmés en zone
  grise + « récupère » seul ne force pas EXTRACT.

### Routing après P1.2

| Mode | SEARCH | SUMMARIZE | CLASSIFY | EXTRACT | Global |
|---|---|---|---|---|---|
| `deterministic_only` | **24/24** | 10/10 | 6/8 | 7/9 | 47/65 (72,3 %) |
| `production_routing` | **24/24** | **10/10** | **8/8** | **9/9** | 51/65 (78,5 %) |

- **RT-040** volontairement laissé en zone grise en `deterministic_only`
  (repli `search` — forcer une règle lexicale « X ou Y ? » risquerait un faux
  positif SEARCH), mais **correctement résolu en CLASSIFY en
  `production_routing`** par le désambiguïsateur borné.
- Bande B : **10/11** résolus en `deterministic_only`, **11/11** en
  `production_routing`.
- SEARCH **24/24 dans les deux modes, avant et après** — aucune régression.

### Smoke CQuAE post-P1.2

`evaluation/reports/cquae_multicapacite/cquae_p1_2.json` :

| Verdict | réf. `20260829-094620` | **P1.2** |
|---|---:|---:|
| PASS | 20 | **21** |
| ANSWER_ONLY | 2 | 3 |
| RETRIEVAL_ONLY | 2 | 1 |
| WRONG | 2 | **1** |
| ABSTAIN_CORRECT | 2 | 2 |
| TECHNICAL_ERROR | 0 | 0 |

- **Un seul changement de routage sur les 28 cas** : `SU-02 : search → summarize`.
  **SU-02 : WRONG → PASS.**
- **Seul WRONG restant : EX-03** (hors périmètre P1.2).
- Les bascules de verdict SEARCH (SQ-02, SQ-13 : PASS → ANSWER_ONLY ; SQ-09,
  SQ-16 : → PASS) ont toutes `detected_tool = search` **inchangé** : variation
  de `groundedness` autour du seuil 0.5 d'un run Ollama à l'autre (limite L7),
  sans lien avec P1.2. Net SEARCH PASS inchangé.

---

<a id="rag-v1--2026-08-29"></a>
## [rag-v1] — 2026-08-29

> **Tag posé** (`git tag -a rag-v1`) sur le commit de clôture P0. Le smoke
> CQuAE `cquae_multicapacite_20260829-094620` valide la condition de gel :
> chaque WRONG = un vrai défaut système (SU-02, EX-03), aucun faux négatif du
> harnais.
>
> À partir de ce tag, `git diff rag-v1 -- src/rag src/tools` doit rester vide :
> toute évolution de `src/rag/` ou `src/tools/` impose un nouveau cycle
> d'évaluation complet et un nouveau nom de socle (voir `docs/DO_NOT_TOUCH.md`).

Instantané du socle RAG figé : modules `src/rag/`, `src/tools/`, `src/agent/`,
`src/llm/`, `src/profiling/` + `config/default.yaml` +
`config/schemas/generic.yaml`. Liste exacte des éléments gelés :
[docs/DO_NOT_TOUCH.md](docs/DO_NOT_TOUCH.md).

### Contenu (résumé de l'historique jusqu'à `938caf5`)

**Couche agentique**
- Graphe LangGraph 4 capacités, routage 100 % déterministe (vocabulaire fermé
  + 2 classifieurs LLM bornés pour les zones grises, jamais de choix d'outil
  par le LLM).
- Branche SEARCH : boucle rechercher → évaluer (pertinence déterministe puis
  suffisance LLM bornée) → répondre | reformuler, refus déterministe si budget
  épuisé sans preuves.
- Outils `search` / `summarize` / `classify` / `extract` : façades sur
  `src/rag/`, `lecture_seule`, agrégation multi-lots déterministe en Python.
- `SessionAgent` / `EtatAgent` : assemblage sans décision, budget de
  tentatives, trace horodatée.

**Cœur RAG**
- Ingestion générique pilotée par profil : loaders multi-format + OCR,
  découpage structure-aware, inférence LLM catégorie/métadonnées,
  normalisation par type, résolution d'entités persistée, embeddings BGE-M3,
  indexation Qdrant.
- Retrieval hybride (dense + sparse BGE-M3, fusion RRF côté Qdrant) +
  reranking BGE-reranker-v2-m3 + diversification par document.
- Résolution documentaire générique par catalogue pondéré IDF (aucun nom en
  dur) ; fast-path identifiant exact.
- Génération sourcée : cloisonnement documentaire, citations `[S1]` validées
  jusqu'au document, réparation unique, refus déterministe sans contexte,
  résistance aux instructions injectées dans les documents.

**Configuration & profils**
- Séparation stricte `.env` / `config/default.yaml` / `config/schemas/*` /
  `profiles/domains/*`.
- CLI de profilage de domaine (`src/profiling`), suggestion LLM sans lecture
  de documents.

**Évaluation**
- Harnais `evaluation/` séparé du code applicatif : retrieval document,
  retrieval evidence, end-to-end avec attribution de cause d'échec, agent
  SEARCH, agent multi-capacités CQuAE, ablation, analyse d'échecs.

### Antériorité (jalons git)

| Commit | Date | Jalon |
|---|---|---|
| `4fddbfd` | 2026-07-31 | pipeline d'ingestion documentaire générique |
| `9b4a85b` | 2026-08-02 | retrieval + génération prêts pour tests RAG |
| `8a42515` | 2026-08-04 | suppression du seuil fixe, ajout d'un validateur de retrieval |
| `a7fa544` | 2026-08-07 | profilage du domaine métier |
| `ad1a64d` | 2026-08-10 | profil de domaine intégré au pipeline RAG |
| `88f1ed9` | 2026-08-22 | **RAG V1 freeze** (1re passe) : 4 bugs bloquants, nettoyage du framework d'éval |
| `ab331ef` → `0b5c907` | 2026-08-23 | première couche agentique minimale |
| `be23906` | 2026-08-25 | outil `summarize` |
| `f897151` | 2026-08-25 | outil `classify` |
| `8d6798b` | 2026-08-25 | outil `extract` |
| `938caf5` | 2026-08-28 | harnais CQuAE multi-capacités |
