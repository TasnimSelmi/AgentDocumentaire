# Changelog

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).
Ce projet ne suit pas encore SemVer : les jalons sont nommés par **version de
socle** (`rag-v1`, …).

---

## [Non publié]

Préparation du gel **RAG V1** et fiabilisation du harnais d'évaluation.

### Ajouté
- `README.md`, `docs/architecture.md`, `docs/DO_NOT_TOUCH.md`, `CHANGELOG.md`.
- `evaluation/cquae_multicapacite.py` : harnais d'évaluation agent **4
  capacités** (SEARCH / SUMMARIZE / CLASSIFY / EXTRACT) sur CQuAE, avec
  frontière anti-fuite gold et préconditions vérifiées avant tout appel agent.
- `evaluation/evaluate_agent.py`, `evaluation/prepare_cquae.py`.
- `tests/evaluation/test_corrections_scoring.py` : couverture ciblée des
  correctifs de scoring (déterministe, sans réseau).
- Livrable `scorecard_reference.md` (scorecard de référence, run
  `20260828-182551`).

### Corrigé — harnais d'évaluation uniquement (aucun module `src/` touché)
- **Groundedness SEARCH** (`evaluate_end_to_end.calculer_groundedness`) :
  l'ancrage était mesuré contre `SourceCitee.extrait`, tronqué à 320
  caractères par la génération, ce qui produisait de faux
  `PROVENANCE_FAILURE`. Mesure désormais sur le **texte complet des passages
  récupérés** (`ReponseRAG.recherche.passages`), repli sûr sur l'extrait si
  aucun rapport de recherche. Seuil de décision (0.5) inchangé.
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
- **SU-02** : « points essentiels » non routé vers SUMMARIZE (défaut routing).
- **EX-03** : résolution documentaire contextuelle d'EXTRACT insuffisante.
- **SQ-16** : gold `cquae:test:11761` générique, à revoir (qualité dataset).
- Smoke CQuAE post-correctif groundedness **non ré-exécuté** (GPU indisponible)
  — condition de clôture J1, voir `scorecard_reference.md` §11.
- `.env` local : `QDRANT_PATH` avec barre oblique de tête → 99 tests en
  `PermissionError`. À corriger dans le `.env` de la machine.

---

## [rag-v1] — À VENIR

> **Tag non posé.** Sera créé une fois le smoke CQuAE J1 ré-exécuté et validé
> (chaque WRONG = un vrai défaut, aucun faux négatif du harnais).

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
