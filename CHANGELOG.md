# Changelog

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).
Ce projet ne suit pas encore SemVer : les jalons sont nommés par **version de
socle** (`rag-v1`, …).

---

## [Non publié]

Fiabilisation du harnais d'évaluation et **clôture P0** (socle RAG V1 gelé,
tag [`rag-v1`](#rag-v1--2026-08-29)), puis **P1.1** (banc de routage),
**P1.2** (lacunes de routage SUMMARIZE / CLASSIFY / EXTRACT, dont **SU-02**),
**P1.3** (contrat de sortie public `AgentResponse`), **P1.4** (détecteur
multi-document déterministe), **P1.5** (capacités COMPARE / SYNTHESIZE),
**P1.6** (défaut **EX-03** — résolution documentaire contextuelle d'EXTRACT
supprimée), **P1.7** (validation finale, cœur agentique **candidat au gel**),
**P2.1** (façade applicative `AgentService`) et **P2.2** (abstraction générique
des sources documentaires).

### P1.3 — contrat de sortie public `AgentResponse` (`src/agent/response.py`)
- Point d'entrée public `executer_agent(requete)` → `AgentResponse`
  (`status` / `capability` / `answer` / `sources` / `citations` / `warnings` /
  `data` / `metadata` / `error`). Enveloppe **fine** après `graph.invoke()` :
  même graphe que `invoquer_agent`, puis normalisation **100 % déterministe,
  aucun appel LLM, aucune reconstruction de provenance**.
- `status="refusal"` (abstention fonctionnelle) strictement distinct de
  `status="error"` (exception technique non prévue). Aucun prompt,
  raisonnement, contenu intégral de document ou objet non sérialisable dans la
  sortie. `tests/agent/test_response.py` (17 tests).

### P1.4 — détecteur multi-document (`src/agent/multidoc.py`)
- Fonction **pure et déterministe** (0 LLM, 0 retrieval) : `is_multidoc` +
  `operation_hint ∈ {compare, synthesize, none}` à partir du seul texte
  (références de fichiers, marqueurs pluriels sur nom de document, marqueurs
  comparatifs / de synthèse, garde-fou déixis singulière). Bilingue FR/EN,
  aucun nom de document en dur. Mesuré par `evaluate_routing` (14/14).

### P1.5 — capacités COMPARE / SYNTHESIZE
- `src/agent/multidoc_pipeline.py` : MAP par document (couverture **intégrale**,
  lots bornés, citations `[D<k>S<j>]`) → REDUCE inter-document qui ne voit que
  les MAP validées. Bornes techniques : `LIMITE_DOCUMENTS=4`, `NB_LOTS_MAX=24`,
  `PROFONDEUR_MAX_AGREGATION=3`, `budget_caracteres_entree_llm()` (source
  unique, contrôle avant chaque envoi). **Aucun search global, aucun repli
  SEARCH** ; abstention déterministe à chaque étape ; divergences /
  contradictions conservées explicitement.
- `src/tools/compare.py`, `src/tools/synthesize.py` : nouvelles façades
  `lecture_seule`. Nouveaux nœuds `noeud_compare` / `noeud_synthesize` +
  arêtes ; routage supplanté seulement depuis SEARCH / SUMMARIZE, jamais
  depuis une demande CLASSIFY / EXTRACT explicite.
- `evaluation/multidoc_benchmark.py` (hors ligne, LLM scripté) : **12/12**.

### P1.6 — défaut EX-03 (résolution documentaire contextuelle d'EXTRACT)
- `noeud_extract` : plus aucun repli `search` global → `extract(document=None)`.
  Ce repli transformait une recherche multi-document en extraction structurée
  implicite dès que le top-k ne faisait ressortir qu'un seul document (choix
  de périmètre par convenance, non déterministe d'un corpus à l'autre).
- Nouvel invariant : EXTRACT n'extrait que sur un `PerimetreDocumentaire`
  résolu de façon **unique et fiable** ; sinon **refus déterministe** sans
  appeler `search` ni `extract`. CLASSIFY / SUMMARIZE inchangés (mode
  contextuel conservé — divergence volontaire propre à EXTRACT).
- Ouverture puis fermeture ponctuelle du gel sur `nodes.py::noeud_extract`
  (fonction + docstring uniquement). Tests génériques ajoutés dans
  `tests/agent/test_nodes.py`, sans dépendance CQuAE.

### P2.1 — façade applicative `AgentService` (`src/agent/service.py`)
- Couche **mince** au-dessus de `executer_agent` : `AgentService.query(requete)
  -> AgentResponse`. Validation légère (chaîne non vide → sinon
  `status="error"`, `error.code="requete_invalide"`, sans appeler le cœur),
  **un seul** appel au point d'entrée public P1, propagation intacte de
  l'`AgentResponse`, **ne lève jamais** (erreur de construction de session →
  `status="error"`).
- Aucune logique de capacité, de routage, de RAG ni de normalisation
  dupliquée. Aucune dépendance FastAPI / UI / connecteur / mémoire / logging.
  `point_entree` et `options_session` injectables (tests hors ligne, sans
  Ollama ni Qdrant).
- Cœur P1 **non modifié** (`graph.py`, `nodes.py`, `state.py`, `session.py`,
  `graph_state.py` intacts) ; `src/agent/__init__.py` exporte `AgentService`.
  `tests/agent/test_service.py` (19 tests). Règle : API/UI appellent
  `AgentService`, jamais LangGraph directement.

### P2.2 — abstraction générique des sources documentaires (`src/sources/`, **gelé** 2026-08-31)
- Découple **l'origine** des documents du **pipeline d'ingestion**. Contrat
  minimal `DocumentSource.materialiser() -> ContextManager[Path]` : une source
  fournit un **snapshot complet** dans un répertoire local, puis le socle gelé
  prend le relais via `ingerer(dossier=…)` — qui accepte déjà un dossier
  arbitraire. Le pipeline reçoit un `Path` (loaders gelés path-based, OCR,
  détection de changement `RegistreFichiers` keyée par chemin).
- **Invariant de sûreté** : le `Path` exposé est TOUJOURS un snapshot complet
  et cohérent — jamais partiel. Une récupération incomplète lève `ErreurSource`
  **avant** le `yield` → `IngestionService.sync` n'appelle pas le pipeline,
  l'index reste intact. Une absence de document n'est lue comme suppression
  **que parce que** le snapshot est garanti complet. Résout le risque
  « erreur de récupération distante interprétée comme suppression ».
- `SnapshotDocumentSource` (`src/sources/snapshot.py`) : base pour source à
  récupération faillible. Sous-classe → `_recuperer(staging)` seulement.
  `materialiser()` hérité : staging à l'écart → si `_recuperer` lève (ou
  staging vide sans `autoriser_snapshot_vide`), staging détruit + **miroir
  précédent intact** + `ErreurSource` ; sinon **remplacement atomique** du
  miroir publié (renommages de répertoires, reprise via `<miroir>.precedent`),
  puis `yield`. Miroir persistant (chemins stables pour le socle).
- `LocalDocumentSource` : adaptateur MVP, satisfait l'invariant trivialement
  (pas de récupération faillible). `materialiser()` rend le dossier de
  l'utilisateur **tel quel** (aucune copie) →
  `IngestionService().sync(LocalDocumentSource(d))` **strictement équivalent** à
  `ingerer(dossier=d)`. `inventaire() -> list[str]` (hors contrat) réutilise
  `decouvrir_fichiers` du socle. `ErreurSource` si le chemin n'est pas un
  répertoire. Pas de modèle `DocumentDescriptor` : aucun usage concret (le
  socle re-découvre depuis le répertoire).
- `IngestionService.sync(source, *, reinitialiser, limite, inferer, nom_profil)`
  : **un seul** appel au pipeline, options transmises telles quelles, rapport
  renvoyé intact. Pipeline injectable (tests hors ligne, sans Ollama ni Qdrant).
- **Aucune** modification de `src/rag/**` (`git diff rag-v1 -- src/rag` **vide**)
  ni du cœur P1. Aucune découverte, hash, détection « inchangé » / suppression,
  parsing, chunking, embedding ou écriture Qdrant réimplémentés : tout reste
  dans `RegistreFichiers` / `ingerer` du socle. Aucune dépendance
  SharePoint / S3 / API entreprise ; `EnterpriseDocumentSource` **non créé**.
- `.gitignore` : l'ignore `docs/` (« documents d'avancement », sur-large, non
  commité) est ramené à `docs/avancement/` — la doc de référence du dépôt sous
  `docs/` redevient versionnable (`P1_CLOTURE.md` en était collatéralement
  exclu).
- `tests/sources/` (26 tests). Documentation :
  [`docs/P2.2_SOURCES.md`](docs/P2.2_SOURCES.md).

### P1.7 — validation finale (aucune nouvelle capacité)
- Audit d'architecture, vérification des invariants, `pytest` **658/658**,
  `evaluate_routing --deterministic_only` (SEARCH **24/24**, multidoc 14/14),
  `multidoc_benchmark` **12/12**, E2E `executer_agent` (refus déterministes
  confirmés live).
- Livrable [`docs/P1_CLOTURE.md`](docs/P1_CLOTURE.md) : architecture finale P1,
  statut des 6 capacités (supporté et testé / supporté avec limitation /
  reporté P2), invariants, verdict **P1 READY TO FREEZE**.
- `docs/architecture.md` mis à jour (COMPARE / SYNTHESIZE, détecteur
  multi-document, `AgentResponse`, discipline de désignation du document).

### Ajouté (P0–P1.2)
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
