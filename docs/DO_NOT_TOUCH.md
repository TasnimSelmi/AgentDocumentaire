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
| `src/agent/nodes.py` | classifieurs LLM bornés (prompts figés), évaluation pertinence/suffisance, boucle reformulation, refus. **Exception : le bloc de détection d'intention** (`_detecter_intention` et ses tables de vocabulaire) est la **surface sanctionnée de l'étape *routing* dédiée** — voir ci-dessous. `noeud_extract` a été ouvert ponctuellement pour la correction P1.6 (défaut EX-03) puis refermé — voir « Défauts système — suivi ». |
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
- ~~**EX-03**~~ — **traité (P1.6, 2026-08-30)** comme étape dédiée. La
  résolution documentaire **contextuelle** d'EXTRACT (requête sans document
  nommé) est **supprimée** : `noeud_extract` ne fait plus de repli
  `search` global -> `extract(document=None)`. Ce repli transformait une
  recherche multi-document en extraction structurée implicite dès que le
  top-k ne faisait ressortir qu'un seul document (choix de périmètre par
  convenance, non déterministe d'un corpus à l'autre). Nouvel invariant :
  EXTRACT n'extrait que sur un périmètre résolu de façon **unique et
  fiable** (`PerimetreDocumentaire` contraignant à une seule valeur) ;
  sinon refus déterministe demandant de préciser le document, sans jamais
  appeler `search` ni `extract`. CLASSIFY/SUMMARIZE **inchangés** (mode
  contextuel conservé) : divergence volontaire, propre à EXTRACT.

  **Ouverture explicite et temporaire du gel** : `src/agent/nodes.py::noeud_extract`
  (fonction et docstring uniquement) a été ouvert pour cette seule
  correction P1.6, puis **refermé**. Aucune autre partie de `nodes.py`
  (prompts, classifieurs LLM bornés, boucle QA, refus RAG, autres nœuds)
  n'a été touchée. `graph.py`, `state.py`, `session.py`, `graph_state.py`,
  `src/rag/**`, `src/tools/**` non touchés. Tests génériques ajoutés dans
  `tests/agent/test_nodes.py` (aucune dépendance CQuAE).

---

## 4. Gelé — couche sources documentaires P2.2 (`src/sources/`)

> Gelée à la clôture **P2.2** (2026-08-31). Découple l'origine des documents du
> pipeline d'ingestion **sans** modifier `src/rag/**` ni le cœur P1.

| Fichier | Rôle gelé |
|---|---|
| `src/sources/base.py` | contrat public `DocumentSource` + `ErreurSource` + invariant de snapshot |
| `src/sources/snapshot.py` | `SnapshotDocumentSource` : staging → publication atomique d'un snapshot complet |
| `src/sources/local.py` | `LocalDocumentSource` : adaptateur local MVP (pass-through du dossier) |
| `src/sources/service.py` | `IngestionService.sync()` : façade d'ingestion P2 |
| `src/sources/__init__.py` | surface publique du paquet |

### Contrat public

- **`DocumentSource.materialiser() -> AbstractContextManager[Path]`** : la seule
  opération. Le `Path` exposé est **toujours un snapshot complet et cohérent**
  de la source — jamais partiel, jamais en cours de construction. En cas
  d'impossibilité, `ErreurSource` est levée **avant** le `yield` (le pipeline
  n'est pas appelé).
- **`LocalDocumentSource`** — adaptateur local MVP : la source *est* un dossier
  déjà présent. `materialiser()` rend le dossier **tel quel** (aucune copie) ;
  `inventaire()` (hors contrat) réutilise `decouvrir_fichiers` du socle.
  `IngestionService().sync(LocalDocumentSource(d))` ≡ `ingerer(dossier=d)`.
- **`SnapshotDocumentSource`** — base imposée pour toute future source
  distante. Une sous-classe n'implémente que `_recuperer(staging)` ;
  `materialiser()` hérité construit la nouvelle matérialisation **à l'écart**,
  ne remplace le miroir publié qu'**après succès complet** et **atomiquement**,
  et laisse le miroir précédent **intact** en cas d'échec.
- **`IngestionService.sync(source, *, reinitialiser, limite, inferer,
  nom_profil)`** — façade d'ingestion P2 : un **seul** appel à
  `ingerer(dossier=…)`, options transmises telles quelles, `RapportIngestion`
  renvoyé intact.

### Invariants normatifs

- **Aucune** future source (API entreprise, GED, SharePoint, S3, montage
  distant…) ne contourne cette couche pour écrire dans l'index ou pour toucher
  `src/rag/**`. Elle implémente `DocumentSource` — via `SnapshotDocumentSource`
  si la récupération est faillible — et passe par `IngestionService.sync()`.
- **Une récupération partielle ou en échec ne doit JAMAIS être assimilée à une
  suppression documentaire.** Le socle lit « présent au registre, absent du
  répertoire » comme une suppression ; n'exposer au pipeline qu'un snapshot
  **garanti complet**. Tout doute (récupération incomplète, staging vide sans
  `autoriser_snapshot_vide`) ⇒ `ErreurSource`, rien n'est publié, l'index reste
  tel quel.
- Le pipeline reçoit un **`Path` de répertoire**, jamais des `bytes` ni un
  `stream` (loaders gelés path-based, OCR, détection de changement de
  `RegistreFichiers` keyée par chemin).
- `src/rag/**` et le cœur P1 (`graph.py`, `nodes.py`, `state.py`, `session.py`,
  `graph_state.py`) restent **gelés** : cette couche les **appelle**, ne les
  modifie pas. `git diff rag-v1 -- src/rag` doit rester **vide**.

### Modification légitime

Comme pour les §1–§3 : correction de défaut documentée dans
[../CHANGELOG.md](../CHANGELOG.md), tests dans `tests/sources/`, sans toucher
`src/rag/**` ni le cœur P1. Ajouter une source concrète (connecteur entreprise)
= **nouveau fichier** implémentant le contrat, jamais une modification des cinq
fichiers ci-dessus. Voir [P2.2_SOURCES.md](P2.2_SOURCES.md).

---

## 4bis. Invariants — observabilité P2.4 (`src/observability/`, gel différé)

> Couche de traçage transverse (P2.4). Gel formel **différé** à la validation
> finale de P2, comme `src/api/`. En attendant, ces invariants sont
> **normatifs**. Voir [P2.4_OBSERVABILITY.md](P2.4_OBSERVABILITY.md).

- **Ne modifie aucune couche gelée.** `src/observability/**` **appelle**
  `AgentService` / `IngestionService` (les enveloppe) ; il ne touche jamais
  `src/rag/**`, `src/agent/**`, `src/tools/**`, `src/sources/**`.
  `git diff rag-v1 -- src/rag` reste **vide**.
- **`AgentResponse` inchangé.** L'observabilité en **dérive** des attributs
  (`capability`, `documents`, `citations`, `source_count`, `refusal_code`,
  `error.*`) ; elle n'ajoute, ne retire ni ne renomme aucun champ, et ne
  modifie jamais l'objet renvoyé.
- **Aucune trace interne exposée.** `EtatAgent.trace` / `EtatGraphe` /
  `SessionAgent` / prompts / chain-of-thought / réponses LLM brutes / requêtes
  reformulées / contenu documentaire complet ne franchissent **jamais** un
  `TraceSink`. La sérialisation passe par une **allow-list** stricte
  (`src/observability/redaction.py`), jamais par `asdict()`.
- **`src/api/routes.py` sans logique d'observabilité.** Le corps de `/query`
  est **strictement identique** à P2.3. OpenAPI expose **uniquement**
  `/health`, `/query`, `/ingestion`.
- **`src/api/errors.py` reste l'unique autorité HTTP.** Un **seul**
  `@app.exception_handler(Exception)` ; il émet `http_unhandled_error` puis
  renvoie **exactement** le même `500` générique P2.3. `ErreurSource` → même
  `503`. Jamais de stack ni de message technique réel au client.
- **Pas de global mutable.** Aucun `get_sink()` / `set_sink()`, aucun sink de
  module. Le `TraceSink` est **injecté** (`install_observability(app, *,
  sink=…)` / `create_app(*, sink=…)`). `structlog` reste confiné à
  `LoggingTraceSink`.
- **Enveloppe d'événement versionnée et fermée** (`schema_version = "1.0"`).
  Changer la forme de l'enveloppe ou d'un bloc `attributes` = bump de
  `schema_version` + entrée [../CHANGELOG.md](../CHANGELOG.md).
- **Configuration** via `Settings` uniquement (`observability_enabled`,
  `observability_emit_start`) — **aucun** `os.getenv` dans
  `src/observability/`, aucun second système de configuration.

Modification légitime : nouveau `TraceSink` (OpenTelemetry, ELK, Loki,
Splunk…) = **nouveau module** implémentant le `Protocol`, injecté via
`create_app(sink=…)` — jamais une modification des sept fichiers du paquet.

---

## 5. Gelé — configuration de référence

| Élément | Valeur gelée |
|---|---|
| LLM | Ollama `qwen3:8b`, `temperature=0.0`, `num_ctx=16384` |
| Embeddings | `BAAI/bge-m3` (dense 1024d + sparse) |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| `config/default.yaml` | chunking, `recherche.*`, `qdrant.*`, `agent.*` — **valeurs figées** |
| `config/schemas/generic.yaml` | profil technique de référence |

Changer une de ces valeurs = nouveau socle + re-mesure complète.

---

## 6. Modifiable SANS toucher au socle

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
  des §1–§5 : ils doivent les appeler, pas les modifier. Les connecteurs de
  sources documentaires (API / GED entreprise) implémentent le contrat de la
  §4 et passent par `IngestionService.sync()` — ils ne touchent ni `src/rag/**`
  ni `src/sources/**`.

---

## 7. Fichier à ne pas déplacer

`test_rag.py` (racine) — fournit `comparer_reponse` / `TOLERANCE_RELATIVE`,
importés par `evaluation/evaluate_end_to_end.py`,
`evaluation/evaluate_agent.py`, `evaluation/cquae_multicapacite.py`. Le
déplacer casse ces imports.
