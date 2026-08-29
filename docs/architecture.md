# Architecture — RAG V1

> État : **gelé** (candidat `rag-v1`). Ce document décrit le système tel qu'il
> est figé. Les modules cités sont ceux de
> [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md).

---

## 1. Vue d'ensemble

Deux chaînes distinctes, sans logique partagée dupliquée :

- **Ingestion** (hors ligne) : fichiers bruts → points indexés dans Qdrant.
- **Requête** (en ligne) : question → réponse sourcée, via un graphe agentique
  déterministe qui appelle le RAG comme un outil.

Trois couches de configuration, jamais mélangées (`src/config.py`) :

| Source | Contenu | Objet |
|---|---|---|
| `.env` | secrets, chemins, modèles | `Settings` (pydantic-settings) |
| `config/default.yaml` | chunking, OCR, seuils, recherche, Qdrant, agent | `ConfigTechnique` |
| `config/schemas/<profil>.yaml` | taxonomie, champs de métadonnées, schéma d'extraction | `Profil` |
| `profiles/domains/<nom>.yaml` | vocabulaire métier pour les prompts | `DomainProfile` (`src/profiling`) |

**Invariant de généricité** : aucun nom de corpus, de société, de fichier ou
de champ métier n'apparaît dans `src/`. Tout élément spécifique à un domaine
transite par un YAML.

---

## 2. Ingestion (`src/rag/ingestion.py`)

Par document :

```
découverte (DOCUMENTS_DIR)
  → hash SHA-256 + déduplication contre le registre (data/vectordb/<...>/registry.json)
  → extraction de texte (src/rag/loaders.py : pdf, docx, xlsx, pptx, html, txt, md)
      → OCR Tesseract si le texte extrait est sous seuil_texte_vide (PDF scanné)
  → découpage structure-aware (src/rag/chunking.py)
  → inférence LLM : catégorie + métadonnées (schéma dérivé du Profil actif)
  → normalisation déterministe par type (src/rag/normalization.py)
  → résolution d'entités : fusion des variantes vers une forme canonique persistée
  → embeddings BGE-M3 (dense 1024d + sparse) — src/rag/embeddings.py
  → indexation Qdrant (vecteurs nommés) — src/rag/vectorstore.py
```

Puis, globalement : un **rapport qualité** (`data/logs/`) exposant taux de
remplissage des champs, échecs, fusions d'entités. C'est la boucle
d'amélioration : ajuster les descriptions du profil YAML, ré-ingérer.

### Découpage structure-aware

Le découpage récursif par taille fixe coupe mal les tableaux (en-tête et
lignes séparés → association colonne/valeur perdue). `chunking.py` segmente
d'abord la page en blocs (`titre`, `paragraphe`, `liste`, `tableau`, `texte`)
puis applique à chaque type le traitement adapté : pour les tableaux,
découpage par lignes avec **en-tête répété** dans chaque chunk. La détection
repose uniquement sur la forme du texte produit par les loaders — jamais sur
un secteur ou un document.

Paramètres (`config/default.yaml → decoupage`) : `taille_chunk=1800`,
`recouvrement=250`, `parent_child` (parent 5000), `voisins` (rayon 1,
expansion ≤ 6 chunks).

### Normalisation & résolution d'entités

- **Normalisation** : mise en forme déterministe par type (texte, date,
  nombre, entier, booléen). Formats de date essayés listés dans le YAML.
- **Résolution** : « Groupe A », « GRP-A », « groupe a » → une forme
  canonique. Seuils : `levenshtein 0.88` (rapidfuzz), `embedding 0.92`
  (cosinus), longueur minimale 3. Persistée dans `entities.json`.

Aucune dépendance directe LLM/Qdrant dans `normalization.py` : testable
unitairement, réutilisable sur tout corpus.

---

## 3. Stockage vectoriel (`src/rag/vectorstore.py`)

- Collection Qdrant à **vecteurs nommés** : chaque point porte `dense` (1024d)
  et `sparse` (lexical BGE-M3). **Fusion RRF côté Qdrant** via l'API Query —
  pas d'index BM25 séparé à maintenir.
- Les **champs filtrables** sont déclarés dans le profil YAML ; le module lit
  cette liste et crée les index de payload. Aucun nom de champ métier dans le
  code.
- Identifiant de point = UUID v5 déterministe sur `(doc_id, chunk_index)`,
  namespace fixe → une réindexation **écrase** au lieu de dupliquer.
- Mode `local` (base fichier embarquée) ou `server` (Qdrant distant).

---

## 4. Retrieval (`src/rag/retrieval.py`)

Déterministe, **sans LLM**. Frontière entre la base documentaire et la
génération.

```
1. validation + normalisation des filtres (champs autorisés = profil + techniques)
2. résolution générique du périmètre documentaire visé (voir 4.1)
3. encodage dense + sparse de la requête (BGE-M3)
4. recherche hybride Qdrant, cloisonnée au périmètre si résolu
5. reranking BGE-reranker-v2-m3 des candidats
6. déduplication + diversification par document
7. attribution d'identifiants de citation stables : S1, S2, …
```

Sortie : `RapportRecherche { requete, passages: list[Passage], perimetre,
candidats_recuperes, reranking_utilise, seuil_applique, … }`.
`Passage { citation, rang, doc_id, texte (complet), source, nom_fichier,
page, categorie, score_recherche, score_reranking, payload }`.

Paramètres (`config/default.yaml → recherche`) : `top_k_dense=20`,
`top_k_sparse=20`, `top_k_final=6`, `fusion=rrf` (`rrf_k=60`),
`score_min=0.30`.

### 4.1 Résolution documentaire générique

Un **catalogue** (`CatalogueDocuments`) est dérivé à l'exécution des
métadonnées déjà présentes dans les payloads Qdrant : identifiant, nom de
fichier, titre, organisation, type, année, alias. Aucun de ces champs n'est
obligatoire.

Le vocabulaire discriminant est pondéré par sa **rareté (IDF)** dans le
catalogue : un jeton partagé par beaucoup de documents pèse presque rien, un
jeton propre à un seul document domine. → fonctionne à l'identique sur
n'importe quel corpus, sans règle métier.

`resoudre_document(requete)` renvoie un `PerimetreDocumentaire` avec un
`statut` :

| statut | sens | filtre Qdrant |
|---|---|---|
| `exact` | un seul document désigné sans ambiguïté | oui |
| `compatible` | plusieurs documents également valables | oui (tous) |
| `ambigu` | candidats trop proches | non |
| `aucun` | la question ne désigne aucun document | non |

Un **fast-path** (`_identifiant_exact_dans`) résout d'abord un identifiant de
document cité verbatim (« … le document rapport_219.txt. ») via
`catalogue().par_identifiant()` — utile quand des noms de fichiers ne
diffèrent que par un nombre. Générique, exact uniquement, retombe sur
l'algorithme lexical historique en cas de doute.

`par_identifiant(id)` apparie sur `document_id` (UUID de version) **ou**
`nom_fichier` **ou** `source` **ou** `titre`, exact puis normalisé.

---

## 5. Génération (`src/rag/generation.py`)

RAG **non agentique** de référence. Passages récupérés → réponse vérifiable.

Garanties :

- réponse construite **uniquement** à partir des passages fournis ;
- **cloisonnement documentaire** : aucune valeur empruntée à un autre
  document que celui visé (contrôlé via `PerimetreDocumentaire.contient`) ;
- provenance explicite de chaque extrait (document, page, chunk) dans le
  prompt ;
- résistance aux instructions malveillantes dans les documents (les extraits
  sont déclarés « données non fiables ») ;
- citations explicites `[S1]`, `[S2]`, … ;
- validation des identifiants cités + **une seule** tentative de réparation
  (`_reparer_citations`) ;
- **refus déterministe sans appel LLM** quand le retrieval ne fournit aucun
  passage exploitable ou aucun passage du périmètre.

Sortie : `ReponseRAG { question, reponse, profil, contexte_suffisant,
citations_valides, citations_reparees, sources: list[SourceCitee],
recherche: RapportRecherche, avertissements, citations_hors_perimetre }`.

`SourceCitee.extrait` est **tronqué à 320 caractères** (affichage). Le texte
complet reste disponible via `recherche.passages[i].texte`. *(Nuance
importante pour l'évaluation — voir `scorecard_reference.md` §4.)*

Validation (`src/rag/validation.py`) : une citation syntaxiquement valide qui
renvoie au **mauvais document** est le pire cas (réponse fausse présentée
comme sourcée). Une citation est jugée sur sa **provenance**, pas seulement sa
forme. Aucun nom de corpus/société/fichier en dur.

---

## 6. Outils de l'agent (`src/tools/`)

Chaque outil est une **façade** : il enveloppe les modules `rag/`, il ne
réimplémente rien. Contrat commun `src/tools/base.py` : nom, schéma typé des
arguments (construit dynamiquement depuis le profil), description, et
`ResultatOutil` normalisé. Le LLM ne reçoit jamais le JSON brut d'un
résultat : `rendu_agent()` produit un texte compact.

Les quatre outils sont **`lecture_seule = True`**. `SessionAgent` n'expose par
défaut que les outils de lecture ; des outils d'écriture (APIs entreprise)
exigeraient une autorisation explicite. Aucune API d'entreprise n'est
supposée, inventée ou appelée.

| Outil | Enveloppe | Cas A (document nommé) | Cas B (contextuel) |
|---|---|---|---|
| `search` | `retrieval.rechercher_passages` | — | recherche hybride complète |
| `summarize` | `retrieval.charger_document` (lecture pure, **pas** de recherche) | document entier → résumé map-reduce borné | résume `ContexteOutil.sources` déjà récupérées |
| `classify` | idem | document entier → **vote majoritaire absolu** par lots, sinon abstention | classe le contexte existant |
| `extract` | idem | document entier → **déduplication** des valeurs par lots (jamais un vote), toutes les valeurs distinctes sourcées conservées | extrait du contexte existant |

Garanties communes SUMMARIZE/CLASSIFY/EXTRACT : aucun appel Qdrant hors
`charger_document`, aucun embedding/reranking, aucune connaissance externe,
citations limitées aux sources réellement présentes, **jamais de succès
silencieux sans provenance**. L'agrégation multi-lots est **déterministe, en
Python, jamais confiée au LLM**.

---

## 7. Couche agentique (`src/agent/`)

### 7.1 Graphe (`graph.py`, `nodes.py`)

Routage **100 % déterministe** — jamais confié au tool-calling du LLM (la
fiabilité du function-calling de qwen3:8b via Ollama n'est pas établie dans ce
dépôt ; le graphe doit rester prévisible).

```
détecter_intention
   ├─ search    → rechercher → évaluer_preuves → (répondre | reformuler → rechercher)
   ├─ summarize → summarize → répondre
   ├─ classify  → classify  → répondre
   └─ extract   → extract   → répondre
```

**Détection d'intention** (`_detecter_intention`) : vocabulaire fermé,
déterministe, testable hors graphe et hors corpus.

1. jetons/bigrammes SUMMARIZE (`résume`, `synthèse`, `points clés`, …) → `summarize`
2. jetons CLASSIFY sûrs / impératif initial → `classify`
3. jetons CLASSIFY ambigus → `ambigu_classify`
4. jetons EXTRACT sûrs / motif structurel `champ : ?` → `extract`
5. marqueurs d'énumération (`,`, ` et `, ` and `, ` ainsi que `) → `ambigu_search_extract`
6. sinon → `search`

Les deux seules zones grises (`ambigu_classify`, `ambigu_search_extract`) sont
tranchées par un **classifieur LLM borné** : une évaluation, ensemble de
sortie fermé (`{search, classify}` ou `{search, extract}`), tout échec ou
sortie invalide retombe sur `search`. Le LLM ne choisit jamais l'outil
lui-même.

`_parser_champs_extraction` (liste des champs demandés par EXTRACT) est un
appel LLM borné distinct du routage : il ne choisit ni outil, ni document, ni
catégorie.

### 7.2 Branche SEARCH : évaluation des preuves (`noeud_evaluer_preuves`)

Deux jugements distincts, journalisés dans la trace et **lus tels quels** par
le routeur (le routage ne diverge jamais du jugement journalisé) :

| Jugement | Niveau | Nature |
|---|---|---|
| `preuves_pertinentes` | 1 | le meilleur score de reranking ≥ `SEUIL_PERTINENCE_MINIMALE` (0.15) — déterministe |
| `preuves_suffisantes` | 2 | ces passages contiennent-ils réellement l'information ? — jugement LLM borné (`_juger_suffisance`) ; reste `None` tant que le niveau 1 échoue |

Une réponse n'est générée que si **les deux** valent `True`. Sinon :
`reformuler` (boucle bornée par `max_tentatives`, défaut 6), avec arrêt
anticipé si `stagnation` (le score de pertinence ne bouge plus). Budget
épuisé sans preuves → **refus déterministe**.

### 7.3 État & session

- `EtatAgent` (`state.py`) : requête courante, budget de tentatives, trace
  d'exécution horodatée. `incrementer_tentative` lève au-delà du budget.
- `SessionAgent` (`session.py`) : assemble LLM + profil de domaine +
  `ContexteOutil` partagé + registre d'outils + `EtatAgent`. Ne prend **aucune
  décision**, ne construit aucun prompt de génération, n'appelle ni Qdrant ni
  les embeddings. Instanciable sans serveur Ollama ni collection Qdrant (seule
  l'exécution effective d'un outil en a besoin).
- `EtatGraphe` (`graph_state.py`) : porté par LangGraph ; `session` n'est
  jamais réassignée, seuls ses attributs internes sont mutés.

---

## 8. LLM (`src/llm/`)

**Point d'accès unique.** `factory.construire_llm()` → `ChatOllama` (LangChain)
paramétré depuis `Settings` (`llm_model`, `llm_temperature`,
`llm_num_ctx`, `num_predict`). `common.py` : helpers d'invocation, extraction
de JSON, et **retrait systématique des blocs `<think>…</think>`** (qwen3 émet
un raisonnement avant sa réponse) de tout texte destiné à l'utilisateur ou
réinjecté dans un prompt.

Aucun autre module n'instancie un client LLM. Changer de modèle Ollama = une
variable `.env`. Changer de fournisseur = ce seul fichier (hors périmètre V1).

---

## 9. Profils de domaine (`src/profiling/`)

`suggest` transforme une saisie administrateur (« Finance et comptabilité »)
en un `DomainProfile` validé via le LLM, **sans lire aucun document** et sans
toucher Qdrant/embeddings/retrieval. Suggestion et persistance séparées.
CLI : `python -m src.profiling.cli {suggest|list|show|delete}`.

Le `DomainProfile` actif (`ACTIVE_DOMAIN_PROFILE`) n'injecte que du
**vocabulaire métier** dans les prompts de génération — il ne modifie ni le
retrieval ni le routage.

---

## 10. Frontières de responsabilité (résumé)

| Question | Répond |
|---|---|
| Comment couper un document ? | `chunking.py` |
| Comment vectoriser / reranker ? | `embeddings.py` |
| Où et comment indexer / chercher dans Qdrant ? | `vectorstore.py` |
| Quels passages pour cette question ? | `retrieval.py` |
| Quel document vise la question ? | `retrieval.py` (`CatalogueDocuments`) |
| Comment rédiger une réponse sourcée ? | `generation.py` |
| Le contexte est-il suffisant ? les citations bonnes ? | `validation.py` |
| Quelle capacité pour cette requête ? | `agent/nodes.py` (déterministe) |
| Boucler / reformuler / refuser ? | `agent/nodes.py` + `agent/graph.py` |
| Exposer une capacité comme outil ? | `tools/*.py` |
| Quel modèle LLM, comment l'appeler ? | `llm/factory.py` + `llm/common.py` |
| Mesurer la qualité ? | `evaluation/` (jamais `src/`) |
