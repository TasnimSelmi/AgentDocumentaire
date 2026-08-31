# Architecture — RAG V1 + cœur agentique P1

> État : socle RAG **gelé** (tag `rag-v1`) ; cœur agentique **candidat au gel
> P1** (clôture P1.7, 2026-08-30). Ce document décrit le système tel qu'il est
> réellement implémenté. Les modules gelés sont ceux de
> [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md).
>
> **Capacités réellement supportées** : SEARCH, SUMMARIZE, CLASSIFY, EXTRACT
> (mono-document) + COMPARE, SYNTHESIZE (multi-document, 2 à 4 documents
> explicitement nommés). Contrat de sortie public unique : `AgentResponse`
> (`src/agent/response.py`, §7.5). Point d'entrée du cœur : `executer_agent`
> (`src/agent/graph.py`). Façade applicative (P2.1) : `AgentService`
> (`src/agent/service.py`, §7.7) — c'est elle que les couches API/UI doivent
> appeler. Voir [P1_CLOTURE.md](P1_CLOTURE.md) pour le statut détaillé
> (supporté / supporté avec limitation / reporté P2).
>
> **Sources documentaires (P2.2, gelé)** : `DocumentSource` /
> `LocalDocumentSource` / `SnapshotDocumentSource` / `IngestionService`
> (`src/sources/`, §2.1) — toute origine de documents (dossier local
> aujourd'hui, API / GED demain) passe par cette couche et par
> `IngestionService.sync()`, jamais par une modification de `src/rag/**`. Voir
> [P2.2_SOURCES.md](P2.2_SOURCES.md) et [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md) §4.

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

### 2.1 Sources documentaires (P2.2 — `src/sources/`, gelé)

`ingerer(dossier=…)` accepte n'importe quel répertoire : l'origine des
documents est donc découplée du pipeline **sans** toucher `src/rag/**`.

```
origine (dossier local ; API / GED / autre demain)
   └─ DocumentSource.materialiser()  ──▶  répertoire = snapshot COMPLET (Path)
        └─ IngestionService.sync(source)  ──▶  ingerer(dossier=…)   [socle gelé]
```

- **`DocumentSource`** (contrat public) : une seule opération,
  `materialiser() -> AbstractContextManager[Path]`. Le `Path` exposé est
  **toujours un snapshot complet et cohérent** — jamais partiel. Impossibilité
  ⇒ `ErreurSource` **avant** le `yield` (le pipeline n'est pas appelé, l'index
  reste tel quel).
- **`LocalDocumentSource`** : adaptateur local MVP — le dossier de
  l'utilisateur *est* le snapshot (pass-through, aucune copie).
  `IngestionService().sync(LocalDocumentSource(d))` ≡ `ingerer(dossier=d)`.
- **`SnapshotDocumentSource`** : base imposée aux futures sources distantes —
  construit la matérialisation à l'écart, publie **atomiquement** après succès
  **complet**, garde le miroir précédent intact en cas d'échec. Une
  récupération partielle n'est donc **jamais** vue comme une suppression.
- **`IngestionService.sync()`** : façade d'ingestion P2 — un seul appel à
  `ingerer`, options transmises telles quelles, `RapportIngestion` intact.

Détails : [P2.2_SOURCES.md](P2.2_SOURCES.md). Gel : [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md) §4.

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

Les six outils sont **`lecture_seule = True`**. `SessionAgent` n'expose par
défaut que les outils de lecture ; des outils d'écriture (APIs entreprise)
exigeraient une autorisation explicite. Aucune API d'entreprise n'est
supposée, inventée ou appelée.

| Outil | Enveloppe | Cas A (document nommé) | Cas B (contextuel) |
|---|---|---|---|
| `search` | `retrieval.rechercher_passages` | — | recherche hybride complète |
| `summarize` | `retrieval.charger_document` (lecture pure, **pas** de recherche) | 1..N documents entiers → résumé map-reduce borné | résume `ContexteOutil.sources` déjà récupérées |
| `classify` | idem | document entier → **vote majoritaire absolu** par lots, sinon abstention | classe le contexte existant |
| `extract` | idem | document entier → **déduplication** des valeurs par lots (jamais un vote), toutes les valeurs distinctes sourcées conservées | Cas B conservé dans l'outil mais **plus jamais atteint par le graphe** depuis P1.6 (voir §7.3) |
| `compare` | `multidoc_pipeline` (`catalogue.par_identifiant` + `charger_document`) | 2..4 documents nommés → MAP par document → REDUCE inter-document | — (jamais de mode contextuel) |
| `synthesize` | idem | 2..4 documents nommés → MAP par document → REDUCE transversal | — |

Garanties communes SUMMARIZE/CLASSIFY/EXTRACT/COMPARE/SYNTHESIZE : aucun appel
Qdrant hors `charger_document` / `catalogue`, aucun embedding/reranking, aucune
connaissance externe, citations limitées aux sources réellement présentes,
**jamais de succès silencieux sans provenance**. L'agrégation multi-lots et
multi-documents est **déterministe, en Python, jamais confiée au LLM** ; le LLM
ne produit que des analyses par lot / par document, chacune revalidée (citation
présente dans le périmètre chargé, catégorie autorisée).

### 6.1 Couverture intégrale du document (mode « document nommé »)

`charger_document` renvoie **tous** les chunks du document, en ordre. Ils sont
partitionnés en lots bornés (`LIMITE_CARACTERES_LOT = 16 000` c) — **jamais de
troncature aux premiers N caractères** : un dépassement démarre un lot
supplémentaire. Chaque lot est analysé (1 appel LLM), puis les lots sont
agrégés (réduction hiérarchique bornée, profondeur ≤ 3). Un lot en échec
devient une **abstention comptée**, jamais un crash.

`multidoc_pipeline` ajoute des garde-fous explicites que SUMMARIZE / CLASSIFY /
EXTRACT n'ont pas encore (limitation connue, backlog P2) : `NB_LOTS_MAX = 24`
lots par document, contrôle `budget_caracteres_entree_llm()` avant chaque
envoi, refus explicite d'un passage unique hors budget. Dans les trois outils
mono-document, le nombre d'appels LLM du mode « document complet » est borné
par la **taille du document** (pas de plafond dur), la profondeur de réduction
reste bornée, la terminaison est garantie.

---

## 7. Couche agentique (`src/agent/`)

### 7.1 Graphe (`graph.py`, `nodes.py`)

Routage **100 % déterministe** — jamais confié au tool-calling du LLM (la
fiabilité du function-calling de qwen3:8b via Ollama n'est pas établie dans ce
dépôt ; le graphe doit rester prévisible).

```
détecter_intention
   ├─ search      → rechercher → évaluer_preuves → (répondre | reformuler → rechercher)
   ├─ summarize   → summarize   → répondre
   ├─ classify    → classify    → répondre
   ├─ extract     → extract     → répondre
   ├─ compare     → compare     → répondre   (P1.5, multi-document)
   └─ synthesize  → synthesize  → répondre   (P1.5, multi-document)
```

Aucune boucle hors branche SEARCH, aucun planner, aucun ReAct.

**Détection d'intention** (`_detecter_intention`) : vocabulaire fermé,
déterministe, testable hors graphe et hors corpus.

1. jetons/bigrammes/expressions SUMMARIZE (`résume`, `synthèse`, `points clés`, `TL;DR`, …) → `summarize`
2. jetons CLASSIFY sûrs / impératif initial / expressions « type de ce document » → `classify`
3. jetons CLASSIFY ambigus (`catégorie`, `classification` nus ; « s'agit-il d'un… ») → `ambigu_classify`
4. jetons EXTRACT sûrs / motif structurel `champ : ?` → `extract`
5. marqueurs d'énumération (`,`, ` et `, ` and `, ` ainsi que `) → `ambigu_search_extract`
6. sinon → `search`

Les deux seules zones grises (`ambigu_classify`, `ambigu_search_extract`) sont
tranchées par un **classifieur LLM borné** : une évaluation, ensemble de
sortie fermé (`{search, classify}` ou `{search, extract}`), tout échec ou
sortie invalide retombe sur `search`. Le LLM ne choisit jamais l'outil
lui-même.

**Signal multi-document** (`src/agent/multidoc.py`, P1.4) : fonction pure,
déterministe, **0 appel LLM, 0 retrieval**. À partir du seul texte, elle
décide `is_multidoc` + `operation_hint ∈ {compare, synthesize, none}` en
croisant références de fichiers explicites (`*.pdf`, `*.txt`, …), marqueurs
pluriels portant sur un nom de document, marqueurs comparatifs / de synthèse,
avec un garde-fou déixis singulière (« ce document »). Appliqué **après**
résolution des zones grises : il ne supplante que `search` / `summarize` ; une
demande explicite CLASSIFY ou EXTRACT n'est jamais détournée. Précédence
inverse (P1.5) : ≥ 2 références de fichiers + verbe compare/synthesize
explicite → `compare`/`synthesize` **sans** appeler de désambiguïsateur LLM.

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

### 7.3 Désignation du document — SUMMARIZE / CLASSIFY / EXTRACT

Chaque nœud résout d'abord un `PerimetreDocumentaire` via `resoudre_document`
(résolution **par identité**, jamais une recherche de contenu) :

| Situation | SUMMARIZE / CLASSIFY | EXTRACT (depuis P1.6, défaut EX-03) |
|---|---|---|
| 1 document résolu de façon **unique et fiable** (`perimetre.contraignant`, une seule valeur) | mode « document complet » | mode « document complet » |
| requête vise un document mais résolution **non fiable** (`compatible` / `ambigu` / `score_insuffisant`) | **refus déterministe**, sans appeler l'outil ni `search` | **refus déterministe**, idem |
| **aucune** référence documentaire fiable dans la requête | mode contextuel historique (`ContexteOutil.sources`) ; jamais de document arbitraire | **refus déterministe** — plus aucun repli `search` global → `extract(document=None)` |

EXTRACT diverge **volontairement** de CLASSIFY/SUMMARIZE : le repli contextuel
transformait une recherche multi-document en extraction structurée implicite
dès que le top-k ne faisait ressortir qu'un seul document (choix de périmètre
par convenance, non déterministe d'un corpus à l'autre). Depuis P1.6, EXTRACT
n'extrait que sur un périmètre résolu de façon unique et fiable.

*Limitation connue* : `resoudre_document` ne distingue pas structurellement
« document nommé mais absent du catalogue » d'« aucune référence
documentaire » — les deux retombent sur `statut="aucun"` /
`raison="aucune_correspondance"`. Conséquence : un document nommé mais absent
donne, pour SUMMARIZE/CLASSIFY, un refus au message générique
(« utilise d'abord search ») plutôt que « document introuvable ». Refus
toujours **sûr** (aucune hallucination, aucun document substitué) ; seule la
qualité du message est en cause. Backlog P2.

### 7.4 Branches COMPARE / SYNTHESIZE (`src/agent/multidoc_pipeline.py`, P1.5)

Activées uniquement quand le signal multi-document est **explicite**
(`is_multidoc` + `operation_hint ∈ {compare, synthesize}`). `references` = noms
de fichiers explicitement cités dans la requête.

```
resoudre_cibles(references)              2..4 documents distincts et fiables,
   │                                     sinon ABSTENTION déterministe
   │                                     (motif : references_insuffisantes /
   │                                      au_dela_limite / document_introuvable /
   │                                      documents_non_distincts / catalogue_indisponible)
   ▼
MAP par document  (couverture INTÉGRALE, lots bornés, citations [D<k>S<j>])
   │              un document sans élément pertinent / en échec est signalé,
   │              jamais inventé
   ▼
diagnostic : < 2 documents exploitables → ABSTENTION déterministe
   ▼
REDUCE inter-document  (ne voit QUE les MAP validées, jamais le corpus)
   │                    chaque point porte ≥ 1 citation [D_S_] du périmètre ;
   │                    citations hors périmètre retirées ; divergences /
   │                    contradictions CONSERVÉES explicitement
   ▼
ResultatOutil { comparaison | synthese, par_document, citations_utilisees }
```

Garde-fous : **aucun search global**, jamais de repli SEARCH en cas d'échec ;
contrôle de taille du prompt REDUCE **avant** envoi (dépassement →
`motif="budget_reduce_depasse"`, jamais de troncature) ; bornes toutes
techniques et configurables (`LIMITE_DOCUMENTS = 4`, `NB_LOTS_MAX = 24`,
`PROFONDEUR_MAX_AGREGATION = 3`, `budget_caracteres_entree_llm()` dérivé de
`num_ctx − num_predict − marge`).

### 7.5 Contrat de sortie public — `AgentResponse` (`src/agent/response.py`, P1.3)

Point d'entrée public : `executer_agent(requete, **kwargs)` → `AgentResponse`.
Enveloppe **fine** autour du graphe déjà compilé : elle exécute exactement le
même graphe que `invoquer_agent` (routage, capacités, RAG V1 inchangés), puis
**normalise** le résultat final de façon **100 % déterministe — aucun appel
LLM, aucune reconstruction de provenance**. `invoquer_agent` reste disponible
(retour brut `ReponseRAG` / `ResultatOutil`) pour compatibilité.

```
AgentResponse {
  status      : "success" | "refusal" | "error"
  capability  : "search" | "summarize" | "classify" | "extract" | "compare" | "synthesize" | ""
  answer      : texte principal (réponse / résumé / conclusion / message de refus)
  sources     : provenance déjà VALIDÉE {citation, document, page, categorie, extrait, hors_perimetre}
  citations   : identifiants de citation réellement utilisés (S1, D1S2, …)
  warnings    : avertissements non bloquants, repris tels quels
  data        : miroir fidèle du résultat interne (gros texte retiré)
  metadata    : durée, nombre de sources, documents résolus, profil
  error       : {code, message} si status != success, sinon None
}
```

- **`refusal`** = abstention FONCTIONNELLE attendue (preuves insuffisantes,
  document introuvable, résolution multi-document non fiable, budget épuisé…).
  Jamais une exception.
- **`error`** = vrai échec technique non prévu (exception non gérée du graphe,
  résultat interne non reconnu). `capability = ""` si l'erreur précède le
  routage.
- Pour SEARCH, succès/refus est tranché **avant génération** par
  `preuves_pertinentes ET preuves_suffisantes` (le graphe ne génère une
  réponse que si les deux valident).
- `AgentResponse` ne transporte jamais : prompts, raisonnement `<think>`,
  contenu intégral des documents, dump d'état du graphe, objets non
  sérialisables, secrets. `vers_dict()` ne renvoie que des types natifs.

### 7.6 État & session

- `EtatAgent` (`state.py`) : requête courante, budget de tentatives, trace
  d'exécution horodatée. `incrementer_tentative` lève au-delà du budget.
- `SessionAgent` (`session.py`) : assemble LLM + profil de domaine +
  `ContexteOutil` partagé + registre d'outils + `EtatAgent`. Ne prend **aucune
  décision**, ne construit aucun prompt de génération, n'appelle ni Qdrant ni
  les embeddings. Instanciable sans serveur Ollama ni collection Qdrant (seule
  l'exécution effective d'un outil en a besoin).
- `EtatGraphe` (`graph_state.py`) : porté par LangGraph ; `session` n'est
  jamais réassignée, seuls ses attributs internes sont mutés.

### 7.7 Façade applicative — `AgentService` (`src/agent/service.py`, P2.1)

Frontière **P1 / P2**. `AgentService` est une couche **mince** au-dessus du
point d'entrée public P1 :

```
API / UI / connecteurs documentaires
        ↓
   AgentService.query(requete) -> AgentResponse
        ↓
   executer_agent(requete, **options_session)      (cœur P1, inchangé)
        ↓
   graphe LangGraph → normaliser_reponse_agent → AgentResponse
```

- **Rôle** : validation légère de l'entrée (chaîne non vide → sinon
  `AgentResponse(status="error", error.code="requete_invalide")`, sans appeler
  le cœur), **un seul** appel à `executer_agent`, propagation **intacte** de
  l'`AgentResponse`. La façade ne lève **jamais** : une erreur de construction
  de session (`ErreurSession`, profil introuvable…) ressort aussi en
  `AgentResponse(status="error")`.
- **Ce qu'elle ne fait pas** : aucun routage, aucun tool, aucun RAG, aucune
  résolution documentaire, aucune génération, aucune logique COMPARE /
  SYNTHESIZE, aucune re-normalisation. Aucune dépendance FastAPI / UI /
  connecteur / mémoire / logging. `options_session` est un simple passe-plat
  vers `construire_session` (`llm`, `profil_domaine`,
  `charger_profil_domaine`, `max_tentatives`, …), jamais interprété.
- **Injection** : `AgentService(point_entree=…, options_session=…)` — le
  `point_entree` par défaut est `executer_agent` ; un fake suffit pour tester
  la façade sans Ollama ni Qdrant.

> **Règle** : toute couche applicative (FastAPI, UI, connecteurs) appelle
> `AgentService`, **jamais** LangGraph, `graph.py`, `nodes.py` ni une
> structure interne du graphe directement.

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
| D'où viennent les documents à ingérer ? | `sources/*.py` (`DocumentSource`, snapshot complet) |
| Lancer une ingestion depuis une source ? | `sources/service.py` (`IngestionService.sync`) |
| Comment couper un document ? | `chunking.py` |
| Comment vectoriser / reranker ? | `embeddings.py` |
| Où et comment indexer / chercher dans Qdrant ? | `vectorstore.py` |
| Quels passages pour cette question ? | `retrieval.py` |
| Quel document vise la question ? | `retrieval.py` (`CatalogueDocuments`) |
| Comment rédiger une réponse sourcée ? | `generation.py` |
| Le contexte est-il suffisant ? les citations bonnes ? | `validation.py` |
| Quelle capacité pour cette requête ? | `agent/nodes.py` (déterministe) |
| La requête vise-t-elle plusieurs documents ? | `agent/multidoc.py` (pur, sans LLM) |
| Boucler / reformuler / refuser ? | `agent/nodes.py` + `agent/graph.py` |
| Comparer / synthétiser 2..4 documents nommés ? | `agent/multidoc_pipeline.py` + `tools/compare.py` / `tools/synthesize.py` |
| Exposer une capacité comme outil ? | `tools/*.py` |
| Contrat de sortie unique pour un consommateur externe ? | `agent/response.py` (`AgentResponse`, déterministe) |
| Quel modèle LLM, comment l'appeler ? | `llm/factory.py` + `llm/common.py` |
| Mesurer la qualité ? | `evaluation/` (jamais `src/`) |
