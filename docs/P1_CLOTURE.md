# Clôture P1 — cœur agentique

> Phase **P1.7** : validation finale avant gel du cœur agentique. Aucune
> nouvelle capacité. Ce document décrit l'état **réel** du système au
> 2026-08-30 et sert de référence de gel (à lire avec
> [architecture.md](architecture.md) et [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md)).

---

## 1. Architecture finale P1

```
requête utilisateur
   │
   ▼
executer_agent(requete)                              (src/agent/graph.py — point d'entrée public)
   │
   ├─ construire_session()  → LLM partagé + profil de domaine + ContexteOutil + registre d'outils + EtatAgent
   │
   ▼
GRAPHE LangGraph compilé, routage 100 % déterministe
   │
   ▼
detecter_intention   ── _detecter_intention (lexical, fermé, sans LLM)
   │                     + detecter_multidoc (pur, sans LLM)
   │                     + 2 désambiguïsateurs LLM bornés (zones grises CLASSIFY/SEARCH, SEARCH/EXTRACT)
   │
   ├─ search      → rechercher → evaluer_preuves ─┬─ pertinent(≥0.15) ET suffisant(LLM borné) → generer_reponse
   │                                              └─ sinon → reformuler → rechercher   (boucle bornée max_tentatives=6,
   │                                                                                    arrêt anticipé si stagnation ;
   │                                                                                    budget épuisé → refus déterministe)
   ├─ summarize   → summarize   (document nommé complet | contexte existant | refus)
   ├─ classify    → classify    (document nommé complet | contexte existant | refus)
   ├─ extract     → extract     (document nommé complet | refus déterministe — jamais de repli search, P1.6)
   ├─ compare     → compare     (2..4 documents nommés : MAP par document → REDUCE inter-document | abstention)
   └─ synthesize  → synthesize  (2..4 documents nommés : MAP par document → REDUCE transversal | abstention)
   │
   ▼
résultat interne  (ReponseRAG pour SEARCH ; ResultatOutil pour les 5 autres)
   │
   ▼
normaliser_reponse_agent(...)        (src/agent/response.py — 100 % déterministe, aucun LLM)
   │
   ▼
AgentResponse { status, capability, answer, sources, citations, warnings, data, metadata, error }
```

Le cœur RAG V1 (`src/rag/**`, tag `rag-v1`) est appelé **comme un outil**, jamais
réimplémenté. `git diff rag-v1 -- src/rag` est **vide**.

---

## 2. Les 6 capacités — condition de routage, périmètre, LLM, provenance, refus, limites

### SEARCH
- **Routage** : cas par défaut (`_detecter_intention` ne trouve aucun déclencheur SUMMARIZE/CLASSIFY/EXTRACT) ; repli de toute zone grise et de toute valeur d'intention inattendue.
- **Mono / multi** : mono-question ; retrieval hybride sur tout le corpus, cloisonné au périmètre documentaire s'il est résolu.
- **Document explicite** : facultatif ; `retrieval` résout un périmètre (`exact` / `compatible` / `ambigu` / `aucun`) et filtre Qdrant si `exact`/`compatible`.
- **LLM** : reranking (modèle dédié, pas un LLM génératif) ; niveau 2 « suffisance » = 1 appel LLM borné ; reformulation = 1 appel LLM borné par tentative ; génération finale = `generation.py` (gelé).
- **Provenance** : citations `[S1]`, `[S2]`… validées jusqu'au document par `validation.py` ; 1 seule tentative de réparation ; refus déterministe sans appel LLM si aucun passage exploitable.
- **Refus sûrs** : preuves non pertinentes (score < 0.15) OU non suffisantes (jugement LLM) → `reformuler` tant qu'il reste du budget ; budget épuisé / stagnation → `AgentResponse(status="refusal", error.code="evidence_insuffisante")`.
- **Limites** : questions hors-corpus mais thématiquement proches → score de reranking élevé, rattrapées par le niveau 2 seulement ; qualité de la reformulation dépend du LLM.

### SUMMARIZE
- **Routage** : jetons/bigrammes/expressions fermés (`résume`, `synthèse`, `points clés`, `TL;DR`, `main takeaways`, …).
- **Mono / multi** : 1..N documents nommés résumés dans leur intégralité (citations renumérotées pour éviter les collisions) ; sinon résume `ContexteOutil.sources`.
- **Document explicite** : si 1 document résolu de façon unique et fiable → mode « document complet » (`charger_document`, **tous** les chunks) ; résolution non fiable → refus déterministe ; aucune référence → mode contextuel.
- **LLM** : 1 appel par lot (map) + réduction hiérarchique bornée (profondeur ≤ 3). Agrégation = concaténation + 1 synthèse finale, jamais un choix du LLM.
- **Provenance** : citations validées contre les passages réellement chargés ; 0 citation valide → échec explicite (`_resultat_sans_provenance`), jamais un succès.
- **Refus sûrs** : document ambigu / inconnu / vide → `ResultatOutil.echec` → `AgentResponse(status="refusal")`.
- **Limites** : pas de plafond dur `NB_LOTS_MAX` (borné par la taille du document) ; message de refus générique pour un document nommé mais absent (voir §4).

### CLASSIFY
- **Routage** : jetons verbaux sûrs (`classifie`, `catégorise`, …), impératif initial « classe … », expressions « type de ce document » ; zone grise (`catégorie`/`classification` nus, « s'agit-il d'un… ») → désambiguïsateur LLM borné → `{search, classify}`.
- **Mono / multi** : mono-document strict.
- **Document explicite** : 1 document unique et fiable → classification hiérarchique par lots ; résolution non fiable → refus déterministe **sans search** ; aucune référence → `search` ciblé pour alimenter le contexte puis classification du contexte (seul cas où ce mode reste sollicité).
- **LLM** : 1 vote par lot, chacun revalidé (catégorie ∈ taxonomie du profil, citation valide) ; **décision finale = vote majoritaire absolu calculé en Python**, sinon abstention.
- **Provenance** : catégories = `profil.classification.noms()` (jamais inventées) ; citations des seuls votes valides pour la catégorie retenue.
- **Refus sûrs** : pas de majorité absolue → abstention explicite avec motif ; document inconnu/vide → échec.
- **Limites** : mêmes que SUMMARIZE (pas de `NB_LOTS_MAX`, message de refus générique document absent).

### EXTRACT
- **Routage** : jetons verbaux sûrs (`extrais`, `extract`, …), motif structurel `champ : ?`, ou « récupère les champs/valeurs… » ; énumération sans verbe d'extraction → zone grise → désambiguïsateur LLM borné → `{search, extract}`.
- **Mono / multi** : mono-document strict — `extract` ne mélange jamais deux documents.
- **Document explicite** : **obligatoire et fiable** depuis P1.6 (défaut EX-03). 1 document unique et fiable → extraction par lots ; **tout le reste → refus déterministe**, sans jamais appeler `search` ni `extract`. Le mode contextuel de l'outil existe encore mais **n'est plus jamais atteint par le graphe**.
- **LLM** : `_parser_champs_extraction` = 1 appel LLM borné (segmente la requête en champs, n'en invente aucun) ; 1 appel d'extraction par lot ; **agrégation = déduplication des valeurs (jamais un vote)** en Python, toutes les valeurs distinctes sourcées conservées.
- **Provenance** : citations validées contre les passages chargés ; aucun champ valide → échec propre (« Aucun champ d'extraction valide n'a été fourni. »).
- **Refus sûrs** : aucun document fiable → `AgentResponse(status="refusal")` demandant de préciser le document.
- **Limites** : pas de `NB_LOTS_MAX`.

### COMPARE
- **Routage** : signal multi-document **explicite** — `is_multidoc` + `operation_hint="compare"` (marqueurs `compare`, `différence`, `en quoi`, `versus`, `points communs`, …) ; précédence sur les zones grises si ≥ 2 références de fichiers + verbe comparatif.
- **Mono / multi** : 2 à 4 documents (`LIMITE_DOCUMENTS = 4`), explicitement nommés par nom de fichier dans la requête.
- **Document explicite** : **obligatoire** ; résolution via `catalogue.par_identifiant` (identité, jamais recherche de contenu).
- **LLM** : MAP = 1 appel par lot par document (couverture intégrale) + agrégation intra-document bornée ; REDUCE = 1 appel inter-document qui **ne voit que les MAP validées**.
- **Provenance** : schéma `[D<k>S<j>]` (k = document) ; chaque point porte ≥ 1 citation du périmètre ; citations hors périmètre retirées ; contradictions/divergences **conservées explicitement**.
- **Refus sûrs** : < 2 références / > 4 / document introuvable / non distincts / catalogue indisponible / < 2 documents exploitables / prompt REDUCE hors budget → **abstention déterministe**, motif explicite, **jamais de repli search**.
- **Limites** : `NB_LOTS_MAX = 24` par document (au-delà : document refusé) ; passage unique hors budget → document refusé (pas de split-avec-provenance en V1) ; pas de compaction pré-REDUCE.

### SYNTHESIZE
- **Routage** : signal multi-document explicite — `operation_hint="synthesize"` (marqueurs `synthèse`, `consolide`, `vue d'ensemble`, `key findings from`, …).
- **Mono / multi**, **document explicite**, **LLM**, **refus sûrs**, **limites** : identiques à COMPARE (même pipeline `multidoc_pipeline`).
- **Provenance** : une synthèse n'efface jamais un désaccord — les divergences sont conservées dans `divergences`, jamais lissées.

---

## 3. Statut des capacités

| Capacité | Statut | Preuve |
|---|---|---|
| SEARCH | **Supporté et testé** | 658 tests (dont branche SEARCH complète) ; routing `deterministic_only` SEARCH **24/24** ; chemin = RAG V1 gelé |
| SUMMARIZE | **Supporté et testé** | tests `tests/tools/test_summarize*.py` + `tests/agent/test_nodes.py` ; routing SUMMARIZE 10/10 |
| CLASSIFY | **Supporté et testé** | tests `tests/tools/test_classify*.py` + nœud ; routing CLASSIFY 6/8 en déterministe, résiduel résolu en production par le désambiguïsateur borné |
| EXTRACT | **Supporté et testé** | tests `tests/tools/test_extract*.py` + `test_nodes.py` (EX-03) ; routing EXTRACT 7/9 en déterministe, résiduel résolu en production |
| COMPARE | **Supporté et testé** | `tests/agent/test_compare.py`, `test_graph_multidoc.py`, `test_multidoc_pipeline.py` ; mini-benchmark multidoc **12/12** ; `executer_agent` bout-en-bout |
| SYNTHESIZE | **Supporté et testé** | `tests/agent/test_synthesize.py` + benchmark multidoc |
| `AgentResponse` (contrat public) | **Supporté et testé** | `tests/agent/test_response.py` (17 tests) : 6 capacités × succès/refus, sérialisation types natifs, aucune perte de provenance, `status="error"` sur résultat inattendu, 2 tests `executer_agent` bout-en-bout |

**Supporté avec limitation**
- Mode « document complet » SUMMARIZE / CLASSIFY / EXTRACT : pas de plafond dur du nombre de lots (borné par la taille du document ; terminaison garantie). `multidoc_pipeline` a ce plafond (`NB_LOTS_MAX`), pas les trois outils mono-document.
- Résolution documentaire : « document nommé mais absent » indistinct de « aucune référence » → message de refus générique pour SUMMARIZE/CLASSIFY (refus toujours sûr).
- Routage `deterministic_only` : CLASSIFY 6/8, EXTRACT 7/9 (les cas manquants retombent sur SEARCH, **jamais l'inverse**) ; résolus en `production_routing` par les désambiguïsateurs LLM bornés.

**Reporté P2** — voir §6.

---

## 4. Invariants vérifiés (P1.7)

| Invariant | Vérdict | Constat |
|---|---|---|
| Aucun hard-code corpus / dataset / métier dans la logique | **OK** (avec réserve cosmétique) | Aucune règle `if corpus == …`, aucun chemin de dataset dans le code de production. Réserve : exemples à saveur corpus (« Absa », « B-BBEE ») dans des **descriptions/docstrings** de `src/tools/classify.py`, `src/tools/extract.py`, `src/agent/session.py` et commentaires de calibration dans `src/agent/nodes.py`. Sans effet fonctionnel ; fichiers gelés → à nettoyer sous exception documentaire P2. |
| `src/rag/**` gelé | **OK** | `git diff rag-v1 -- src/rag` **vide**. |
| SEARCH reste générique | **OK** | `src/tools/search.py` = façade `retrieval.rechercher_passages` ; schéma dérivé du profil ; aucune logique métier. |
| Ciblage full-document couvre tous les passages OU refuse explicitement | **OK** | `charger_document` → tous les chunks ; partitionnement sans troncature ; `multidoc_pipeline` refuse explicitement (NB_LOTS_MAX, passage hors budget) ; SUMMARIZE/CLASSIFY/EXTRACT couvrent tout (plafond = taille document). |
| EXTRACT ne choisit plus implicitement un document via top-k | **OK** | `noeud_extract` (P1.6) : `document_complet` seulement si `perimetre.contraignant` et 1 seule valeur ; sinon refus déterministe, **jamais** `search` ni `extract(document=None)`. |
| COMPARE / SYNTHESIZE travaillent uniquement sur documents résolus | **OK** | `resoudre_cibles` → 2..4 documents distincts et fiables du catalogue, sinon abstention ; MAP borné à `cible.doc_id`. |
| Aucun fallback SEARCH caché en cas d'échec multi-doc | **OK** | `comparer` / `synthetiser_documents` : tous les chemins d'échec → `ResultatOutil.echec` avec motif ; aucun appel à `search` / `rechercher_passages`. |
| Budgets LLM bornés | **OK** (avec limitation) | `budget_caracteres_entree_llm()` = source unique ; contrôle **avant** chaque envoi dans `multidoc_pipeline` ; `PROFONDEUR_MAX_AGREGATION`/`_PROFONDEUR_MAX_SYNTHESE` = 3 ; `max_tentatives` = 6. Limitation : pas de `NB_LOTS_MAX` dans les 3 outils mono-document. |
| Provenance / citations jamais inventées | **OK** | `validation.py` (SEARCH) ; `_valider_citations` contre passages chargés (SUMMARIZE/CLASSIFY/EXTRACT) ; `valider_citations` contre `citations_autorisees` + `retirer_citations_invalides` (COMPARE/SYNTHESIZE) ; 0 citation valide → échec, jamais un succès. |
| `AgentResponse` transporte sans nouvelle génération LLM | **OK** | `normaliser_reponse_agent` = lecture d'attributs / dict pure ; `_ANSWER_PAR_CAPACITE` / `_CITATIONS_PAR_CAPACITE` = accès dict ; aucun import LLM dans `response.py`. |
| Refus fonctionnels ≠ erreurs techniques | **OK** | `status="refusal"` (abstention prévue) vs `status="error"` (exception non gérée / résultat non reconnu) ; `executer_agent` n'émet `error` que sur exception du graphe. Confirmé en E2E live (refus multidoc / summarize → `status="refusal"`). |

**Aucune violation critique.** Points relevés = limitations connues ou réserves
cosmétiques, tous listés ci-dessus et en §6.

---

## 5. Résultats de validation P1.7

### pytest
`658 passed` (suite complète), exécution ~3 s.

> Sur cet environnement, `.env` pointe `QDRANT_PATH` vers un `/data/vectordb/…`
> **absolu non inscriptible** → 161 échecs `PermissionError` lors de
> `Settings.creer_dossiers()`, **sans rapport avec le code** (gotcha déjà
> documenté dans `.env.example` et le CHANGELOG). Avec `QDRANT_PATH` redirigé
> vers un dossier inscriptible du projet : **658/658 PASS**. Sous-ensembles :
> `tests/agent` 308/308, `tests/tools` 79/79, `tests/agent/test_response.py`
> 17/17, `tests/evaluation` 112/112.

### Benchmarks (mesure / non-régression uniquement — aucun patch)
| Banc | Résultat |
|---|---|
| `evaluate_routing --mode deterministic_only` | 57/65 (87,7 %). **SEARCH 24/24** (invariant), SUMMARIZE 10/10, COMPARE 6/6, SYNTHESIZE 4/4, CLASSIFY 6/8, EXTRACT 7/9, CLARIFY 0/4 (capacité non implémentée — reporté P2). Détecteur multi-document : **14/14** (is_multidoc + operation_hint). |
| `multidoc_benchmark` (hors ligne, LLM scripté) | **12/12 (100 %)** — routing 12/12, aucune fuite tierce 7/7, contradictions conservées 5/5, refus sûrs 2/2, aucun repli search 2/2. |
| `evaluate_routing --mode production_routing` | **non complété** sur cet environnement : le LLM local (qwen3:8b, quasi tout en CPU) est trop lent pour 65 cas dans le temps imparti. Aucun impact sur le verdict : l'invariant SEARCH 24/24 est déjà établi en déterministe, et `production_routing` ne peut qu'**améliorer** CLASSIFY/EXTRACT via les désambiguïsateurs. |
| Smoke CQuAE (`cquae_multicapacite`) | **non ré-exécuté** sur cet environnement : `config/default.yaml → qdrant.nom_collection` (gelé) ne correspond pas à la collection indexée localement, et BGE-M3 ne peut pas se charger (un processus tiers occupe 21 Go / 24 Go de VRAM → CUDA OOM). Référence P0 inchangée : 20/28 PASS, run `20260829-094620`. |

### End-to-end `executer_agent` → `AgentResponse`
- Contrat validé par `tests/agent/test_response.py` (17 tests) : 6 capacités succès + refus, 2 tests bout-en-bout, sérialisation 100 % native, aucune perte de provenance.
- Vérification **live** (agent réel, Ollama réel) — partielle (contrainte VRAM tierce) :
  - `Compare docA_inexistant.pdf et docB_inexistant.pdf` → `AgentResponse(status="refusal", capability="compare", error.code="document_introuvable")`, 2,2 s, 0 appel LLM. ✅
  - `Résume le document inexistant_zzz_999.txt` → `AgentResponse(status="refusal", capability="summarize")`, message générique (limitation §4), aucun document substitué. ✅
  - EXTRACT sans document fiable / CLASSIFY document nommé : lancés, non complétés dans le temps imparti (lenteur LLM CPU) — couverts par les tests unitaires.
  - SEARCH / COMPARE-succès / SYNTHESIZE-succès live : **impossibles ici** (CUDA OOM sur BGE-M3, cause externe) ; couverts par les 658 tests + benchmark hors ligne.

---

## 6. Éléments explicitement reportés à P2

- **Capacité CLARIFY** : requêtes vagues sans intention actionnable (« Analyse ça. », « Compare. ») → actuellement routées vers SEARCH (repli sûr, aboutit à un refus). Aucune demande de précision.
- **Plafond `NB_LOTS_MAX`** pour le mode « document complet » de SUMMARIZE / CLASSIFY / EXTRACT (symétrie avec `multidoc_pipeline`).
- **Split-avec-provenance** d'un passage unique hors budget (COMPARE/SYNTHESIZE le refusent aujourd'hui).
- **Compaction pré-REDUCE** quand les analyses par document dépassent le budget REDUCE (refus déterministe aujourd'hui).
- **Résolveur documentaire** : distinguer « document nommé absent » de « aucune référence » pour un message de refus précis.
- **Nettoyage des exemples à saveur corpus** dans les descriptions/docstrings de `src/tools/classify.py`, `src/tools/extract.py`, `src/agent/session.py`, commentaires de `src/agent/nodes.py` (sous exception documentaire, sans changement fonctionnel).
- **Traces détaillées / tracing** exposées au consommateur externe (hors périmètre `AgentResponse` P1).
- **API / UI / SSE**, mémoire conversationnelle, planner/ReAct, nouveau resolver, nouveau LLM : hors P1 par construction.

---

## 7. Verdict

## ✅ P1 READY TO FREEZE

Le cœur agentique P1 est cohérent, générique et multi-corpus, couvre les
6 capacités annoncées (SEARCH, SUMMARIZE, CLASSIFY, EXTRACT, COMPARE,
SYNTHESIZE) plus le contrat public `AgentResponse`, respecte tous les
invariants critiques, et est couvert par 658 tests verts + 2 benchmarks
hors-ligne au vert. Le socle `src/rag/**` reste gelé (`git diff rag-v1`
vide). Les points ouverts sont des **limitations connues** ou des
**améliorations P2**, aucun n'est un blocker de sûreté ou de correction.

**Aucun blocker critique P1.**
