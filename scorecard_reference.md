# Scorecard de référence — CQuAE multi-capacités

> **P0 / Étape 1 — « Fiabiliser le benchmark et publier un scorecard de référence »**

---

## 0. Statut

**J1 / P0 Étape 1 : NON TERMINÉ.**

Motif unique : la condition de sortie (« un run CQuAE dont chaque WRONG
correspond à un vrai défaut identifié, plus aucun faux négatif du harnais »)
exige un **run agent complet post-correctif #1 (groundedness)**. Ce run n'a
pas pu aboutir dans la session : le GPU de la machine est monopolisé à 98–100 %
par un entraînement tiers (`salem/src/train.py`, ~3 h), Ollama bascule sur CPU,
et une seule génération `qwen3:8b` (modèle *reasoning*) a dépassé **19 minutes
sans terminer** — un run de 28 cas est non réalisable tant que le GPU n'est pas
rendu.

Les trois correctifs sont **implémentés et couverts par des tests unitaires
déterministes** (voir §7–9). Les correctifs **#2 (UUID/nom de fichier)** et
**#3 (EXTRACT)** sont en outre **validés end-to-end** par le dernier run agent
réel (`20260828-182551`). Le correctif **#1 (groundedness)** est validé
unitairement mais son effet sur les verdicts SEARCH reste à confirmer sur un
run agent (analyse hors-ligne en §4).

---

## 1. Run de référence

| | |
|---|---|
| **Run ID de référence** | `cquae_multicapacite_20260828-182551` |
| Fichiers | `evaluation/reports/cquae_multicapacite/cquae_multicapacite_20260828-182551.{json,csv}` |
| Date | 2026-08-28 18:25 |
| Correctifs harnais actifs dans ce run | #2 (UUID→nom de fichier), #3 (EXTRACT : `donnees["extractions"]` + `_associer_champ` + valeurs multiples), + exactitude SEARCH `_reponse_couvre_gold` |
| Correctif **non** présent dans ce run | **#1 groundedness** (dénominateur = passages récupérés complets) — ajouté après ce run, testé unitairement, **run de confirmation en attente GPU** |
| Run de confirmation #1 | **à relancer** — commande en §11 |

### Configuration (préconditions vérifiées par le harnais avant exécution)

| Paramètre | Valeur | Source |
|---|---|---|
| `ACTIVE_PROFILE` | `generic` | env |
| `ACTIVE_DOMAIN_PROFILE` | `histoire-culture-humaines` | env |
| Collection Qdrant | `cquae_eval` — 2240 points, statut `green` | surcharge `_runner_config` |
| `QDRANT_PATH` | `data/vectordb/qdrant_cquae_eval` | env |
| LLM | Ollama `qwen3:8b`, `temperature=0.0`, `num_ctx=16384` | `.env` / `src/llm/factory.py` (abstraction inchangée) |
| Embeddings / reranker | `BAAI/bge-m3` / `BAAI/bge-reranker-v2-m3` | `.env` (device sans effet sur les valeurs) |
| Gold | `cquae_agent_gold.jsonl` — 240 questions, **1 exclue** (`cquae_doc_2262.txt` absent de l'index → `cquae:test:8298`), 239 évaluables |
| Cas smoke | `cquae_smoke_cases.jsonl` — 28 cas |
| Latence agent | moyenne 12,6 s / cas ; max 28,0 s (run avec GPU) |

> **Blocage d'environnement constaté (hors périmètre correctifs)** : le `.env`
> local porte `QDRANT_PATH=/data/vectordb/qdrant_cquae_eval` (barre oblique de
> tête). `src/config.py::creer_dossiers()` tente alors `mkdir /data` →
> `PermissionError`, ce qui fait **échouer 99 tests** et bloquerait le smoke.
> Corrigé pour la validation par surcharge `QDRANT_PATH=data/vectordb/qdrant_cquae_eval`
> (valeur conforme à `.env.example` et à la docstring du harnais). **À rétablir
> dans le `.env` : retirer la barre oblique de tête.**

---

## 2. Résultats globaux — run de référence `20260828-182551`

| Verdict | Nombre | Cas |
|---|---:|---|
| **PASS** | **14** | SQ-02, SQ-04, SQ-06, SQ-07, SQ-10, SQ-12 ; EX-01, EX-02, EX-04, EX-05 ; CL-01, CL-02 ; SU-01, SU-03 |
| ANSWER_ONLY | 8 | SQ-01, SQ-03, SQ-05, SQ-08, SQ-09, SQ-13, SQ-14, SQ-15 — *tous PROVENANCE_FAILURE via groundedness < 0.5* |
| RETRIEVAL_ONLY | 2 | SQ-11, SQ-16 — GENERATION_FAILURE |
| UNSUPPORTED | 0 | — |
| **WRONG** | **2** | EX-03 (EXTRACTION_FAILURE), SU-02 (ROUTING_FAILURE) |
| ABSTAIN_CORRECT | 2 | CL-03 (document absent), AH-01 (hors-corpus) |
| TECHNICAL_ERROR | 0 | — |

### Évolution du score

| Run | PASS | ANSWER_ONLY | RETRIEVAL_ONLY | WRONG | ABSTAIN_CORRECT | Correctifs |
|---|---:|---:|---:|---:|---:|---|
| `20260827-134434` (pré-audit) | **0** | 0 | 16 | 10 | 2 | aucun |
| `20260828-124343` | 6 | 10 | 2 | 8 | 2 | exactitude SEARCH + fast-path retrieval |
| `20260828-182551` **(réf.)** | **14** | 8 | 2 | 2 | 2 | + #2 UUID/nom fichier + #3 EXTRACT |
| *post-#1 (attendu, à confirmer)* | *≥ 18* | *≤ 4* | *2* | *2* | *2* | + #1 groundedness |

- **Ancien score** (pré-audit, `20260827-134434`) : **0 / 28 PASS**.
- **Score de référence actuel** (`20260828-182551`) : **14 / 28 PASS**.
- Les 5 non-PASS restants attendus après #1 : **EX-03, SU-02** (défauts système réels),
  **SQ-11, SQ-16** (génération vs gold — voir §6), et 0 à 2 SEARCH si la
  groundedness lexicale reste sous le seuil malgré le bon dénominateur (§4).

---

## 3. Résultats SEARCH (16 cas)

Run de référence : **6 PASS**, 8 ANSWER_ONLY, 2 RETRIEVAL_ONLY.

| Cas | Verdict réf. | exactitude | groundedness (réf., extrait tronqué) | couverture_rel. | Lecture |
|---|---|---|---:|---:|---|
| SQ-02, 04, 06, 07, 10, 12 | PASS | ✔ | ≥ 0.50 | 1.0 | corrects et ancrés |
| SQ-01 | ANSWER_ONLY | ✔ | 0.25 | 1.0 | **faux PROVENANCE_FAILURE** — cf. §4 |
| SQ-03 | ANSWER_ONLY | ✔ | 0.37 | 1.0 | idem |
| SQ-05 | ANSWER_ONLY | ✔ | 0.40 | 0.96 | idem |
| SQ-08 | ANSWER_ONLY | ✔ | 0.35 | 1.0 | idem |
| SQ-09 | ANSWER_ONLY | ✔ | 0.18 | 0.86 | idem + dérivation arithmétique (« 35 ans ») — cf. §4 |
| SQ-13 | ANSWER_ONLY | ✔ | 0.47 | 1.0 | idem |
| SQ-14 | ANSWER_ONLY | ✔ | 0.29 | 1.0 | idem + méta-commentaire abondant — cf. §4 |
| SQ-15 | ANSWER_ONLY | ✔ | 0.38 | 1.0 | idem |
| SQ-11 | RETRIEVAL_ONLY | ✘ | 0.59 | 1.0 | génération incomplète vs gold à 2 assertions — §6 |
| SQ-16 | RETRIEVAL_ONLY | ✘ | 0.40 | 0.91 | gold générique de faible qualité — §6 |

**Interprétation.** `exactitude=True` sur 14/16 (contre 0/16 avant
`_reponse_couvre_gold`). Les 8 ANSWER_ONLY sont **exclusivement** dus à
`groundedness < 0.5`, mesurée dans ce run sur `SourceCitee.extrait` **tronqué
à 320 caractères** — précisément le défaut de l'audit. Le correctif #1 corrige
le dénominateur (§4).

---

## 4. Correctif #1 — groundedness : analyse et confirmation en attente

### Chemin de scoring constaté (avant correctif)

`noter_search` → `evaluate_end_to_end.calculer_groundedness(reponse)` →
part des **jetons porteurs** de la réponse présents dans le **contexte cité**,
ce contexte étant reconstruit à partir de `reponse.sources[i].extrait`.

`src/rag/generation.py` (l. 624-626) tronque cet extrait :
`extrait = " ".join(passage.texte.split()); if len(extrait) > 320: extrait = extrait[:317] + "..."`.

Le texte complet des passages **est disponible** dans le résultat via
`reponse.recherche.passages[i].texte` (`Passage`, `src/rag/retrieval.py`),
jamais tronqué. Il n'était pas utilisé.

### Correctif appliqué (harnais uniquement)

`calculer_groundedness` reconstruit désormais le contexte de référence à
partir du **texte complet de TOUS les passages récupérés**
(`reponse.recherche.passages`) — dénominateur standard d'une mesure de
fidélité RAG (« la réponse est-elle soutenue par ce que la recherche a
ramené ? »). Repli sur l'union des `SourceCitee.extrait` **seulement** si
`reponse.recherche is None` ou sans passage (`ReponseRAG` reconstruit hors
pipeline). **Le seuil de décision (0.5) est inchangé** ; seul le dénominateur
est corrigé. `src/rag/generation.py` **non modifié**.

### Effet attendu sur les 8 ANSWER_ONLY (analyse hors-ligne, run de confirmation requis)

| Cas | Cause dominante du score bas | Effet attendu du correctif |
|---|---|---|
| SQ-05, SQ-08, SQ-13, SQ-15 | élaboration correcte présente dans le passage cité **au-delà du 320ᵉ car.** et/ou dans un autre passage récupéré | **→ PASS** (dénominateur = passage complet + tous les passages) |
| SQ-01, SQ-03 | idem + méta-commentaire modéré (« aucun autre extrait ne mentionne… ») | **probable PASS** (les jetons WWI « 1914/1918/1919 » proviennent de passages S2–S12 réellement récupérés) |
| SQ-14 | méta-commentaire abondant sur les autres sources | **PASS ou proche du seuil** — à confirmer |
| SQ-09 | dérivation arithmétique : « 35 ans » **calculé** (1804−1769), absent verbatim de tout passage ; phrases de raisonnement non citées | **peut rester &lt; 0.5** — dans ce cas c'est une **limite légitime de la métrique lexicale** (le système synthétise), pas un faux négatif du harnais |

**Conclusion #1 :** le correctif lève le faux PROVENANCE_FAILURE structurel.
Un résidu possible (SQ-09, éventuellement SQ-14) relèverait d'une limite
connue de la groundedness lexicale face à une réponse qui *dérive* ou
*commente*, à documenter comme telle — **et non** à masquer par un seuil plus
laxiste. Chiffres définitifs : **run de confirmation** (§11).

---

## 5. Résultats EXTRACT / CLASSIFY / SUMMARIZE

### EXTRACT (5 cas) — 4 PASS, 1 WRONG

| Cas | Verdict | Détail |
|---|---|---|
| EX-01 | PASS | 2 champs trouvés, `precision_valeur = 1.0` chacun |
| EX-02 | PASS | 2 champs (dont routage EXTRACT implicite via classifieur LLM) |
| EX-04 | PASS | champ absent correctement non trouvé (anti-hallucination) |
| EX-05 | PASS | champ absent correctement non trouvé (anti-hallucination) |
| **EX-03** | **WRONG** | **défaut système réel** — voir §6 |

Le correctif #3 (lecture de `donnees["extractions"]`, `_associer_champ` à
3 passes, `_valeur_champ_extrait` pour les valeurs multiples) fait passer
EX-01/02/04/05 de WRONG (ancien scoring : champs lus à la racine → tous
`trouve=False`) à un verdict juste.

### CLASSIFY (3 cas) — 2 PASS, 1 ABSTAIN_CORRECT

| Cas | Verdict | Détail |
|---|---|---|
| CL-01, CL-02 | PASS | `mode=document_complet`, `document_demande_ok=True` **après résolution UUID→nom de fichier** (#2), catégorie ∈ taxonomie, provenance OK |
| CL-03 | ABSTAIN_CORRECT | document `cquae_doc_2262.txt` absent de l'index → refus explicite (`mode=document_vise_non_resolu`), **aucune classification inventée** ✔ |

### SUMMARIZE (3 cas) — 2 PASS, 1 WRONG

| Cas | Verdict | Détail |
|---|---|---|
| SU-01, SU-03 | PASS | `structural_ok=True` **après résolution UUID→nom de fichier** (#2), aucun search interne, sources dans le périmètre, checkpoints factuels touchés |
| **SU-02** | **WRONG** | **défaut système réel** (ROUTING_FAILURE) — voir §6 |

---

## 6. Erreurs réelles restantes — système vs benchmark vs dataset

| Cas | Verdict | Catégorie | Nature | À traiter |
|---|---|---|---|---|
| **SU-02** | WRONG | ROUTING_FAILURE | **Défaut système.** « Donne-moi les points essentiels du document … » routé vers `search` au lieu de `summarize` (« points essentiels » absent des marqueurs lexicaux, classifieur LLM ne rattrape pas). Attendu et documenté dans le dataset ; **mesuré, non corrigé** (hors périmètre J1). | Étape routing (P1) |
| **EX-03** | WRONG | EXTRACTION_FAILURE | **Défaut système.** « Corvée : ? Année de création du corps des Ponts et Chaussées : ? » ne nomme aucun document → `noeud_extract` mode `contexte_existant` → `search` sur une requête mal formée → `extract(document=None)` échoue (`documents_disponibles`). La résolution documentaire **contextuelle** de EXTRACT est plus faible que celle de CLASSIFY/SUMMARIZE. | Étape extract/routing (P1) |
| **SQ-11** | RETRIEVAL_ONLY | GENERATION_FAILURE | **Limite légitime / gold à 2 assertions.** Retrieval parfait (couv 1.0). La réponse identifie correctement Colbert comme *premier secrétaire d'État à la Marine sous Louis XIV* (assertion 1 du gold) mais **n'énonce pas** l'assertion 2 (« partant de presque rien, créer une puissance navale ») et pivote vers la politique mercantiliste. `exactitude=False` est un choix **strict mais défendable**. | Candidat : gestion des golds multi-assertions (à évaluer hors J1, sans laxisme) |
| **SQ-16** | RETRIEVAL_ONLY | GENERATION_FAILURE | **Qualité du gold (dataset).** `expected_answer` est un paragraphe **générique** sur la Renaissance qui ne répond pas à la question « pourquoi … au nord de l'Italie ». La réponse agent est détaillée, spécifique et correcte, mais partage peu de jetons porteurs avec ce gold faible. | Revue du gold `cquae:test:11761` |
| SQ-09, (SQ-14) | ANSWER_ONLY → *à revoir post-#1* | PROVENANCE_FAILURE | **Limite de la métrique lexicale** si le score reste &lt; 0.5 après #1 : réponse correcte qui *dérive* (« 35 ans » calculé) ou *commente les sources*. Pas un faux négatif du harnais au sens strict (les jetons concernés ne sont réellement dans aucun passage). | Documenté ; ne PAS assouplir le seuil |

---

## 7. Refus / anti-hallucination

| Contrôle | Cas | Résultat | Statut |
|---|---|---|---|
| Question hors-corpus → refus attendu | **AH-01** (score finale CdM 2022) | `refus=True` → **ABSTAIN_CORRECT** | ✔ strict — jamais PASS possible pour ce cas par construction |
| Document absent de l'index → refus attendu | **CL-03** (`cquae_doc_2262.txt`) | refus explicite, aucune classification de repli → **ABSTAIN_CORRECT** | ✔ |
| Champ absent → ne pas inventer | **EX-04, EX-05** | `trouve=False`, aucune valeur inventée → **PASS** | ✔ |
| Provenance EXTRACT hors périmètre | intégré à `noter_extract` (`sources_ok`) | — | inchangé |
| Groundedness : seuil de décision | 0.5 (SEARCH), 0.3 (UNSUPPORTED) | **inchangés** par les correctifs | ✔ aucune métrique rendue plus permissive |

Aucun correctif n'assouplit un refus, une abstention ou un contrôle de
provenance. Le correctif #1 corrige un **dénominateur**, pas un seuil.

---

## 8. Corrections apportées au harnais (liste exacte)

Périmètre : `evaluation/` **uniquement**. `src/rag/`, `src/tools/`,
`src/agent/` **non modifiés**.

### #1 — `evaluation/evaluate_end_to_end.py::calculer_groundedness`
- **Avant** : contexte de référence = union des `reponse.sources[i].extrait`
  (tronqués à 320 car. par `src/rag/generation.py`).
- **Après** : contexte de référence = **texte complet de tous les passages
  récupérés** `reponse.recherche.passages[i].texte`. Repli sur les extraits
  seulement si `reponse.recherche is None` / sans passage. Seuil 0.5
  inchangé. Jamais d'exception (repli sûr).

### #2 — `evaluation/cquae_multicapacite.py::_cle_document_resolu` (nouveau)
- Traduit l'identifiant enregistré dans la trace (`document_demande` /
  `documents_demandes`, un **UUID de version documentaire** produit par
  `noeud_classify` / `noeud_summarize`) en **nom de fichier canonique** via
  `src.rag.retrieval.catalogue().par_identifiant(...)` (lecture seule),
  **puis** `cle_document(...)` pour comparaison au gold.
- Repli sûr sur `cle_document(identifiant_brut)` si le catalogue est
  indisponible ou l'identifiant inconnu — **jamais** d'exception, **jamais**
  de PASS accordé par erreur (test : mauvais document → reste WRONG).
- Appliqué dans `noter_classify` (`document_demande`) et `noter_summarize`
  (`documents_demandes`). **Aucune table CQuAE, aucun UUID ni préfixe
  `cquae_doc_*` codé en dur.**

### #3 — `evaluation/cquae_multicapacite.py`, scoring EXTRACT
- `noter_extract` lit les champs sous **`donnees["extractions"]`** (et non à
  la racine de `donnees` — bug de l'ancien scoring qui rendait tous les
  champs `trouve=False`).
- `_associer_champ` réécrit en **3 passes déterministes** : (1) égalité
  exacte après `normaliser_texte` ; (2) inclusion d'un libellé dans l'autre ;
  (3) Jaccard ≥ **0.34** (écarte les appariements fortuits sur un jeton
  commun). Tolère les paraphrases raisonnables (« date de la bataille de
  Valmy » ↔ `date_bataille_valmy`), rejette les champs réellement différents.
- `_valeur_champ_extrait` (nouveau) : pour la mesure de précision, retombe
  sur la concaténation de `valeurs[].valeur` quand `valeur` vaut `None`
  (pluralité légitime), au lieu de mesurer sur une chaîne vide.

### Ajustement de tests existants
- `tests/evaluation/test_cquae_multicapacite.py` : 3 fixtures EXTRACT
  migrées vers la structure réelle `donnees={"extractions": {...}}`.

---

## 9. Tests

### Ajoutés — `tests/evaluation/test_corrections_scoring.py` (nouveau, 31 tests, 100 % synthétiques)

**Correctif #1 (groundedness)**
- `test_groundedness_reponse_reellement_ancree_est_acceptee` — réponse soutenue → score = 1.0 (≥ seuil).
- `test_groundedness_reponse_non_supportee_est_rejetee` — faits absents de tout passage → score &lt; 0.2 (**non permissif**).
- `test_groundedness_info_au_dela_du_320e_caractere_nest_plus_un_faux_negatif` — info après le 320ᵉ car. : `< 0.5` sur extrait tronqué, `≥ 0.5` sur passage complet.
- `test_groundedness_utilise_tous_les_passages_recuperes_pas_seulement_les_cites`.
- `test_groundedness_repli_sur_extrait_quand_recherche_absente`, `..._repli_extrait_pauvre_reste_bas`, `..._repli_sur_extrait_si_rapport_sans_passage`.

**Correctif #2 (UUID / nom de fichier)**
- `test_cle_document_resolu_uuid_vers_nom_fichier` — bon document mais UUID interne → **PASS documentaire**.
- `test_cle_document_resolu_uuid_inconnu_repli_sur_identifiant`, `..._catalogue_indisponible_repli`, `..._vide`.
- `test_cle_document_resolu_nom_fichier_et_source_equivalents` — chemin complet ↔ nom nu → identité reconnue.
- `test_noter_classify_ne_souffre_plus_du_bug_uuid_vs_filename`, `test_noter_summarize_ne_souffre_plus_du_bug_uuid_vs_filename` (SU-01/CL-01/CL-02).
- `test_noter_classify_mauvais_document_reste_wrong`, `test_noter_summarize_mauvais_document_reste_wrong` — **mauvais document → FAIL**.
- `test_noter_summarize_document_absent_conserve_l_abstention` — **document absent → pas de PASS silencieux**.

**Correctif #3 (EXTRACT)**
- `test_noter_extract_lit_donnees_extractions`, `test_noter_extract_ignore_les_champs_a_la_racine_ancien_bug`.
- `test_associer_champ_egalite_exacte_normalisee`, `..._inclusion`, `..._jaccard_au_dessus_du_seuil` — **variation raisonnable → match**.
- `test_associer_champ_rejet_sous_le_seuil` — **champ réellement différent → pas de match**.
- `test_noter_extract_champ_absent_evalue_correctement`, `test_noter_extract_champ_present_alors_qu_attendu_absent_est_wrong`.
- `test_valeur_champ_extrait_*` (valeur unique / multiples / vide), `test_noter_extract_champ_multi_valeurs_precision_mesuree_sur_toutes` — **multi-valeurs**.
- `test_noter_extract_valeur_fausse_nest_pas_creditee` — **mauvaise valeur → jamais PASS**.
- `test_noter_extract_multi_valeurs_dont_une_hors_sujet_reste_acceptable`.

### Exécutés

| Commande | Résultat |
|---|---|
| `pytest tests/evaluation/test_corrections_scoring.py` | **31 passed** |
| `pytest tests/evaluation/` | **66 passed** |
| `pytest tests/` (baseline : arbre de travail à l'ouverture de la session, même env) | 463 passed |
| `pytest tests/` (après correctif #1 + tests ajoutés) | **472 passed**, 0 échec |
| **Smoke CQuAE post-#1** | **NON ABOUTI** — GPU tiers à 100 %, `qwen3:8b` CPU > 19 min / génération |

> Sans la surcharge `QDRANT_PATH`, `pytest tests/` remonte **99 échecs
> `PermissionError: '/data'`** — cause : la barre oblique de tête dans le
> `.env` local (§1), **antérieure et étrangère aux correctifs**.

### Zéro régression

Confirmée : **463 → 472 passed** (delta = **+9** : la section groundedness de
`test_corrections_scoring.py` passe de 4 à 7 tests, +6 tests #2/#3 ajoutés),
**aucun test préexistant cassé**. `test_corrections_scoring.py` était présent
mais non suivi (`??`) à l'ouverture de la session ; il en compte 31 après J1.
Les correctifs #2/#3 sont en outre non-régressifs sur le run agent réel
(`20260828-182551` : 14 PASS, 0 `TECHNICAL_ERROR`).

---

## 10. Vérification des 4 propriétés majeures

### 1. Généricité
- **#1** : `reponse.recherche.passages` est un champ générique de `ReponseRAG` ;
  aucun nom de corpus.
- **#2** : `catalogue().par_identifiant()` est le résolveur générique déjà
  utilisé par le pipeline ; aucune table CQuAE, aucun UUID, **aucun préfixe
  `cquae_doc_*`**, aucune correspondance codée en dur. Fonctionne sur tout
  corpus dont les fiches exposent un `nom_fichier` / `source` / `titre`.
- **#3** : `_associer_champ` n'utilise que `normaliser_texte` / Jaccard
  (linguistique, pas métier) ; le seuil 0.34 est une constante d'appariement
  lexical, pas une connaissance du corpus.
- Vérifié : `grep -rniE 'cquae_doc|cquae:test|histoire-culture' evaluation/cquae_multicapacite.py evaluation/evaluate_end_to_end.py` → occurrences =
  (a) **constantes de préconditions** `PROFIL_DOMAINE_ATTENDU`,
  `DOCUMENT_MANQUANT_CONNU`, id exclu `cquae:test:8298` — **vérification
  d'environnement, jamais un ajustement de verdict**, présentes avant J1 ;
  (b) un **exemple en docstring** de `_cle_document_resolu` (« cquae_doc_219.txt »),
  pas du code. Aucune occurrence dans une branche de décision ajoutée par J1.

### 2. RAG comme socle gelable
- **Aucun** fichier de `src/rag/`, `src/tools/`, `src/agent/` modifié pour J1
  (`git diff --stat` : `evaluation/` uniquement + tests).
- Embeddings, chunking, retrieval, reranking, génération, prompts :
  **inchangés**. Le fast-path `src/rag/retrieval.py::_identifiant_exact_dans`
  et le garde-fou map-reduce SUMMARIZE mentionnés dans le brief sont
  **antérieurs à J1** (commits `be23906`, `136ed43`) et hors de ce diff.

### 3. Indépendance du fournisseur LLM
- **#1, #2, #3** : purement déterministes (jetons, ensembles, catalogue,
  Jaccard). **Aucun appel LLM.**
- Le juge LLM secondaire de SUMMARIZE (`_juger_contradiction_llm`) est
  **inchangé**, **optionnel** (`--llm-judge`, non activé dans le run de
  référence), **jamais décisionnaire seul**, et passe par l'abstraction
  `src.llm.factory.construire_llm` / `src.llm.common.invoquer_llm`. Aucune
  référence à Qwen/Ollama introduite.

### 4. Anti-hallucination / provenance
- Seuils de décision **inchangés** : groundedness 0.5 / 0.3, `SEUIL_RECOUVREMENT_*`,
  `SEUIL_COUVERTURE`.
- Tests explicites de non-régression de la sévérité : réponse non supportée →
  rejetée ; mauvais document → WRONG ; document absent → abstention conservée ;
  mauvaise valeur EXTRACT → jamais PASS ; champ inventé → WRONG.
- **#1** élargit le *dénominateur* (contexte réellement récupéré), pas la
  *tolérance* : un jeton fabriqué reste non ancré.
- AH-01 et CL-03 : abstention correcte préservée.

---

## 11. Limites connues restantes

| # | Limite | Impact | Étape |
|---|---|---|---|
| L1 | **Run agent post-#1 non exécuté** (GPU tiers). Verdicts SEARCH définitifs non confirmés empiriquement. | Bloque la clôture J1. | **immédiat, dès GPU libre** |
| L2 | `.env` local : `QDRANT_PATH=/data/...` (barre oblique de tête) → 99 tests en `PermissionError`, smoke bloqué sans surcharge. | Environnement. | retirer la barre oblique |
| L3 | **SU-02** — routage `summarize` non déclenché par « points essentiels ». | 1 WRONG (défaut système réel, visible). | routing (P1) |
| L4 | **EX-03** — résolution documentaire contextuelle faible pour EXTRACT sans document nommé. | 1 WRONG (défaut système réel, visible). | extract/routing (P1) |
| L5 | **SQ-16** — gold `cquae:test:11761` générique, ne répond pas à la question posée. | 1 RETRIEVAL_ONLY (qualité dataset). | revue gold |
| L6 | **SQ-11** — `exactitude` binaire face à un gold à 2 assertions ; réponse correcte sur l'assertion principale marquée non conforme. | 1 RETRIEVAL_ONLY (borderline). | à évaluer sans laxisme |
| L7 | groundedness lexicale : une réponse qui **dérive** (calcul) ou **commente ses sources** peut rester sous 0.5 même après #1 (SQ-09, éventuellement SQ-14). | 0–2 SEARCH. | documenté ; **ne pas** assouplir le seuil |
| L8 | `calculer_groundedness` mesure contre **tous** les passages récupérés, pas contre le sous-ensemble réellement inséré dans le contexte (budget caractères) — le harnais n'y a pas accès. Écart marginal en pratique. | négligeable. | acceptée |

### Commande du run de confirmation (#1)

```bash
QDRANT_PATH=data/vectordb/qdrant_cquae_eval \
ACTIVE_PROFILE=generic \
ACTIVE_DOMAIN_PROFILE=histoire-culture-humaines \
python -m evaluation._runner_config \
    --surcharges '{"qdrant.nom_collection": "cquae_eval"}' \
    -- evaluation.cquae_multicapacite --executer --nom cquae_reference --verbose
```

Critère de clôture J1 après ce run : chaque non-PASS restant ∈ {SU-02, EX-03
(défauts système), SQ-16 (gold), SQ-11 / SQ-09 / SQ-14 (limites documentées
§6/L6/L7)} — **aucun faux négatif harnais de type provenance/UUID/EXTRACT**.

---

## 12. Verdict

> ### J1 / P0 Étape 1 — **NON TERMINÉ**
>
> Correctifs harnais implémentés et testés unitairement (472 tests, 0
> régression) ; #2 et #3 validés end-to-end sur run agent réel. **Reste** : un
> run agent complet post-correctif #1 (groundedness), bloqué uniquement par la
> monopolisation externe du GPU. La condition de sortie de l'audit — « chaque
> WRONG = un vrai défaut, plus aucun faux négatif du harnais » — sera vérifiée
> à l'issue de ce run (§11).
