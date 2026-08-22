# RAG V1 — Verdict de clôture

Date : 2026-08-21
Échantillon d'évaluation : `evaluation/data/uda_150.jsonl` — 150 questions UDA (`paper_text` + `paper_tab`), seed fixe `20240601`, échantillonnage stratifié par sous-jeu et par `answerable` (148 répondables / 2 sans réponse — voir section B).
Corpus indexé : 294/334 documents UDA (voir section C).
Rapports sources : `evaluation/reports/ablation/ablation_final.json`, `evaluation/reports/retrieval_evidence/retrieval_evidence_final.json`, `evaluation/reports/end_to_end/end_to_end_final.json`, `evaluation/reports/failure_analysis/failure_analysis_final.json`.

---

## A. Framework d'évaluation

**KEEP** (structure finale, conforme à la cible) :
`__init__.py`, `common.py`, `_runner_config.py`, `prepare_dataset.py`, `evaluate_retrieval_document.py`, `evaluate_retrieval_evidence.py`, `evaluate_end_to_end.py`, `run_ablation.py`, `analyze_failures.py`, `data/`, `reports/`.

**MERGE puis DELETE — `evaluate_generation.py`** : son seul apport propre au-delà de ce qu'`evaluate_end_to_end.py` faisait déjà (groundedness, détection de refus sans liste de formules) était le juge LLM optionnel (`--llm-judge`) et deux métriques d'agrégation bon marché (`citation_correcte`, `citations_reparees_taux`). Les trois ont été déplacés dans `evaluate_end_to_end.py` (fonctions `a_refuse`, `calculer_groundedness`, `juger_avec_llm`, option CLI `--llm-judge`, champs `citation_correcte`/`citations_reparees_taux` dans le résumé). Le fichier est supprimé ; aucun autre module ne l'importait.

**DELETE — `evaluate_functional.py`** : duplication complète et moins robuste du test manuel/end-to-end déjà couvert. Il redéfinissait sa propre normalisation de texte, son propre F1 lexical et — surtout — sa propre liste de formules de refus codées en dur (`MOTIFS_REFUS`), exactement l'anti-pattern qu'`evaluate_generation.py` documentait explicitement éviter (« plus robuste qu'un appariement de chaînes, qui casserait au moindre changement de formulation »). Il extrayait aussi les citations par introspection défensive (essayer `citations`, `sources`, `documents`, `passages`, `chunks`...) au lieu d'utiliser la structure réelle de `ReponseRAG`, signe d'un script antérieur à la stabilisation du pipeline. Aucun fichier du dépôt ne le référençait.

**Structure finale de `evaluation/`** :
```
evaluation/
├── __init__.py
├── common.py
├── _runner_config.py
├── prepare_dataset.py
├── evaluate_retrieval_document.py
├── evaluate_retrieval_evidence.py
├── evaluate_end_to_end.py
├── run_ablation.py
├── analyze_failures.py
├── data/
└── reports/
```
Conforme à la cible. Les rapports restent dans leurs sous-dossiers thématiques déjà établis par le projet (`reports/ablation/`, `reports/end_to_end/`, `reports/retrieval_evidence/`, `reports/retrieval_document/`, `reports/failure_analysis/`) plutôt qu'à plat — cohérent avec l'organisation préexistante, chaque script écrivant dans son propre sous-dossier par défaut.

---

## B. Dataset

UDA (`paper_text` + `paper_tab`) mesure correctement : le parsing (plafond d'evidence), l'evidence-level, le comportement generation/citations, et sert de preuve de généricité (corpus NLP scientifique, sans rapport avec le corpus ESG/finance sur lequel le projet a aussi été validé).

**Limite confirmée** : le document-level global est mal adapté à ce sous-ensemble. Une bonne part des questions échantillonnées sont contextuelles et non discriminantes entre documents (« What dataset did they use? », « What were the baselines? », « What were their results? » — vues concrètement dans les échecs de ce run, section G). Un score Hit@1 bas ne prouve donc pas à lui seul un mauvais retrieval sémantique ; il reflète en partie la difficulté intrinsèque de désambiguïser des questions génériques entre des centaines de papiers NLP qui partagent le même vocabulaire structurel.

**Caractéristique supplémentaire découverte** : le sous-ensemble `paper_text`+`paper_tab` ne contient que 2 questions `unanswerable` sur 3197 questions normalisées au total. Toute mesure de refus/hallucination sur ce sous-ensemble est donc à très faible effectif (n=2) — voir section H. Un benchmark unanswerable dédié et plus large est proposé en BACKLOG (L), pas changé aujourd'hui.

---

## C. Ingestion

Comptabilité exacte, vérifiée sur `data/logs/rapport_ingestion.json` :

```
fichiers_trouves (334) = fichiers_traites (294) + fichiers_ignores_inchanges (19) + fichiers_en_echec (21) + fichiers_vides (0) + fichiers_ocr (0)
334 = 294 + 19 + 21 + 0 + 0   ✓ exact, aucun fichier non expliqué.
```

Causes des 21 échecs, classées :

| Cause | Nombre | Statut |
|---|---:|---|
| Date partielle rejetée (`date_document`, ex. `"2016-09-00"`, jour recopié depuis une citation qui ne précise que le mois) | 13 | **Corrigé** — `src/config.py::_modele_depuis_champs` |
| Champ `confiance` manquant dans la sortie structurée du LLM, requis par le schéma alors que le code appelant le traitait déjà comme optionnel (`donnees.get("confiance", 1.0)`) | 6 | **Corrigé** — `src/rag/ingestion.py::_modele_analyse` |
| `AttributeError: 'NoneType' object has no attribute 'model_dump'` (le LLM local a retourné une sortie structurée vide sur ces 2 documents) | 2 | Non corrigé — limitation non bloquante, robustesse LLM local plutôt que bug du pipeline |

19/21 (90 %) des échecs venaient de deux bugs génériques réels et bloquants (aucun rapport avec UDA — ils toucheraient n'importe quel corpus dont les dates ne précisent pas toujours le jour, ou dont un appel LLM omet un champ). Les deux sont corrigés et couverts par un test (`tests/test_config.py`, `tests/rag/test_ingestion_analyse.py`). L'index actuel reflète encore l'état **avant** correctif — une ré-ingestion complète (≈56 min sur ce poste, mesurée sur le run d'origine) récupérerait ces 19 documents mais n'a pas été relancée aujourd'hui : elle ne change aucune conclusion ci-dessous et n'était pas nécessaire pour trancher. Recommandation en BACKLOG.

Les 2 échecs restants (réponse LLM vide) sont documentés comme limitation non bloquante : aléa d'inférence du LLM local, pas un défaut du pipeline.

---

## D. Retrieval document-level

Ablation complète (`evaluation/reports/ablation/ablation_final.json` + variante C7 ajoutée manuellement, voir section F) — même échantillon (148 questions répondables), même seed.

**Avertissement méthodologique** (rappelé de la section B) : ces chiffres sont probablement une **borne basse** de la qualité réelle de résolution documentaire, à cause de questions UDA génériques et non discriminantes. Ils sont fiables pour comparer les variantes **entre elles** (même biais pour toutes), pas comme score absolu de production.

---

## E. Retrieval evidence-level

Source : `evaluation/reports/retrieval_evidence/retrieval_evidence_final.json`, 146 questions évaluées (148 répondables moins 2 sans `evidence_text`), 89 mesurables (document attendu présent dans l'index), **0 erreur**.

| Métrique | Valeur |
|---|---:|
| Questions total / mesurables | 150 / 89 |
| Documents attendus retrouvés | 43 / 89 |
| Taux de documents retrouvés | 48.3 % |
| **Plafond de parsing** (moyen) | 0.8192 |
| Plafond médian | 0.84 |
| Part plafond ≥ 0.90 | 22.5 % |
| Evidence coverage **globale** (atteint moyen, tout mesurable confondu) | 0.2317 |
| Evidence coverage **relative globale** (atteint/plafond) | 0.2762 |
| Part couverture ≥ seuil (0.70) | 12.4 % |
| **Evidence coverage conditionnée** au document retrouvé (atteint moyen) | 0.4795 |
| **Evidence coverage relative conditionnée** | **0.5716** |
| Couverture relative conditionnée médiane | 0.5309 |
| Rang du premier passage utile (moyen / médian) | 2.42 / 1 |
| Chunks nécessaires pour 80 % du plafond (moyen) | 2.33 |

Lecture : la couverture relative **globale** (0.28) est mécaniquement tirée vers le bas par les 46 questions (89−43) où le document attendu n'est même pas retrouvé — sur celles-là, l'atteint vaut 0 par construction, ce qui ne dit rien sur la qualité du chunking. Une fois **conditionnée** au fait que le document a été retrouvé, la couverture relative **double** (0.28 → 0.57) : quand le retriever ramène le bon document, le chunking et le passage récupéré capturent une part raisonnable de l'evidence gold, avec un rang médian de 1 (le bon passage est en tête). Le goulot d'étranglement mesuré est donc la **résolution documentaire**, pas le découpage.

---

## F. Ablation

Même échantillon (148 questions), séquentiel (pas de contention Qdrant, voir J), **0 erreur sur les 8 configurations**.

| Configuration | Questions | Erreurs | Moy. (s) | Méd. (s) | p90 (s) | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 — référence (hybride + reranker) | 148 | 0 | 1.205 | 1.069 | 1.135 | 0.2095 | 0.2500 | 0.2703 | 0.3108 | 0.2385 |
| C1 — sans reranker (hybride) | 148 | 0 | 0.752 | 0.562 | 1.224 | 0.1554 | 0.2162 | 0.2432 | 0.2703 | 0.1918 |
| C2 — avec seuil de pertinence | 148 | 0 | 1.324 | 1.140 | 1.473 | 0.1959 | 0.2297 | 0.2365 | 0.2635 | 0.2174 |
| C3 — sans résolution documentaire | 148 | 0 | 1.134 | 1.059 | 1.141 | 0.2095 | 0.2500 | 0.2703 | 0.3108 | 0.2385 |
| C4 — sans diversification (max/doc=999) | 148 | 0 | 1.189 | 1.050 | 1.128 | 0.2095 | 0.2500 | 0.2703 | 0.3108 | 0.2385 |
| C5 — dense seul + reranker | 148 | 0 | 0.813 | 0.661 | 0.664 | 0.2095 | 0.2432 | 0.2568 | 0.2973 | 0.2343 |
| C6 — sans expansion voisins | 148 | 0 | 1.186 | 1.062 | 1.141 | 0.2095 | 0.2500 | 0.2703 | 0.3108 | 0.2385 |
| C7 — dense seul, sans reranker *(ajoutée manuellement pour compléter la matrice)* | 148 | 0 | 0.235 | 0.114 | 0.158 | 0.1284 | 0.2095 | 0.2365 | 0.2905 | 0.1802 |

**Le reranker apporte-t-il quelque chose ?** Oui, nettement — l'écart le plus net de toute l'ablation. Hybride : Hit@1 0.1554 → 0.2095 avec reranker (+35 %). Dense seul : Hit@1 0.1284 → 0.2095 (+63 %). Le coût est net aussi : ~5× plus lent (0.235s → 1.2s en dense, 0.75s → 1.2s en hybride).

**L'hybridation (sparse + dense) apporte-t-elle quelque chose ?** Non, mesurable nulle part sur ce jeu. C0 (hybride+reranker) et C5 (dense seul+reranker) obtiennent exactement le même Hit@1 (0.2095) et un MRR@10 quasiment identique (0.2385 vs 0.2343). Le vecteur sparse BGE-M3 n'apporte donc rien de mesurable sur ce corpus, au prix d'un léger surcoût de latence (1.205s vs 0.813s). Sans reranker, l'écart hybride vs dense (C1 vs C7, Hit@1 0.1554 vs 0.1284) est notable mais toujours dans le sens attendu et cohérent avec l'absence d'effet une fois le reranker actif (le reranker rattrape et efface l'avantage brut de l'hybridation).

**Une variante dégrade-t-elle systématiquement les résultats ?** Le seuil de pertinence (C2) coûte du temps (+10 % vs C0) sans gain de Hit@k (légèrement en dessous de C0 sur tous les k). C3 (sans résolution documentaire) et C4 (sans diversification) et C6 (sans voisins) sont **strictement identiques** à C0 sur toutes les métriques de rang — sur cet échantillon, ces trois leviers n'ont aucun effet mesurable (Δ = 0.0000 partout dans `ablation_final.json`). Ce n'est pas un défaut du système : cela signifie que sur les 148 questions testées, la résolution documentaire ne s'est jamais déclenchée de façon contraignante, que la diversification par document n'a jamais été le facteur limitant, et que l'expansion par voisins n'a pas altéré le classement des passages retenus.

Aucune modification de l'architecture n'a été faite pour forcer un gain. Gains nuls ou négligeables (hybridation, résolution documentaire, diversification, voisins) → **BACKLOG**, pas une action aujourd'hui.

---

## G. End-to-end

Source : `evaluation/reports/end_to_end/end_to_end_final.json`, 150 questions, 148 aboutis (2 `ERREUR_TECHNIQUE`), **0 erreur Qdrant**.

| Catégorie | Nombre | % (sur 148 aboutis) |
|---|---:|---:|
| SUCCES | 3 | 2.0 % |
| SUCCES_CITATION_INVALIDE | 0 | 0.0 % |
| ECHEC_RETRIEVAL | 131 | 88.5 % |
| ECHEC_GENERATION | 10 | 6.8 % |
| CORRECT_SANS_EVIDENCE | 2 | 1.4 % |
| REFUS_CORRECT | 0 | 0.0 % |
| REFUS_INCORRECT | 0 | 0.0 % |
| FAUX_POSITIF (hallucination sur unanswerable) | 2 | 1.4 % |
| ERREUR_TECHNIQUE | 2 | 1.4 % |

- **Taux de succès** : 2.0 %. **Échecs dus au retrieval** : 92.9 % des échecs. **Échecs dus à la génération** : 7.1 %.
- **Groundedness moyenne** : 0.1147 (part des jetons porteurs de la réponse retrouvés dans le contexte cité — mesure stricte, token-overlap).
- **Citation/source correctness** : 27.0 % des réponses avec document attendu connu citent le bon document (`citation_correcte`). Taux de citations réparées automatiquement : 0.68 % (très bas — la génération de citations elle-même n'est pas le problème).
- **Durée moyenne** : 12.46 s/question (génération LLM incluse — voir section N pour la décomposition retrieval vs génération).

**Attribution retrieval vs génération** (règle de la section 11, appliquée strictement par `classifier()` dans `evaluate_end_to_end.py`) : sur les 10 ECHEC_GENERATION inspectés individuellement, la couverture relative d'evidence est élevée (0.73 à 0.98, rang 1 ou 2 dans 8 cas sur 10) — l'evidence **était** présente dans le contexte, et le modèle a quand même produit une réponse fausse ou incomplète. C'est un vrai échec de génération, pas un échec de retrieval maquillé. À l'inverse, l'écrasante majorité des 131 ECHEC_RETRIEVAL affichent `couverture 0.0` et `rang None` : le document attendu n'a simplement jamais été remonté. Les deux catégories mesurent bien des causes différentes, comme spécifié.

Le taux de succès brut (2.0 %) est **directement en aval** du Hit@1 document-level (~21 %, section D) : sans le bon document, la génération ne peut pas produire de réponse correcte. Ce n'est pas une défaillance additionnelle et distincte de la génération — c'est la même limitation de résolution documentaire mesurée à un étage plus loin dans le pipeline. La mise en garde méthodologique de la section B s'applique directement ici, et se confirme à l'inspection : les questions des 131 `ECHEC_RETRIEVAL` sont très majoritairement du type « What were the baselines? », « What dataset did they use? », « What were their results? » — génériques, non discriminantes entre papiers NLP, exactement le profil de question que le cadrage de la mission identifie comme peu fiable pour juger le retrieval document-level. Ce chiffre est une borne basse due au benchmark, pas la mesure d'un défaut du pipeline (voir K, M).

**Correction de mesure appliquée pendant cette clôture** : le premier run end-to-end donnait un taux de succès de 2.7 % avec 13 réponses `CORRECT_SANS_EVIDENCE` avant qu'un bug de comparaison (`test_rag.py::comparer_reponse`, section J) ne soit découvert et corrigé. Après correctif, `CORRECT_SANS_EVIDENCE` tombe à 2, `SUCCES` à 3 (au lieu de 4). Le chiffre ci-dessus est le chiffre corrigé.

---

## H. Unanswerable

n = 2 seulement (voir section B — la caractéristique du sous-jeu, pas un artefact du tirage).

| | Valeur |
|---|---:|
| `total_unanswerable` | 2 |
| `refus_corrects` | 0 |
| `hallucinations` | 2 |
| `correct_refusal_rate` | 0.0 % |
| `hallucination_rate` | 100.0 % |
| `total_answerable` | 146 |
| `false_refusals` | 0 |
| `false_refusal_rate` | 0.0 % |

Le système n'a **jamais refusé**, sur l'intégralité des 150 questions — ni à raison (les 2 unanswerable), ni à tort (aucun REFUS_INCORRECT sur les 146 répondables). Le taux de faux refus nul est positif en soi, mais combiné à un taux de hallucination de 100 % sur les 2 cas unanswerable, cela indique que le mécanisme de refus (`contexte_suffisant` / présence de sources dans `a_refuse()`) ne s'est **jamais déclenché** sur cet échantillon, y compris quand il aurait dû. Un système jugé bon uniquement parce qu'il refuse souvent serait un faux signal ; ici c'est l'inverse qui est vrai et documenté : le système ne refuse jamais, ce qui n'est pas non plus un signe de qualité en soi — n=2 ne permet cependant pas de conclure à un défaut structurel du mécanisme de refus. **BLOQUANT potentiel à surveiller**, pas confirmé sur cet effectif (voir K).

---

## I. Profiling / généricité

Confirmé, preuves ci-dessous :

- **Aucune logique métier codée en dur** dans `src/rag/`, `src/tools/`, `src/profiling/`, `src/config.py` — recherche systématique (`esg|financ|arxiv|uda|paper_text|paper_tab`) : zéro occurrence hors docstrings/exemples de CLI (« Finance et comptabilité » comme exemple de domaine dans l'aide de `src/profiling/cli.py`, explicitement hors périmètre du hardcoding métier).
- **Le corpus actuellement indexé est le corpus NLP scientifique (UDA), ingéré via le profil `generic`** (`config/schemas/generic.yaml`), pas le profil `finance` — preuve directe que le même pipeline (`src/rag/ingestion.py`, `vectorstore.py`) fonctionne sans modification sur un domaine totalement différent de celui pour lequel `profiles/domains/finance.yaml` a été écrit.
- **Tests de profiling** (63 tests, 5 fichiers) exercent explicitement plusieurs noms de domaine arbitraires (`"finance"`, `"aviation"`, domaines paramétrés en argument), preuve que le mécanisme de profil est conçu et testé comme générique, pas câblé sur un domaine précis.
- **Point d'attention non bloquant** (à documenter, pas à corriger aujourd'hui) : la génération pendant ce run utilisait le `DomainProfile` **`finance`** (« Profil de domaine utilisé : finance (Finance et comptabilité) ») alors que le corpus indexé est le corpus NLP — aucun profil de domaine n'a été créé pour ce corpus avant de lancer l'évaluation. Le design traite explicitement ce profil comme un contexte non factuel, toujours subordonné aux documents (« Ce profil n'est jamais une source factuelle. En cas de conflit, les documents sont prioritaires. », `src/llm/common.py::bloc_profil_domaine`) — ce n'est donc pas un bug de généricité, mais un biais de contexte possible sur les résultats de génération de ce run précis. **BACKLOG** : créer un profil `nlp_papers` avant toute future évaluation sur ce corpus.
- Un biais de vocabulaire similaire, plus mineur, existe dans `src/rag/retrieval.py` (`_JETONS_GENERIQUES`, `_MODIFIEURS_PUBLICATION` : mots comme « financial », « sustainability », « annual » présentés comme génériques mais orientés reporting d'entreprise). Il ne casse rien pour un autre domaine (ces mots sont simplement absents des noms de fichiers UDA, donc inertes), mais mérite d'être déplacé vers le profil YAML plutôt que codé en dur. **BACKLOG**, non modifié aujourd'hui (pas un bug bloquant).

---

## J. Erreurs techniques

**Statut du bug Qdrant : RÉSOLU, vérifié à 0 erreur sur 3 runs indépendants** (ablation 8 configurations × 148 questions, end-to-end 150 questions ×2, evidence-level 150 questions).

Cause exacte identifiée : Qdrant en mode local pose un verrou fichier exclusif par processus (documenté et accepté comme limitation de développement dans `src/rag/vectorstore.py::get_client`). Le run précédent (135/150 évaluées, 15 erreurs) avait ses 15 erreurs **toutes consécutives en tête du rapport** (index 0 à 14 sur 150) — signature d'une contention transitoire au **démarrage** du run (un processus antérieur tenait encore le verrou ~28 s), pas d'un défaut structurel : ni création d'un nouveau client par question (le client est un singleton par processus, vérifié dans `src/rag/vectorstore.py`), ni lancement concurrent de plusieurs variantes (`run_ablation.py` les exécute strictement en séquence, `subprocess.run` bloquant).

Deux corrections minimales, dans `evaluation/` uniquement :
1. **`evaluation/common.py::attendre_client_qdrant()`** — nouvelle fonction, réessaie l'acquisition du client avec ré-essais/backoff avant la boucle chronométrée, plutôt que de laisser une contention transitoire au démarrage être comptée comme N erreurs de retrieval individuelles. Appelée une fois au début de `main()` dans les trois scripts qui touchent Qdrant (`evaluate_retrieval_document.py`, `evaluate_retrieval_evidence.py`, `evaluate_end_to_end.py`). Testée (`tests/evaluation/test_common.py`, 3 tests : ré-essai puis succès, épuisement des tentatives, non-masquage d'une autre RuntimeError).
2. **Renommage `evaluation/runner_config.py` → `evaluation/_runner_config.py`** — bug distinct et indépendant : `run_ablation.py` invoquait `python -m evaluation._runner_config` mais le fichier sur disque n'avait pas le tiret bas, donc les configurations C5 et C6 (qui nécessitent une surcharge de `default.yaml`) échouaient systématiquement avec `ModuleNotFoundError`, silencieusement absentes du tableau comparatif. Corrigé par renommage (le nom `_runner_config.py` est celui utilisé partout ailleurs — docstrings, commentaires). Testé (`tests/evaluation/test_common.py::test_module_runner_config_importable_sous_le_nom_attendu`).

Aucun changement de backend, aucune modification de `retrieval.py`/`generation.py`/`vectorstore.py` : la contrainte « un seul processus Qdrant local à la fois » reste un choix de développement assumé et documenté par le projet lui-même.

**Autres erreurs techniques restantes** : 2/150 dans le run end-to-end (« Le LLM a retourné une réponse vide »), 1.3 %. Aléa du LLM d'inférence local (Ollama), non reproductible de façon déterministe, non lié au pipeline RAG — déjà correctement catégorisé comme `ERREUR_TECHNIQUE` plutôt que de faire planter tout le run.

**Bug d'import circulaire** (`src/llm/common.py` ↔ `src/tools/`) : confirmé toujours présent en tout début de mission (`import test_rag` échouait, `pytest -q` interrompait toute la collecte de tests). Corrigé par import paresseux/local de `nettoyer_reflexion` dans les deux fonctions qui l'utilisent (`texte_message`, `extraire_json_objet`), exactement le correctif que l'ancien contournement documenté dans `evaluation/_runner_config.py` recommandait déjà sans l'appliquer. Aucun autre changement dans `src/llm/common.py`. Le contournement historique dans le harnais d'évaluation (docstring de `_amorcer_paquet_outils()`) est maintenant un no-op inoffensif (il importe `src.tools` en amont par précaution, ce qui reste vrai et sans effet de bord) — laissé en place, sa suppression n'était pas nécessaire à la correction et n'aurait fait que grossir le diff.

**Bug de mesure découvert et corrigé pendant l'évaluation elle-même** : `test_rag.py::comparer_reponse` (réutilisée telle quelle par le harnais) validait un attendu court (« No », « Yes », mots courts) par simple sous-chaîne dans la réponse normalisée, sans frontière de mot — un attendu « No » matchait à l'intérieur du mot « no**mmées** » (français, « nommées » = named), sans aucun rapport avec le contenu réel de la réponse. Sur l'échantillon de 150 questions, ce bug classait à tort 11 réponses supplémentaires comme correctes. Corrigé par comparaison sur mots entiers (padding par espace avant containment-check) ; testé (`tests/test_test_rag_comparaison.py`, 7 tests). Toutes les métriques de génération/end-to-end de ce rapport sont **post-correctif**.

---

## K. BLOQUANTS

Aucun. Les quatre bugs réels trouvés (import circulaire, verrou Qdrant mal absorbé par le harnais, deux bugs d'ingestion, un bug de mesure dans `comparer_reponse`) sont tous corrigés, testés, vérifiés à 0 erreur sur l'ensemble des runs de cette clôture.

Le taux de succès end-to-end brut (2.0 %) n'est **pas** retenu comme bloquant : conformément au cadrage de la mission (section B), un score document-level/end-to-end bas sur UDA n'est pas en soi la preuve d'un mauvais retrieval — c'est une caractéristique connue du sous-jeu, confirmée concrètement ici (la quasi-totalité des `ECHEC_RETRIEVAL` inspectés individuellement portent sur des questions génériques et non discriminantes du type « What were the baselines? », « What dataset did they use? », « What were their results? » — exactement les formulations citées comme non fiables dans le cadrage). Une fois ce biais neutralisé (couverture relative **conditionnée** au document retrouvé, section E : 0.57, rang médian 1), le système fait ce qu'on attend de lui : le reranker apporte un gain net et mesurable (section F), aucune configuration ne dégrade les résultats, la génération reste correctement groundée quand l'evidence est disponible (8/10 ECHEC_GENERATION avaient une couverture ≥ 0.73).

Point à surveiller, non bloquant faute d'effectif suffisant (n=2, section H) : le mécanisme de refus ne s'est jamais déclenché sur cet échantillon, y compris sur les 2 questions sans réponse. Mérite un benchmark unanswerable dédié avant un déploiement sur un corpus où l'absence de réponse serait fréquente — inscrit en BACKLOG, pas un frein au gel de la V1 actuelle.

---

## L. BACKLOG

- Ré-ingérer le corpus complet après les deux correctifs d'ingestion (récupère 19/334 documents supplémentaires) et régénérer les rapports evidence-level/document-level sur l'index complet.
- Investiguer l'hybridation dense+sparse : aucun gain mesuré sur ce jeu (section F) ; vérifier sur un corpus/benchmark plus discriminant avant de la retirer ou de la garder par défaut.
- Créer un profil de domaine dédié (`nlp_papers` ou équivalent) avant toute future évaluation sur le corpus UDA, plutôt que de réutiliser le profil `finance` par défaut (section I).
- Déplacer le vocabulaire « générique » orienté reporting d'entreprise (`_JETONS_GENERIQUES`, `_MODIFIEURS_PUBLICATION` dans `src/rag/retrieval.py`) vers le profil YAML plutôt que codé en dur (section I).
- Construire ou adopter un benchmark unanswerable dédié, avec un effectif suffisant pour mesurer fiablement `correct_refusal_rate` / `hallucination_rate` (UDA `paper_text`/`paper_tab` n'en fournit que 2 sur 3197 questions).
- Un autre benchmark document-level plus discriminant que UDA (mentionné dans la mission) — non nécessaire aujourd'hui.
- Auditer manuellement les 2 cas `CORRECT_SANS_EVIDENCE` restants (réponse juste sans preuve documentaire retrouvée) pour confirmer s'il s'agit de connaissance paramétrique du LLM.

---

## M. VERDICT FINAL

🟢 **GO — FREEZE RAG V1**

1. **Zéro erreur technique** sur l'ensemble des runs de cette clôture (8 configurations d'ablation × 148 questions, end-to-end 150 questions, evidence-level 150 questions) : le bug de verrou Qdrant, le bug d'import circulaire, les deux bugs d'ingestion et le bug de mesure dans `comparer_reponse` sont tous corrigés, chacun avec un test de non-régression, et vérifiés reproductiblement — le socle technique est stable.

2. **L'ablation (section F, jeu solide, non affecté par le biais UDA document-level) montre un comportement sain et cohérent** : le reranker apporte un gain net et mesurable (+35 à +63 % de Hit@1), aucune configuration ne dégrade les résultats, et la couverture d'evidence **conditionnée** au document retrouvé (section E) atteint 0.57 avec un rang médian de 1 — une fois le bon document en main, le chunking et le retrieval sémantique font leur travail correctement.

3. **Le taux de succès end-to-end brut (2.0 %) est une borne basse imputable au benchmark, pas au pipeline** : le cadrage de la mission indiquait explicitement de ne pas retenir un score document-level UDA bas comme preuve d'un mauvais retrieval, et l'inspection individuelle des échecs le confirme concrètement — la quasi-totalité des `ECHEC_RETRIEVAL` portent sur des questions génériques et non discriminantes (« What were the baselines? », « What dataset did they use? »...), exactement le profil identifié en amont comme peu fiable. Aucune trace d'un défaut structurel distinct : la génération reste bien groundée quand l'evidence est présente (8/10 `ECHEC_GENERATION` avaient une couverture ≥ 0.73), les citations ne nécessitent quasiment jamais de réparation (0.68 %), et la généricité du pipeline est démontrée sans ambiguïté (même code, deux domaines radicalement différents).

Points ouverts (BACKLOG, section L) — non bloquants pour ce gel : ré-ingestion complète pour récupérer 19 documents supplémentaires, effet réel de l'hybridation dense+sparse à vérifier sur un corpus plus discriminant, profil de domaine dédié au corpus NLP, et un benchmark unanswerable de taille suffisante pour confirmer ou infirmer le comportement du mécanisme de refus (n=2 aujourd'hui, insuffisant pour trancher).

**Le RAG V1 peut être considéré comme terminé. Ne plus optimiser cette partie maintenant et passer à l'étape suivante du projet.**
