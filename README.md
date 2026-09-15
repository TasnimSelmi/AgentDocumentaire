# Agent documentaire — RAG générique + couche agentique + API + frontend

Agent de questions-réponses et de traitement documentaire construit sur un
**RAG local** (embeddings BGE-M3, Qdrant hybride, reranking BGE-v2-m3, LLM via
Ollama), une **couche agentique déterministe** (LangGraph) qui route une
requête vers l'une de **six capacités** — **SEARCH**, **SUMMARIZE**,
**CLASSIFY**, **EXTRACT** (mono-document), **COMPARE**, **SYNTHESIZE**
(multi-document) —, une **API HTTP** (FastAPI) exposant le tout en
**multi-corpus**, et un **frontend statique** servi par cette même API.

Le système est **agnostique au domaine** : taxonomie, métadonnées, schéma
d'extraction et vocabulaire métier proviennent tous de fichiers de
configuration (`config/schemas/*.yaml`, `profiles/domains/*.yaml`). Aucun nom
de corpus, de société ou de champ métier n'est codé en dur.

> ## Statut : socle RAG gelé, couche produit (API + multi-corpus + frontend) livrée
>
> Le socle RAG (ingestion, retrieval, génération, outils, cœur agentique) est
> **stable et gelé** — voir **[docs/DO_NOT_TOUCH.md](docs/DO_NOT_TOUCH.md)**
> pour la liste exacte des modules à ne pas modifier sans un cycle
> d'évaluation complet. Au-dessus : API FastAPI multi-corpus, sources
> documentaires (upload/import URL), profilage de domaine, observabilité, et
> un frontend statique — voir §14 pour les limitations connues et §15 pour ce
> qui reste hors périmètre (P3).

---

## 1. Architecture en un coup d'œil

```
                 ┌─────────────────────── INGESTION (hors ligne) ───────────────────────┐
  fichiers ──►   loaders → OCR → chunking structure-aware → inférence LLM (catégorie +
                 métadonnées) → normalisation → résolution d'entités → embeddings BGE-M3
                 → indexation Qdrant (vecteurs nommés dense + sparse, 1 collection/corpus)
                 └──────────────────────────────────────────────────────────────────────┘

                 ┌──────────────────────── REQUÊTE (en ligne) ──────────────────────────┐
  question ─►   détecter_intention (routage 100 % déterministe, vocabulaire fermé +
                 2 classifieurs LLM bornés pour les zones grises)
                     │
                     ├─ SEARCH ──► retrieval hybride + rerank ──► évaluer preuves
                     │              (pertinence déterministe, puis suffisance LLM bornée)
                     │              ──► générer réponse sourcée │ reformuler (boucle bornée)
                     │                                          │ refus déterministe
                     ├─ SUMMARIZE ─► charger le document entier ──► résumé map-reduce borné
                     ├─ CLASSIFY ──► charger le document entier ──► vote majoritaire par lots
                     ├─ EXTRACT ───► charger le document entier ──► extraction sourcée par lots
                     ├─ COMPARE ───► charger 2 à 4 documents nommés ──► comparaison sourcée
                     └─ SYNTHESIZE ► charger 2 à 4 documents nommés ──► synthèse sourcée
                 └──────────────────────────────────────────────────────────────────────┘

                 ┌────────────────────────── PRODUIT (P2) ──────────────────────────────┐
  navigateur ─►  frontend statique (src/ui/*.html) ──► API FastAPI (src/api/**)
                 ──► AgentService / IngestionService ──► cœur ci-dessus
                 (multi-corpus, sources upload/import URL, profils de domaine,
                 observabilité — 1 seul process, 1 seul port : voir §5)
                 └──────────────────────────────────────────────────────────────────────┘
```

Détails : **[docs/architecture.md](docs/architecture.md)**.

Principes structurants (invariants du projet) :

1. **Généricité** — aucune connaissance de corpus/métier dans le code.
2. **RAG gelable** — le benchmark mesure le système ; le système ne se déforme
   pas pour le benchmark.
3. **Indépendance du fournisseur LLM** — un seul point d'accès
   (`src/llm/factory.py`), aucune logique dépendante d'un modèle particulier.
4. **Anti-hallucination / provenance** — refus déterministe sans contexte,
   citations `[S1]` validées jusqu'au document, cloisonnement documentaire.
5. **Isolation multi-corpus** — 1 corpus logique = 1 collection Qdrant + 1
   registre de fichiers + 1 profil, jamais partagés entre corpus.

---

## 2. Fonctionnalités

| Domaine | Fonctionnalités |
|---|---|
| **Capacités agent** | SEARCH (question sourcée), SUMMARIZE, CLASSIFY, EXTRACT (un document), COMPARE, SYNTHESIZE (2 à 4 documents nommés) — réponses `success` / `success` avec avertissements (« PARTIAL ») / `refusal` / `error`, toujours sourcées ou refusées explicitement |
| **Multi-corpus** | Création / liste / détail / suppression de corpus, isolation stricte (collection Qdrant, registre de fichiers, profil dédiés) |
| **Ingestion** | Dossier local partagé (source `local`) ou stockage géré par corpus alimenté par upload de dossier / import URL (source `managed`) ; documents longs, OCR (PDF scannés), tableaux, formats PDF/DOCX/TXT/MD/XLSX/CSV/PPTX/HTML |
| **Profilage de domaine** | Proposition (LLM) puis validation d'un profil de vocabulaire métier par corpus |
| **Sources documentaires** | `GET /sources` (sources réellement enregistrées côté serveur) |
| **Observabilité** | Corrélation `X-Request-Id` / `X-Execution-Id`, traces JSON structurées (durée, statut, capacité, documents, erreurs — jamais de contenu ni de chain-of-thought) |
| **Frontend** | Gestion des corpus (créer, uploader, importer une URL, indexer, profiler, supprimer), Agent Documentaire (interroger, sources/citations, historique de session local, PARTIAL/REFUSAL/ERROR) |
| **Sécurité applicative** | Validation stricte des chemins d'upload (anti-traversée), SSRF minimal sur l'import URL (DNS, plages privées, redirections bornées), pas de trace technique exposée au client |

---

## 3. Prérequis

| Composant | Version / détail |
|---|---|
| Python | **3.11** (venv du dépôt) — le code cible 3.10+ |
| [Ollama](https://ollama.com) | serveur local, modèle `qwen3:8b` (`ollama pull qwen3:8b`) |
| GPU | optionnel — CUDA accélère embeddings/reranking et Ollama ; sinon CPU |
| Tesseract OCR + Poppler | requis seulement pour les PDF scannés (`OCR_ENABLED=true`, activé par défaut — voir commandes d'installation ci-dessous) |
| RAM | ~6 Go pour BGE-M3 + reranker chargés en mémoire |
| Accès réseau sortant | requis pour le **frontend** (React/Babel via `unpkg.com`, polices via `fonts.googleapis.com`) — l'API et le RAG, eux, sont 100 % locaux et fonctionnent sans Internet |

Modèles Hugging Face (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`) : téléchargés
au premier usage, puis mis en cache.

### Installer Tesseract + Poppler (PDF scannés)

`OCR_ENABLED=true` par défaut : sans ces deux binaires système, l'import d'un
PDF scanné échoue avec une erreur système (pas une erreur applicative claire).
`pytesseract` et `pdf2image` (dans `requirements.txt`) sont de simples
wrappers Python : ils n'installent **pas** ces binaires eux-mêmes.

`OCR_LANGUAGES=fra+ara+eng` par défaut : les paquets de langue française et
arabe ne sont **pas** inclus dans l'installation de base de Tesseract et
doivent être installés explicitement.

| OS | Commande |
|---|---|
| Ubuntu / Debian | `sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara poppler-utils` |
| macOS (Homebrew) | `brew install tesseract tesseract-lang poppler` |
| Windows | Installer [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (cocher les paquets de langue French/Arabic pendant l'installation) et [Poppler pour Windows](https://github.com/oschwartz10612/poppler-windows/releases/) ; ajouter les deux au `PATH`, ou renseigner `TESSERACT_CMD` dans `.env` si le binaire Tesseract n'est pas sur le `PATH` |

Vérifier l'installation :

```bash
tesseract --list-langs   # doit lister fra, ara, eng
pdftoppm -h               # confirme que Poppler est bien sur le PATH
```

Pour désactiver l'OCR (pas de PDF scanné à traiter) : `OCR_ENABLED=false`
dans `.env`.

---

## 4. Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                 # puis adapter (voir §5)
ollama pull qwen3:8b
```

> **Installation hors ligne (Windows).** `wheels/` contient un cache optionnel
> de roues `torch` / `sympy` / `mpmath` pour un poste Windows / CPython 3.12
> sans accès PyPI (`pip install --no-index --find-links wheels/ torch`). Ce
> dossier est ignoré par git et **peut être supprimé** si l'installation se
> fait en ligne.

---

## 5. Configuration

Trois niveaux, volontairement séparés (`src/config.py`) :

| Fichier | Rôle | Varie selon |
|---|---|---|
| `.env` | secrets, chemins, choix des modèles | la machine |
| `config/default.yaml` | chunking, OCR, seuils, recherche, Qdrant, agent | le réglage (**gelé**, voir DO_NOT_TOUCH §5) |
| `config/schemas/<profil>.yaml` | taxonomie, métadonnées, schéma d'extraction | le domaine |
| `profiles/domains/<nom>.yaml` | vocabulaire métier injecté aux prompts | le domaine |
| `config/corpus.yaml` | registre des corpus déclarés (multi-corpus) | l'usage — géré par l'API, pas à éditer à la main |

Clés `.env` essentielles (liste complète et commentée : `.env.example`) :

| Clé | Défaut | Note |
|---|---|---|
| `LLM_MODEL` | `qwen3:8b` | Ollama uniquement |
| `LLM_BASE_URL` | `http://localhost:11434` | serveur Ollama |
| `EMBEDDING_DEVICE` | `cpu` | `cuda` si GPU |
| `QDRANT_MODE` | `local` | `server` pour un Qdrant distant |
| `QDRANT_PATH` | `data/vectordb` | **chemin relatif** ou absolu inscriptible — jamais `/data/...` |
| `DOCUMENTS_DIR` | `data/documents` | dossier de la source `local` (historique/partagée) |
| `ACTIVE_PROFILE` | `generic` | `config/schemas/<profil>.yaml` |
| `ACTIVE_DOMAIN_PROFILE` | *(vide)* | `profiles/domains/<nom>.yaml`, facultatif — repli du corpus « default » uniquement |

Le nom de la collection Qdrant du corpus « default » n'est **pas** dans
`.env` : il vit dans `config/default.yaml → qdrant.nom_collection` (gelé).
Tout autre corpus obtient automatiquement sa propre collection dérivée de son
`corpus_id` — jamais éditée à la main.

`.env` n'est jamais commité (voir `.gitignore`) : à la livraison, seul
`.env.example` — générique, sans référence à un corpus ou un dataset
particulier — fait foi.

---

## 6. Lancement — une seule commande

```bash
python scripts/run.py
```

Ce script : vérifie qu'Ollama est joignable et que le modèle configuré y est
disponible (erreur claire sinon, avant de démarrer quoi que ce soit), crée les
dossiers runtime nécessaires (`data/...`), puis démarre l'API FastAPI **et**
sert le frontend statique **dans le même process, sur le même port** — aucun
second terminal, aucun `python -m http.server` séparé, aucun souci CORS.

Ouvrir ensuite :

```
http://127.0.0.1:8000/
```

→ redirige automatiquement vers la page de gestion des corpus. `Ctrl+C`
arrête proprement le serveur (rien d'autre à arrêter).

Lancement manuel équivalent (sans les vérifications de `scripts/run.py`) :

```bash
uvicorn "src.api:create_app" --factory --host 127.0.0.1 --port 8000
```

---

## 7. Workflow d'utilisation

1. **Créer un corpus** (page « Gestion des corpus » → « Nouveau corpus », ou
   `POST /corpora {"corpus_id": "...", "source": "managed"}`).
2. **Importer des documents** : upload d'un dossier, ou import par URL
   (source `managed`) — ou déposer des fichiers dans `DOCUMENTS_DIR` pour le
   corpus `default` (source `local`).
3. **Indexer** : bouton de synchronisation (`POST /ingestion`).
4. **Profiler** (optionnel mais recommandé) : proposer puis valider un profil
   de domaine — injecte du vocabulaire métier dans les réponses générées.
5. **Interroger** : page « Agent Documentaire » (`?corpus_id=...`) —
   question libre, réponse sourcée (SEARCH) ou traitement documentaire
   (SUMMARIZE/CLASSIFY/EXTRACT/COMPARE/SYNTHESIZE selon la formulation).

En CLI (diagnostic, sans passer par l'API) :

```bash
python -m src.rag.ingestion --verbose                # indexe DOCUMENTS_DIR (corpus default)
python -m src.rag.generation "Qu'est-ce qu'une caravelle ?"   # RAG brut, une recherche/génération
python scripts/demo_agent.py "Résume rapport_2024.txt" --verbose  # agent complet
python -m src.profiling.cli suggest --domain "Finance et comptabilité" --save
```

---

## 8. Multi-corpus

Un corpus logique (`corpus_id`) = une collection Qdrant + un registre de
fichiers + un profil, **strictement isolés** (`src/rag/corpus.py`) :
supprimer, réingérer ou profiler un corpus n'affecte jamais les autres. Le
corpus `default` préserve le comportement historique mono-corpus (collection
= `config/default.yaml → qdrant.nom_collection`, dossier = `DOCUMENTS_DIR`) ;
tout autre corpus obtient une collection et un registre dédiés, dérivés de
façon déterministe de son `corpus_id`. `default` ne peut pas être supprimé
via l'API (`DELETE /corpora/default` → `403`).

---

## 9. Sources documentaires

Deux sources logiques, jamais un chemin fourni par le client :

- **`local`** — dossier serveur partagé et historique (`DOCUMENTS_DIR`).
- **`managed`** — stockage propre à un corpus, alimenté par upload de dossier
  ou import URL (`data/corpora/<corpus_id>/documents/`).

`GET /sources` liste les sources réellement enregistrées côté backend.
Ajouter un connecteur d'entreprise (GED, SharePoint, API) = implémenter le
contrat `DocumentSource` (`src/sources/base.py`) — voir
[docs/P2.2_SOURCES.md](docs/P2.2_SOURCES.md) — jamais une modification du
socle RAG.

---

## 10. Tests

```bash
pytest -q                              # suite complète (~1050 cas)
pytest tests/api tests/rag tests/tools tests/agent tests/observability tests/sources -q
```

> **Piège d'environnement connu.** Si `.env` contient
> `QDRANT_PATH=/data/vectordb/...` (barre oblique de tête), `mkdir /data`
> échoue au démarrage et la quasi-totalité des tests tombent en
> `PermissionError`. Utiliser un chemin **relatif** (`data/vectordb/...`)
> comme dans `.env.example`.

---

## 11. Structure du dépôt

```
src/
  config.py            Settings (.env) + Technique (yaml) + Profil (schema) + registre corpus
  llm/                  point d'accès unique au LLM (Ollama)
  rag/                  ingestion, chunking, embeddings, vectorstore,
                        retrieval, generation, normalization, validation, loaders, corpus
  tools/                façades agent : search / summarize / classify / extract / compare / synthesize
  agent/               graphe LangGraph, nœuds, routeurs, état, session, service, multidoc
  profiling/           génération/gestion des profils de domaine (LLM)
  sources/             abstraction DocumentSource (local, managed) + IngestionService
  observability/       corrélation, traces JSON structurées, redaction
  api/                  FastAPI : routes, schémas, uploads, import URL, erreurs, frontend statique
  ui/                   pages HTML statiques (Gestion des corpus, Agent Documentaire, Connexion)
config/                default.yaml (gelé) + schemas/<profil>.yaml + corpus.yaml (registre)
profiles/domains/      profils de vocabulaire métier
evaluation/            harnais d'évaluation, séparé du code applicatif (non commité, voir §12)
tests/                 miroir de src/ + tests/evaluation/
scripts/
  run.py                lancement unique (API + frontend)
  demo_agent.py         démo CLI de l'agent complet
test_rag.py            comparateur de réponses réutilisé par evaluation/ (ne pas déplacer)
wheels/                cache d'install hors ligne, optionnel (git-ignored)
docs/                  architecture.md, DO_NOT_TOUCH.md, P2.2/P2.3/P2.4
CHANGELOG.md           historique des versions
```

---

## 12. Évaluation

`evaluation/` héberge un harnais d'évaluation **séparé du code applicatif**,
non versionné (`.gitignore`) — présent seulement sur les postes qui en ont
besoin, jamais nécessaire au fonctionnement normal de l'application :

| Script | Mesure |
|---|---|
| `evaluate_retrieval_document.py` | résolution documentaire (Hit@k, MRR) |
| `evaluate_retrieval_evidence.py` | couverture d'evidence |
| `evaluate_end_to_end.py` | chaîne complète + attribution de la cause d'échec |
| `evaluate_agent.py` | agent, capacité SEARCH |
| `cquae_multicapacite.py` | agent, smoke CQuAE multi-capacités |
| `run_ablation.py` | ablation des leviers de retrieval |

Le harnais **ne modifie ni n'importe** de logique de `src/rag/`, `src/tools/`,
`src/agent/` : il consomme leurs points d'entrée publics. Dernier scorecard :
**[`scorecard_reference.md`](scorecard_reference.md)**.

---

## 13. Authentification — état actuel

Une page `Connexion.html` existe mais **n'est pas branchée** à un backend
d'authentification : formulaire de démonstration (délais simulés, aucun appel
réseau, aucune redirection effective). L'API n'a **aucune** authentification
(MVP, cf. `src/api/app.py`) : ne pas exposer publiquement en l'état. Voir §14
pour ce que cela implique concrètement.

---

## 14. Limitations actuelles

- **Aucune authentification réelle** ni ACL — page de connexion cosmétique
  uniquement (§13). Bloque une mise en production multi-utilisateur ; ne
  bloque pas un usage interne sur réseau restreint.
- **CORS ouvert** (`allow_origins=["*"]`) — cohérent avec l'absence
  d'authentification et de cookies transportés, mais à restreindre avant
  toute exposition au-delà d'un réseau de confiance.
- **Frontend dépendant d'Internet** au chargement (CDN `unpkg.com`, polices
  Google) — l'API et le RAG restent 100 % locaux.
- **Qdrant en mode local embarqué** (fichier) — pas de scalabilité multi-
  process ni de tolérance aux pannes ; `QDRANT_MODE=server` bascule vers un
  Qdrant distant sans changement de code, mais n'a pas été validé en
  production par ce projet.
- **Historique de conversation local à la page**, non persisté entre
  rechargements — un rafraîchissement du navigateur le vide.
- **Pas de file d'attente / worker asynchrone** : une ingestion ou une
  requête longue bloque le thread qui la sert (acceptable pour un usage MVP
  mono-utilisateur ; à revoir avant un usage concurrent important).
- **Pas de CI/CD** configurée dans ce dépôt.

---

## 15. Perspectives d'industrialisation (P3)

Hors périmètre de ce livrable, à considérer pour une mise en production
réelle :

1. Authentification / SSO, gestion des rôles et ACL par corpus.
2. Qdrant en mode serveur (cluster, sauvegardes automatisées).
3. Connecteurs d'entreprise (GED, SharePoint, API documentaire) —
   implémentent `DocumentSource`, sans toucher au socle.
4. File d'attente / workers asynchrones pour les ingestions longues.
5. Monitoring industriel (export `TraceSink` vers OpenTelemetry/ELK/Loki/Splunk).
6. CI/CD, conteneurisation, charte graphique officielle INSY2S.
7. **Historique de conversation par compte**, une fois l'authentification (1)
   en place : à stocker **côté serveur, associé à l'utilisateur authentifié**
   — pas en `localStorage` navigateur, qui ne suit que l'appareil/le
   navigateur et ne peut pas offrir une vraie isolation par compte (un poste
   partagé verrait un historique partagé). Implique un endpoint dédié
   (ex. `GET/POST /conversations`) et un modèle de stockage encore à définir ;
   ne pas anticiper cette persistance côté frontend avant que (1) existe.

Aucun de ces chantiers ne doit modifier les modules listés dans
[docs/DO_NOT_TOUCH.md](docs/DO_NOT_TOUCH.md) sans un cycle d'évaluation complet
et une nouvelle version de socle.
