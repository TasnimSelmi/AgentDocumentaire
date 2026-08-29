# Agent documentaire — RAG générique + couche agentique

Agent de questions-réponses et de traitement documentaire construit sur un
**RAG local** (embeddings BGE-M3, Qdrant hybride, reranking BGE-v2-m3, LLM via
Ollama) et une **couche agentique déterministe** (LangGraph) qui route une
requête vers l'une de quatre capacités : **SEARCH**, **SUMMARIZE**,
**CLASSIFY**, **EXTRACT**.

Le système est **agnostique au domaine** : taxonomie, métadonnées, schéma
d'extraction et vocabulaire métier proviennent tous de fichiers de
configuration (`config/schemas/*.yaml`, `profiles/domains/*.yaml`). Aucun nom
de corpus, de société ou de champ métier n'est codé en dur.

> ## Statut : gel RAG V1 en préparation
>
> Le socle RAG (ingestion, retrieval, génération, outils) est considéré
> **stable et gelé**. Voir **[docs/DO_NOT_TOUCH.md](docs/DO_NOT_TOUCH.md)**
> pour la liste exacte des modules à ne pas modifier sans un cycle
> d'évaluation complet.
>
> Le **tag `rag-v1` n'est pas encore posé** : il attend la ré-exécution et la
> validation du smoke CQuAE J1 (voir
> [`scorecard_reference.md`](scorecard_reference.md) §11). Le travail de
> fiabilisation du harnais d'évaluation (J1) est terminé côté code et tests ;
> seule la confirmation empirique reste en attente d'un GPU disponible.

---

## 1. Architecture en un coup d'œil

```
                 ┌─────────────────────── INGESTION (hors ligne) ───────────────────────┐
  fichiers ──►   loaders → OCR → chunking structure-aware → inférence LLM (catégorie +
                 métadonnées) → normalisation → résolution d'entités → embeddings BGE-M3
                 → indexation Qdrant (vecteurs nommés dense + sparse)
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
                     └─ EXTRACT ───► charger le document entier ──► extraction sourcée par lots
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

---

## 2. Prérequis

| Composant | Version / détail |
|---|---|
| Python | **3.11** (venv du dépôt) — le code cible 3.10+ |
| [Ollama](https://ollama.com) | serveur local, modèle `qwen3:8b` (`ollama pull qwen3:8b`) |
| GPU | optionnel — CUDA accélère embeddings/reranking et Ollama ; sinon CPU |
| Tesseract OCR | requis seulement pour les PDF scannés (`OCR_ENABLED=true`) |
| RAM | ~6 Go pour BGE-M3 + reranker chargés en mémoire |

Modèles Hugging Face (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`) : téléchargés
au premier usage, puis mis en cache.

---

## 3. Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                 # puis adapter (voir §4)
ollama pull qwen3:8b
```

> **Installation hors ligne (Windows).** `wheels/` contient un cache optionnel
> de roues `torch` / `sympy` / `mpmath` pour un poste Windows / CPython 3.12
> sans accès PyPI (`pip install --no-index --find-links wheels/ torch`). Ce
> dossier est ignoré par git et **peut être supprimé** si l'installation se
> fait en ligne.

---

## 4. Configuration

Trois niveaux, volontairement séparés (`src/config.py`) :

| Fichier | Rôle | Varie selon |
|---|---|---|
| `.env` | secrets, chemins, choix des modèles | la machine |
| `config/default.yaml` | chunking, OCR, seuils, recherche, Qdrant, agent | le réglage |
| `config/schemas/<profil>.yaml` | taxonomie, métadonnées, schéma d'extraction | le domaine |
| `profiles/domains/<nom>.yaml` | vocabulaire métier injecté aux prompts | le domaine |

Clés `.env` essentielles (liste complète et commentée : `.env.example`) :

| Clé | Défaut | Note |
|---|---|---|
| `LLM_MODEL` | `qwen3:8b` | Ollama uniquement |
| `LLM_BASE_URL` | `http://localhost:11434` | serveur Ollama |
| `EMBEDDING_DEVICE` | `cpu` | `cuda` si GPU |
| `QDRANT_MODE` | `local` | `server` pour un Qdrant distant |
| `QDRANT_PATH` | `data/vectordb` | **chemin relatif** ou absolu inscriptible — jamais `/data/...` |
| `ACTIVE_PROFILE` | `generic` | `config/schemas/<profil>.yaml` |
| `ACTIVE_DOMAIN_PROFILE` | *(vide)* | `profiles/domains/<nom>.yaml`, facultatif |

Le nom de la collection Qdrant n'est **pas** dans `.env` : il vit dans
`config/default.yaml → qdrant.nom_collection`.

---

## 5. Utilisation

### 5.1 Indexer un corpus

```bash
# Placer les fichiers sous DOCUMENTS_DIR (data/documents/ par défaut)
python -m src.rag.ingestion --verbose
python -m src.rag.ingestion --reset          # vide collection + registre puis réindexe
python -m src.rag.ingestion --limit 10 --no-llm   # test rapide, sans inférence LLM
```

Un rapport qualité est écrit sous `data/logs/` (taux de remplissage des
champs, échecs, fusions d'entités).

### 5.2 Interroger

```bash
# RAG brut (une recherche, une génération) — sans boucle agentique
python -m src.rag.generation "Qu'est-ce qu'une caravelle ?"

# Retrieval seul (diagnostic)
python -m src.rag.retrieval "Comment installer le produit ?"

# Agent complet (LangGraph : boucle rechercher → évaluer → répondre | reformuler)
python scripts/demo_agent.py "Résume le document rapport_2024.txt" --verbose
```

### 5.3 Gérer les profils de domaine

```bash
python -m src.profiling.cli suggest --domain "Finance et comptabilité" --save
python -m src.profiling.cli list
python -m src.profiling.cli show finance
```

---

## 6. Tests

```bash
pytest tests/                         # suite complète (~470 cas)
pytest tests/tools/ tests/rag/        # ciblé
```

> **Piège d'environnement connu.** Si `.env` contient
> `QDRANT_PATH=/data/vectordb/...` (barre oblique de tête), `mkdir /data`
> échoue au démarrage et ~99 tests tombent en `PermissionError`. Utiliser un
> chemin **relatif** (`data/vectordb/...`) comme dans `.env.example`.

---

## 7. Évaluation

`evaluation/` héberge un harnais d'évaluation **séparé du code applicatif** :

| Script | Mesure |
|---|---|
| `evaluate_retrieval_document.py` | résolution documentaire (Hit@k, MRR) |
| `evaluate_retrieval_evidence.py` | couverture d'evidence (plafond de parsing, couverture relative) |
| `evaluate_end_to_end.py` | chaîne complète + attribution de la cause d'échec |
| `evaluate_agent.py` | agent, capacité SEARCH |
| `cquae_multicapacite.py` | agent, **4 capacités** (smoke CQuAE, 28 cas) |
| `run_ablation.py` | ablation des leviers de retrieval |
| `analyze_failures.py` | tri des échecs end-to-end |

Le harnais **ne modifie ni n'importe** de logique de `src/rag/`, `src/tools/`,
`src/agent/` : il consomme leurs points d'entrée publics.

Dernier scorecard : **[`scorecard_reference.md`](scorecard_reference.md)**.

---

## 8. Structure du dépôt

```
src/
  config.py            Settings (.env) + Technique (yaml) + Profil (schema)
  llm/                  point d'accès unique au LLM (Ollama)
  rag/                  ingestion, chunking, embeddings, vectorstore,
                        retrieval, generation, normalization, validation, loaders
  tools/                façades agent : search / summarize / classify / extract
  agent/               graphe LangGraph, nœuds, routeurs, état, session
  profiling/           génération/gestion des profils de domaine (LLM)
config/                default.yaml + schemas/<profil>.yaml
profiles/domains/      profils de vocabulaire métier
evaluation/            harnais d'évaluation (voir §7)
tests/                 miroir de src/ + tests/evaluation/
scripts/demo_agent.py  démo CLI de l'agent complet
test_rag.py            comparateur de réponses réutilisé par evaluation/ (ne pas déplacer)
wheels/                cache d'install hors ligne, optionnel (git-ignored)
docs/                  architecture.md, DO_NOT_TOUCH.md
CHANGELOG.md           historique des versions
```

---

## 9. Feuille de route (post-gel)

Hors périmètre du gel V1, dans l'ordre :

1. Fiabilisation routing (ex. « points essentiels » non routé vers SUMMARIZE ;
   résolution documentaire contextuelle d'EXTRACT).
2. Capacités multi-documents (COMPARE / SYNTHESIZE).
3. Couche service (`AgentResponse`, `AgentService`).
4. API (FastAPI) et UI (Streamlit / React).
5. Connecteurs entreprise, observabilité persistée, sécurité, conteneurisation.

Aucun de ces chantiers ne doit modifier les modules listés dans
[docs/DO_NOT_TOUCH.md](docs/DO_NOT_TOUCH.md) sans un cycle d'évaluation complet
et une nouvelle version de socle.
