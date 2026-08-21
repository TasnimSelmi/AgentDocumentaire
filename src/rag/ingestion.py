"""
Pipeline d'ingestion : du fichier brut au point indexé dans Qdrant.

Enchaînement, par document :
    découverte -> hash/dédup -> extraction texte (+OCR) -> découpage
    -> inférence LLM (catégorie + métadonnées) -> normalisation
    -> résolution d'entités -> embeddings -> indexation

Puis, globalement : un rapport qualité qui rend visibles les taux de
remplissage, les échecs et les fusions d'entités. Ce rapport est ce qui
permet d'améliorer les descriptions du profil YAML et de relancer —
c'est la boucle qui fait progresser la qualité du RAG.

Rien de spécifique à un domaine : catégories, champs et consignes
viennent tous du profil actif.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, create_model

from src.config import (
    Profil,
    get_config_technique,
    get_profil,
    get_settings,
)
from src.llm.factory import construire_llm
from src.rag.chunking import (
    Chunk,
    decouper_pages_recursif,
    decouper_pages_structure,
)
from src.rag.embeddings import encoder, encoder_dense_seul, precharger_modeles
from src.rag.loaders import ErreurChargement, Page, charger_document
from src.rag.normalization import (
    RegistreEntites,
    creer_registre,
    normaliser_metadonnees,
)
from src.rag.vectorstore import (
    ChunkIndexable,
    creer_collection,
    fermer_client,
    indexer,
    info_collection,
    supprimer_document,
)

logger = logging.getLogger(__name__)


# À incrémenter lorsqu'une modification du code change le résultat de
# l'ingestion sans être déjà représentée dans la configuration.
_VERSION_PIPELINE_INGESTION = 1

# Espace de noms stable pour les UUID déterministes des versions documentaires.
_NAMESPACE_DOCUMENT = uuid.UUID("ec75e166-4f47-4e11-9c70-b84b9afbf681")


# ===========================================================================
# 1. Registre des fichiers indexés
# ===========================================================================

class RegistreFichiers:
    """
    Associe chaque fichier à son empreinte, à la signature du pipeline et
    à l'identifiant de sa version actuellement indexée.

    Un fichier n'est considéré comme inchangé que si :
      - son contenu est identique ;
      - la configuration qui produit ses chunks et métadonnées est identique.
    """

    def __init__(self, chemin: Path) -> None:
        self.chemin = chemin
        self._donnees: dict[str, dict[str, Any]] = {}

        if chemin.exists():
            donnees = json.loads(chemin.read_text(encoding="utf-8"))
            # Normalisation des anciennes clés du registre vers des chemins
            # absolus résolus. Cela assure une comparaison stable sous Windows.
            self._donnees = {
                self._cle(Path(cle)): valeur
                for cle, valeur in donnees.items()
            }

    @staticmethod
    def _cle(chemin_fichier: Path) -> str:
        return str(chemin_fichier.resolve(strict=False))

    def doc_id(self, chemin_fichier: Path) -> str | None:
        entree = self._donnees.get(self._cle(chemin_fichier))
        return entree.get("doc_id") if entree else None

    def est_inchange(
        self,
        chemin_fichier: Path,
        empreinte: str,
        signature_pipeline: str,
    ) -> bool:
        entree = self._donnees.get(self._cle(chemin_fichier))
        return (
            entree is not None
            and entree.get("hash") == empreinte
            and entree.get("signature_pipeline") == signature_pipeline
        )

    def enregistrer(
        self,
        chemin_fichier: Path,
        empreinte: str,
        signature_pipeline: str,
        doc_id: str,
        nb_chunks: int,
    ) -> None:
        self._donnees[self._cle(chemin_fichier)] = {
            "hash": empreinte,
            "signature_pipeline": signature_pipeline,
            "doc_id": doc_id,
            "chunks": nb_chunks,
            "indexe_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def entrees_absentes(
        self,
        chemins_presents: list[Path],
    ) -> list[tuple[str, str]]:
        """
        Renvoie les entrées du registre dont le fichier source n'existe plus
        parmi les documents actuellement éligibles à l'ingestion.

        Le résultat contient des couples (clé_du_registre, doc_id).
        """
        cles_presentes = {self._cle(chemin) for chemin in chemins_presents}
        absentes: list[tuple[str, str]] = []

        for cle, entree in self._donnees.items():
            doc_id = entree.get("doc_id")
            if cle not in cles_presentes and doc_id:
                absentes.append((cle, str(doc_id)))

        return absentes

    def retirer_cle(self, cle: str) -> None:
        """Retire une entrée après suppression réussie de ses points Qdrant."""
        self._donnees.pop(cle, None)

    def sauvegarder(self) -> None:
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self.chemin.write_text(
            json.dumps(self._donnees, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def vider(self) -> None:
        self._donnees = {}


def empreinte_fichier(chemin: Path, algo: str = "sha256") -> str:
    """Empreinte du contenu, calculée par blocs pour ne pas charger le fichier en RAM."""
    h = hashlib.new(algo)
    with chemin.open("rb") as fh:
        for bloc in iter(lambda: fh.read(65536), b""):
            h.update(bloc)
    return h.hexdigest()


def calculer_signature_pipeline(profil: Profil, inferer: bool) -> str:
    """
    Calcule une signature déterministe des réglages qui influencent le
    contenu indexé.

    Ainsi, un document est retraité même si son fichier n'a pas changé,
    dès qu'un élément important change : profil, découpage, OCR,
    normalisation, résolution d'entités, modèle d'embedding ou LLM.

    Les secrets, chemins locaux, tailles de lots et paramètres de recherche
    ne sont volontairement pas inclus, car ils ne changent pas le contenu
    logique produit par l'ingestion.
    """
    settings = get_settings()
    tech = get_config_technique()

    configuration_llm: dict[str, Any] = {"active": inferer}
    if inferer:
        configuration_llm.update(
            {
                "provider": settings.llm_provider,
                "model": settings.llm_model,
                "base_url": settings.llm_base_url or "",
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
            }
        )

    composants: dict[str, Any] = {
        "version_pipeline": _VERSION_PIPELINE_INGESTION,
        "profil": {
            "profile_name": profil.profile_name,
            "classification": profil.classification.model_dump(mode="json"),
            "champs_metadonnees": [
                champ.model_dump(mode="json")
                for champ in profil.champs_metadonnees
            ],
        },
        "decoupage": tech.decoupage.model_dump(mode="json"),
        "ingestion": tech.ingestion.model_dump(mode="json"),
        "ocr": {
            "active": settings.ocr_enabled,
            "configuration": tech.ocr.model_dump(mode="json"),
        },
        "normalisation": tech.normalisation.model_dump(mode="json"),
        "resolution_entites": tech.resolution_entites.model_dump(mode="json"),
        "embedding": {
            "model": settings.embedding_model,
            "taille_vecteur_dense": tech.qdrant.taille_vecteur_dense,
            "sparse_active": tech.qdrant.sparse_active,
        },
        "llm": configuration_llm,
    }

    representation = json.dumps(
        composants,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(representation.encode("utf-8")).hexdigest()


def identifiant_version_document(
    chemin: Path,
    racine_documents: Path,
    empreinte: str,
    signature_pipeline: str,
) -> str:
    """
    Identifiant déterministe d'une version documentaire.

    Il varie si le chemin, le contenu ou la configuration d'ingestion varie.
    Deux fichiers identiques placés à deux chemins différents ne peuvent donc
    pas se remplacer mutuellement dans Qdrant.
    """
    source = chemin.resolve().relative_to(racine_documents.resolve()).as_posix()
    valeur = f"{source}:{empreinte}:{signature_pipeline}"
    return str(uuid.uuid5(_NAMESPACE_DOCUMENT, valeur))


# ===========================================================================
# 2. Découpage
# ===========================================================================

# `Chunk` est défini dans `src.rag.chunking` et ré-exporté ici : les champs
# historiques (index, texte, page) sont inchangés, les champs de structure
# sont optionnels et valent None quand l'information n'existe pas.


def _construire_splitter() -> Any:
    """Construit le splitter récursif LangChain à partir de la configuration."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    cfg = get_config_technique().decoupage
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.taille_chunk,
        chunk_overlap=cfg.recouvrement,
        separators=cfg.separateurs,
        length_function=len,
    )


def decouper(pages: list[Page]) -> list[Chunk]:
    """
    Découpe page par page, en conservant le numéro de page d'origine.

    Découper page par page plutôt que sur le texte concaténé permet de
    citer une page exacte dans les réponses — sans quoi les citations
    seraient approximatives.

    Deux stratégies sont disponibles, choisies par `decoupage.strategie` :

    - ``recursive`` : découpage historique par taille fixe ;
    - ``structure_aware`` : segmentation en blocs (titre, paragraphe, liste,
      tableau) avant découpage, avec en-tête de tableau répété et
      regroupement parent-child.

    La signature et le type de retour sont inchangés.
    """
    cfg = get_config_technique().decoupage
    splitter = _construire_splitter()

    if cfg.strategie != "structure_aware":
        return decouper_pages_recursif(pages, splitter)

    return decouper_pages_structure(
        pages,
        taille_chunk=cfg.taille_chunk,
        taille_parent=cfg.parent_child.taille_parent,
        parent_child_actif=cfg.parent_child.actif,
        tables_actives=cfg.tables.actif,
        lignes_par_chunk=cfg.tables.lignes_par_chunk,
        recouvrement_lignes=cfg.tables.recouvrement_lignes,
        conserver_entete=cfg.tables.conserver_entete,
        lignes_table_min=cfg.tables.lignes_min,
        splitter=splitter,
    )


# ===========================================================================
# 3. Inférence LLM (catégorie + métadonnées)
# ===========================================================================


def _modele_analyse(profil: Profil) -> type[BaseModel]:
    """
    Assemble un schéma unique : catégorie + tous les champs de métadonnées.

    Le modèle hérite du modèle dynamique de métadonnées afin de conserver
    ses validators et sa logique de normalisation.
    """
    ModeleMeta = profil.modele_metadonnees()
    CategorieEnum = profil.classification.en_enum()

    modele = create_model(
        "AnalyseDocument",
        __base__=ModeleMeta,
        categorie=(
            CategorieEnum,
            Field(
                ...,
                description="Catégorie du document parmi la liste fournie.",
            ),
        ),
        confiance=(
            float,
            Field(
                ...,
                ge=0.0,
                le=1.0,
                description="Confiance dans la catégorie, entre 0 et 1.",
            ),
        ),
    )  # type: ignore[call-overload]

    modele.__doc__ = (
        "Analyse structurée d'un document : catégorie et métadonnées."
    )

    return modele


def _prompt_analyse(profil: Profil, nom_fichier: str, extrait: str) -> str:
    """
    Prompt d'inférence, construit intégralement depuis le profil.

    Les descriptions écrites dans le YAML deviennent littéralement les
    instructions du LLM : leur précision conditionne la qualité des
    métadonnées, donc celle du filtrage.
    """
    return f"""Tu analyses un document pour l'indexer dans une base documentaire.

CATÉGORIES POSSIBLES :
{profil.classification.bloc_prompt()}

CHAMPS À RENSEIGNER :
{profil.bloc_prompt_metadonnees()}

RÈGLES :
- N'invente jamais une valeur. Si une information est absente, laisse le champ vide.
- Recopie les valeurs telles qu'elles apparaissent dans le document.
- Les dates au format AAAA-MM-JJ.
- Si aucune catégorie ne convient clairement, utilise « {profil.classification.categorie_defaut} ».

NOM DU FICHIER : {nom_fichier}

EXTRAIT DU DOCUMENT :
---
{extrait}
---"""


def analyser_document(
    nom_fichier: str,
    texte: str,
    profil: Profil,
    llm: Any,
) -> dict[str, Any]:
    """
    Un appel LLM en sortie structurée : catégorie, confiance, métadonnées.

    Le schéma Pydantic généré depuis le YAML est passé en function calling :
    le modèle ne peut structurellement pas produire d'autres champs ni
    d'autres catégories.
    """
    cfg = get_config_technique().ingestion
    extrait = texte[: cfg.chars_pour_inference]

    Modele = _modele_analyse(profil)
    prompt = _prompt_analyse(profil, nom_fichier, extrait)

    resultat = llm.with_structured_output(Modele).invoke(prompt)
    donnees = resultat.model_dump()

    # Sous le seuil de confiance, on retombe sur la catégorie par défaut
    # plutôt que de propager un classement douteux dans les filtres.
    if donnees.get("confiance", 1.0) < profil.classification.seuil_confiance:
        donnees["categorie"] = profil.classification.categorie_defaut

    return donnees


# ===========================================================================
# 4. Rapport qualité
# ===========================================================================

@dataclass
class RapportIngestion:
    """Bilan chiffré d'une exécution. Sérialisé en JSON pour le README."""

    profil: str = ""
    debut: str = ""
    duree_secondes: float = 0.0

    fichiers_trouves: int = 0
    fichiers_ignores_inchanges: int = 0
    fichiers_traites: int = 0
    fichiers_en_echec: int = 0
    fichiers_vides: int = 0
    fichiers_ocr: int = 0
    fichiers_supprimes: int = 0
    chunks_indexes: int = 0

    par_categorie: dict[str, int] = field(default_factory=dict)
    remplissage_champs: dict[str, float] = field(default_factory=dict)
    entites: dict[str, dict[str, int]] = field(default_factory=dict)
    valeurs_rares: dict[str, list[str]] = field(default_factory=dict)
    erreurs: list[dict[str, str]] = field(default_factory=list)
    avertissements: list[dict[str, str]] = field(default_factory=list)

    def sauvegarder(self, chemin: Path) -> None:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def afficher(self) -> None:
        print("\n" + "=" * 62)
        print(f"  RAPPORT D'INGESTION — profil « {self.profil} »")
        print("=" * 62)
        print(f"  Durée                  : {self.duree_secondes:.1f}s")
        print(f"  Fichiers trouvés       : {self.fichiers_trouves}")
        print(f"  Inchangés (ignorés)    : {self.fichiers_ignores_inchanges}")
        print(f"  Traités                : {self.fichiers_traites}")
        print(f"  En échec               : {self.fichiers_en_echec}")
        print(f"  Vides (non indexés)    : {self.fichiers_vides}")
        print(f"  Passés par OCR         : {self.fichiers_ocr}")
        print(f"  Supprimés du corpus    : {self.fichiers_supprimes}")
        print(f"  Chunks indexés         : {self.chunks_indexes}")

        if self.par_categorie:
            print("\n  Répartition par catégorie")
            for cat, n in sorted(self.par_categorie.items(), key=lambda x: -x[1]):
                print(f"    {cat:<24} {n}")

        if self.remplissage_champs:
            print("\n  Taux de remplissage des champs")
            for champ, taux in sorted(self.remplissage_champs.items(), key=lambda x: -x[1]):
                alerte = "  <-- à revoir" if taux < 50 else ""
                print(f"    {champ:<24} {taux:5.1f}%{alerte}")

        if self.entites:
            print("\n  Résolution d'entités")
            for champ, stats in self.entites.items():
                print(
                    f"    {champ:<24} {stats['canoniques']} canoniques "
                    f"/ {stats['variantes']} variantes fusionnées"
                )

        if self.erreurs:
            print(f"\n  {len(self.erreurs)} erreur(s) — détail dans le rapport JSON")
            for e in self.erreurs[:5]:
                print(f"    {e['fichier']} : {e['message'][:60]}")

        print("=" * 62 + "\n")


# ===========================================================================
# 5. Pipeline
# ===========================================================================

def decouvrir_fichiers(dossier: Path, extensions: list[str]) -> list[Path]:
    """Parcours récursif, filtré par extension et par taille."""
    cfg = get_config_technique().ingestion
    taille_max = cfg.taille_max_mo * 1024 * 1024

    fichiers = [
        p
        for p in sorted(dossier.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in extensions
        and not p.name.startswith("~$")
    ]

    retenus = []
    for p in fichiers:
        if p.stat().st_size > taille_max:
            logger.warning("%s ignoré : dépasse %d Mo.", p.name, cfg.taille_max_mo)
            continue
        retenus.append(p)
    return retenus


def _payload_structure(chunk: Chunk, doc_id: str) -> dict[str, Any]:
    """
    Champs de structure ajoutés au payload Qdrant.

    Tous sont optionnels : un chunk produit par la stratégie `recursive` n'en
    porte aucun, et les anciens payloads restent lisibles tels quels. Les
    identifiants de parent et de tableau sont préfixés par le `doc_id` pour
    rester uniques dans toute la collection.
    """
    champs: dict[str, Any] = {}
    if chunk.type_bloc:
        champs["type_bloc"] = chunk.type_bloc
    if chunk.section_title:
        champs["section_title"] = chunk.section_title
    if chunk.parent_id:
        champs["parent_id"] = f"{doc_id}:{chunk.parent_id}"
    if chunk.table_id:
        champs["table_id"] = f"{doc_id}:{chunk.table_id}"
    if chunk.ordre_dans_parent is not None:
        champs["ordre_dans_parent"] = chunk.ordre_dans_parent
    if chunk.is_table:
        champs["is_table"] = True
        champs["header_repeated"] = chunk.header_repeated
    return champs


def _traiter_fichier(
    chemin: Path,
    doc_id: str,
    empreinte: str,
    signature_pipeline: str,
    profil: Profil,
    llm: Any,
    registre_entites: RegistreEntites,
    rapport: RapportIngestion,
    inferer: bool,
    racine_documents: Path,
) -> list[ChunkIndexable]:
    """Traite un fichier et renvoie ses chunks prêts à indexer."""

    doc = charger_document(chemin)

    if doc.ocr_utilise:
        rapport.fichiers_ocr += 1
    for avertissement in doc.avertissements:
        rapport.avertissements.append({"fichier": chemin.name, "message": avertissement})

    if doc.est_vide:
        rapport.fichiers_vides += 1
        return []

    chunks = decouper(doc.pages)
    if not chunks:
        rapport.fichiers_vides += 1
        return []

    # --- Inférence, une seule fois pour tout le document ---
    if inferer:
        analyse = analyser_document(chemin.name, doc.texte_complet, profil, llm)
    else:
        analyse = {"categorie": profil.classification.categorie_defaut, "confiance": 1.0}

    categorie_brute = analyse.pop(
        "categorie", profil.classification.categorie_defaut
    )
    categorie = (
        str(categorie_brute.value)
        if hasattr(categorie_brute, "value")
        else str(categorie_brute)
    )
    analyse.pop("confiance", None)

    rapport.par_categorie[categorie] = rapport.par_categorie.get(categorie, 0) + 1

    # --- Normalisation + résolution d'entités ---
    metadonnees = normaliser_metadonnees(analyse, profil=profil, registre=registre_entites)

    # --- Payload commun à tous les chunks du document ---
    payload_commun: dict[str, Any] = {
        "source": chemin.relative_to(racine_documents).as_posix(),
        "nom_fichier": chemin.name,
        "hash_contenu": empreinte,
        "signature_pipeline": signature_pipeline,
        "categorie": categorie,
        "ocr": doc.ocr_utilise,
        **{k: v for k, v in metadonnees.items() if v is not None},
    }

    # --- Vectorisation ---
    enc = encoder([c.texte for c in chunks])

    return [
        ChunkIndexable(
            doc_id=doc_id,
            chunk_index=c.index,
            texte=c.texte,
            dense=enc.dense[i],
            sparse=enc.sparse[i],
            payload={**payload_commun, "page": c.page, **_payload_structure(c, doc_id)},
        )
        for i, c in enumerate(chunks)
    ]


def ingerer(
    reinitialiser: bool = False,
    limite: int | None = None,
    inferer: bool = True,
    nom_profil: str | None = None,
    dossier: Path | None = None,
) -> RapportIngestion:
    """
    Exécute le pipeline complet sur data/documents/.

    reinitialiser : vide la collection et le registre avant de recommencer.
    limite        : n'ingère que les N premiers fichiers (mise au point).
    inferer       : désactive les appels LLM (test rapide, sans coût).
    """
    from tqdm import tqdm

    debut = time.time()
    s = get_settings()
    racine_documents = (
        dossier.resolve()
    if dossier is not None
    else s.documents_dir.resolve()
)
    tech = get_config_technique()
    profil = get_profil(nom_profil)
    signature_pipeline = calculer_signature_pipeline(profil, inferer)

    rapport = RapportIngestion(
        profil=profil.profile_name,
        debut=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    # --- Préparation ---
    precharger_modeles(avec_reranker=False)
    creer_collection(reinitialiser=reinitialiser, profil=profil)

    registre_fichiers = RegistreFichiers(s.chemin_registre)
    if reinitialiser:
        registre_fichiers.vider()

    registre_entites = creer_registre(fonction_embedding=encoder_dense_seul)
    llm = construire_llm() if inferer else None

    # Découverte complète avant d'appliquer --limit. Le nettoyage des fichiers
    # supprimés doit toujours comparer le registre au corpus complet, sinon
    # les documents hors limite seraient supprimés par erreur.
    tous_les_fichiers = decouvrir_fichiers(
        racine_documents, tech.ingestion.extensions_supportees
    )
    rapport.fichiers_trouves = len(tous_les_fichiers)

    # Retrait des documents qui n'existent plus dans le corpus. L'entrée du
    # registre n'est supprimée qu'après la réussite de la suppression Qdrant ;
    # en cas d'échec, la prochaine exécution pourra donc réessayer.
    for cle_registre, doc_id_absent in registre_fichiers.entrees_absentes(
        tous_les_fichiers
    ):
        try:
            supprimer_document(doc_id_absent)
            registre_fichiers.retirer_cle(cle_registre)
            rapport.fichiers_supprimes += 1
            logger.info("Document retiré du corpus : %s", cle_registre)
        except Exception as exc:  # noqa: BLE001
            rapport.fichiers_en_echec += 1
            rapport.erreurs.append(
                {"fichier": cle_registre, "message": repr(exc)}
            )
            logger.exception(
                "Impossible de supprimer de Qdrant le document absent %s",
                cle_registre,
            )

    fichiers = (
        tous_les_fichiers[:limite]
        if limite is not None
        else tous_les_fichiers
    )

    if not fichiers:
        logger.warning(
    "Aucun fichier exploitable dans %s",
    racine_documents,
)

    # --- Suivi du remplissage des champs ---
    noms_champs = [c.nom for c in profil.champs_metadonnees]
    remplis = {nom: 0 for nom in noms_champs}

    # --- Boucle principale ---
    for chemin in tqdm(fichiers, desc="Ingestion", unit="doc"):
        try:
            empreinte = empreinte_fichier(chemin, tech.ingestion.algo_hash)

            if registre_fichiers.est_inchange(
                chemin, empreinte, signature_pipeline
            ):
                rapport.fichiers_ignores_inchanges += 1
                continue

            ancien_doc_id = registre_fichiers.doc_id(chemin)
            nouveau_doc_id = identifiant_version_document(
    chemin=chemin,
    racine_documents=racine_documents,
    empreinte=empreinte,
    signature_pipeline=signature_pipeline,
)

            # La nouvelle version est entièrement préparée avant toute
            # suppression. Si le chargement, le LLM ou l'embedding échoue,
            # l'ancienne version reste disponible dans Qdrant.
            chunks = _traiter_fichier(
                chemin=chemin,
                doc_id=nouveau_doc_id,
                empreinte=empreinte,
                signature_pipeline=signature_pipeline,
                profil=profil,
                racine_documents=racine_documents,
                llm=llm,
                registre_entites=registre_entites,
                rapport=rapport,
                inferer=inferer,
            )

            if not chunks:
                if ancien_doc_id:
                    supprimer_document(ancien_doc_id)

                registre_fichiers.enregistrer(
                    chemin_fichier=chemin,
                    empreinte=empreinte,
                    signature_pipeline=signature_pipeline,
                    doc_id=nouveau_doc_id,
                    nb_chunks=0,
                )
                continue

            n = indexer(chunks)

            # La nouvelle version est maintenant complète dans Qdrant.
            if ancien_doc_id and ancien_doc_id != nouveau_doc_id:
                supprimer_document(ancien_doc_id)

            rapport.chunks_indexes += n
            rapport.fichiers_traites += 1

            for nom in noms_champs:
                if chunks[0].payload.get(nom) not in (None, "", []):
                    remplis[nom] += 1

            registre_fichiers.enregistrer(
                chemin_fichier=chemin,
                empreinte=empreinte,
                signature_pipeline=signature_pipeline,
                doc_id=nouveau_doc_id,
                nb_chunks=n,
            )

        except ErreurChargement as exc:
            rapport.fichiers_en_echec += 1
            rapport.erreurs.append({"fichier": chemin.name, "message": str(exc)})
            logger.warning("Échec : %s", exc)

        except Exception as exc:  # noqa: BLE001 — un fichier ne doit jamais tout arrêter
            rapport.fichiers_en_echec += 1
            rapport.erreurs.append({"fichier": chemin.name, "message": repr(exc)})
            logger.exception("Erreur inattendue sur %s", chemin.name)

    # --- Clôture ---
    registre_fichiers.sauvegarder()
    registre_entites.sauvegarder()

    traites = max(rapport.fichiers_traites, 1)
    rapport.remplissage_champs = {
        nom: round(100 * n / traites, 1) for nom, n in remplis.items()
    }
    rapport.entites = registre_entites.statistiques()
    rapport.valeurs_rares = registre_entites.valeurs_rares()
    rapport.duree_secondes = round(time.time() - debut, 1)

    rapport.sauvegarder(s.logs_dir / "rapport_ingestion.json")
    return rapport


# ===========================================================================
# 6. Interface en ligne de commande
# ===========================================================================

def main() -> None:

    parseur = argparse.ArgumentParser(description="Ingestion documentaire")
    parseur.add_argument(
    "--dossier",
    type=Path,
    default=None,
    help="dossier documentaire à ingérer (défaut : dossier configuré)",
)
    parseur.add_argument("--reset", action="store_true", help="vide la collection et le registre")
    parseur.add_argument("--limit", type=int, default=None, help="n'ingérer que N fichiers")
    parseur.add_argument("--no-llm", action="store_true", help="sans inférence LLM (test rapide)")
    parseur.add_argument("--profil", type=str, default=None, help="profil à utiliser")
    parseur.add_argument("--verbose", action="store_true")
    args = parseur.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    try:
        rapport = ingerer(
            reinitialiser=args.reset,
            limite=args.limit,
            inferer=not args.no_llm,
            nom_profil=args.profil,
            dossier=args.dossier,
)
        rapport.afficher()
        print(f"  Collection : {info_collection()}")
        print(f"  Rapport    : {get_settings().logs_dir / 'rapport_ingestion.json'}\n")
    finally:
        fermer_client()


if __name__ == "__main__":
    main()