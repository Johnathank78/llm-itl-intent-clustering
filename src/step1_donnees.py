"""Étape 1 : données MASSIVE fr-FR et embeddings Gemini.

Les fichiers produits par cette étape sont déjà versionnés dans data/ :
  - massive_fr/massive_validation.jsonl  (partition dev de MASSIVE fr-FR, 2 033 phrases, 59 intentions)
  - text.txt, true_labels.json, true_labelname.json
  - embeddings_massive.npy               (Gemini, 3 072 dimensions, float32)
Relancer cette étape n'est donc utile que pour changer de partition ou de modèle d'embedding.
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

from src.config import charger_config, chemin, cle_gemini

logger = logging.getLogger(__name__)


def telecharger_massive(cfg: dict) -> None:
    """Télécharge la partition MASSIVE demandée si elle n'est pas déjà présente."""
    dossier = chemin(cfg["donnees"]["dossier"]) / "massive_fr"
    partition = cfg["donnees"]["partition"]
    sortie = dossier / f"massive_{partition}.jsonl"
    if sortie.exists():
        logger.info("MASSIVE déjà présent : %s", sortie)
        return

    from datasets import load_dataset  # import tardif : dépendance lourde

    dossier.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("AmazonScience/massive", cfg["donnees"]["locale"], split=partition, trust_remote_code=True)
    n = cfg["donnees"].get("taille_echantillon")
    if n and n < len(ds):
        # Échantillonnage stratifié : préserve la proportion de chaque intention
        ds = ds.train_test_split(train_size=n, stratify_by_column="intent", seed=42)["train"]
    ds.to_json(str(sortie), force_ascii=False)
    logger.info("Écrit : %s (%d lignes)", sortie, len(ds))


def preparer_textes(cfg: dict) -> list[str]:
    """Écrit text.txt, true_labels.json et true_labelname.json à partir du jsonl MASSIVE."""
    dossier = chemin(cfg["donnees"]["dossier"])
    source = dossier / "massive_fr" / f"massive_{cfg['donnees']['partition']}.jsonl"
    textes, labels = [], []
    with open(source, encoding="utf-8") as f:
        for ligne in f:
            item = json.loads(ligne)
            textes.append(item["utt"].strip())
            labels.append(int(item["intent"]))

    with open(dossier / "text.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(textes) + "\n")
    with open(dossier / "true_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f)

    noms_existants = dossier / "true_labelname.json"
    if not noms_existants.exists():
        from datasets import load_dataset

        noms = load_dataset("AmazonScience/massive", cfg["donnees"]["locale"], split="train",
                            trust_remote_code=True).features["intent"].names
        with open(noms_existants, "w", encoding="utf-8") as f:
            json.dump({int(l): noms[l] for l in sorted(set(labels))}, f, indent=4, ensure_ascii=False)
    return textes


def calculer_embeddings(cfg: dict, textes: list[str]) -> np.ndarray:
    """Vectorise les phrases avec Gemini, avec reprise sur point de contrôle."""
    dossier = chemin(cfg["donnees"]["dossier"])
    final = dossier / cfg["embeddings"]["fichier"]
    partiel = dossier / "embeddings_partiel.npy"
    if final.exists():
        X = np.load(final)
        if len(X) != len(textes):
            raise ValueError(f"{final} contient {len(X)} vecteurs pour {len(textes)} phrases")
        logger.info("Embeddings déjà calculés : %s %s", final, X.shape)
        return X

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=cle_gemini())
    vecteurs: list[list[float]] = np.load(partiel).tolist() if partiel.exists() else []
    if vecteurs:
        logger.info("Reprise du calcul à l'index %d", len(vecteurs))

    lot = int(cfg["embeddings"]["taille_lot"])
    pause = float(cfg["embeddings"]["pause_entre_lots_s"])
    i = len(vecteurs)
    attente_429 = 30.0
    while i < len(textes):
        morceau = textes[i:i + lot]
        try:
            rep = client.models.embed_content(
                model=cfg["embeddings"]["modele"],
                contents=morceau,
                config=types.EmbedContentConfig(task_type=cfg["embeddings"]["task_type"]),
            )
        except Exception as e:  # le SDK ne type pas finement les erreurs de quota
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.warning("Quota atteint, nouvel essai du même lot dans %.0fs", attente_429)
                time.sleep(attente_429)
                attente_429 = min(attente_429 * 2, 900)
                continue  # on rejoue le MÊME lot : pas de décalage possible
            raise
        if len(rep.embeddings) != len(morceau):
            raise RuntimeError(f"Lot {i}: {len(rep.embeddings)} vecteurs reçus pour {len(morceau)} phrases")
        vecteurs.extend(e.values for e in rep.embeddings)
        np.save(partiel, np.asarray(vecteurs, dtype=np.float32))
        i += len(morceau)
        attente_429 = 30.0
        logger.info("Embeddings : %d/%d", i, len(textes))
        if i < len(textes):
            time.sleep(pause)

    X = np.asarray(vecteurs, dtype=np.float32)
    np.save(final, X)
    os.remove(partiel)
    logger.info("Matrice finale %s écrite dans %s", X.shape, final)
    return X


def executer(cfg: dict | None = None) -> np.ndarray:
    cfg = cfg or charger_config()
    telecharger_massive(cfg)
    textes = preparer_textes(cfg)
    return calculer_embeddings(cfg, textes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    executer()
