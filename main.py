"""Point d'entrée du pipeline.

Exemples :
    python main.py                    # nouveau run (refuse d'écraser un run existant)
    python main.py --reprendre        # reprend un run interrompu là où il s'est arrêté
    python main.py --reinitialiser    # efface le run courant puis repart de zéro
    python main.py --recalculer-embeddings   # ignore data/embeddings_massive.npy (nécessite GEMINI_API_KEY)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys

from src.config import charger_config, chemin
from src.llm import LLMIndisponible


def main() -> int:
    parser = argparse.ArgumentParser(description="Tamis récursif LLM-in-the-loop sur MASSIVE fr-FR")
    groupe = parser.add_mutually_exclusive_group()
    groupe.add_argument("--reprendre", action="store_true", help="reprendre un run interrompu")
    groupe.add_argument("--reinitialiser", action="store_true", help="effacer le run courant avant de lancer")
    parser.add_argument("--recalculer-embeddings", action="store_true")
    parser.add_argument("--config", default=None, help="chemin vers un config.json alternatif")
    args = parser.parse_args()

    cfg = charger_config(args.config)
    dossier = chemin(cfg["sorties"]["dossier_run"])
    if args.reinitialiser and dossier.exists():
        # Avant d'ouvrir pipeline.log : sous Windows, un fichier ouvert ne peut pas être supprimé
        shutil.rmtree(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s : %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(dossier / "pipeline.log", encoding="utf-8")],
    )
    log = logging.getLogger("main")
    log.info("Modèle LLM : %s @ %s", cfg["llm"]["model"], cfg["llm"]["base_url"])

    from src import step1_donnees, step3_tamis

    if args.recalculer_embeddings:
        f = chemin(cfg["donnees"]["dossier"]) / cfg["embeddings"]["fichier"]
        if f.exists():
            f.rename(f.with_suffix(".ancien.npy"))
    step1_donnees.executer(cfg)

    try:
        step3_tamis.executer(cfg, reprendre=args.reprendre)
    except LLMIndisponible:
        return 2
    except FileExistsError as e:
        log.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
