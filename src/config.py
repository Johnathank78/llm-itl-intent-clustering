"""Chargement de la configuration et des secrets.

Les paramètres vivent dans config/config.json et les prompts dans config/prompts.json.
Les secrets (clé Gemini, éventuelle clé du serveur LLM) sont lus dans les variables
d'environnement, éventuellement depuis un fichier .env à la racine du dépôt.
Aucune clé ne doit être écrite dans le code ni dans config.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]

try:  # python-dotenv est optionnel
    from dotenv import load_dotenv

    load_dotenv(RACINE / ".env")
except ImportError:  # pragma: no cover
    pass


def charger_config(chemin: str | Path | None = None) -> dict:
    """Lit config.json puis applique les surcharges par variables d'environnement."""
    chemin = Path(chemin) if chemin else RACINE / "config" / "config.json"
    with open(chemin, encoding="utf-8") as f:
        cfg = json.load(f)

    # Surcharges utiles sans toucher au fichier versionné
    cfg["llm"]["model"] = os.getenv("LLM_MODEL", cfg["llm"]["model"])
    cfg["llm"]["base_url"] = os.getenv("LLM_BASE_URL", cfg["llm"]["base_url"])
    cfg["llm"]["api_key"] = os.getenv("LLM_API_KEY") or "ollama"  # vide dans .env -> valeur par défaut
    return cfg


def charger_prompts(chemin: str | Path | None = None) -> dict:
    chemin = Path(chemin) if chemin else RACINE / "config" / "prompts.json"
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def cle_gemini() -> str:
    cle = os.getenv("GEMINI_API_KEY", "").strip()
    if not cle:
        raise RuntimeError(
            "Variable GEMINI_API_KEY absente. Copie .env.example vers .env et renseigne ta clé, "
            "ou utilise les embeddings déjà fournis dans data/embeddings_massive.npy."
        )
    return cle


def chemin(relatif: str | Path) -> Path:
    """Résout un chemin relatif à la racine du dépôt."""
    p = Path(relatif)
    return p if p.is_absolute() else RACINE / p
