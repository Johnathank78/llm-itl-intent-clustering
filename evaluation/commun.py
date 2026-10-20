"""Fonctions partagées par les scripts d'évaluation.

La vérité terrain (intentions MASSIVE) n'est JAMAIS utilisée par le pipeline : elle ne sert
qu'ici, après coup, pour mesurer la qualité du résultat.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

DOSSIER_RESULTATS = RACINE / "evaluation" / "results"
DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)
MAPPING_REFERENCE = RACINE / "outputs" / "run_mars_2026" / "final_mapping.json"


def charger_corpus() -> tuple[list[str], np.ndarray, dict[int, str]]:
    textes = (RACINE / "data" / "text.txt").read_text(encoding="utf-8").splitlines()
    y = np.array(json.loads((RACINE / "data" / "true_labels.json").read_text()))
    noms = {int(k): v for k, v in json.loads((RACINE / "data" / "true_labelname.json").read_text()).items()}
    return textes, y, noms


def charger_embeddings() -> np.ndarray:
    return np.load(RACINE / "data" / "embeddings_massive.npy")


def charger_scenarios() -> list[str]:
    """Scénario MASSIVE (alarm, calendar, iot...) de chaque phrase, dans l'ordre de text.txt."""
    scen = []
    with open(RACINE / "data" / "massive_fr" / "massive_validation.jsonl", encoding="utf-8") as f:
        for ligne in f:
            scen.append(str(json.loads(ligne)["scenario"]))
    return scen


def charger_mapping(chemin: str | Path = MAPPING_REFERENCE) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (indices des phrases étiquetées, tag prédit pour chacune).

    Le mapping du run de mars 2026 ne contient que le texte : on le réaligne sur text.txt en
    consommant les doublons dans l'ordre (16 phrases sont présentes deux fois dans le corpus).
    Les runs produits par la version actuelle contiennent directement l'index.
    """
    textes, _, _ = charger_corpus()
    lignes = json.loads(Path(chemin).read_text(encoding="utf-8"))
    if lignes and "index" in lignes[0]:
        return np.array([l["index"] for l in lignes]), np.array([l["intent"] for l in lignes])

    positions = collections.defaultdict(list)
    for i, t in enumerate(textes):
        positions[t].append(i)
    deja = set()
    idx, tags = [], []
    for l in lignes:
        libres = [i for i in positions[l["text"]] if i not in deja]
        if not libres:
            raise ValueError(f"Phrase introuvable dans text.txt : {l['text']!r}")
        deja.add(libres[0])
        idx.append(libres[0])
        tags.append(l["intent"])
    return np.array(idx), np.array(tags)


def purete(vrai, pred) -> float:
    """Part des phrases qui portent l'intention majoritaire de leur cluster."""
    par_cluster = collections.defaultdict(collections.Counter)
    for v, p in zip(vrai, pred):
        par_cluster[p][v] += 1
    return sum(c.most_common(1)[0][1] for c in par_cluster.values()) / len(pred)
