"""Chargement des résultats pour l'application. Aucun calcul lourd, aucun appel LLM :
tout vient des fichiers déjà produits par le pipeline et par evaluation/."""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parents[1]
RES = RACINE / "evaluation" / "results"
RUN = RACINE / "outputs" / "run_mars_2026"

TYPES = {"pur": "Pur", "pollue_facile": "Pollué facile", "pollue_difficile": "Pollué difficile"}
STATUTS = {"pur": "accepté", "impur": "rejeté", "invalide": "réponse invalide"}

FICHIERS_JUGE = {
    ("v1 (prompt du run)", 0): "juge_qwen2.5_14b",
    ("v2 (verdict par phrase)", 0): "juge_qwen2.5_14b_prompts_v2",
    ("v1 (prompt du run)", 1): "juge_qwen2.5_14b_graine1",
    ("v2 (verdict par phrase)", 1): "juge_qwen2.5_14b_prompts_v2_graine1",
}


def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data
def corpus() -> pd.DataFrame:
    textes = (RACINE / "data" / "text.txt").read_text(encoding="utf-8").splitlines()
    y = _json(RACINE / "data" / "true_labels.json")
    noms = {int(k): v for k, v in _json(RACINE / "data" / "true_labelname.json").items()}
    return pd.DataFrame({"index": range(len(textes)), "phrase": textes, "intention_massive": [noms[int(v)] for v in y]})


@st.cache_data
def mapping() -> pd.DataFrame:
    """Une ligne par phrase du corpus : tag attribué (ou None), itération, intention MASSIVE."""
    df = corpus().copy()
    lignes = _json(RUN / "final_mapping.json")
    positions = collections.defaultdict(list)
    for i, t in enumerate(df["phrase"]):
        positions[t].append(i)
    pris: set[int] = set()
    tag = [None] * len(df)
    it = [None] * len(df)
    for l in lignes:
        i = l["index"] if "index" in l else next(j for j in positions[l["text"]] if j not in pris)
        pris.add(i)
        tag[i], it[i] = l["intent"], l["iteration_found"]
    df["tag"] = tag
    df["iteration"] = it
    return df


@st.cache_data
def comparaison() -> pd.DataFrame:
    return pd.read_csv(RES / "comparaison_baseline.csv")


@st.cache_data
def purete_tags() -> pd.DataFrame:
    return pd.read_csv(RES / "purete_par_tag.csv")


@st.cache_data
def eclatement() -> pd.DataFrame:
    return pd.read_csv(RES / "eclatement_par_intention.csv")


@st.cache_data
def oracle() -> dict | None:
    p = RES / "plafond_juge_oracle.json"
    return _json(p) if p.exists() else None


@st.cache_data
def iterations() -> pd.DataFrame:
    return pd.read_csv(RUN / "resume_iterations.csv")


@st.cache_data
def juge() -> pd.DataFrame:
    """Tous les cas d'évaluation du juge disponibles, une ligne par cluster soumis."""
    lignes = []
    for (prompt, graine), base in FICHIERS_JUGE.items():
        f = RES / f"{base}_details.jsonl"
        if not f.exists():
            continue
        for k, l in enumerate(f.read_text(encoding="utf-8").splitlines()):
            d = json.loads(l)
            r = d.get("reponse_juge") or {}
            lignes.append({
                "prompt": prompt, "graine": graine, "cas": k + 1, "type": TYPES[d["type"]],
                "pur_attendu": d["pur_attendu"], "decision": STATUTS[d["statut"]],
                "correct": (d["statut"] == "pur") == d["pur_attendu"],
                "central": d["central"], "extremes": d["extremes"], "intentions": d["intentions"],
                "confiance": r.get("niveau_confiance"), "analyse": r.get("analyse_extremes"),
                "verdicts": r.get("verdicts"), "latence_s": d.get("latence_s"),
            })
    return pd.DataFrame(lignes)
