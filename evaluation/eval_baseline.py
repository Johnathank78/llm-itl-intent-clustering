"""Le tamis LLM fait-il mieux qu'un simple K-Means ?

Compare le résultat du pipeline (final_mapping.json) à des baselines sans LLM, sur les mêmes
embeddings Gemini :
  - K-Means sur les embeddings normalisés (similarité cosinus) ;
  - K-Means après UMAP (15 dimensions, n_neighbors=15, min_dist=0), comme dans le pipeline.

Information sur K donnée aux baselines (colonne « info_K ») :
  - « aucune »   : K choisi par un critère interne, sans étiquette (silhouette ou Davies-Bouldin, sur
                   K = 20, 30, ..., 160). C'est la situation réelle, comme pour le tamis ;
  - « K du tamis » : K = nombre de tags produits par le tamis (comparaison à complexité égale) ;
  - « vrai K »   : K = 59, le vrai nombre d'intentions, information indisponible en pratique.
Toutes les baselines sont moyennées sur 3 graines.

Attention au choix du critère : en situation réelle, on choisit son critère AVANT de voir les
résultats. Présenter seulement le meilleur des deux critères avantagerait K-Means.

Deux périmètres d'évaluation :
  - « phrases retenues » : uniquement les phrases étiquetées par le pipeline (86,4 % en mars 2026),
    baselines évaluées sur exactement les mêmes phrases ;
  - « corpus complet » : les phrases restées dans le tamis sont rattachées au tag dont le
    centroïde est le plus proche (cosinus), pour comparer sur 100 % des phrases.

Usage : python evaluation/eval_baseline.py [--mapping chemin/final_mapping.json]
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

from commun import DOSSIER_RESULTATS, MAPPING_REFERENCE, charger_corpus, charger_embeddings, charger_mapping, purete

warnings.filterwarnings("ignore")


def scores(vrai, pred) -> dict:
    return {"ARI": adjusted_rand_score(vrai, pred), "NMI": normalized_mutual_info_score(vrai, pred),
            "purete": purete(vrai, pred)}


def main() -> None:
    import umap

    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default=str(MAPPING_REFERENCE))
    ap.add_argument("--graines", type=int, default=3)
    args = ap.parse_args()

    textes, y, _ = charger_corpus()
    X = normalize(charger_embeddings())
    idx, tags = charger_mapping(args.mapping)
    n_tags = len(set(tags))
    n_vrai = len(set(y.tolist()))

    # Pipeline, corpus complet : rattachement des phrases restantes au centroïde de tag le plus proche
    liste_tags = sorted(set(tags))
    centroides = normalize(np.stack([X[idx[tags == t]].mean(axis=0) for t in liste_tags]))
    complet = np.empty(len(y), dtype=object)
    complet[idx] = tags
    reste = np.setdiff1d(np.arange(len(y)), idx)
    complet[reste] = np.array(liste_tags)[np.argmax(X[reste] @ centroides.T, axis=1)]

    # Ablation : sans la fusion opérée par le nommeur (réutilisation d'un tag existant), un même tag
    # trouvé à deux itérations différentes formerait deux groupes distincts.
    iterations = np.array([l["iteration_found"] for l in json.loads(open(args.mapping, encoding="utf-8").read())])
    sans_fusion = np.array([f"{t}#{i}" for t, i in zip(tags, iterations)])

    lignes = [
        {"methode": f"Tamis LLM ({n_tags} tags)", "info_K": "aucune", "K": str(n_tags),
         "perimetre": "phrases retenues", **scores(y[idx], tags), "ecart_type_ARI": 0.0},
        {"methode": "Tamis LLM sans fusion des tags entre itérations", "info_K": "aucune",
         "K": str(len(set(sans_fusion))), "perimetre": "phrases retenues", **scores(y[idx], sans_fusion),
         "ecart_type_ARI": 0.0},
        {"methode": f"Tamis LLM ({n_tags} tags)", "info_K": "aucune", "K": str(n_tags),
         "perimetre": "corpus complet", **scores(y, complet.astype(str)), "ecart_type_ARI": 0.0},
    ]

    def ajouter(nom: str, info_k: str, resultats: list[tuple[int, np.ndarray]]) -> None:
        """resultats : une paire (K, étiquettes) par graine."""
        ks = sorted({k for k, _ in resultats})
        k_txt = str(ks[0]) if len(ks) == 1 else f"{ks[0]}-{ks[-1]}"
        for perim, sel in [("phrases retenues", idx), ("corpus complet", np.arange(len(y)))]:
            df = pd.DataFrame([scores(y[sel], lab[sel]) for _, lab in resultats])
            lignes.append({"methode": nom, "info_K": info_k, "K": k_txt, "perimetre": perim,
                           **df.mean().to_dict(), "ecart_type_ARI": df["ARI"].std()})

    grille_k = list(range(20, 161, 10))
    fixes = {"cos": {n_tags: [], n_vrai: []}, "umap": {n_tags: [], n_vrai: []}}
    par_critere = {"silhouette": [], "Davies-Bouldin": []}
    for g in range(args.graines):
        Z = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=15, metric="cosine", random_state=g).fit_transform(X)
        for k in (n_tags, n_vrai):
            fixes["cos"][k].append((k, KMeans(n_clusters=k, n_init=10, random_state=g).fit_predict(X)))
            fixes["umap"][k].append((k, KMeans(n_clusters=k, n_init=10, random_state=g).fit_predict(Z)))
        balayage = [(k, KMeans(n_clusters=k, n_init=10, random_state=g).fit_predict(Z)) for k in grille_k]
        sil = [silhouette_score(Z, lab) for _, lab in balayage]
        db = [davies_bouldin_score(Z, lab) for _, lab in balayage]
        par_critere["silhouette"].append(balayage[int(np.argmax(sil))])
        par_critere["Davies-Bouldin"].append(balayage[int(np.argmin(db))])

    for critere, res in par_critere.items():
        ajouter(f"UMAP 15D + K-Means, K choisi par {critere}", "aucune", res)
    ajouter(f"UMAP 15D + K-Means, K = {n_tags} (repris du tamis)", "K du tamis", fixes["umap"][n_tags])
    ajouter(f"K-Means cosinus, K = {n_tags} (repris du tamis)", "K du tamis", fixes["cos"][n_tags])
    ajouter(f"UMAP 15D + K-Means, K = {n_vrai} (vrai nombre)", "vrai K", fixes["umap"][n_vrai])
    ajouter(f"K-Means cosinus, K = {n_vrai} (vrai nombre)", "vrai K", fixes["cos"][n_vrai])

    out = pd.DataFrame(lignes)
    out = out[["perimetre", "methode", "info_K", "K", "ARI", "NMI", "purete", "ecart_type_ARI"]].sort_values(
        ["perimetre", "ARI"], ascending=[False, False])
    out.to_csv(DOSSIER_RESULTATS / "comparaison_baseline.csv", index=False, float_format="%.4f")
    resume = {"phrases_etiquetees": int(len(idx)), "phrases_total": int(len(y)),
              "couverture": round(len(idx) / len(y), 4), "n_tags": n_tags, "n_intentions_vraies": n_vrai}
    (DOSSIER_RESULTATS / "resume_run.json").write_text(json.dumps(resume, indent=2), encoding="utf-8")
    print(json.dumps(resume, indent=2))
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
