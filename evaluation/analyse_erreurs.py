"""Où le tamis se trompe-t-il ?

Produit trois tableaux dans evaluation/results/ :
  - purete_par_tag.csv          : pour chaque tag créé, sa taille, sa pureté et les 3 intentions MASSIVE
                                  qu'il contient le plus (tags « fourre-tout » = pureté faible) ;
  - eclatement_par_intention.csv: pour chaque intention MASSIVE, en combien de tags elle a été
                                  répartie (intentions « éclatées ») ;
  - composition_residu.csv      : les intentions des phrases que le tamis n'a jamais étiquetées.

Usage : python evaluation/analyse_erreurs.py [--mapping chemin/final_mapping.json]
"""

from __future__ import annotations

import argparse
import collections

import numpy as np
import pandas as pd

from commun import DOSSIER_RESULTATS, MAPPING_REFERENCE, charger_corpus, charger_mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default=str(MAPPING_REFERENCE))
    args = ap.parse_args()

    _, y, noms = charger_corpus()
    idx, tags = charger_mapping(args.mapping)
    vrai = [noms[int(v)] for v in y[idx]]

    par_tag = collections.defaultdict(collections.Counter)
    par_intention = collections.defaultdict(collections.Counter)
    for t, v in zip(tags, vrai):
        par_tag[t][v] += 1
        par_intention[v][t] += 1

    lignes = []
    for t, c in par_tag.items():
        n = sum(c.values())
        lignes.append({"tag": t, "phrases": n, "purete": c.most_common(1)[0][1] / n,
                       "intentions_massive": ", ".join(f"{k} ({v})" for k, v in c.most_common(3))})
    df_tag = pd.DataFrame(lignes).sort_values(["purete", "phrases"], ascending=[True, False])
    df_tag.to_csv(DOSSIER_RESULTATS / "purete_par_tag.csv", index=False, float_format="%.3f")

    lignes = []
    for v, c in par_intention.items():
        n = sum(c.values())
        lignes.append({"intention_massive": v, "phrases_etiquetees": n, "nb_tags": len(c),
                       "part_tag_principal": c.most_common(1)[0][1] / n,
                       "tags": ", ".join(f"{k} ({m})" for k, m in c.most_common(4))})
    df_int = pd.DataFrame(lignes).sort_values("nb_tags", ascending=False)
    df_int.to_csv(DOSSIER_RESULTATS / "eclatement_par_intention.csv", index=False, float_format="%.3f")

    reste = np.setdiff1d(np.arange(len(y)), idx)
    total = collections.Counter(noms[int(v)] for v in y)
    res = collections.Counter(noms[int(y[i])] for i in reste)
    df_res = pd.DataFrame([{"intention_massive": k, "phrases_non_etiquetees": v, "total_intention": total[k],
                            "part_non_etiquetee": v / total[k]} for k, v in res.most_common()])
    df_res.to_csv(DOSSIER_RESULTATS / "composition_residu.csv", index=False, float_format="%.3f")

    tranches = pd.cut(df_tag["purete"], [0, .4, .6, .7, .8, .9, .999, 1.0],
                      labels=["< 40 %", "40-59 %", "60-69 %", "70-79 %", "80-89 %", "90-99 %", "100 %"],
                      include_lowest=True)
    print("Répartition des tags par pureté :")
    print(df_tag.groupby(tranches, observed=False)["phrases"].agg(["count", "sum"]).rename(
        columns={"count": "tags", "sum": "phrases"}).to_string())
    print("\nTags les moins purs :")
    print(df_tag.head(8).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nIntentions les plus éclatées :")
    print(df_int.head(8).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    absentes = sorted(set(total) - set(par_intention))
    print(f"\nRésidu : {len(reste)} phrases. Intentions jamais étiquetées : {absentes or 'aucune'}")


if __name__ == "__main__":
    main()
