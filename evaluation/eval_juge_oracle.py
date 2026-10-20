"""Plafond de la méthode : que donnerait le tamis avec un juge et un nommeur PARFAITS ?

On remplace le LLM par un « oracle » qui connaît les vraies intentions MASSIVE :
  - juge oracle   : accepte un cluster si le centroïde et ses extrêmes ont la même intention ;
  - nommeur oracle: renvoie le nom de l'intention du centroïde.
Tout le reste (UMAP, K-Means, extrêmes, boucle) est le code du pipeline.

Cela sépare deux sources d'erreur :
  - l'heuristique « centroïde + 3 extrêmes » (même parfaite, elle accepte des clusters impurs
    dont les phrases intermédiaires sont mal classées) ;
  - les erreurs propres au LLM (écart entre ce plafond et le résultat réel).

lambda (pénalité de coût) dépend du temps de réponse du LLM ; l'oracle répondant instantanément,
on fixe lambda à la valeur moyenne mesurée pendant le run de mars 2026 (~1,3).

Usage : python evaluation/eval_juge_oracle.py [--rapide]
  --rapide réduit les grilles UMAP (2 valeurs de n_neighbors, 4 dimensions) : ~15 min au lieu de ~1 h.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from commun import DOSSIER_RESULTATS, RACINE, charger_corpus, charger_mapping, purete
from src import llm as module_llm
from src import step2_optimisation, step3_tamis
from src.config import charger_config

LAMBDA_RUN_MARS = 1.32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapide", action="store_true")
    args = ap.parse_args()

    textes, y, noms = charger_corpus()
    premiere_occurrence = {}
    for t, l in zip(textes, y.tolist()):
        premiere_occurrence.setdefault(t, l)

    def oracle(self, system, user, temperature=0.0):
        phrases = re.findall(r'Centroïde : "(.*)"', user) + re.findall(r'Extrême \d+ : "(.*)"', user)
        labels = {premiere_occurrence[p] for p in phrases}
        if "intention_nommee" in user:
            return json.dumps({"justification_nom": "oracle", "intention_nommee": noms[premiere_occurrence[phrases[0]]].upper()})
        return json.dumps({"action_centroide": "oracle", "analyse_extremes": "oracle",
                           "niveau_confiance": 100, "reclusteriser": len(labels) > 1})

    module_llm.ClientLLM.brut = oracle
    step2_optimisation.OptimiseurIteration.profiler_llm = lambda self: (LAMBDA_RUN_MARS, 0.0)

    cfg = charger_config()
    cfg["sorties"]["dossier_run"] = "outputs/oracle"
    cfg["sorties"]["sauver_figures"] = False
    if args.rapide:
        cfg["umap"].update(grille_n_neighbors=[15, 50], grille_min_dist=[0.0], grille_n_components=[5, 10, 15, 20])
    shutil.rmtree(RACINE / cfg["sorties"]["dossier_run"], ignore_errors=True)
    sortie = step3_tamis.executer(cfg)

    idx, tags = charger_mapping(sortie)
    vrai = y[idx]
    res = {"parametres": "rapide" if args.rapide else "complets", "lambda": LAMBDA_RUN_MARS,
           "couverture": len(idx) / len(y), "n_tags": len(set(tags)),
           "ARI_phrases_retenues": adjusted_rand_score(vrai, tags),
           "NMI_phrases_retenues": normalized_mutual_info_score(vrai, tags),
           "purete_phrases_retenues": purete(vrai, tags)}
    (DOSSIER_RESULTATS / "plafond_juge_oracle.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
