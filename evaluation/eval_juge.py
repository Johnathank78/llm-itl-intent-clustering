"""Le juge LLM sait-il reconnaître un cluster pur ?

On fabrique des clusters de 4 phrases dont on connaît la réponse grâce aux intentions MASSIVE,
puis on les soumet au juge avec EXACTEMENT le prompt et la règle de décision du pipeline
(reclusteriser == false et niveau_confiance >= seuil -> « pur »).

Trois types de clusters (même nombre de chaque) :
  - pur              : 4 phrases de la même intention ;
  - pollué facile    : 3 phrases d'une intention + 1 intrus d'un autre scénario
                       (ex. alarme + météo) ;
  - pollué difficile : 3 phrases d'une intention + 1 intrus du MÊME scénario mais d'une autre
                       intention (ex. allumer vs éteindre la lumière, lists_query vs lists_remove).
L'intrus est placé parmi les « extrêmes », comme dans le pipeline.

Sorties :
  - evaluation/results/juge_<modele>.json         : métriques (précision, rappel, taux de rejet...)
  - evaluation/results/juge_<modele>_details.jsonl: chaque cas avec la réponse brute du modèle

Usage :
    python evaluation/eval_juge.py --n 30                         # modèle de config.json (Ollama)
    python evaluation/eval_juge.py --n 30 --modele qwen2.5:7b
    python evaluation/eval_juge.py --n 30 --modele juge-ft        # modèle fine-tuné importé dans Ollama
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import time

from pathlib import Path

from commun import DOSSIER_RESULTATS, RACINE, charger_corpus, charger_scenarios
from src.config import charger_config, charger_prompts
from src.llm import ClientLLM, LLMIndisponible
from src.step3_tamis import INVALIDE, PUR, juger


def fabriquer_cas(n_par_type: int, graine: int) -> list[dict]:
    textes, y, noms = charger_corpus()
    scen = charger_scenarios()
    rng = random.Random(graine)
    par_int = collections.defaultdict(list)
    for i, l in enumerate(y.tolist()):
        par_int[l].append(i)
    intentions = [l for l, v in par_int.items() if len(v) >= 4]

    def tirer(intention, k):
        return rng.sample(par_int[intention], k)

    cas = []
    for type_cas in ["pur", "pollue_facile", "pollue_difficile"]:
        produits = 0
        while produits < n_par_type:
            l = rng.choice(intentions)
            if type_cas == "pur":
                ids = tirer(l, 4)
                intrus = None
            else:
                ids = tirer(l, 3)
                s = scen[ids[0]]
                candidates = [m for m in intentions if m != l and
                              ((scen[par_int[m][0]] == s) == (type_cas == "pollue_difficile"))]
                if not candidates:
                    continue  # scénario avec une seule intention : pas d'intrus « difficile » possible
                m = rng.choice(candidates)
                intrus = rng.choice(par_int[m])
                ids.insert(rng.randint(1, 3), intrus)  # jamais en position de centroïde
            cas.append({"type": type_cas, "pur_attendu": type_cas == "pur",
                        "central": textes[ids[0]], "extremes": [textes[i] for i in ids[1:]],
                        "intentions": [noms[int(y[i])] for i in ids]})
            produits += 1
    rng.shuffle(cas)
    return cas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="nombre de clusters par type")
    ap.add_argument("--graine", type=int, default=0)
    ap.add_argument("--modele", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--prompts", default=None, help="fichier de prompts alternatif (ex. config/prompts_v2.json)")
    args = ap.parse_args()

    cfg = charger_config()
    if args.modele:
        cfg["llm"]["model"] = args.modele
    if args.base_url:
        cfg["llm"]["base_url"] = args.base_url
    prompts = charger_prompts(RACINE / args.prompts if args.prompts else None)
    client = ClientLLM(cfg["llm"])
    cas = fabriquer_cas(args.n, args.graine)
    tag_modele = re.sub(r"[^A-Za-z0-9._-]+", "_", cfg["llm"]["model"])
    if args.prompts:
        tag_modele += "_" + Path(args.prompts).stem
    if args.graine:
        tag_modele += f"_graine{args.graine}"

    details = []
    for k, c in enumerate(cas, 1):
        t0 = time.time()
        try:
            statut, trace = juger(client, prompts, cfg, c["central"], c["extremes"])
        except LLMIndisponible as e:
            print(f"Serveur LLM indisponible : {e}")
            break
        details.append({**c, "statut": statut, "latence_s": round(time.time() - t0, 2),
                        "reponse_juge": trace.get("juge"),
                        "incoherence_decision": trace.get("incoherence_decision", False)})
        print(f"[{k}/{len(cas)}] {c['type']:<17} -> {statut}")

    if not details:
        return
    # Même règle que le pipeline : seule une acceptation explicite compte comme « pur ».
    def predit_pur(d):
        return d["statut"] == PUR

    vp = sum(d["pur_attendu"] and predit_pur(d) for d in details)
    fp = sum((not d["pur_attendu"]) and predit_pur(d) for d in details)
    fn = sum(d["pur_attendu"] and not predit_pur(d) for d in details)
    par_type = collections.defaultdict(lambda: [0, 0])
    for d in details:
        par_type[d["type"]][0] += 1
        par_type[d["type"]][1] += int(predit_pur(d) == d["pur_attendu"])
    metriques = {
        "modele": cfg["llm"]["model"], "prompts": args.prompts or "config/prompts.json",
        "n_cas": len(details), "graine": args.graine,
        "decisions_corrigees_par_les_verdicts": sum(bool(d.get("incoherence_decision")) for d in details),
        "precision_pur": vp / (vp + fp) if vp + fp else None,
        "rappel_pur": vp / (vp + fn) if vp + fn else None,
        "taux_bonnes_decisions_par_type": {t: ok / n for t, (n, ok) in par_type.items()},
        "taux_reponses_invalides": sum(d["statut"] == INVALIDE for d in details) / len(details),
        "latence_moyenne_s": sum(d["latence_s"] for d in details) / len(details),
        "note": "précision_pur = part des clusters acceptés qui étaient vraiment purs ; "
                "rappel_pur = part des clusters purs que le juge a acceptés.",
    }
    (DOSSIER_RESULTATS / f"juge_{tag_modele}.json").write_text(
        json.dumps(metriques, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(DOSSIER_RESULTATS / f"juge_{tag_modele}_details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps(metriques, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
