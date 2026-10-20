"""Le modèle fine-tuné est-il meilleur que le modèle de base ?

Rejoue le jeu de test du fine-tuning (jamais vu à l'entraînement) sur un ou plusieurs modèles
servis par une API compatible OpenAI (Ollama par exemple), et mesure l'accord avec la réponse
de référence.

ATTENTION à l'interprétation : les réponses de référence ont été produites par des LLM « professeurs »
(Llama 70B, Gemini, Claude), pas par des humains. Ce script mesure donc l'accord avec le professeur,
pas une exactitude absolue. Pour une mesure contre la vérité terrain MASSIVE, utiliser eval_juge.py.

  - juge    (fine_tuning/data/juge/test.jsonl, 29 cas)    : accord sur la décision reclusteriser ;
  - nommeur (fine_tuning/data/nommeur/test.jsonl, 23 cas) : tag identique au tag de référence.

Usage :
    python evaluation/eval_finetune.py --modeles qwen2.5:7b juge-ft --tache juge
    python evaluation/eval_finetune.py --modeles qwen2.5:7b nommeur-ft --tache nommeur
"""

from __future__ import annotations

import argparse
import json
import re
import time

from commun import DOSSIER_RESULTATS, RACINE
from src.config import charger_config
from src.llm import ClientLLM, LLMIndisponible, ReponseInvalide
from src.step3_tamis import _booleen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeles", nargs="+", required=True)
    ap.add_argument("--tache", choices=["juge", "nommeur"], default="juge")
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()

    cas = [json.loads(l) for l in open(RACINE / "fine_tuning" / "data" / args.tache / "test.jsonl", encoding="utf-8")]
    cfg = charger_config()
    if args.base_url:
        cfg["llm"]["base_url"] = args.base_url

    bilan = {}
    for modele in args.modeles:
        cfg["llm"]["model"] = modele
        client = ClientLLM(cfg["llm"])
        ok = invalides = 0
        t_total = 0.0
        for c in cas:
            system, user, attendu = (m["content"] for m in c["messages"])
            attendu = json.loads(attendu)
            t0 = time.time()
            try:
                rep = client.json(system, user, temperature=0.0)
            except ReponseInvalide:
                invalides += 1
                continue
            except LLMIndisponible as e:
                raise SystemExit(f"{modele} : serveur indisponible ({e})")
            finally:
                t_total += time.time() - t0
            if args.tache == "juge":
                ok += int(_booleen(rep.get("reclusteriser")) == attendu["reclusteriser"])
            else:
                ok += int(str(rep.get("intention_nommee", "")).strip().upper() == attendu["intention_nommee"].upper())
        bilan[modele] = {"tache": args.tache, "n_cas": len(cas), "accord_avec_reference": ok / len(cas),
                         "reponses_invalides": invalides, "latence_moyenne_s": t_total / len(cas)}
        print(modele, bilan[modele])

    nom = "finetune_" + args.tache + "_" + "_vs_".join(re.sub(r"[^A-Za-z0-9.-]+", "-", m) for m in args.modeles)
    (DOSSIER_RESULTATS / f"{nom}.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
