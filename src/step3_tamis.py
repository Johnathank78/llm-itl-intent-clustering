"""Étape 3 : le tamis récursif (boucle LLM-in-the-loop).

À chaque itération, sur les phrases encore non étiquetées :
  1. optimisation UMAP + K (step2_optimisation) ;
  2. K-Means dans l'espace réduit ;
  3. pour chaque cluster, le juge LLM reçoit la phrase la plus proche du centroïde et les
     n_extremes phrases les plus éloignées. Cluster accepté si reclusteriser == false ET
     niveau_confiance >= seuil_confiance ;
  4. si accepté, le nommeur LLM lui donne un tag DOMAINE_ACTION, en réutilisant si possible un
     tag déjà créé (lexique cumulatif) ; toutes les phrases du cluster sont étiquetées ;
  5. les phrases des clusters refusés repartent au tour suivant.
Arrêt : moins de phrases_restantes_min phrases, ou aucun cluster accepté sur un tour complet.

Le test de pureté est une heuristique : si le centroïde et les phrases les plus excentrées
partagent la même intention, on suppose que les phrases intermédiaires aussi. Ce n'est pas une
garantie (voir l'analyse d'erreurs dans evaluation/).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from src.config import charger_config, charger_prompts, chemin
from src.llm import ClientLLM, LLMIndisponible, ReponseInvalide
from src.step2_optimisation import OptimiseurIteration

logger = logging.getLogger(__name__)

PUR, IMPUR, INVALIDE = "pur", "impur", "invalide"


def _booleen(v) -> bool | None:
    """Interprète true/false même si le modèle renvoie une chaîne ("false", "oui"...)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "vrai", "oui", "yes", "1"}:
            return True
        if s in {"false", "faux", "non", "no", "0"}:
            return False
    return None


def formater_extremes(extremes: list[str]) -> str:
    # Même format que lors du run de mars 2026
    return "\n".join(f"    - Extrême {i + 1} : \"{t}\"" for i, t in enumerate(extremes))


def juger(client: ClientLLM, prompts: dict, cfg: dict, central: str, extremes: list[str]) -> tuple[str, dict]:
    """Étape 1 seule (le juge). Renvoie (PUR | IMPUR | INVALIDE, trace)."""
    p1 = prompts["etape_1_purete"]
    try:
        r1 = client.json(p1["system"],
                         p1["user_template"].format(central=central, outliers_text=formater_extremes(extremes)),
                         temperature=cfg["llm"]["temperature_juge"])
    except ReponseInvalide as e:
        return INVALIDE, {"erreur_juge": str(e)}
    trace = {"juge": r1}
    reclusteriser = _booleen(r1.get("reclusteriser"))
    verdicts = r1.get("verdicts")
    if isinstance(verdicts, list) and verdicts:
        # Prompt v2 : la décision est DÉDUITE des verdicts par phrase, pour éviter qu'un raisonnement
        # qui détecte un intrus soit suivi d'un « reclusteriser: false » (observé avec le prompt v1).
        valeurs = [_booleen(v.get("meme_intention")) if isinstance(v, dict) else None for v in verdicts]
        if any(v is None for v in valeurs):
            return INVALIDE, trace
        deduit = not all(valeurs)
        trace["incoherence_decision"] = reclusteriser is not None and reclusteriser != deduit
        reclusteriser = deduit
    try:
        confiance = float(r1.get("niveau_confiance"))
    except (TypeError, ValueError):
        confiance = None
    if reclusteriser is None or confiance is None:
        # Réponse hors schéma : rejet prudent, compté à part pour pouvoir le mesurer
        return INVALIDE, trace
    if reclusteriser or confiance < cfg["tamis"]["seuil_confiance"]:
        return IMPUR, trace
    return PUR, trace


def auditer_cluster(client: ClientLLM, prompts: dict, cfg: dict, central: str, extremes: list[str],
                    tags: list[str]) -> tuple[str, str | None, dict]:
    """Juge puis nommeur. Renvoie (statut, tag, trace). Lève LLMIndisponible si le serveur ne répond plus."""
    statut, trace = juger(client, prompts, cfg, central, extremes)
    if statut != PUR:
        return statut, None, trace
    texte_ext = formater_extremes(extremes)

    p2 = prompts["etape_2_nommage"]
    liste = "\n".join(f"- {t}" for t in tags) if tags else "Aucun tag."
    try:
        r2 = client.json(p2["system"],
                         p2["user_template"].format(tags_existants=liste, central=central, outliers_text=texte_ext),
                         temperature=cfg["llm"]["temperature_nommeur"])
    except ReponseInvalide as e:
        trace["erreur_nommeur"] = str(e)
        return INVALIDE, None, trace
    trace["nommeur"] = r2
    tag = str(r2.get("intention_nommee", "")).strip().upper()
    if not tag:
        return INVALIDE, None, trace
    return PUR, tag, trace


class EtatTamis:
    """Persistance incrémentale : une ligne JSON par cluster validé."""

    def __init__(self, dossier: Path):
        self.dossier = dossier
        self.fichier = dossier / "etat_tamis.jsonl"
        self.fichier_tags = dossier / "taxonomie_tags.txt"
        self.fichier_iter = dossier / "resume_iterations.csv"

    def charger(self) -> tuple[dict[int, dict], list[str], int]:
        attribues: dict[int, dict] = {}
        tags: list[str] = []
        derniere_iter = 0
        if self.fichier.exists():
            with open(self.fichier, encoding="utf-8") as f:
                for ligne in f:
                    c = json.loads(ligne)
                    for i in c["indices"]:
                        attribues[i] = {"intent": c["tag"], "iteration_found": c["iteration"]}
                    if c["tag"] not in tags:
                        tags.append(c["tag"])
                    derniere_iter = max(derniere_iter, c["iteration"])
        if self.fichier_iter.exists():
            with open(self.fichier_iter, encoding="utf-8") as f:
                for ligne in csv.DictReader(f):
                    derniere_iter = max(derniere_iter, int(ligne["iteration"]))
        return attribues, tags, derniere_iter

    def ajouter_cluster(self, iteration: int, tag: str, indices: list[int], trace: dict) -> None:
        with open(self.fichier, "a", encoding="utf-8") as f:
            f.write(json.dumps({"iteration": iteration, "tag": tag, "indices": indices, "trace": trace},
                               ensure_ascii=False) + "\n")

    def ecrire_tags(self, tags: list[str]) -> None:
        self.fichier_tags.write_text("\n".join(tags) + ("\n" if tags else ""), encoding="utf-8")

    def ajouter_iteration(self, ligne: dict) -> None:
        nouveau = not self.fichier_iter.exists()
        with open(self.fichier_iter, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ligne))
            if nouveau:
                w.writeheader()
            w.writerow(ligne)


def ecrire_mapping_final(dossier: Path, textes: list[str], attribues: dict[int, dict]) -> Path:
    lignes = [{"index": i, "text": textes[i], **attribues[i]} for i in sorted(attribues)]
    sortie = dossier / "final_mapping.json"
    with open(sortie, "w", encoding="utf-8") as f:
        json.dump(lignes, f, indent=2, ensure_ascii=False)
    return sortie


def executer(cfg: dict | None = None, reprendre: bool = False) -> Path:
    cfg = cfg or charger_config()
    prompts = charger_prompts()
    dossier = chemin(cfg["sorties"]["dossier_run"])
    dossier.mkdir(parents=True, exist_ok=True)

    X_tout = np.load(chemin(cfg["donnees"]["dossier"]) / cfg["embeddings"]["fichier"])
    textes = (chemin(cfg["donnees"]["dossier"]) / "text.txt").read_text(encoding="utf-8").splitlines()
    if len(textes) != len(X_tout):
        raise ValueError(f"{len(textes)} phrases pour {len(X_tout)} embeddings : données désalignées")

    etat = EtatTamis(dossier)
    if not reprendre and etat.fichier.exists():
        raise FileExistsError(f"{etat.fichier} existe déjà. Utilise --reprendre ou --reinitialiser.")
    attribues, tags, iteration = etat.charger() if reprendre else ({}, [], 0)
    restants = np.array([i for i in range(len(textes)) if i not in attribues])
    logger.info("Départ : %d phrases déjà étiquetées, %d restantes, %d tags", len(attribues), len(restants), len(tags))

    client = ClientLLM(cfg["llm"])
    optimiseur = OptimiseurIteration(cfg, prompts, client, textes)
    n_ext = cfg["tamis"]["n_extremes"]

    try:
        while len(restants) > cfg["tamis"]["phrases_restantes_min"]:
            iteration += 1
            t0 = time.time()
            logger.info("=== Itération %d : %d phrases restantes ===", iteration, len(restants))
            params = optimiseur.executer(X_tout[restants], iteration)
            X_red = params.pop("X_reduit")
            etiquettes = KMeans(n_clusters=params["k"], n_init=cfg["kmeans"]["n_init"],
                                random_state=cfg["kmeans"]["random_state"]).fit_predict(X_red)

            a_garder: list[int] = []
            compte = {PUR: 0, IMPUR: 0, INVALIDE: 0}
            for c in range(params["k"]):
                locaux = np.where(etiquettes == c)[0]
                if len(locaux) == 0:
                    continue
                centre = X_red[locaux].mean(axis=0)
                d = np.linalg.norm(X_red[locaux] - centre, axis=1)
                ordre = np.argsort(d)
                central = textes[restants[locaux[ordre[0]]]]
                ext_locaux = [j for j in ordre[::-1] if j != ordre[0]][:n_ext]
                extremes = [textes[restants[locaux[j]]] for j in ext_locaux]

                statut, tag, trace = auditer_cluster(client, prompts, cfg, central, extremes, tags)
                compte[statut] += 1
                globaux = [int(restants[j]) for j in locaux]
                if statut == PUR:
                    for g in globaux:
                        attribues[g] = {"intent": tag, "iteration_found": iteration}
                    if tag not in tags:
                        tags.append(tag)
                    etat.ajouter_cluster(iteration, tag, globaux, trace)
                    etat.ecrire_tags(tags)
                else:
                    a_garder.extend(globaux)

            extraites = len(restants) - len(a_garder)
            etat.ajouter_iteration({
                "iteration": iteration, "phrases_restantes": len(restants), "n_neighbors": params["n_neighbors"],
                "min_dist": params["min_dist"], "n_components": params["n_components"], "k": params["k"],
                "lambda": round(params["lambda"], 3), "temps_appel_moyen_s": round(params["temps_appel_moyen_s"], 2),
                "clusters_purs": compte[PUR], "clusters_impurs": compte[IMPUR], "reponses_invalides": compte[INVALIDE],
                "phrases_extraites": extraites, "duree_s": round(time.time() - t0, 1),
            })
            logger.info("Itération %d : %d purs, %d impurs, %d invalides, %d phrases extraites",
                        iteration, compte[PUR], compte[IMPUR], compte[INVALIDE], extraites)
            if extraites == 0:
                logger.info("Aucun cluster accepté sur ce tour : arrêt (stagnation).")
                break
            restants = np.array(a_garder)
    except LLMIndisponible:
        logger.error("Serveur LLM indisponible. Les clusters déjà validés sont sauvegardés ; "
                     "relance avec --reprendre une fois le serveur rétabli.")
        raise
    finally:
        sortie = ecrire_mapping_final(dossier, textes, attribues)
        logger.info("%d/%d phrases étiquetées, %d tags -> %s", len(attribues), len(textes), len(tags), sortie)
    return sortie


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    executer()
