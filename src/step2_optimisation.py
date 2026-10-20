"""Étape 2 : choix des hyperparamètres UMAP et du nombre de clusters K, à chaque tour du tamis.

Trois sous-étapes, recalculées sur le sous-corpus restant à chaque itération :
  1. UMAP : grille (n_neighbors x min_dist) notée par la trustworthiness, puis choix du nombre
     de dimensions sur la courbe trustworthiness / dimensions (front de Pareto) ;
  2. profilage du LLM : temps moyen d'un appel du juge -> pénalité lambda = t_moyen * K_max / budget ;
  3. K-Means : score(K) = inertie(K) / inertie_max + lambda * K / K_max, puis départage par le rayon
     sémantique (percentile 95 des rayons maximaux des clusters).
"""

from __future__ import annotations

import logging
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness

from src.config import chemin
from src.llm import ClientLLM, LLMIndisponible, ReponseInvalide

logger = logging.getLogger(__name__)


class OptimiseurIteration:
    def __init__(self, cfg: dict, prompts: dict, client: ClientLLM, textes: list[str]):
        self.cfg = cfg
        self.prompts = prompts
        self.client = client
        self.textes = textes
        self.dossier = chemin(cfg["sorties"]["dossier_run"])
        self.dossier_figures = self.dossier / "figures"
        self.dossier_figures.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ UMAP
    def optimiser_umap(self, X: np.ndarray, iteration: int) -> dict:
        c = self.cfg["umap"]
        rs = c["random_state"]
        if len(X) > c["max_echantillons"]:
            idx = np.random.default_rng(rs).choice(len(X), size=c["max_echantillons"], replace=False)
            X = X[idx]
        n = len(X)
        k_tw = max(1, min(c["k_trustworthiness"], n // 2 - 1))

        voisins = [v for v in c["grille_n_neighbors"] if v < n] or [max(2, n - 1)]
        grille = []
        for md in c["grille_min_dist"]:
            for nn in voisins:
                emb = umap.UMAP(n_neighbors=nn, min_dist=md, n_components=2, metric="cosine",
                                random_state=rs).fit_transform(X)
                tw = trustworthiness(X, emb, n_neighbors=k_tw, metric="cosine")
                grille.append({"n_neighbors": nn, "min_dist": md, "trustworthiness": tw})
        df = pd.DataFrame(grille)
        meilleur = df.loc[df["trustworthiness"].idxmax()]
        nn, md = int(meilleur["n_neighbors"]), float(meilleur["min_dist"])

        dims = []
        for nc in c["grille_n_components"]:
            if nc >= n - 1:
                break
            emb = umap.UMAP(n_neighbors=nn, min_dist=md, n_components=nc, metric="cosine",
                            random_state=rs).fit_transform(X)
            dims.append({"n_components": nc,
                         "trustworthiness": trustworthiness(X, emb, n_neighbors=k_tw, metric="cosine")})
        dd = pd.DataFrame(dims)
        seuil = dd["trustworthiness"].max() * (1 - c["tolerance_trustworthiness"])
        valides = dd[dd["trustworthiness"] >= seuil].copy()
        # Toutes les dimensions <= dimension_sans_penalite sont jugées équivalentes en coût :
        # parmi elles on garde la meilleure trustworthiness, sinon la plus petite dimension.
        valides["cout"] = valides["n_components"].clip(lower=c["dimension_sans_penalite"])
        nc = int(valides.sort_values(["cout", "trustworthiness"], ascending=[True, False]).iloc[0]["n_components"])

        df.to_csv(self.dossier / f"umap_grille_iter_{iteration}.csv", index=False)
        dd.to_csv(self.dossier / f"umap_dimensions_iter_{iteration}.csv", index=False)
        if self.cfg["sorties"].get("sauver_figures", True):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(dd["n_components"], dd["trustworthiness"], marker="o")
            ax.axhline(seuil, ls=":", c="grey", label="seuil de tolérance")
            ax.scatter([nc], dd.loc[dd["n_components"] == nc, "trustworthiness"], s=150, marker="*",
                       c="green", zorder=3, label=f"choix : {nc} dimensions")
            ax.set_xlabel("dimensions UMAP"); ax.set_ylabel(f"trustworthiness (k={k_tw})")
            ax.set_title(f"Itération {iteration} : choix du nombre de dimensions"); ax.legend()
            fig.savefig(self.dossier_figures / f"umap_dimensions_iter_{iteration}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        return {"n_neighbors": nn, "min_dist": md, "n_components": nc}

    # ------------------------------------------------------------- lambda
    def profiler_llm(self) -> tuple[float, float]:
        """Mesure le temps moyen d'un appel du juge et en déduit lambda."""
        c = self.cfg["kmeans"]
        p = self.prompts["etape_1_purete"]
        temps = []
        for _ in range(int(c["appels_profilage"])):
            ech = random.sample(self.textes, 4)
            user = p["user_template"].format(central=ech[0], outliers_text="\n".join(f"- {t}" for t in ech[1:]))
            t0 = time.time()
            try:
                self.client.json(p["system"], user, temperature=0.0)
            except ReponseInvalide:
                pass  # le temps reste représentatif même si le JSON est illisible
            temps.append(time.time() - t0)  # LLMIndisponible remonte : inutile de continuer
        t_moyen = float(np.mean(temps))
        # lambda = part du budget LLM que consommerait l'audit de K_ref clusters (K_ref = 200 dans le run de mars)
        return t_moyen * c["k_max_reference_lambda"] / c["budget_llm_s"], t_moyen

    # ------------------------------------------------------------- K-Means
    def optimiser_k(self, X: np.ndarray, params_umap: dict, lam: float, iteration: int) -> tuple[int, np.ndarray]:
        c = self.cfg["kmeans"]
        k_max = max(2, min(c["k_max_plafond"], len(X) // c["phrases_min_par_cluster"]))
        k_min = max(2, k_max // 4)
        pas = c["pas"] if k_max - k_min > 20 else 1
        grille_k = list(range(k_min, k_max + 1, pas)) or [2]

        X_red = umap.UMAP(n_neighbors=params_umap["n_neighbors"], min_dist=params_umap["min_dist"],
                          n_components=params_umap["n_components"], metric="cosine",
                          random_state=self.cfg["umap"]["random_state"]).fit_transform(X)
        lignes = []
        for k in grille_k:
            km = KMeans(n_clusters=k, n_init=c["n_init"], random_state=c["random_state"]).fit(X_red)
            rayons = [np.linalg.norm(X_red[km.labels_ == i] - km.cluster_centers_[i], axis=1).max()
                      for i in range(k) if np.any(km.labels_ == i)]
            lignes.append({"K": k, "rayon_p95": float(np.percentile(rayons, 95)), "inertie": km.inertia_})
        df = pd.DataFrame(lignes)
        df["score"] = df["inertie"] / df["inertie"].max() + lam * df["K"] / df["K"].max()
        df.to_csv(self.dossier / f"grille_k_iter_{iteration}.csv", index=False)

        # Départage : parmi les 4 meilleurs scores, on préfère un K plus grand
        # s'il réduit le rayon du pire cluster d'au moins gain_rayon_min.
        top = df.nsmallest(4, "score")
        choix = top.iloc[0]
        for _, ligne in top.sort_values("K").iterrows():
            if ligne["K"] > choix["K"] and (choix["rayon_p95"] - ligne["rayon_p95"]) / choix["rayon_p95"] >= c["gain_rayon_min"]:
                choix = ligne
        return int(choix["K"]), X_red

    # ------------------------------------------------------------- tout
    def executer(self, X: np.ndarray, iteration: int) -> dict:
        params = self.optimiser_umap(X, iteration)
        lam, t_moyen = self.profiler_llm()
        k, X_red = self.optimiser_k(X, params, lam, iteration)
        params.update({"k": k, "lambda": lam, "temps_appel_moyen_s": t_moyen, "X_reduit": X_red})
        return params


__all__ = ["OptimiseurIteration", "LLMIndisponible"]
