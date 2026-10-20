# Découverte d'intentions par LLM-in-the-loop (MASSIVE, français)

Regrouper automatiquement 2 033 requêtes d'assistant vocal en intentions nommées (`ALARME_CREER`,
`METEO_QUERY`...), sans aucune étiquette, en combinant embeddings, clustering et un LLM qui
contrôle et nomme les clusters. Le projet adapte au français la méthode *Dial-In LLM*
(Hong et al., 2024) et la compare honnêtement à des baselines sans LLM.

**En une phrase :** sans connaître le nombre d'intentions, le tamis fait jeu égal avec un K-Means dont le
nombre de clusters est choisi automatiquement, mais il produit une taxonomie **nommée, plus fine et plus
pure**, et il mesure lui-même ses limites (évaluation du juge LLM contre la vérité terrain).

**Application de démonstration :** [llm-itl-intent-clustering-g3vgnyyd96mhjwp73trl4c.streamlit.app](https://llm-itl-intent-clustering.streamlit.app/)

**Projet de fin d'année, Master 2 MLSD, Université Paris Cité, 2025-2026.**
Équipe : Ibrahim Assebbane, Rayane Zekri, Johnathan Kancel, Jibril Ben Dhaou.
Encadrant : Imed Keraghel.
Dépôt d'origine de l'équipe : [Kaamek/LLM-in-the-Loop-Intent-Clustering](https://github.com/Kaamek/LLM-in-the-Loop-Intent-Clustering).

---

## Résultats

Run de référence : 20-21 mars 2026, Qwen 2.5 14B servi par Ollama, environ 8 h, 31 itérations.
Les intentions MASSIVE ne sont jamais vues par le pipeline : elles servent uniquement à l'évaluation.

| Indicateur | Résultat |
|---|---|
| Phrases étiquetées | 1 756 / 2 033 (**86,4 %**) ; 277 restent non classées |
| Tags créés | **79** (pour 59 intentions réelles) |

**Comparaison à des baselines sans LLM** (mêmes embeddings Gemini ; baselines : moyenne sur 3 graines),
évaluée sur les 1 756 phrases étiquetées par le pipeline. La colonne « K connu ? » indique ce que la
méthode sait du nombre d'intentions.

| Méthode | K connu ? | K obtenu | ARI | NMI | Pureté |
|---|---|---|---|---|---|
| UMAP 15D + K-Means, K choisi par Davies-Bouldin | non | 30-40 | **0,574** | 0,778 | 0,702 |
| **Tamis LLM** | non | 79 | 0,561 | 0,774 | 0,762 |
| UMAP 15D + K-Means, K choisi par silhouette | non | 20-30 | 0,532 | 0,745 | 0,586 |
| UMAP 15D + K-Means, K = 59 | oui (vrai nombre)* | 59 | 0,519 | **0,780** | 0,764 |
| UMAP 15D + K-Means, K = 79 | K repris du tamis | 79 | 0,453 | 0,775 | **0,798** |
| K-Means cosinus, K = 79 | K repris du tamis | 79 | 0,427 | 0,754 | 0,762 |
| Tamis LLM **sans** fusion des tags entre itérations | non | 117 | 0,375 | 0,762 | 0,816 |

\* information indisponible en pratique.

Sur le corpus complet (les 277 phrases restantes rattachées au tag le plus proche), le classement est le
même : ARI 0,529 (Davies-Bouldin), 0,506 (tamis), 0,479 (silhouette).
Détails : `evaluation/results/comparaison_baseline.csv`.

**Lecture honnête de ces chiffres**

- **En situation réelle (K inconnu), le tamis fait jeu égal avec K-Means.** Selon le critère utilisé pour
  choisir K, K-Means fait un peu mieux (Davies-Bouldin) ou un peu moins bien (silhouette, le plus courant)
  en ARI et NMI. Les écarts sont de l'ordre de la variance entre graines (écart-type de l'ARI : 0,03 à
  0,07). Attention : retenir après coup le meilleur critère avantagerait K-Means, puisqu'en pratique on
  choisit son critère avant de voir les résultats.
- **Ce que le tamis apporte :** des intentions **nommées** (`ALARME_SUPPRIMER`, `TRANSPORT_RESERVER`...),
  plus **fines** (79 groupes, quand les critères internes en retiennent 20 à 40 pour 59 intentions
  réelles) et plus **pures** que celles de K-Means à K choisi automatiquement (0,762 contre 0,702 et
  0,586). Il laisse aussi de côté les phrases trop ambiguës au lieu de les forcer dans un cluster.
- À **nombre de clusters égal** (79), le tamis gagne nettement en ARI (+0,11).
- Ce gain vient surtout du **nommeur** : en réutilisant un tag déjà créé, il fusionne des clusters
  de la même intention trouvés à des itérations différentes. Sans cette fusion, l'ARI tombe à 0,375 :
  les groupes sont plus purs (0,816), mais chaque intention est éclatée entre davantage de groupes (117).
- Un nommage par LLM peut aussi être appliqué **après** un K-Means (un appel par cluster). Cette baseline
  « K-Means + nommage LLM » n'a pas été testée : c'est la comparaison suivante à faire pour isoler l'apport
  de la boucle elle-même.
- MASSIVE est un jeu de données favorable aux embeddings (phrases courtes, domaines bien séparés). Le papier
  défend l'intérêt du LLM sur des corpus plus bruités : ce projet ne permet pas de conclure pour ce cas.
- **Plafond avec un juge parfait** (`evaluation/eval_juge_oracle.py`, le LLM remplacé par un oracle qui
  connaît les vraies intentions, grilles UMAP réduites) : 72,3 % des phrases étiquetées, 40 tags,
  ARI 0,826, NMI 0,895, pureté 0,889. Le LLM réel est donc plus permissif qu'un juge parfait
  (86,4 % de couverture) au prix de la qualité. Et même un juge parfait laisse passer 11 % de phrases
  mal classées : c'est la limite de l'heuristique « centroïde + 3 extrêmes ».
- Coût : environ 8 h de calcul local pour 2 033 phrases.

**Le juge LLM contre la vérité terrain** (`evaluation/eval_juge.py`, Qwen 2.5 14B servi par Ollama)

On soumet au juge des clusters de 4 phrases construits à partir des intentions MASSIVE : purs, pollués
« faciles » (un intrus d'un autre scénario) et pollués « difficiles » (un intrus du même scénario, autre
intention). Même règle de décision que dans le pipeline. 90 clusters par graine, deux graines.

| Mesure | v1, graine 0 | v1, graine 1 | v2, graine 0 | v2, graine 1 |
|---|---|---|---|---|
| Purs acceptés (rappel) | 12/30 | 8/30 | 8/30 | 3/30 |
| Pollués faciles rejetés | 29/30 | 29/30 | 30/30 | 30/30 |
| Pollués difficiles rejetés | 26/30 | 29/30 | 29/30 | 30/30 |
| Précision (clusters acceptés vraiment purs) | 0,71 | 0,80 | 0,89 | 1,00 |
| Bonnes décisions | 67/90 | 66/90 | 67/90 | 63/90 |

v1 = prompt du run ; v2 = `config/prompts_v2.json`. Environ 4,4 s par appel ; 3 réponses JSON invalides
sur 180 avec v1, aucune avec v2.

- Le juge du run (v1) repère bien les intrus (94 % des pollués rejetés) mais il est **trop sévère** : il
  rejette les deux tiers des clusters purs, souvent pour une simple différence de variable (deux villes,
  deux devises), alors que son prompt demande d'ignorer les variables. Avec v1, le raisonnement et la
  décision peuvent aussi se contredire (intrus signalé dans l'analyse, puis `reclusteriser: false`).
- Le **prompt v2** devait corriger ces deux défauts : règle des variables explicitée, verdict demandé phrase
  par phrase, décision déduite des verdicts par le code. Il supprime bien les contradictions, mais il rend
  le juge **encore plus sévère** : plus précis, moins de clusters purs acceptés, et pas meilleur au total.
  Demander un verdict par phrase donne trois occasions de dire non. **Résultat négatif, conservé comme tel.**
- Une partie du désaccord tient au découpage de référence : le modèle sépare *pause* et *reprise* d'un livre
  audio, *rappel* et *événement*, que MASSIVE regroupe, et MASSIVE contient du bruit d'étiquetage
  (« google traduction » étiqueté `lists_query`). Une règle abstraite ne transmet pas la bonne finesse au
  modèle : l'étape suivante serait de lui fournir la taxonomie de référence avec quelques exemples annotés.
- Sur ce test, le juge accepte moins de clusters que pendant le run (86 % de couverture). Hypothèse non
  vérifiée : les clusters de test sont tirés au hasard dans chaque intention, donc plus variés que les
  clusters compacts produits par K-Means.

**Analyse d'erreurs** (`evaluation/analyse_erreurs.py`)

- 23 tags sont purs à 100 %, mais 16 sont sous 60 % de pureté.
- Tags « fourre-tout » : `GENERAL_QUERY` (61 phrases, 28 % de pureté) mélange questions générales,
  questions factuelles et demandes d'heure.
- Règle non respectée : `LUMIERE_MODIFIER` mélange allumer, éteindre, tamiser et changer la couleur,
  alors que le prompt du juge interdit explicitement de regrouper des actions opposées. Le juge ne
  voit que 4 phrases par cluster (centroïde et 3 extrêmes) : les phrases intermédiaires échappent au
  contrôle.
- Découpages défendables : `calendar_set` (MASSIVE) est réparti entre `CALENDRIER_CREE`,
  `REMINDER_CREER` et `CALENDRIER_REMINDER`, ce qui distingue rappels et événements.
- Phrases non classées : surtout `general_quirky`, `calendar_query`, `calendar_remove`, `calendar_set` ;
  l'intention `iot_wemo_on` (7 phrases) n'est jamais étiquetée.

---

## Méthode

```text
phrases ─► embeddings Gemini (3 072 dim.)
              │
              ▼  à chaque itération, sur les phrases non encore étiquetées :
        ┌──────────────────────────────────────────────────────────────┐
        │ 1. UMAP : réglage par trustworthiness, choix des dimensions   │
        │ 2. K-Means : K = compromis inertie / coût d'appel du LLM (λ)  │
        │ 3. Juge LLM : centroïde + 3 phrases extrêmes -> pur / impur   │
        │ 4. Nommeur LLM : tag DOMAINE_ACTION, réutilise les tags connus│
        └──────────────────────────────────────────────────────────────┘
              │ clusters purs -> étiquetés      clusters impurs -> tour suivant
              ▼
        arrêt quand aucun cluster n'est accepté sur un tour complet
```

**Ce qui vient du papier** (Hong et al., algorithme 1) : la boucle itérative elle-même, qui garde les
clusters jugés cohérents par un LLM et re-regroupe le reste ; le nommage des intentions par LLM.

**Ce que nous avons changé**

| Aspect | Papier | Ce projet |
|---|---|---|
| Langue, données | Appels de service client en chinois, benchmarks anglais | MASSIVE fr-FR (2 033 phrases, 59 intentions) |
| Embeddings | BGE (`bge-large-zh-v1.5`) | Gemini `gemini-embedding-2-preview` |
| Choix de K | Le LLM évalue une partition par K candidat, on garde le meilleur ratio bons / mauvais | Sans appel LLM : `inertie(K)/inertie_max + λ·K/K_max`, λ mesuré à partir du temps de réponse du LLM ; départage par le rayon du pire cluster |
| Réduction de dimension | Aucune | UMAP réoptimisé à chaque itération |
| Entrée du juge | 20 phrases représentatives par cluster | Centroïde + 3 phrases les plus éloignées |
| LLM | Modèles fine-tunés pour juger et nommer | Qwen 2.5 14B sans fine-tuning (fine-tuning d'un 7B exploré ensuite) |
| Fusion des clusters | Après coup, par similarité des noms | Pendant le run : le nommeur reçoit la liste des tags existants |

À titre indicatif seulement, le papier rapporte une NMI de 0,781 sur MASSIVE anglais (table 5),
dans d'autres conditions (langue anglaise, protocole d'évaluation différent) : ce n'est pas une
comparaison directe.

**Fine-tuning (exploratoire)** : distillation des réponses de LLM « professeurs » vers un Qwen 2.5 7B
4 bits, avec LoRA et MLX. Le modèle fine-tuné n'a pas encore été évalué : voir `fine_tuning/LISEZMOI.md`.

---

## Limites

- Un seul run complet : pas de mesure de variance du pipeline LLM (les baselines, elles, sont moyennées
  sur 3 graines).
- Le test de pureté sur 4 phrases est une heuristique : un cluster peut être accepté avec des phrases
  intermédiaires mal classées (voir `LUMIERE_MODIFIER`).
- Le seuil de confiance (80) et le format des tags ont été fixés à la main.
- Les métriques « phrases retenues » portent sur les 86,4 % que le pipeline a acceptés : c'est pourquoi
  les baselines sont évaluées sur exactement les mêmes phrases, et le corpus complet est aussi rapporté.
- Les données de fine-tuning ont été annotées par des LLM, pas par des humains.

---

## Application de présentation des résultats

**En ligne :** [llm-itl-intent-clustering-g3vgnyyd96mhjwp73trl4c.streamlit.app](https://llm-itl-intent-clustering-g3vgnyyd96mhjwp73trl4c.streamlit.app/)

`app/streamlit_app.py` : résultats contre les baselines, exploration des 79 intentions créées (et de leur
contenu réel), évaluation du juge cas par cas, déroulé du run. Elle ne lit que les fichiers du dépôt :
aucun LLM, aucun calcul lourd.

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Déployée sur Streamlit Community Cloud, avec `app/streamlit_app.py` comme fichier principal.

---

## Reproduire

```bash
git clone https://github.com/Johnathank78/llm-itl-intent-clustering.git && cd llm-itl-intent-clustering
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

**Évaluation (sans LLM, quelques minutes)** : les embeddings et le résultat du run de mars 2026 sont
fournis.

```bash
python evaluation/eval_baseline.py      # tableau de comparaison ci-dessus
python evaluation/analyse_erreurs.py    # pureté par tag, intentions éclatées, résidu
```

**Pipeline complet** : nécessite un serveur LLM compatible OpenAI, par défaut [Ollama](https://ollama.com)
avec `ollama pull qwen2.5:14b`.

```bash
cp .env.example .env              # facultatif : changer de modèle ou de serveur
python main.py                    # nouveau run dans outputs/run_courant/
python main.py --reprendre        # reprendre après une interruption
```

**Évaluer le juge LLM** contre la vérité terrain (clusters purs, pollués « faciles » et « difficiles ») :

```bash
python evaluation/eval_juge.py --n 30 --modele qwen2.5:14b
python evaluation/eval_juge.py --n 30 --modele qwen2.5:14b --prompts config/prompts_v2.json
python evaluation/eval_juge.py --n 30 --modele qwen2.5:14b --prompts config/prompts_v2.json --graine 1
python evaluation/eval_juge_oracle.py --rapide   # plafond de la méthode avec un juge parfait
```

---

## Structure

```text
config/        paramètres (config.json) et prompts du juge et du nommeur (prompts.json)
data/          MASSIVE fr-FR (partition dev), étiquettes de référence, embeddings Gemini (float32)
src/           pipeline : step1_donnees, step2_optimisation, step3_tamis, client LLM
evaluation/    baselines, analyse d'erreurs, évaluation du juge, plafond « juge oracle » ; résultats
app/           application Streamlit de présentation des résultats
fine_tuning/   données et configurations LoRA (MLX), exploratoire
outputs/       run_mars_2026 : résultat de référence (mapping final, taxonomie, paramètres par itération)
```

## Données, licence et références

- Données : MASSIVE (Amazon Science), FitzGerald et al., 2022, partition fr-FR, publiée sous licence
  CC BY 4.0.
- Méthode : Hong, Ng, Zhang, Song, Jiang. *Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering
  for Customer Service Dialogues*. [arXiv:2412.09049](https://arxiv.org/abs/2412.09049).
- Licence du code : à définir par l'équipe.
