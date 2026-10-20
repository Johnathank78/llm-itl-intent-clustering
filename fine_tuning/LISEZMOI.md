# Fine-tuning LoRA du juge et du nommeur (exploratoire)

Objectif : remplacer Qwen 2.5 14B (utilisé pendant le run) par un Qwen 2.5 7B quantifié en 4 bits,
spécialisé par LoRA, plus rapide sur un portable. **Ce volet est exploratoire : le modèle fine-tuné
n'a pas encore été évalué** (voir « Évaluer » plus bas).

## Données

| Fichier | Contenu |
|---|---|
| `data/selection/finetuning_easy.jsonl` | 1 193 phrases du run jugées bien classées |
| `data/selection/finetuning_hard_to_annotate.jsonl` | 549 phrases jugées douteuses, à faire annoter |
| `data/dataset_for_juge.json` | 271 clusters annotés pour le juge (décision + raisonnement) |
| `data/dataset_for_nommeur.json` | 220 clusters annotés pour le nommeur (tag + justification), dont 214 répartis dans les splits |
| `data/juge/{train,valid,test}.jsonl` | 216 / 26 / 29 exemples au format `messages` |
| `data/nommeur/{train,valid,test}.jsonl` | 170 / 21 / 23 exemples au format `messages` |

La séparation « faciles / douteuses » a été faite avec un modèle de mélange gaussien (GMM) ajusté
autour des tags finaux ; le script correspondant n'a pas été conservé. Le cosinus au centroïde
présent dans ces fichiers ne suffit pas à lui seul à séparer les deux groupes (médianes 0,83 et 0,81).

Les annotations ont été produites par des LLM « professeurs » (Llama 70B, Gemini, Claude), **pas par
des humains** : c'est de la distillation. Aucun prompt du jeu de test n'apparaît dans le jeu
d'entraînement (vérifié). Les phrases viennent en revanche du même corpus MASSIVE que celui que le
pipeline regroupe : le modèle fine-tuné ne doit pas être présenté comme évalué « hors domaine ».

Petit écart à connaître : dans les données d'entraînement, les extrêmes sont listés au format
`- phrase`, alors que le pipeline envoie `- Extrême 1 : "phrase"`.

## Entraîner (Mac Apple Silicon)

```bash
pip install -r fine_tuning/requirements-mlx.txt
# Placer le modèle de base Qwen 2.5 7B Instruct 4 bits au format MLX dans ./qwen-base-4bit
mlx_lm.lora --config fine_tuning/lora_config_juge.yaml
mlx_lm.lora --config fine_tuning/lora_config_nommeur.yaml
```

Hyperparamètres : LoRA rang 16, 8 couches adaptées, taux d'apprentissage 1e-4, batch 1,
600 itérations (juge) et 500 (nommeur), perte calculée uniquement sur la réponse (`mask_prompt`).

## Évaluer (à faire)

Servir le modèle de base et le modèle fine-tuné (par exemple dans Ollama), puis :

```bash
python evaluation/eval_finetune.py --modeles qwen2.5:7b <modele-fine-tune> --tache juge
python evaluation/eval_juge.py --n 30 --modele <modele-fine-tune>
```

Le premier script mesure l'accord avec le professeur, le second la qualité contre la vérité terrain.
`Modelfile.base` charge le modèle de base GGUF dans Ollama, **sans adaptateur** : pour servir la
version fine-tunée, il faut d'abord fusionner l'adaptateur (`mlx_lm.fuse`) puis convertir le
résultat dans un format qu'Ollama accepte. Les commandes exactes dépendent des versions de mlx_lm et
d'Ollama : à vérifier dans leur documentation.
