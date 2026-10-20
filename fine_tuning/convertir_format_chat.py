"""Convertit les jeux de fine-tuning du format texte ChatML vers le format « messages » de mlx_lm.

Les fichiers de fine_tuning/data/{juge,nommeur} sont DÉJÀ convertis : ce script est conservé pour
traçabilité et ne modifie pas un fichier déjà au bon format (il est donc sans risque à relancer).

Format d'entrée  : {"text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"}
Format de sortie : {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent / "data"
MOTIF = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.DOTALL)


def convertir(fichier: Path) -> None:
    lignes = [json.loads(l) for l in fichier.read_text(encoding="utf-8").splitlines() if l.strip()]
    if all("messages" in l for l in lignes):
        print(f"{fichier} : déjà au format messages ({len(lignes)} exemples), inchangé")
        return
    sortie = []
    for l in lignes:
        segments = MOTIF.findall(l["text"])
        if len(segments) != 3:
            print(f"  exemple ignoré : {len(segments)} segments au lieu de 3")
            continue
        sortie.append({"messages": [{"role": r, "content": c.strip()} for r, c in segments]})
    fichier.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in sortie) + "\n", encoding="utf-8")
    print(f"{fichier} : {len(sortie)} exemples convertis")


if __name__ == "__main__":
    for tache in ["juge", "nommeur"]:
        for partition in ["train", "valid", "test"]:
            f = DOSSIER / tache / f"{partition}.jsonl"
            if f.exists():
                convertir(f)
