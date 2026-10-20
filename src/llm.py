"""Client LLM minimal (API compatible OpenAI : Ollama, vLLM, Groq...).

On distingue trois cas :
  - réponse JSON valide                  -> dict renvoyé
  - réponse reçue mais JSON illisible    -> ReponseInvalide
  - serveur injoignable après tentatives -> LLMIndisponible (le run s'arrête proprement)
"""

from __future__ import annotations

import json
import logging
import re
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

logger = logging.getLogger(__name__)

_MOTIF_JSON = re.compile(r"\{.*\}", re.DOTALL)


class LLMIndisponible(RuntimeError):
    """Le serveur LLM ne répond pas : continuer produirait des résultats faux."""


class ReponseInvalide(ValueError):
    """Le modèle a répondu, mais pas avec un objet JSON exploitable."""


def extraire_json(texte: str) -> dict:
    """Extrait le premier objet JSON d'une réponse de modèle."""
    m = _MOTIF_JSON.search(texte or "")
    if not m:
        raise ReponseInvalide("aucun objet JSON dans la réponse")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ReponseInvalide(f"JSON mal formé : {e}") from e
    if not isinstance(obj, dict):
        raise ReponseInvalide("la réponse JSON n'est pas un objet")
    return obj


class ClientLLM:
    def __init__(self, cfg_llm: dict):
        self.model = cfg_llm["model"]
        self.tentatives = int(cfg_llm.get("tentatives", 3))
        self.client = OpenAI(
            api_key=cfg_llm.get("api_key", "ollama"),
            base_url=cfg_llm["base_url"],
            timeout=float(cfg_llm.get("timeout_s", 300)),
            max_retries=0,
        )

    def brut(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Appel brut avec nouvelles tentatives sur erreurs réseau et 429/5xx."""
        derniere_erreur: Exception | None = None
        for essai in range(1, self.tentatives + 1):
            try:
                rep = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=temperature,
                )
                return rep.choices[0].message.content or ""
            except (APIConnectionError, APITimeoutError) as e:
                derniere_erreur = e
            except APIStatusError as e:
                if e.status_code == 429 or e.status_code >= 500:
                    derniere_erreur = e
                else:  # 400, 401, 404 (modèle absent)... : inutile de réessayer
                    raise LLMIndisponible(f"Erreur {e.status_code} du serveur LLM : {e}") from e
            attente = 5 * essai
            logger.warning("Appel LLM échoué (essai %d/%d) : %s. Nouvel essai dans %ds",
                           essai, self.tentatives, derniere_erreur, attente)
            time.sleep(attente)
        raise LLMIndisponible(f"Serveur LLM injoignable après {self.tentatives} essais : {derniere_erreur}")

    def json(self, system: str, user: str, temperature: float = 0.0) -> dict:
        return extraire_json(self.brut(system, user, temperature))
