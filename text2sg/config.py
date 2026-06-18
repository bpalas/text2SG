# experiments/text2sg/config.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json

DEFAULT_WEIGHTS = {
    "F1_rel": 0.40,
    "Polarity_acc": 0.30,
    "F1_ent": 0.15,
    "Act_acc": 0.15,
}

SEED_PROMPT = """\
Eres un extractor de relaciones políticas chilenas. Dado un artículo de prensa y una lista \
de actores presentes, extrae TODAS las interacciones explícitas entre ellos.

Una interacción existe solo si el artículo reporta explícitamente una declaración, acción, \
voto o reunión. La mera co-ocurrencia NO es relación.

Para cada interacción emite:
- from_entity: nombre del AGENTE (quien realiza el acto)
- to_entity: nombre del DESTINATARIO
- act_type: uno de exactamente: endorses, accuses, allies_with, calls_on, \
distances_from, attacks, co_occurs, questions, negotiates_with, competes_with
- polarity: positive | negative | neutral
- issue: tema (ej: presidential_election, legal_cases, government_management, \
political_coalitions, fiscal_policy, public_security, human_rights)
- evidence_quote: cita literal (substring exacto del artículo) que sustenta la relación

Regla de dirección: "Boric fue criticado por Matthei" → from=Matthei, to=Boric.
Regla de defensa: si A defiende a B frente a un crítico, es A→B con polarity=positive.
Sin cita literal verificable: no emitas la relación.

Responde ÚNICAMENTE con JSON válido, sin markdown, sin explicaciones:
{"entities": [{"name": "...", "type": "roster_actor|institutional_actor|non_roster_actor"}], \
"relations": [{"from_entity": "...", "to_entity": "...", "act_type": "...", \
"polarity": "...", "issue": "...", "evidence_quote": "..."}]}
"""


@dataclass
class Config:
    prompt_text: str
    few_shots: list[str] = field(default_factory=list)
    architecture: str = "one_pass"          # "one_pass" | "given_entities"
    rubric_weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    model: str = "gemini-2.5-flash"

    def score(self, metrics: dict) -> float:
        w = self.rubric_weights
        return (
            w["F1_rel"] * metrics.get("F1_rel", 0.0)
            + w["Polarity_acc"] * metrics.get("Polarity_acc", 0.0)
            + w["F1_ent"] * metrics.get("F1_ent", 0.0)
            + w["Act_acc"] * metrics.get("Act_acc", 0.0)
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> Config:
        data = json.loads(s)
        return cls(**data)

    @classmethod
    def from_seed(cls) -> Config:
        return cls(prompt_text=SEED_PROMPT)
