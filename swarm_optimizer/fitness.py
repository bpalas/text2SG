"""Fitness precision-weighted (F0.5) + piso de recall + penalización de costo."""
from __future__ import annotations

DEFAULT_FITNESS_WEIGHTS = {"rel": 0.45, "ent": 0.25, "pol": 0.15, "act": 0.15}
RECALL_FLOOR = 0.15          # recall_rel mínimo para ser campeón
COST_LAMBDA = 0.10           # peso de la penalización de costo
COST_REF_TOKENS = 4000       # tokens/artículo de referencia (modelo barato) para normalizar
COST_CAP = 2.0               # tope del costo normalizado (deja gradiente por encima de Flash)
DISQUALIFY_PENALTY = 1.0     # se resta si recall_rel < piso

# Multiplicador de precio relativo POR TOKEN respecto al modelo barato (Flash = 1.0).
# Hace que el cost penalty mida COSTO (precio), no solo cantidad de tokens. Aprox., tunable.
PRICE_MULT = {
    "gemini-2.5-flash": 1.0,
    "gemini-2.5-pro": 16.0,
    # Claude (aprox. precio por token vs Flash; tunable):
    "claude-haiku-4-5": 3.0,
    "claude-sonnet-4-6": 9.0,
}
DEFAULT_PRICE_MULT = 1.0


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


# Objetivo de selección precision-first (función de pérdida del usuario:
# "prefiero perderme una relación a emitir una falsa", sin colapsar a no-emitir-nada).
PRECISION_FIRST_FLOOR = 0.80     # piso de recall DURO; bajo esto, descalificado
DISTRACTOR_FP_PENALTY = 0.02     # cada relación inventada en un distractor (alucinación pura)


def selection_score(precision: float, recall: float, distractor_fp: int = 0,
                    beta: float = 0.5, recall_floor: float = PRECISION_FIRST_FLOOR,
                    distractor_penalty: float = DISTRACTOR_FP_PENALTY) -> float:
    """f_beta inclinada a precisión, con piso de recall DURO y castigo a alucinaciones.

    - recall ≥ piso: score = f_beta(P,R,β) − penalización·alucinaciones  ∈ banda positiva.
    - recall < piso: banda negativa (−1+recall), ordenada por recall pero SIEMPRE bajo los
      válidos. Así el piso descalifica de verdad (evita el colapso id4/id5: P alta, R 0.45)
      en vez de penalizar apenas. β=0.5 → precisión 2:1; β=0.33 → 3:1."""
    penalty = distractor_penalty * max(distractor_fp, 0)
    if recall < recall_floor:
        return -1.0 + recall - penalty
    return f_beta(precision, recall, beta) - penalty


def fitness(
    metrics: dict,
    tokens_per_article: float,
    model: str = "gemini-2.5-flash",
    weights: dict | None = None,
    recall_floor: float = RECALL_FLOOR,
    cost_lambda: float = COST_LAMBDA,
) -> float:
    w = weights or DEFAULT_FITNESS_WEIGHTS
    f05_rel = f_beta(metrics.get("Precision_rel", 0.0), metrics.get("Recall_rel", 0.0), 0.5)
    f05_ent = f_beta(metrics.get("Precision_ent", 0.0), metrics.get("Recall_ent", 0.0), 0.5)

    quality = (
        w["rel"] * f05_rel
        + w["ent"] * f05_ent
        + w["pol"] * metrics.get("Polarity_acc", 0.0)
        + w["act"] * metrics.get("Act_acc", 0.0)
    )

    # Costo = tokens × precio relativo del modelo (price-aware), normalizado y con tope.
    price = PRICE_MULT.get(model, DEFAULT_PRICE_MULT)
    cost_norm = min(tokens_per_article * price / COST_REF_TOKENS, COST_CAP)
    score = quality - cost_lambda * cost_norm

    # Penalización de recall GRADUADA: bajo el piso, peor mientras más lejos del piso,
    # pero el recall sigue importando entre genomas descalificados. Evita que, cuando TODOS
    # están bajo el piso, un -1.0 plano se cancele y el ruido de métricas secundarias
    # invierta la selección (hallazgo del smoke test del 2026-06-09).
    rr = metrics.get("Recall_rel", 0.0)
    if rr < recall_floor:
        score -= DISQUALIFY_PENALTY * (1.0 - rr / recall_floor)

    return score
