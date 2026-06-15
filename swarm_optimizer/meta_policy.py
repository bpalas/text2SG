"""Meta-agente de evolución (Capa 3): bandit contextual Thompson Sampling.

Elige (operador, sesgo de padre) en cada iteración. Aprende de fitness_delta
(recompensa densa ya persistida en el archivo) + crédito retrospectivo de
championship. Sin LLM, puro Python (informe 2026-06-09 §2.2).

Garantía anti-degeneración: ε-floor — cada brazo conserva probabilidad mínima
de ser elegido (equivalente funcional de los stepping stones de DGM en el
espacio de operadores).
"""
from __future__ import annotations

import json
from pathlib import Path

OPERATORS = ("diff_a", "diff_b", "cross", "fresh")
PARENT_BIASES = ("exploit", "explore")
ARMS = [(op, bias) for op in OPERATORS for bias in PARENT_BIASES]

EPSILON_FLOOR = 0.05          # prob. mínima por brazo (×n_brazos = prob. de uniforme)
REWARD_CLIP = 0.2             # clip de fitness_delta a [−0.2, +0.2]
CREDIT_ALPHA = 0.5            # peso del crédito retrospectivo de championship


def _arm_key(arm: tuple[str, str]) -> str:
    return f"{arm[0]}|{arm[1]}"


def reward_from_delta(delta: float, clip: float = REWARD_CLIP) -> float:
    """Mapea fitness_delta ∈ [−clip, +clip] a recompensa fraccional ∈ [0, 1]."""
    d = max(-clip, min(clip, delta))
    return (d + clip) / (2 * clip)


class MetaPolicy:
    """Thompson Sampling con posteriors Beta y updates fraccionales."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self._alpha: dict[str, float] = {_arm_key(a): 1.0 for a in ARMS}
        self._beta: dict[str, float] = {_arm_key(a): 1.0 for a in ARMS}
        if self.path and self.path.exists():
            self._load()

    # ── decisión ──────────────────────────────────────────────────── #
    def choose(self, rng, available_ops: list[str] | None = None) -> tuple[str, str]:
        """Elige (operador, sesgo_padre). rng = np.random.Generator.

        available_ops restringe los operadores válidos (p.ej. cross/fresh
        requieren ≥2 entradas en el archivo).
        """
        ops = available_ops or list(OPERATORS)
        arms = [a for a in ARMS if a[0] in ops]
        # ε-floor: con prob ε·n_brazos, elegir uniforme
        if rng.random() < EPSILON_FLOOR * len(arms):
            return arms[int(rng.integers(len(arms)))]
        # Thompson: muestrear de cada posterior Beta y tomar argmax
        best, best_sample = arms[0], -1.0
        for arm in arms:
            k = _arm_key(arm)
            sample = rng.beta(self._alpha[k], self._beta[k])
            if sample > best_sample:
                best, best_sample = arm, sample
        return best

    # ── aprendizaje ───────────────────────────────────────────────── #
    def update(self, arm: tuple[str, str], reward: float) -> None:
        """Update fraccional del posterior Beta. reward ∈ [0, 1]."""
        r = max(0.0, min(1.0, reward))
        k = _arm_key(arm)
        self._alpha[k] += r
        self._beta[k] += 1.0 - r
        self._persist()

    def update_from_delta(self, arm: tuple[str, str], fitness_delta: float) -> None:
        self.update(arm, reward_from_delta(fitness_delta))

    def credit_championship(self, window_arms: list[tuple[str, str]],
                            score_delta: float, alpha: float = CREDIT_ALPHA) -> None:
        """Crédito retrospectivo: reparte el delta de championship_score a las
        acciones de la ventana (corrige el ruido del skirmish con la métrica anclada)."""
        if not window_arms:
            return
        bonus = max(0.0, min(1.0, 0.5 + alpha * score_delta / REWARD_CLIP))
        for arm in window_arms:
            r = bonus
            k = _arm_key(arm)
            self._alpha[k] += r
            self._beta[k] += 1.0 - r
        self._persist()

    # ── introspección ─────────────────────────────────────────────── #
    def posterior_means(self) -> dict[str, float]:
        return {
            k: self._alpha[k] / (self._alpha[k] + self._beta[k])
            for k in self._alpha
        }

    # ── persistencia ──────────────────────────────────────────────── #
    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"alpha": self._alpha, "beta": self._beta}, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for k, v in data.get("alpha", {}).items():
            if k in self._alpha:
                self._alpha[k] = float(v)
        for k, v in data.get("beta", {}).items():
            if k in self._beta:
                self._beta[k] = float(v)
