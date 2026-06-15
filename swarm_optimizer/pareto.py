"""Archivo de Pareto sobre (precisión, recall) — reemplaza el campeón-único/ELO.
Un archivo por tier de modelo. Determinista; sin dependencias del ELO.

Soporte multi-gradiente (Quality-Diversity, estilo MAP-Elites): además del frente global
(P,R), cada entrada puede guardar métricas por subconjunto (`subset_metrics`) y los
gradientes en los que mejoró a su padre (`gradient_tags`). Así conservamos al campeón de
cada gradiente aunque esté dominado globalmente, y exploramos hacia el gradiente menos
cubierto en vez de buscar un único máximo escalar."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import json
import random
from pathlib import Path

# Gradientes rastreados: <eje>_<métrica>. eje ∈ {directed, undirected}, métrica ∈ {P, R}.
# directed = scoring ordenado (a→b); undirected = par no ordenado {a,b} = el techo si la
# dirección fuera perfecta. El gap directed→undirected mide cuánto cuesta hoy la dirección.
# El parsing es genérico (rsplit por "_"); formal/informal se siguen guardando en
# subset_metrics como desglose diagnóstico, pero ya no son ejes de gradiente.
GRADIENTS = ("directed_P", "directed_R", "undirected_P", "undirected_R")


@dataclass
class ParetoEntry:
    id: int
    genome: dict
    P: float
    R: float
    parent_id: int | None = None
    expansions: int = 0
    preds_path: str | None = None
    split: str | None = None     # split/dataset bajo los que se puntuó (para regenerar diag igual)
    dataset: str = "v1"
    gradient_tags: list[str] = field(default_factory=list)   # gradientes en que superó al padre
    subset_metrics: dict = field(default_factory=dict)       # {subset: {"P":.., "R":..}}
    per_instance_scores: dict = field(default_factory=dict)  # {article_id: score} para win-count GEPA


def _dominates(a: ParetoEntry, b: ParetoEntry) -> bool:
    """a domina b si a >= b en P y R, y es estrictamente mejor en al menos uno."""
    return a.P >= b.P and a.R >= b.R and (a.P > b.P or a.R > b.R)


class ParetoArchive:
    def __init__(self, entries: list[ParetoEntry] | None = None, next_id: int = 0):
        self._entries: list[ParetoEntry] = list(entries or [])
        self._next_id = next_id

    def add(self, genome: dict, P: float, R: float, parent_id: int | None = None,
            preds_path: str | None = None, split: str | None = None,
            dataset: str = "v1", gradient_tags: list[str] | None = None,
            subset_metrics: dict | None = None,
            per_instance_scores: dict | None = None) -> ParetoEntry:
        e = ParetoEntry(id=self._next_id, genome=genome, P=P, R=R,
                        parent_id=parent_id, preds_path=preds_path,
                        split=split, dataset=dataset,
                        gradient_tags=list(gradient_tags or []),
                        subset_metrics=dict(subset_metrics or {}),
                        per_instance_scores=dict(per_instance_scores or {}))
        self._next_id += 1
        self._entries.append(e)
        return e

    @staticmethod
    def gradient_value(entry: ParetoEntry, gradient: str) -> float | None:
        """Valor de un gradiente '<subset>_<metric>' sobre una entrada, o None si falta."""
        subset, metric = gradient.rsplit("_", 1)
        return entry.subset_metrics.get(subset, {}).get(metric)

    def gradient_champion(self, gradient: str) -> ParetoEntry | None:
        """Entrada con el mayor valor en el gradiente (empate: menos expansiones, menor id)."""
        scored = [(self.gradient_value(e, gradient), e) for e in self._entries]
        scored = [(v, e) for v, e in scored if v is not None]
        if not scored:
            return None
        return max(scored, key=lambda ve: (ve[0], -ve[1].expansions, -ve[1].id))[1]

    def gradient_champions(self) -> dict[str, ParetoEntry]:
        """{gradiente: entrada campeona} para cada gradiente con datos."""
        out = {}
        for g in GRADIENTS:
            champ = self.gradient_champion(g)
            if champ is not None:
                out[g] = champ
        return out

    def all(self) -> list[ParetoEntry]:
        return list(self._entries)

    def frontier(self, include_champions: bool = False) -> list[ParetoEntry]:
        front = [e for e in self._entries
                 if not any(_dominates(o, e) for o in self._entries if o.id != e.id)]
        if include_champions:
            ids = {e.id for e in front}
            for champ in self.gradient_champions().values():
                if champ.id not in ids:
                    front.append(champ)
                    ids.add(champ.id)
        return front

    def gradient_coverage(self) -> dict[str, int]:
        """Cuántas entradas reclaman cada gradiente (vía gradient_tags)."""
        return {g: sum(g in e.gradient_tags for e in self._entries) for g in GRADIENTS}

    def pick_to_expand(self, prefer_gradient: str | None = None) -> ParetoEntry | None:
        """Elige el siguiente genoma a mutar.

        - prefer_gradient: devuelve el campeón de ese gradiente (empuja ese eje).
        - si hay tags en el archivo: empuja el gradiente MENOS cubierto (su campeón).
        - fallback: miembro del frente global menos expandido (comportamiento previo).
        """
        if prefer_gradient:
            champ = self.gradient_champion(prefer_gradient)
            if champ is not None:
                return champ
        front = self.frontier(include_champions=True)
        if not front:
            return None
        if any(e.gradient_tags for e in self._entries):
            cov = self.gradient_coverage()
            target = min(GRADIENTS, key=lambda g: (cov[g], g))
            champ = self.gradient_champion(target)
            if champ is not None:
                return champ
        return min(front, key=lambda e: (e.expansions, e.id))

    def win_counts(self) -> dict[int, int]:
        """Selección GEPA: para cada artículo, qué candidato tiene el mayor score por
        instancia → cuenta cuántos artículos 'gana' cada candidato. Preserva diversidad:
        un candidato mediocre en promedio pero el mejor en los artículos difíciles igual
        acumula wins. Empate de score → gana el id menor (determinista)."""
        counts = {e.id: 0 for e in self._entries}
        art_ids = {a for e in self._entries for a in e.per_instance_scores}
        for art_id in art_ids:
            best_score, best_id = -1.0, None
            for e in sorted(self._entries, key=lambda x: x.id):
                s = e.per_instance_scores.get(art_id, -1.0)
                if s > best_score:
                    best_score, best_id = s, e.id
            if best_id is not None:
                counts[best_id] += 1
        return counts

    def pick_to_expand_gepa(self, rng: random.Random | None = None) -> ParetoEntry | None:
        """Padre a mutar muestreado del frente con prob ∝ win-count por instancia (GEPA).
        Sin per_instance_scores (legacy) → uniforme sobre el frente. rng inyectable
        para reproducibilidad; el peso mínimo es 1 para no excluir a nadie del frente."""
        front = self.frontier(include_champions=True)
        if not front:
            return None
        rng = rng or random.Random()
        counts = self.win_counts()
        weights = [max(counts.get(e.id, 0), 1) for e in front]
        return rng.choices(front, weights=weights, k=1)[0]

    # ── merge por componente (GEPA proposer/merge.py) ──────────────────── #

    def _by_id(self) -> dict[int, ParetoEntry]:
        return {e.id: e for e in self._entries}

    def ancestors(self, entry_id: int) -> list[int]:
        """Cadena de ancestros (parent_id) desde el más cercano al más lejano."""
        by = self._by_id()
        chain, seen = [], set()
        cur = by[entry_id].parent_id if entry_id in by else None
        while cur is not None and cur not in seen and cur in by:
            seen.add(cur)
            chain.append(cur)
            cur = by[cur].parent_id
        return chain

    def lowest_common_ancestor(self, a_id: int, b_id: int) -> int | None:
        b_anc = set(self.ancestors(b_id))
        for anc in self.ancestors(a_id):   # del más cercano hacia arriba → primero = más bajo
            if anc in b_anc:
                return anc
        return None

    @staticmethod
    def _f05(P: float, R: float) -> float:
        return (1.25 * P * R / (0.25 * P + R)) if (0.25 * P + R) > 0 else 0.0

    def merge_pairs(self) -> list[tuple[ParetoEntry, ParetoEntry, ParetoEntry]]:
        """Pares (a, b, ancestro) elegibles para merge: comparten ancestro común, ninguno
        es ancestro del otro, y AMBOS le ganan al ancestro en f05. Ordenados por f05 combinado
        descendente (el mejor par primero)."""
        by = self._by_id()
        front = self.frontier()
        out = []
        for i, a in enumerate(front):
            for b in front[i + 1:]:
                if a.id in self.ancestors(b.id) or b.id in self.ancestors(a.id):
                    continue
                anc_id = self.lowest_common_ancestor(a.id, b.id)
                if anc_id is None or anc_id not in by:
                    continue
                anc = by[anc_id]
                fa, fb, fanc = self._f05(a.P, a.R), self._f05(b.P, b.R), self._f05(anc.P, anc.R)
                if fa > fanc and fb > fanc:
                    out.append((a, b, anc, fa + fb))
        out.sort(key=lambda t: t[3], reverse=True)
        return [(a, b, anc) for a, b, anc, _ in out]

    def mark_expanded(self, entry_id: int) -> None:
        for e in self._entries:
            if e.id == entry_id:
                e.expansions += 1
                return

    def to_json(self) -> str:
        return json.dumps({"next_id": self._next_id,
                           "entries": [asdict(e) for e in self._entries]},
                          ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, s: str) -> "ParetoArchive":
        d = json.loads(s)
        entries = [ParetoEntry(**e) for e in d.get("entries", [])]
        return cls(entries=entries, next_id=d.get("next_id", len(entries)))

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ParetoArchive":
        p = Path(path)
        return cls.from_json(p.read_text(encoding="utf-8")) if p.exists() else cls()
