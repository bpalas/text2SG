"""Archivo evolutivo open-ended (DGM): conserva todos los genomas con ELO + linaje."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from swarm_optimizer.elo import ELO_BASE, sample_parent
from swarm_optimizer.genome import Genome


@dataclass
class ArchiveEntry:
    id: int
    genome: Genome
    elo: float = ELO_BASE
    children: int = 0
    parent_id: int | None = None
    mutation_type: str | None = None        # "seed"|"diff_a"|"diff_b"|"cross"
    artifact_touched: str | None = None      # "A"|"B"|None
    championship_score: float | None = None
    metrics: dict = field(default_factory=dict)
    diagnosis: str | None = None          # diagnóstico que originó esta mutación (memoria v2)
    fitness_delta: float | None = None    # margen vs campeón en el skirmish (memoria v2)
    meta_review: str | None = None        # síntesis transversal del meta-revisor (AI Co-Scientist)


class Archive:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[int, ArchiveEntry] = {}
        self._next_id = 0
        if self.path.exists():
            self._reload()

    # ── construcción ──────────────────────────────────────────────── #
    def add(self, genome: Genome, parent_id: int | None = None,
            mutation_type: str = "seed", artifact_touched: str | None = None,
            elo: float = ELO_BASE, diagnosis: str | None = None) -> int:
        eid = self._next_id
        self._next_id += 1
        entry = ArchiveEntry(
            id=eid, genome=genome, elo=elo, parent_id=parent_id,
            mutation_type=mutation_type, artifact_touched=artifact_touched,
            diagnosis=diagnosis,
        )
        self._entries[eid] = entry
        if parent_id is not None and parent_id in self._entries:
            self._entries[parent_id].children += 1
        self._persist(entry)
        return eid

    # ── updates ───────────────────────────────────────────────────── #
    def record_elo(self, eid: int, new_elo: float) -> None:
        self._entries[eid].elo = new_elo
        self._persist(self._entries[eid])

    def record_championship(self, eid: int, score: float, metrics: dict) -> None:
        e = self._entries[eid]
        e.championship_score = score
        e.metrics = metrics
        self._persist(e)

    def record_delta(self, eid: int, fitness_delta: float) -> None:
        self._entries[eid].fitness_delta = fitness_delta
        self._persist(self._entries[eid])

    def record_meta_review(self, eid: int, text: str) -> None:
        self._entries[eid].meta_review = text
        self._persist(self._entries[eid])

    # ── consultas ─────────────────────────────────────────────────── #
    def get(self, eid: int) -> ArchiveEntry:
        return self._entries[eid]

    def all(self) -> list[ArchiveEntry]:
        return list(self._entries.values())

    def select_parent(self, rng) -> int:
        entries = self.all()
        idx = sample_parent(
            [{"elo": e.elo, "children": e.children} for e in entries], rng
        )
        return entries[idx].id

    def top_by_elo(self, n: int) -> list[ArchiveEntry]:
        return sorted(self.all(), key=lambda e: e.elo, reverse=True)[:n]

    def champion(self) -> ArchiveEntry | None:
        scored = [e for e in self.all() if e.championship_score is not None]
        if not scored:
            return None
        return max(scored, key=lambda e: e.championship_score)

    def total_tokens(self) -> int:
        return sum(int(e.metrics.get("tokens", 0)) for e in self.all())

    def lineage(self, eid: int) -> list[ArchiveEntry]:
        """Cadena ancestral desde eid hasta la semilla (incluye eid)."""
        chain, cur = [], self._entries.get(eid)
        seen: set[int] = set()
        while cur is not None and cur.id not in seen:
            chain.append(cur)
            seen.add(cur.id)
            cur = self._entries.get(cur.parent_id) if cur.parent_id is not None else None
        return chain

    def memory_snippets(self, eid: int, n: int = 3) -> str:
        """Memoria retrospectiva v1 (MLEvolve): top-n mutaciones con mayor delta
        positivo y top-n con delta más negativo del linaje de eid, formateadas
        para inyectar en el prompt de propose()."""
        tried = [e for e in self.lineage(eid) if e.fitness_delta is not None]
        if not tried:
            return ""
        ordered = sorted(tried, key=lambda e: e.fitness_delta, reverse=True)
        best = [e for e in ordered[:n] if e.fitness_delta > 0]
        worst = [e for e in ordered[-n:] if e.fitness_delta < 0]

        def _fmt(e: ArchiveEntry) -> str:
            diag = (e.diagnosis or "").strip().replace("\n", " ")[:120]
            extra = f" | diagnóstico: {diag}" if diag else ""
            return (f"- {e.mutation_type} sobre artefacto {e.artifact_touched or '?'}: "
                    f"delta {e.fitness_delta:+.3f}{extra}")

        parts = []
        if best:
            parts.append("Funcionó:\n" + "\n".join(_fmt(e) for e in best))
        if worst:
            parts.append("NO funcionó:\n" + "\n".join(_fmt(e) for e in worst))
        return "\n".join(parts)

    # ── persistencia ──────────────────────────────────────────────── #
    def _persist(self, entry: ArchiveEntry) -> None:
        rec = {
            "id": entry.id,
            "genome": entry.genome.to_dict(),
            "elo": entry.elo,
            "children": entry.children,
            "parent_id": entry.parent_id,
            "mutation_type": entry.mutation_type,
            "artifact_touched": entry.artifact_touched,
            "championship_score": entry.championship_score,
            "metrics": entry.metrics,
            "diagnosis": entry.diagnosis,
            "fitness_delta": entry.fitness_delta,
            "meta_review": entry.meta_review,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _reload(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            entry = ArchiveEntry(
                id=rec["id"],
                genome=Genome.from_dict(rec["genome"]),
                elo=rec["elo"],
                children=rec["children"],
                parent_id=rec["parent_id"],
                mutation_type=rec["mutation_type"],
                artifact_touched=rec["artifact_touched"],
                championship_score=rec["championship_score"],
                metrics=rec.get("metrics", {}),
                diagnosis=rec.get("diagnosis"),
                fitness_delta=rec.get("fitness_delta"),
                meta_review=rec.get("meta_review"),
            )
            self._entries[entry.id] = entry      # última línea gana (updates)
            self._next_id = max(self._next_id, entry.id + 1)
