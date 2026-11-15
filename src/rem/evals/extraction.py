"""Extraction eval: run a real LLM over a labelled dataset and score what it extracts.

Entirely in-process. Every case gets a fresh in-memory Memory over a temporary data dir, so
the eval never touches the user's real store and never needs the daemon.
"""

import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rem.config import RemConfig, VectorStoreConfig
from rem.embed.base import Embedder
from rem.errors import LLMError
from rem.extract import extract
from rem.llm.base import LLM
from rem.memory import Memory
from rem.models import Provenance, Scope, Status
from rem.storage.queue import Turn
from rem.storage.vector.memory import InMemoryVectorStore

DEFAULT_DATASET = Path(__file__).parent / "data" / "extraction.jsonl"

_SEED_TRUST = {
    "verified": (Provenance.USER_ASSERTED, Status.VERIFIED),
    "unverified": (Provenance.ASSISTANT_GENERATED, Status.UNVERIFIED),
}


@dataclass
class EvalReport:
    cases: int
    precision: float
    recall: float
    scope_acc: float
    origin_acc: float
    json_valid_rate: float
    dedup_acc: float
    latency_p50: float
    latency_p95: float
    failures: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_cases(path: Path = DEFAULT_DATASET) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def match_facts(
    predicted: list[str], expected: list[str], embedder: Embedder, threshold: float = 0.80
) -> list[tuple[int, int]]:
    """Greedy one-to-one (predicted index, expected index) matching by cosine, highest first."""
    if not predicted or not expected:
        return []
    pv, ev = embedder.embed(predicted), embedder.embed(expected)
    pairs = sorted(
        ((_cosine(p, e), i, j) for i, p in enumerate(pv) for j, e in enumerate(ev)), reverse=True
    )
    used_p: set[int] = set()
    used_e: set[int] = set()
    matched: list[tuple[int, int]] = []
    for score, i, j in pairs:
        if score < threshold:
            break
        if i in used_p or j in used_e:
            continue
        used_p.add(i)
        used_e.add(j)
        matched.append((i, j))
    return sorted(matched)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def _ratio(num: int, den: int) -> float:
    return num / den if den else 1.0


def run_eval(llm: LLM, embedder: Embedder, cases: list[dict], cfg: RemConfig) -> EvalReport:
    settings = cfg.extraction
    predicted_total = expected_total = matched_total = 0
    scope_ok = origin_ok = 0
    valid = 0
    dedup_cases = dedup_ok = 0
    latencies: list[float] = []
    failures: list[dict] = []

    for case in cases:
        with tempfile.TemporaryDirectory(prefix="rem-eval-") as tmp:
            case_cfg = cfg.model_copy(
                update={"data_dir": Path(tmp), "vector_store": VectorStoreConfig(provider="memory")}
            )
            mem = Memory(case_cfg, vector_store=InMemoryVectorStore(), embedder=embedder)
            try:
                project_id = mem.db.upsert_project(tmp, "evalproj", None)
                seeded: list[tuple[str, str]] = []  # (memory id, text)
                for e in case.get("existing", []):
                    provenance, status = _SEED_TRUST[e["status"]]
                    rec = mem.add(
                        e["text"], scope=Scope(e["scope"]), provenance=provenance, status=status,
                        source="eval", actor="eval", project_id=project_id,
                    )
                    seeded.append((rec.id, rec.text))

                turns = [Turn(t["role"], t["content"]) for t in case["turns"]]
                context = [Turn(t["role"], t["content"]) for t in case.get("context", [])]
                related = mem.related("\n".join(t.content for t in turns), project_id, settings.related_limit)

                started = time.perf_counter()
                try:
                    extraction = extract(llm, turns, context, related, settings.max_candidates)
                except LLMError as e:
                    latencies.append(time.perf_counter() - started)
                    expected_total += len(case["expected"])
                    if case.get("expect_corroborate"):
                        dedup_cases += 1
                    failures.append({"id": case["id"], "error": str(e)})
                    continue
                latencies.append(time.perf_counter() - started)
                valid += 1

                candidates = extraction.candidates
                expected = case["expected"]
                pred_texts = [c.text for c in candidates]
                pairs = match_facts(pred_texts, [x["text"] for x in expected], embedder)
                predicted_total += len(candidates)
                expected_total += len(expected)
                matched_total += len(pairs)
                wrong_scope, wrong_origin = [], []
                for i, j in pairs:
                    if str(candidates[i].scope) == expected[j]["scope"]:
                        scope_ok += 1
                    else:
                        wrong_scope.append({"text": candidates[i].text, "got": str(candidates[i].scope),
                                            "want": expected[j]["scope"]})
                    if candidates[i].origin == expected[j]["origin"]:
                        origin_ok += 1
                    else:
                        wrong_origin.append({"text": candidates[i].text, "got": candidates[i].origin,
                                             "want": expected[j]["origin"]})

                problems: dict = {}
                extra = [pred_texts[i] for i in range(len(pred_texts)) if i not in {p for p, _ in pairs}]
                missing = [expected[j]["text"] for j in range(len(expected)) if j not in {q for _, q in pairs}]
                if extra:
                    problems["extra"] = extra
                if missing:
                    problems["missing"] = missing
                if wrong_scope:
                    problems["wrong_scope"] = wrong_scope
                if wrong_origin:
                    problems["wrong_origin"] = wrong_origin

                result = mem.apply_candidates(
                    candidates, job_key=f"job:eval:{case['id']}", project_id=project_id,
                    session_id=None, source="eval", threshold=settings.dedup_threshold,
                )
                if case.get("expect_corroborate"):
                    dedup_cases += 1
                    added_texts = [t for t in (getattr(mem.get(i), "text", None) for i in result.added) if t]
                    bad: list[str] = []
                    for text in case["expect_corroborate"]:
                        hit = match_facts([text], [t for _, t in seeded], embedder)
                        sid = seeded[hit[0][1]][0] if hit else None
                        if sid is None or sid not in result.corroborated or match_facts([text], added_texts, embedder):
                            bad.append(text)
                    if bad:
                        problems["not_corroborated"] = bad
                    else:
                        dedup_ok += 1
                if problems:
                    failures.append({"id": case["id"], **problems})
            finally:
                mem.close()

    return EvalReport(
        cases=len(cases),
        precision=_ratio(matched_total, predicted_total),
        recall=_ratio(matched_total, expected_total),
        scope_acc=_ratio(scope_ok, matched_total),
        origin_acc=_ratio(origin_ok, matched_total),
        json_valid_rate=_ratio(valid, len(cases)),
        dedup_acc=_ratio(dedup_ok, dedup_cases),
        latency_p50=_percentile(latencies, 0.50),
        latency_p95=_percentile(latencies, 0.95),
        failures=failures,
    )
