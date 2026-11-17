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
from rem.entities import normalize_name
from rem.embed.base import Embedder
from rem.errors import LLMError
from rem.extract import extract
from rem.llm.base import LLM
from rem.memory import Memory
from rem.models import ENTITY_TYPES, Candidate, Provenance, Scope, Status
from rem.storage.queue import Turn
from rem.storage.vector.memory import InMemoryVectorStore

DEFAULT_DATASET = Path(__file__).parent / "data" / "extraction.jsonl"

_SEED_TRUST = {
    "verified": (Provenance.USER_ASSERTED, Status.VERIFIED),
    "unverified": (Provenance.ASSISTANT_GENERATED, Status.UNVERIFIED),
}


@dataclass
class EvalReport:
    # A rate is None when there was nothing to measure it on (e.g. no dedup cases).
    cases: int
    precision: float | None
    recall: float | None
    scope_acc: float | None
    origin_acc: float | None
    json_valid_rate: float | None
    dedup_acc: float | None
    latency_p50: float
    latency_p95: float
    entity_precision: float | None = None
    entity_recall: float | None = None
    entity_type_acc: float | None = None
    invalid_type_rate: float | None = None
    relation_precision: float | None = None
    relation_recall: float | None = None
    relation_type_acc: float | None = None
    labelled_cases: int = 0
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


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def is_labelled(case: dict) -> bool:
    """A case is labelled iff every expected fact has an `entities` key (an empty `expected` counts)."""
    return all("entities" in x for x in case["expected"])


@dataclass
class LabelScore:
    ent_pred: int = 0
    ent_exp: int = 0
    ent_match: int = 0
    type_mentions: int = 0
    type_ok: int = 0
    rel_pred: int = 0
    rel_exp: int = 0
    rel_match: int = 0
    rel_type_ok: int = 0
    problems: dict = field(default_factory=dict)


def _expected_entities(expected: list[dict]) -> list[dict]:
    """One entry per label across all expected facts, de-duplicated by normalised name."""
    out: dict[str, dict] = {}
    for fact in expected:
        for e in fact.get("entities", []):
            norm = normalize_name(e["name"])
            if norm not in out:
                keys = {norm} | {normalize_name(a) for a in e.get("aliases", [])}
                out[norm] = {"name": e["name"], "norm": norm, "type": e["type"], "keys": keys}
    return list(out.values())


def _expected_relations(expected: list[dict]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for fact in expected:
        for r in fact.get("relations", []):
            rel = (normalize_name(r["source"]), normalize_name(r["target"]), r["type"])
            if rel not in out:
                out.append(rel)
    return out


def invalid_type_mentions(candidates: list[Candidate]) -> tuple[int, int]:
    """(mentions whose type is not a known entity type, all mentions)."""
    mentions = [m for c in candidates for m in c.entities]
    return sum(1 for m in mentions if m.type not in ENTITY_TYPES), len(mentions)


def score_labels(candidates: list[Candidate], expected: list[dict]) -> LabelScore:
    exp_ents = _expected_entities(expected)
    mentions = [m for c in candidates for m in c.entities]
    pred_names: dict[str, str] = {}  # normalised -> first spelling seen
    for m in mentions:
        pred_names.setdefault(normalize_name(m.name), m.name)

    used: set[str] = set()
    matched: dict[str, dict] = {}  # predicted normalised name -> expected entity
    for e in exp_ents:
        hit = next((n for n in pred_names if n not in used and n in e["keys"]), None)
        if hit is not None:
            used.add(hit)
            matched[hit] = e

    score = LabelScore(ent_pred=len(pred_names), ent_exp=len(exp_ents), ent_match=len(matched))
    problems = score.problems
    wrong_type = []
    for m in mentions:
        e = matched.get(normalize_name(m.name))
        if e is None:
            continue
        score.type_mentions += 1
        if m.type == e["type"]:
            score.type_ok += 1
        else:
            wrong_type.append({"name": m.name, "got": m.type, "want": e["type"]})
    missing = [e["name"] for e in exp_ents if e["norm"] not in {x["norm"] for x in matched.values()}]
    extra = [pred_names[n] for n in pred_names if n not in matched]
    invalid, seen = [], set()
    for m in mentions:
        if m.type not in ENTITY_TYPES and (m.name, m.type) not in seen:
            seen.add((m.name, m.type))
            invalid.append({"name": m.name, "type": m.type})
    for key, val in (("missing_entities", missing), ("extra_entities", extra),
                     ("wrong_entity_type", wrong_type), ("invalid_entity_type", invalid)):
        if val:
            problems[key] = val

    canon = {}  # key -> canonical normalised name, for matched expected entities
    for e in matched.values():
        for k in e["keys"]:
            canon.setdefault(k, e["norm"])

    def resolve(name: str) -> str:
        n = normalize_name(name)
        return canon.get(n, n)

    pred_rels: list[tuple[str, str, str]] = []
    for c in candidates:
        for r in c.relations:
            rel = (resolve(r.source), resolve(r.target), r.type)
            if rel not in pred_rels:
                pred_rels.append(rel)
    exp_rels = _expected_relations(expected)
    score.rel_pred, score.rel_exp = len(pred_rels), len(exp_rels)
    used_p: set[int] = set()
    matched_exp: set[int] = set()
    wrong_rel_type = []
    for j, (src, tgt, typ) in enumerate(exp_rels):
        i = next((i for i, p in enumerate(pred_rels) if i not in used_p and p[:2] == (src, tgt)), None)
        if i is None:
            continue
        used_p.add(i)
        matched_exp.add(j)
        score.rel_match += 1
        if pred_rels[i][2] == typ:
            score.rel_type_ok += 1
        else:
            wrong_rel_type.append({"relation": f"{src} -{typ}-> {tgt}", "got": pred_rels[i][2], "want": typ})
    miss_rel = [f"{s} -{t}-> {d}" for j, (s, d, t) in enumerate(exp_rels) if j not in matched_exp]
    extra_rel = [f"{s} -{t}-> {d}" for i, (s, d, t) in enumerate(pred_rels) if i not in used_p]
    for key, val in (("missing_relations", miss_rel), ("extra_relations", extra_rel),
                     ("wrong_relation_type", wrong_rel_type)):
        if val:
            problems[key] = val
    return score


def run_eval(llm: LLM, embedder: Embedder, cases: list[dict], cfg: RemConfig) -> EvalReport:
    settings = cfg.extraction
    predicted_total = expected_total = matched_total = 0
    scope_ok = origin_ok = 0
    valid = 0
    dedup_cases = dedup_ok = 0
    latencies: list[float] = []
    failures: list[dict] = []
    ent_pred = ent_exp = ent_match = type_mentions = type_ok = 0
    rel_pred = rel_exp = rel_match = rel_type_ok = 0
    invalid_total = mention_total = labelled_cases = 0

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
                    if is_labelled(case):
                        labelled_cases += 1
                        ent_exp += len(_expected_entities(case["expected"]))
                        rel_exp += len(_expected_relations(case["expected"]))
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
                    session_id=None, source="eval", restate_min_similarity=settings.restate_min_similarity,
                    has_user_turn=any(t.role == "user" for t in turns),
                    user_texts=[t.content for t in turns if t.role == "user"],
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
                bad_types, n_mentions = invalid_type_mentions(candidates)
                invalid_total += bad_types
                mention_total += n_mentions
                if is_labelled(case):
                    ls = score_labels(candidates, expected)
                    labelled_cases += 1
                    ent_pred += ls.ent_pred
                    ent_exp += ls.ent_exp
                    ent_match += ls.ent_match
                    type_mentions += ls.type_mentions
                    type_ok += ls.type_ok
                    rel_pred += ls.rel_pred
                    rel_exp += ls.rel_exp
                    rel_match += ls.rel_match
                    rel_type_ok += ls.rel_type_ok
                    problems.update(ls.problems)
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
        entity_precision=_ratio(ent_match, ent_pred),
        entity_recall=_ratio(ent_match, ent_exp),
        entity_type_acc=_ratio(type_ok, type_mentions),
        invalid_type_rate=_ratio(invalid_total, mention_total),
        relation_precision=_ratio(rel_match, rel_pred),
        relation_recall=_ratio(rel_match, rel_exp),
        relation_type_acc=_ratio(rel_type_ok, rel_match),
        labelled_cases=labelled_cases,
        failures=failures,
    )
