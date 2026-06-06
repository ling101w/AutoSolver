from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from autosolver_agent import AutoSolverLangChainAgent
from autosolver_agent.caseio import CaseParseError, aggregate_features, dataset_features, load_cases, parse_case, score_answer
from autosolver_agent.cli import build_parser
from autosolver_agent.framework import FrameworkStore, FrameworkUpdate, FrameworkValidationError, SolverFramework
from autosolver_agent.llm.schema import parse_candidate_envelope
from autosolver_agent.memory import MemoryStore
from autosolver_agent.memory.store import MEMORY_SCHEMA_VERSION
from autosolver_agent.models import Case, ScoreResult
from autosolver_agent.runtime import run_candidate
from autosolver_agent.tools import Validator
from autosolver_agent.workflow.parallel import ParallelAutoSolverRunner, ParallelRunConfig
from solvers import seed_solvers

CASE_TEXT = """task_id_list\tcourier_id\ttotal_score\twillingness
t0\tc0\t10\t0.8
t0\tc1\t30\t0.3
t1\tc1\t12\t0.7
t0,t1\tc2\t40\t0.6
"""


VALID_SOLVER = r'''
def solve(input_text: str) -> list:
    rows = []
    for line in input_text.strip().splitlines():
        if not line or line.startswith("task_id_list"):
            continue
        task_key, courier, score, willingness = line.split("\t")[:4]
        rows.append((task_key, courier, float(score), float(willingness)))
    if any(row[0] == "t0,t1" for row in rows):
        return [("t0,t1", ["c2"])]
    return []
'''


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = []


class FakeLLM:
    def __init__(
        self,
        outputs=None,
        *,
        plan_outputs=None,
        candidate_outputs=None,
        repair_outputs=None,
        framework_outputs=None,
        interpretation_outputs=None,
        framework_update_outputs=None,
    ):
        self.outputs = list(outputs or [])
        self.plan_outputs = list(plan_outputs or [])
        self.candidate_outputs = list(candidate_outputs or [])
        self.repair_outputs = list(repair_outputs or [])
        self.framework_outputs = list(framework_outputs or [])
        self.interpretation_outputs = list(interpretation_outputs or [])
        self.framework_update_outputs = list(framework_update_outputs or [])
        self.lock = threading.Lock()

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        with self.lock:
            prompt = "\n".join(str(message) for message in messages)
            if "SolverFramework schema" in prompt:
                return FakeResponse(_pop_or_default(self.framework_outputs, structured_framework()))
            if "InstanceInterpretation schema" in prompt:
                return FakeResponse(_pop_or_default(self.interpretation_outputs, structured_interpretation()))
            if "FrameworkUpdate schema" in prompt:
                return FakeResponse(_pop_or_default(self.framework_update_outputs, structured_framework_update()))
            if "planning controller" in prompt or "SolverPlan schema" in prompt:
                queue = self.plan_outputs or self.outputs
            elif "Repair attempt" in prompt:
                queue = self.repair_outputs or self.outputs
            else:
                queue = self.candidate_outputs or self.outputs
            if not queue:
                raise RuntimeError("no fake outputs left")
            return FakeResponse(queue.pop(0))


def _pop_or_default(queue, default):
    if queue:
        return queue.pop(0)
    return default


def fake_candidate(name: str = "fake_solver") -> str:
    return structured_candidate(name)


def structured_plan(name: str = "structured_plan", strategies=None) -> str:
    return json.dumps(
        {
            "name": name,
            "strategy_combination": list(strategies or ["risk_balanced_cover", "adaptive_pair_merge"]),
            "parameter_changes": {"beam_width": 8},
            "exploration_mode": "parallel_strategy",
            "reasoning": "use tool-provided features and memory to generate independent candidates",
            "risk_control": "keep standard-library solve with validator repair",
            "generation_directives": [
                "Generate a complete solve(input_text: str) implementation.",
                "Prioritize valid disjoint task and courier assignments.",
            ],
        },
        ensure_ascii=False,
    )


def structured_candidate(name: str = "structured_solver", code: str = VALID_SOLVER) -> str:
    return json.dumps(
        {
            "rationale": {
                "name": name,
                "idea": "structured bundle smoke",
                "strategy_combination": ["risk_balanced_cover"],
                "parameter_changes": {"pair_weight": 10},
                "expected_effect": "cover both tasks",
                "risk_control": "validator repair",
            },
            "code": code,
        },
        ensure_ascii=False,
    )


def structured_framework() -> str:
    return json.dumps(
        {
            "feature_dimensions": [
                {
                    "name": "bundle_opportunity",
                    "description": "Detect whether multi-task groups can cover tasks efficiently.",
                    "signals": ["pair_ratio", "bundle_ratio", "bundle_task_coverage"],
                    "interpretation_notes": ["Compare bundle coverage with singleton availability."],
                    "confidence": 0.7,
                }
            ],
            "strategies": [
                {
                    "name": "risk_balanced_cover",
                    "description": "Construct disjoint task coverage while balancing willingness and score.",
                    "applicable_tags": ["bundle_opportunity"],
                    "feature_signals": ["avg_willingness", "total_score", "bundle_ratio"],
                    "implementation_notes": "Parse TSV rows, choose valid groups, then add couriers only when penalty improves.",
                    "recommended_parameters": {"candidate_limit": 32},
                    "risks": ["Must preserve global courier uniqueness."],
                    "confidence": 0.7,
                },
                {
                    "name": "adaptive_pair_merge",
                    "description": "Try pair groups when they reduce expected penalty and preserve coverage.",
                    "applicable_tags": ["bundle_opportunity"],
                    "feature_signals": ["pair_ratio"],
                    "implementation_notes": "Compare singleton cover with candidate pair groups under the same penalty model.",
                    "recommended_parameters": {"pair_limit": 16},
                    "risks": ["Pair groups may block better single assignments."],
                    "confidence": 0.6,
                },
            ],
            "skills": [
                {
                    "name": "safe_standard_solver",
                    "strategy_names": ["risk_balanced_cover", "adaptive_pair_merge"],
                    "construction_notes": "Generate one self-contained standard-library solve(input_text: str) implementation.",
                    "code_contract": "Return a Python list and keep validation, scoring, runtime, and parser contracts unchanged.",
                    "constraints": ["No duplicate tasks.", "No duplicate couriers."],
                    "examples": ["Use deterministic tie-breakers for reproducible artifacts."],
                    "confidence": 0.8,
                }
            ],
        },
        ensure_ascii=False,
    )


def structured_interpretation() -> str:
    return json.dumps(
        {
            "tags": ["bundle_opportunity", "compact_case"],
            "opportunities": ["The pair row can cover both tasks with one courier."],
            "risks": ["A duplicate task assignment is invalid."],
            "recommended_focus": ["risk_balanced_cover", "adaptive_pair_merge"],
            "feature_notes": {"pair_ratio": "pair rows are available"},
            "reasoning": "Use the maintained framework to prefer valid compact coverage.",
            "confidence": 0.75,
        },
        ensure_ascii=False,
    )


def structured_framework_update(strategy_name: str = "risk_balanced_cover") -> str:
    return json.dumps(
        {
            "update_reason": "Record evidence from the latest candidate evaluation.",
            "source_experiments": ["fake_solver"],
            "confidence": 0.65,
            "strategies": [
                {
                    "name": strategy_name,
                    "description": "Updated from validation and scoring evidence.",
                    "applicable_tags": ["bundle_opportunity"],
                    "feature_signals": ["pair_ratio"],
                    "implementation_notes": "Keep full coverage first, then reduce penalty.",
                    "recommended_parameters": {"candidate_limit": 24},
                    "risks": ["Reject invalid duplicate assignments."],
                    "confidence": 0.65,
                }
            ],
        },
        ensure_ascii=False,
    )


INVALID_DUPLICATE_SOLVER = r'''
def solve(input_text: str) -> list:
    return [("t0", ["c0"]), ("t0", ["c1"])]
'''


class ModularAgentTests(unittest.TestCase):
    def test_parse_objective_features_and_score(self):
        parsed = parse_case(CASE_TEXT)
        self.assertEqual(parsed.all_tasks, ["t0", "t1"])
        features = dataset_features(parsed)
        aggregate = aggregate_features([features])
        self.assertNotIn("recommended_focus", aggregate)
        self.assertNotIn("tags", aggregate)
        self.assertIn("bundle_ratio", aggregate)
        scored = score_answer(parsed, [("t0,t1", ["c2"])])
        self.assertTrue(scored["valid"])
        self.assertEqual(scored["covered"], 2)

    def test_framework_store_bootstraps_updates_and_rejects_unsafe_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FrameworkStore(tmp)
            self.assertTrue(store.is_empty())
            framework = SolverFramework.model_validate(json.loads(structured_framework()))
            applied = store.bootstrap(framework, source="test_bootstrap")
            self.assertEqual(applied["action"], "bootstrap_applied")
            self.assertFalse(store.is_empty())
            self.assertIn("risk_balanced_cover", store.candidate_strategy_names())

            update = json.loads(structured_framework_update("latency_aware_repair"))
            update["strategies"][0]["applicable_tags"] = ["compact_case"]
            applied = store.apply_update(
                FrameworkUpdate.model_validate(update),
                source="test_update",
                iteration=1,
            )
            self.assertEqual(applied["action"], "update_applied")
            self.assertIn("latency_aware_repair", store.candidate_strategy_names())

            bad_payload = json.loads(structured_framework_update("unsafe_strategy"))
            bad_payload["strategies"][0]["implementation_notes"] = "call open('x')"
            with self.assertRaises(FrameworkValidationError):
                store.apply_update(
                    FrameworkUpdate.model_validate(bad_payload),
                    source="test_update",
                    iteration=2,
                )

    def test_seed_solvers_are_directly_callable_and_valid(self):
        parsed = parse_case(CASE_TEXT)
        for name, solver in seed_solvers.SEED_SOLVERS.items():
            with self.subTest(strategy=name):
                answer = solver(CASE_TEXT)
                scored = score_answer(parsed, answer)
                self.assertTrue(scored["valid"])
                self.assertGreater(scored["covered"], 0)

        default_scored = score_answer(parsed, seed_solvers.solve(CASE_TEXT))
        self.assertTrue(default_scored["valid"])

    def test_validator_rejects_duplicate_and_dangerous_code(self):
        parsed = parse_case(CASE_TEXT)
        validator = Validator(smoke_timeout=1.0)
        dangerous = "import os\n\ndef solve(input_text: str):\n    return []\n"
        self.assertFalse(validator.validate_static(dangerous).valid)
        duplicate = "def solve(input_text: str):\n    return [('t0', ['c0']), ('t0', ['c1'])]\n"
        result = validator.validate(duplicate, [Case("case.txt", CASE_TEXT)], [parsed])
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0]["type"], "invalid_output")

    def test_sandbox_allows_safe_solver_and_rejects_escape_paths(self):
        safe = """
import heapq
import math

def solve(input_text: str) -> list:
    heap = [math.sqrt(4)]
    heapq.heapify(heap)
    return [("t0,t1", ["c2"])] if heapq.heappop(heap) == 2 else []
"""
        validator = Validator(smoke_timeout=1.0)
        self.assertTrue(validator.validate_static(safe).valid)
        run = run_candidate(safe, CASE_TEXT, timeout=1.0)
        self.assertEqual(run["status"], "ok")

        forbidden_snippets = [
            "def solve(input_text: str):\n    open('x', 'w')\n    return []\n",
            "def solve(input_text: str):\n    eval('1 + 1')\n    return []\n",
            "def solve(input_text: str):\n    exec('x = 1')\n    return []\n",
            "def solve(input_text: str):\n    compile('1', '<x>', 'eval')\n    return []\n",
            "def solve(input_text: str):\n    __import__('os')\n    return []\n",
            "def solve(input_text: str):\n    globals()\n    return []\n",
            "def solve(input_text: str):\n    return [(().__class__.__name__, [])]\n",
            "import time\n\ndef solve(input_text: str):\n    deadline = time.time() + 1\n    return []\n",
            "from time import time\n\ndef solve(input_text: str):\n    deadline = time() + 1\n    return []\n",
        ]
        for code in forbidden_snippets:
            with self.subTest(code=code):
                self.assertFalse(validator.validate_static(code).valid)

    def test_sandbox_times_out_busy_loop(self):
        code = "def solve(input_text: str):\n    while True:\n        pass\n"
        run = run_candidate(code, CASE_TEXT, timeout=0.2)
        self.assertEqual(run["status"], "timeout")

    def test_case_parser_rejects_malformed_rows(self):
        bad_text = CASE_TEXT + "bad-row\n\ttmp\t1\t0.5\n"
        with self.assertRaises(CaseParseError) as context:
            parse_case(bad_text, case_name="bad.txt")
        self.assertEqual([item.code for item in context.exception.diagnostics], ["malformed_row", "empty_task_key"])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(bad_text)
            with self.assertRaises(CaseParseError) as context:
                load_cases([path], max_cases=1)
            self.assertEqual([item.code for item in context.exception.diagnostics], ["malformed_row", "empty_task_key"])

    def test_memory_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            store.save(os.path.join(tmp, "short.json"))
            self.assertTrue(os.path.exists(os.path.join(tmp, "long_term_memory.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "short.json")))

    def test_memory_rejects_unsupported_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            long_term_path = os.path.join(tmp, "long_term_memory.json")
            old_schema = {
                "schema_version": 1,
                "created_at": "old",
                "updated_at": "old",
                "strategy_history": [{"name": f"s{i}"} for i in range(4)],
                "feature_strategy_effects": [{"name": f"f{i}"} for i in range(4)],
                "experiments": [],
                "bandit_arms": {},
                "metadata": {"retention": {"max_items_per_list": 2}},
            }
            with open(long_term_path, "w", encoding="utf-8") as handle:
                json.dump(old_schema, handle)

            with self.assertRaisesRegex(RuntimeError, "unsupported memory schema_version"):
                MemoryStore(tmp, max_long_term_items=2)

    def test_memory_trims_current_schema_long_term_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            long_term_path = os.path.join(tmp, "long_term_memory.json")
            current = {
                "schema_version": MEMORY_SCHEMA_VERSION,
                "created_at": "current",
                "updated_at": "current",
                "strategy_history": [{"name": f"s{i}"} for i in range(4)],
                "feature_strategy_effects": [{"name": f"f{i}"} for i in range(4)],
                "experiments": [],
                "bandit_arms": {},
                "metadata": {"retention": {"max_items_per_list": 4}},
            }
            with open(long_term_path, "w", encoding="utf-8") as handle:
                json.dump(current, handle)

            store = MemoryStore(tmp, max_long_term_items=2)
            self.assertEqual(store.long_term["schema_version"], MEMORY_SCHEMA_VERSION)
            self.assertEqual(len(store.long_term["strategy_history"]), 2)
            self.assertEqual(len(store.long_term["feature_strategy_effects"]), 2)

    def test_memory_save_merges_latest_long_term_records_under_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = MemoryStore(tmp, max_long_term_items=10)
            second = MemoryStore(tmp, max_long_term_items=10)

            first.record_experiment(
                iteration=1,
                candidate_name="first",
                features={},
                strategy=["simple_cover_seed"],
                params={},
            )
            first.save()

            second.record_experiment(
                iteration=2,
                candidate_name="second",
                features={},
                strategy=["risk_balanced_cover"],
                params={},
            )
            second.save()

            loaded = MemoryStore(tmp, max_long_term_items=10)
            candidates = {record["candidate"] for record in loaded.long_term["experiments"]}
            self.assertTrue({"first", "second"}.issubset(candidates))
            self.assertEqual(loaded.long_term["bandit_arms"]["simple_cover_seed"]["count"], 1)
            self.assertEqual(loaded.long_term["bandit_arms"]["risk_balanced_cover"]["count"], 1)
            self.assertTrue(os.path.exists(os.path.join(tmp, "long_term_memory.json.lock")))

    def test_structured_candidate_schema(self):
        envelope = parse_candidate_envelope(structured_candidate("schema_solver"))
        self.assertEqual(envelope.rationale.name, "schema_solver")
        self.assertIn("def solve", envelope.code)

    def test_memory_similarity_and_bandit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            features = {
                "task_count": 2,
                "courier_count": 3,
                "pair_ratio": 0.25,
                "avg_willingness": 0.6,
                "capacity_ratio": 1.5,
                "tags": ["small_task_count", "high_pair_ratio"],
            }
            score = ScoreResult(
                name="good",
                rank=(0, -2, 30.0, 0.01),
                total_covered=2,
                total_tasks=2,
                total_penalty=30.0,
                total_runtime=0.01,
                failures=0,
                cases=[],
            )
            store.record_experiment(
                iteration=1,
                candidate_name="good",
                features=features,
                strategy=["risk_balanced_cover"],
                params={"pair_weight": 10},
                score=score,
            )
            similar = store.retrieve_similar(dict(features), top_k=1)
            self.assertEqual(similar[0]["candidate"], "good")
            recs = store.bandit_recommendations(["new_strategy"], limit=2)
            self.assertEqual(recs[0]["arm"], "new_strategy")
            self.assertEqual(recs[0]["mode"], "explore_cold_start")

    def test_agent_strategy_workers_with_fake_llm_use_parallel_batch_path(self):
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_run_candidate(code, case_text, timeout):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.05)
                return {"status": "ok", "answer": [("t0,t1", ["c2"])], "runtime": 0.01, "error": None}
            finally:
                with lock:
                    active -= 1

        with tempfile.TemporaryDirectory() as tmp:
            case_paths = []
            for idx in range(2):
                path = os.path.join(tmp, f"case_{idx}.txt")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(CASE_TEXT)
                case_paths.append(path)

            agent = AutoSolverLangChainAgent(
                case_paths=case_paths,
                output_path=os.path.join(tmp, "generated_submit_solution.py"),
                budget_seconds=10,
                per_case_timeout=2,
                search_per_case_timeout=1,
                iterations=1,
                memory_dir=os.path.join(tmp, "memory"),
                artifact_dir=os.path.join(tmp, "artifacts"),
                llm=FakeLLM(
                    plan_outputs=[structured_plan("parallel_plan")],
                    candidate_outputs=[structured_candidate("parallel_a"), structured_candidate("parallel_b")],
                ),
                max_cases=2,
                verbose=False,
                finalize_top_k=1,
                strategy_workers=2,
            )
            with patch("autosolver_agent.tools.validator.run_candidate", side_effect=fake_run_candidate), patch(
                "autosolver_agent.tools.scorer.run_candidate",
                side_effect=fake_run_candidate,
            ):
                report = agent.run()

        self.assertGreaterEqual(max_active, 2)
        self.assertEqual(report["strategy_workers"], 2)
        self.assertGreaterEqual(report["summary"]["candidates_generated"], 2)

    def test_parallel_runner_merges_worker_reports_and_finalizes_global_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Case("case.txt", CASE_TEXT)
            parsed = parse_case(CASE_TEXT)

            def fake_worker_entry(worker_id, iteration_counter, iteration_lock, cases, parsed_cases, config, queue):
                claimed = []
                while True:
                    with iteration_lock:
                        if iteration_counter.value >= config.iterations:
                            break
                        iteration_counter.value += 1
                        iteration = iteration_counter.value
                    claimed.append(iteration)
                worker_dir = os.path.join(config.artifact_dir, f"worker_{worker_id:02d}")
                os.makedirs(worker_dir, exist_ok=True)
                code_path = os.path.join(worker_dir, f"worker_{worker_id}_solver.py")
                with open(code_path, "w", encoding="utf-8") as handle:
                    handle.write(VALID_SOLVER)
                queue.put(
                    (
                        "ok",
                        {
                            "worker_id": worker_id,
                            "iterations_completed": len(claimed),
                            "claimed_iterations": claimed,
                            "artifacts": [
                                {
                                    "iteration": claimed[0] if claimed else worker_id + 1,
                                    "candidate_name": f"worker_{worker_id}_solver",
                                    "code_path": code_path,
                                }
                            ],
                            "experiments": [
                                {
                                    "iteration": claimed[0] if claimed else worker_id + 1,
                                    "candidate": f"worker_{worker_id}_solver",
                                    "strategy": ["risk_balanced_cover"],
                                    "params": {},
                                    "score": {
                                        "rank": [0, -2, 30.0 + worker_id, 0.01],
                                        "covered": 2,
                                        "tasks": 2,
                                        "penalty": 30.0 + worker_id,
                                        "runtime": 0.01,
                                        "failures": 0,
                                    },
                                }
                            ],
                        },
                    )
                )

            config = ParallelRunConfig(
                iterations=5,
                deadline=time.time() + 10,
                per_case_timeout=2,
                search_per_case_timeout=1,
                output_path=os.path.join(tmp, "generated_submit_solution.py"),
                memory_dir=os.path.join(tmp, "memory"),
                artifact_dir=os.path.join(tmp, "artifacts"),
                llm_model=None,
                llm_base_url=None,
                verbose=False,
                finalize_top_k=1,
                max_repair_attempts=0,
                memory_top_k=1,
                bandit_exploration=1.4,
                strategy_workers=2,
                summary_output_path=None,
                event_log_path=None,
            )
            with patch("autosolver_agent.workflow.parallel._worker_entry", side_effect=fake_worker_entry):
                report = ParallelAutoSolverRunner([case], [parsed], config).run()

            self.assertEqual(report["run_mode"], "parallel_workers")
            self.assertEqual(report["parallel_workers"], 2)
            self.assertEqual(report["iterations_requested"], 5)
            self.assertEqual(report["iterations_completed"], 5)
            claimed = sorted(
                iteration
                for worker_report in report["worker_reports"]
                for iteration in worker_report.get("claimed_iterations", [])
            )
            self.assertEqual(claimed, [1, 2, 3, 4, 5])
            self.assertTrue(os.path.exists(config.output_path))
            self.assertEqual(report["best"]["name"], "worker_0_solver")
            self.assertEqual(len(report["worker_reports"]), 2)

    def test_agent_with_fake_llm_creates_artifacts_and_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_path = os.path.join(tmp, "case.txt")
            out_path = os.path.join(tmp, "generated_submit_solution.py")
            with open(case_path, "w", encoding="utf-8") as handle:
                handle.write(CASE_TEXT)
            agent = AutoSolverLangChainAgent(
                case_paths=[case_path],
                output_path=out_path,
                budget_seconds=10,
                per_case_timeout=2,
                search_per_case_timeout=1,
                iterations=2,
                memory_dir=os.path.join(tmp, "memory"),
                artifact_dir=os.path.join(tmp, "artifacts"),
                llm=FakeLLM(
                    plan_outputs=[structured_plan("plan_a"), structured_plan("plan_b")],
                    candidate_outputs=[
                        fake_candidate("fake_a"),
                        fake_candidate("fake_b"),
                        fake_candidate("fake_c"),
                        fake_candidate("fake_d"),
                    ],
                ),
                max_cases=1,
                verbose=False,
                strategy_workers=2,
            )
            report = agent.run()
            self.assertTrue(os.path.exists(out_path))
            self.assertTrue(os.path.exists(out_path + ".report.json"))
            self.assertEqual(report["iterations_completed"], 2)
            self.assertTrue(report["artifacts"])
            self.assertIsNotNone(report["best"])
            with open(out_path + ".report.json", "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            self.assertIn("instance_features", loaded)
            self.assertIn("planner_trace", loaded)
            self.assertIn("bandit", loaded)
            self.assertIn("summary", loaded)
            self.assertIn("run_id", loaded)
            self.assertIn("event_log_path", loaded)
            self.assertIn("timings", loaded)
            self.assertIn("candidate_hashes", loaded)
            self.assertTrue(os.path.exists(report["event_log_path"]))
            with open(report["event_log_path"], "r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.assertTrue(events)
            self.assertTrue(all("run_id" in item and "phase" in item and "event" in item for item in events))

    def test_agent_strategy_workers_one_runs_single_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_path = os.path.join(tmp, "case.txt")
            out_path = os.path.join(tmp, "generated_submit_solution.py")
            with open(case_path, "w", encoding="utf-8") as handle:
                handle.write(CASE_TEXT)
            agent = AutoSolverLangChainAgent(
                case_paths=[case_path],
                output_path=out_path,
                budget_seconds=10,
                per_case_timeout=2,
                search_per_case_timeout=1,
                iterations=1,
                memory_dir=os.path.join(tmp, "memory"),
                artifact_dir=os.path.join(tmp, "artifacts"),
                llm=FakeLLM(
                    plan_outputs=[structured_plan("single_worker_plan")],
                    candidate_outputs=[fake_candidate("single_worker_solver")],
                ),
                max_cases=1,
                verbose=False,
                strategy_workers=1,
            )
            report = agent.run()
            self.assertTrue(os.path.exists(out_path))
            self.assertEqual(report["strategy_workers"], 1)
            self.assertEqual(report["iterations_completed"], 1)
            self.assertIsNotNone(report["best"])

    def test_agent_rejects_malformed_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_path = os.path.join(tmp, "case.txt")
            with open(case_path, "w", encoding="utf-8") as handle:
                handle.write(CASE_TEXT + "bad-row\n")
            agent = AutoSolverLangChainAgent(
                case_paths=[case_path],
                output_path=os.path.join(tmp, "generated_submit_solution.py"),
                budget_seconds=10,
                per_case_timeout=2,
                search_per_case_timeout=1,
                iterations=1,
                memory_dir=os.path.join(tmp, "memory"),
                artifact_dir=os.path.join(tmp, "artifacts"),
                llm=FakeLLM(candidate_outputs=[structured_candidate("unused")]),
                max_cases=1,
                verbose=False,
                strategy_workers=2,
            )
            with self.assertRaises(CaseParseError):
                agent.run()

    def test_agent_repairs_schema_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_path = os.path.join(tmp, "case.txt")
            out_path = os.path.join(tmp, "generated_submit_solution.py")
            with open(case_path, "w", encoding="utf-8") as handle:
                handle.write(CASE_TEXT)
            agent = AutoSolverLangChainAgent(
                case_paths=[case_path],
                output_path=out_path,
                budget_seconds=10,
                per_case_timeout=2,
                search_per_case_timeout=1,
                iterations=1,
                memory_dir=os.path.join(tmp, "memory"),
                artifact_dir=os.path.join(tmp, "artifacts"),
                llm=FakeLLM(
                    plan_outputs=[structured_plan("schema_plan")],
                    candidate_outputs=["not json", structured_candidate("schema_parallel")],
                    repair_outputs=[structured_candidate("schema_repair")],
                ),
                max_cases=1,
                verbose=False,
                strategy_workers=2,
            )
            report = agent.run()
            self.assertTrue(any(item.get("candidate") == "schema_repair" for item in report["repair_history"]))
            self.assertTrue(any(item.get("reason") == "schema_error" for item in report["repair_history"]))

    def test_agent_repairs_validation_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_path = os.path.join(tmp, "case.txt")
            out_path = os.path.join(tmp, "generated_submit_solution.py")
            with open(case_path, "w", encoding="utf-8") as handle:
                handle.write(CASE_TEXT)
            agent = AutoSolverLangChainAgent(
                case_paths=[case_path],
                output_path=out_path,
                budget_seconds=10,
                per_case_timeout=2,
                search_per_case_timeout=1,
                iterations=1,
                memory_dir=os.path.join(tmp, "memory"),
                artifact_dir=os.path.join(tmp, "artifacts"),
                llm=FakeLLM(
                    plan_outputs=[structured_plan("validation_plan")],
                    candidate_outputs=[
                        structured_candidate("bad_duplicate", INVALID_DUPLICATE_SOLVER),
                        structured_candidate("validation_parallel", VALID_SOLVER),
                    ],
                    repair_outputs=[structured_candidate("validation_repair", VALID_SOLVER)],
                ),
                max_cases=1,
                verbose=False,
                strategy_workers=2,
            )
            report = agent.run()
            self.assertTrue(any(item.get("candidate") == "validation_repair" for item in report["repair_history"]))
            self.assertTrue(any(item.get("reason") == "validation_error" for item in report["repair_history"]))
            self.assertTrue(report["experiments"])

    def test_validation_repair_continues_after_bad_repair_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_path = os.path.join(tmp, "case.txt")
            out_path = os.path.join(tmp, "generated_submit_solution.py")
            with open(case_path, "w", encoding="utf-8") as handle:
                handle.write(CASE_TEXT)
            agent = AutoSolverLangChainAgent(
                case_paths=[case_path],
                output_path=out_path,
                budget_seconds=10,
                per_case_timeout=2,
                search_per_case_timeout=1,
                iterations=1,
                memory_dir=os.path.join(tmp, "memory"),
                artifact_dir=os.path.join(tmp, "artifacts"),
                llm=FakeLLM(
                    plan_outputs=[structured_plan("bad_repair_plan")],
                    candidate_outputs=[
                        structured_candidate("bad_duplicate", INVALID_DUPLICATE_SOLVER),
                        structured_candidate("bad_repair_parallel", VALID_SOLVER),
                    ],
                    repair_outputs=["bad repair response", structured_candidate("second_repair", VALID_SOLVER)],
                ),
                max_cases=1,
                verbose=False,
                strategy_workers=2,
            )
            report = agent.run()
            self.assertTrue(any("error" in item for item in report["repair_history"]))
            self.assertTrue(any(item.get("candidate") == "second_repair" for item in report["repair_history"]))

    def test_agent_without_llm_fails_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_path = os.path.join(tmp, "case.txt")
            with open(case_path, "w", encoding="utf-8") as handle:
                handle.write(CASE_TEXT)
            old_key = os.environ.pop("OPENAI_API_KEY", None)
            old_alt_key = os.environ.pop("OPENAI_KEY", None)
            try:
                agent = AutoSolverLangChainAgent(
                    case_paths=[case_path],
                    output_path=os.path.join(tmp, "generated_submit_solution.py"),
                    budget_seconds=5,
                    per_case_timeout=1,
                    search_per_case_timeout=1,
                    iterations=1,
                    memory_dir=os.path.join(tmp, "memory"),
                    artifact_dir=os.path.join(tmp, "artifacts"),
                    max_cases=1,
                    verbose=False,
                )
                with self.assertRaisesRegex(RuntimeError, "requires OPENAI_API_KEY or OPENAI_KEY"):
                    agent.run()
            finally:
                if old_key is not None:
                    os.environ["OPENAI_API_KEY"] = old_key
                if old_alt_key is not None:
                    os.environ["OPENAI_KEY"] = old_alt_key

    def test_llm_environment_accepts_openai_compatible_key_alias(self):
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        old_alt_key = os.environ.get("OPENAI_KEY")
        os.environ["OPENAI_KEY"] = "test-key"
        try:
            from autosolver_agent.llm import LLMCodeGenerator

            LLMCodeGenerator.validate_environment()
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key
            if old_alt_key is not None:
                os.environ["OPENAI_KEY"] = old_alt_key
            else:
                os.environ.pop("OPENAI_KEY", None)

    def test_cli_parses_new_agent_options(self):
        args = build_parser().parse_args(
            [
                "--cases",
                "examples/demo_case.txt",
                "--max-repair-attempts",
                "3",
                "--memory-top-k",
                "7",
                "--bandit-exploration",
                "2.0",
                "--strategy-workers",
                "3",
                "--summary-out",
                "runs/summary.json",
                "--event-log",
                "runs/events.jsonl",
            ]
        )
        self.assertEqual(args.max_repair_attempts, 3)
        self.assertEqual(args.memory_top_k, 7)
        self.assertEqual(args.bandit_exploration, 2.0)
        self.assertEqual(args.strategy_workers, 3)
        self.assertEqual(args.summary_out, "runs/summary.json")
        self.assertEqual(args.event_log, "runs/events.jsonl")


if __name__ == "__main__":
    unittest.main()
