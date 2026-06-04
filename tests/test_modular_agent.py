from __future__ import annotations

import json
import os
import tempfile
import textwrap
import unittest

from autosolver_agent import AutoSolverLangChainAgent
from autosolver_agent.caseio import CaseParseError, load_cases_with_diagnostics, parse_case, parse_case_with_diagnostics, score_answer
from autosolver_agent.llm.schema import parse_candidate_envelope
from autosolver_agent.memory import MemoryStore
from autosolver_agent.memory.store import MEMORY_SCHEMA_VERSION
from autosolver_agent.models import Case, ScoreResult
from autosolver_agent.runtime import run_candidate
from autosolver_agent.skills import SolverSkillLibrary, StrategyLibrary
from autosolver_agent.tools import InstanceClassifier, Validator
from langchain_autosolver_agent import build_parser
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


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def invoke(self, messages):
        if not self.outputs:
            raise RuntimeError("no fake outputs left")
        return FakeResponse(self.outputs.pop(0))


def fake_candidate(name: str = "fake_solver") -> str:
    metadata = json.dumps(
        {
            "name": name,
            "idea": "bundle smoke",
            "strategy_combination": ["bundle_first"],
            "parameter_changes": {},
            "expected_effect": "cover both tasks",
            "risk_control": "simple",
        },
        ensure_ascii=False,
    )
    return textwrap.dedent(
        f"""
        ```json
        {metadata}
        ```
        ```python
        {VALID_SOLVER}
        ```
        """
    )


def structured_candidate(name: str = "structured_solver", code: str = VALID_SOLVER) -> str:
    return json.dumps(
        {
            "rationale": {
                "name": name,
                "idea": "structured bundle smoke",
                "strategy_combination": ["bundle_first"],
                "parameter_changes": {"pair_weight": 10},
                "expected_effect": "cover both tasks",
                "risk_control": "validator repair",
            },
            "code": code,
        },
        ensure_ascii=False,
    )


INVALID_DUPLICATE_SOLVER = r'''
def solve(input_text: str) -> list:
    return [("t0", ["c0"]), ("t0", ["c1"])]
'''


class ModularAgentTests(unittest.TestCase):
    def test_parse_classify_and_score(self):
        parsed = parse_case(CASE_TEXT)
        self.assertEqual(parsed.all_tasks, ["t0", "t1"])
        features = InstanceClassifier().classify([Case("case.txt", CASE_TEXT)], [parsed])
        self.assertIn("recommended_focus", features["aggregate"])
        scored = score_answer(parsed, [("t0,t1", ["c2"])])
        self.assertTrue(scored["valid"])
        self.assertEqual(scored["covered"], 2)

    def test_strategy_and_solver_context_include_reference_examples(self):
        strategy_payload = json.loads(
            StrategyLibrary().as_prompt_context(
                {
                    "task_count": 2,
                    "courier_count": 3,
                    "pair_ratio": 0.25,
                    "avg_willingness": 0.6,
                    "capacity_ratio": 1.5,
                }
            )
        )
        strategy_by_name = {item["name"]: item for item in strategy_payload["strategies"]}
        self.assertIn("basic_seed_solver_pack", strategy_by_name["bundle_first"]["reference_examples"])
        self.assertIn("task_first_greedy_repair_reference", strategy_by_name["bundle_first"]["reference_examples"])

        solver_payload = json.loads(SolverSkillLibrary().as_prompt_context())
        example_by_name = {item["name"]: item for item in solver_payload["solver_examples"]}
        self.assertEqual(example_by_name["basic_seed_solver_pack"]["source_file"], "solvers/seed_solvers.py")
        self.assertEqual(example_by_name["task_first_greedy_repair_reference"]["source_file"], "solvers/solver.py")
        self.assertEqual(example_by_name["multi_start_hybrid_reference"]["source_file"], "solvers/solver_70433_best_E1.py")
        self.assertIn("init_min_cost_flow_single", example_by_name["multi_start_hybrid_reference"]["prompt_excerpt"])

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
        ]
        for code in forbidden_snippets:
            with self.subTest(code=code):
                self.assertFalse(validator.validate_static(code).valid)

    def test_sandbox_times_out_busy_loop(self):
        code = "def solve(input_text: str):\n    while True:\n        pass\n"
        run = run_candidate(code, CASE_TEXT, timeout=0.2)
        self.assertEqual(run["status"], "timeout")

    def test_parse_diagnostics_default_compatible_and_strict_failure(self):
        bad_text = CASE_TEXT + "bad-row\n\ttmp\t1\t0.5\n"
        parsed, diagnostics = parse_case_with_diagnostics(bad_text, case_name="bad.txt")
        self.assertEqual(parsed.all_tasks, ["t0", "t1"])
        self.assertEqual([item.code for item in diagnostics], ["malformed_row", "empty_task_key"])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(bad_text)
            cases, file_diagnostics = load_cases_with_diagnostics([path], max_cases=1)
            self.assertEqual(len(cases), 1)
            self.assertEqual(len(file_diagnostics), 2)
            with self.assertRaises(CaseParseError):
                load_cases_with_diagnostics([path], max_cases=1, strict=True)

    def test_memory_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            store.save(os.path.join(tmp, "short.json"))
            self.assertTrue(os.path.exists(os.path.join(tmp, "long_term_memory.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "short.json")))

    def test_memory_migrates_and_trims_long_term_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            long_term_path = os.path.join(tmp, "long_term_memory.json")
            legacy = {
                "created_at": "legacy",
                "strategy_history": [{"name": f"s{i}"} for i in range(4)],
                "feature_strategy_effects": [{"name": f"f{i}"} for i in range(4)],
                "experiments": [
                    {
                        "candidate": f"c{i}",
                        "features": {},
                        "tags": [],
                        "strategy": ["expected_greedy"],
                        "params": {},
                        "score": None,
                        "reward": float(i),
                    }
                    for i in range(4)
                ],
            }
            with open(long_term_path, "w", encoding="utf-8") as handle:
                json.dump(legacy, handle)

            store = MemoryStore(tmp, max_long_term_items=2)
            self.assertEqual(store.long_term["schema_version"], MEMORY_SCHEMA_VERSION)
            self.assertEqual(store.long_term["metadata"]["retention"]["max_items_per_list"], 2)
            self.assertEqual([item["candidate"] for item in store.long_term["experiments"]], ["c2", "c3"])
            self.assertEqual(len(store.long_term["strategy_history"]), 2)

            store.save()
            with open(long_term_path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(saved["schema_version"], MEMORY_SCHEMA_VERSION)
            self.assertEqual(len(saved["experiments"]), 2)
            self.assertEqual(saved["bandit_arms"]["expected_greedy"]["count"], 4)

    def test_memory_save_merges_latest_long_term_records_under_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = MemoryStore(tmp, max_long_term_items=10)
            second = MemoryStore(tmp, max_long_term_items=10)

            first.record_experiment(
                iteration=1,
                candidate_name="first",
                features={},
                strategy=["expected_greedy"],
                params={},
            )
            first.save()

            second.record_experiment(
                iteration=2,
                candidate_name="second",
                features={},
                strategy=["bundle_first"],
                params={},
            )
            second.save()

            loaded = MemoryStore(tmp, max_long_term_items=10)
            candidates = {record["candidate"] for record in loaded.long_term["experiments"]}
            self.assertTrue({"first", "second"}.issubset(candidates))
            self.assertEqual(loaded.long_term["bandit_arms"]["expected_greedy"]["count"], 1)
            self.assertEqual(loaded.long_term["bandit_arms"]["bundle_first"]["count"], 1)
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
                strategy=["bundle_first"],
                params={"pair_weight": 10},
                score=score,
            )
            similar = store.retrieve_similar(dict(features), top_k=1)
            self.assertEqual(similar[0]["candidate"], "good")
            recs = store.bandit_recommendations(["new_strategy"], limit=2)
            self.assertEqual(recs[0]["arm"], "new_strategy")
            self.assertEqual(recs[0]["mode"], "explore_cold_start")

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
                llm=FakeLLM([fake_candidate("fake_a"), fake_candidate("fake_b")]),
                max_cases=1,
                verbose=False,
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
            self.assertIn("case_diagnostics", loaded)
            self.assertIn("timings", loaded)
            self.assertIn("candidate_hashes", loaded)
            self.assertTrue(os.path.exists(report["event_log_path"]))
            with open(report["event_log_path"], "r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.assertTrue(events)
            self.assertTrue(all("run_id" in item and "phase" in item and "event" in item for item in events))

    def test_agent_strict_cases_fails_on_parse_diagnostics(self):
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
                llm=FakeLLM([structured_candidate("unused")]),
                max_cases=1,
                verbose=False,
                strict_cases=True,
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
                llm=FakeLLM(["not json", structured_candidate("schema_repair")]),
                max_cases=1,
                verbose=False,
            )
            report = agent.run()
            self.assertEqual(report["best"]["name"], "schema_repair")
            self.assertEqual(report["repair_history"][0]["reason"], "schema_error")

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
                    [
                        structured_candidate("bad_duplicate", INVALID_DUPLICATE_SOLVER),
                        structured_candidate("validation_repair", VALID_SOLVER),
                    ]
                ),
                max_cases=1,
                verbose=False,
            )
            report = agent.run()
            self.assertEqual(report["best"]["name"], "validation_repair")
            self.assertEqual(report["repair_history"][0]["reason"], "validation_error")
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
                    [
                        structured_candidate("bad_duplicate", INVALID_DUPLICATE_SOLVER),
                        "bad repair response",
                        structured_candidate("second_repair", VALID_SOLVER),
                    ]
                ),
                max_cases=1,
                verbose=False,
            )
            report = agent.run()
            self.assertEqual(report["best"]["name"], "second_repair")
            self.assertIn("error", report["repair_history"][0])

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
                with self.assertRaisesRegex(RuntimeError, "requires OPENAI_API_KEY"):
                    agent.run()
            finally:
                if old_key is not None:
                    os.environ["OPENAI_API_KEY"] = old_key
                if old_alt_key is not None:
                    os.environ["OPENAI_KEY"] = old_alt_key

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
                "--summary-out",
                "runs/summary.json",
                "--strict-cases",
                "--event-log",
                "runs/events.jsonl",
            ]
        )
        self.assertEqual(args.max_repair_attempts, 3)
        self.assertEqual(args.memory_top_k, 7)
        self.assertEqual(args.bandit_exploration, 2.0)
        self.assertEqual(args.summary_out, "runs/summary.json")
        self.assertTrue(args.strict_cases)
        self.assertEqual(args.event_log, "runs/events.jsonl")


if __name__ == "__main__":
    unittest.main()
