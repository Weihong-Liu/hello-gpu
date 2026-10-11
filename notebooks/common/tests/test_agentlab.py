"""CPU contracts for the adapter; fixture measurements are never GPU evidence."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "hello_gpu"
# Keep CPU contracts runnable without importing the facade's ROCm/Torch helpers.
PACKAGE = "_hello_gpu_agentlab_contract"
package = ModuleType(PACKAGE)
package.__path__ = [str(SOURCE_ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + ".agentlab", SOURCE_ROOT / "agentlab.py")
agentlab = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = agentlab
spec.loader.exec_module(agentlab)


def fixture_evaluation(*args, **kwargs):
    """Deterministic fixture to check data flow, not a performance experiment."""
    source = Path(args[1]).read_text()
    if "output = x - y" in source:
        return {"status": "wrong_answer", "latencyMs": None, "benchmark": None,
                "correctness": {"passed": False, "reason": "fixture arithmetic mismatch"}}
    result = {"status": "ok", "latencyMs": 0.8,
              "benchmark": {"samplesMs": [0.8, 0.7, 0.9, 0.8, 0.8]},
              "correctness": {"passed": True}, "environment": {"fixture": True}}
    if kwargs.get("correctness_only"):
        result.update(latencyMs=None, benchmark=None)
    if kwargs.get("incumbent_path") is not None:
        result["comparison"] = {"incumbentLatencyMs": 1.0,
                                "incumbentBenchmark": {"samplesMs": [1.0] * 5},
                                "improvementFraction": 0.2, "pairedImprovements": [0.2] * 5}
    return result


class AgentLabContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        for patch in (mock.patch.object(agentlab, "artifact_root", return_value=self.directory),
                      mock.patch.object(agentlab, "_gpu_ready"),
                      mock.patch.object(agentlab._backend(), "_evaluator", return_value=fixture_evaluation)):
            patch.start()
            self.addCleanup(patch.stop)
        self.cfg = {"n": 4097, "block": 256, "warmup": 1, "repeat": 5, "seed": 17}
        self.lab = agentlab.Lab("chapter15", self.cfg)
        self.candidate = self.lab.baseline_source.replace("block_size = 256", "block_size = 512")

    def test_cpu_tools_need_no_model_gpu_or_terminal_imports(self):
        script = f"""
import importlib.util, pathlib, socket, sys, types
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network forbidden'))
root = pathlib.Path({str(SOURCE_ROOT)!r})
p = types.ModuleType('adapter_cpu'); p.__path__ = [str(root)]; sys.modules['adapter_cpu'] = p
s = importlib.util.spec_from_file_location('adapter_cpu.agentlab', root/'agentlab.py')
m = importlib.util.module_from_spec(s); sys.modules[s.name] = m; s.loader.exec_module(m)
ex = m.ToolExecutor(); ex.register('double', 'CPU tool', lambda x: str(x*2))
assert ex.call('double', {{'x':4}}) == '8'
assert not ({{'torch','triton','litellm','prompt_toolkit'}} & set(sys.modules))
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_task_preserves_exact_elements_dtype_and_benchmark_settings(self):
        for n, shape in ((4097, {"rows": 1, "cols": 4097}), (1 << 20, {"rows": 512, "cols": 2048})):
            task = agentlab.task_spec(self.cfg | {"n": n})
            self.assertEqual(task["shape"], shape)
            self.assertEqual(task["costModel"]["bytes"], n * 6)
            self.assertEqual(task["costModel"]["flops"], n)
            self.assertEqual(task["tensors"][-1]["value"], n)
            self.assertEqual(task["correctness"]["seeds"], [17, 18])
            self.assertEqual(task["benchmark"]["samples"], 5)
            self.assertEqual(task["benchmark"]["innerRepeats"], 5)
            self.assertFalse(task["reference"]["selfCheckPassed"])
        with self.assertRaises(ValueError):
            agentlab.task_spec(self.cfg | {"repeat": 4})
        with self.assertRaises(ValueError):
            agentlab.task_spec(self.cfg | {"n": True})

    def test_fresh_workspaces_and_visible_source_are_frozen(self):
        other = agentlab.Lab("chapter15", self.cfg, baseline_source=self.candidate.rstrip("\n"))
        self.assertNotEqual(other.root, self.lab.root)
        self.assertEqual(other.baseline_source, self.candidate)
        self.assertEqual(other.best_path.read_text(), self.candidate)
        self.assertEqual(other.task["entrypoint"], "launch")

    def test_observation_has_exact_artifact_without_stale_fallback(self):
        first = self.lab.call("compile_kernel", source=self.candidate)
        second = self.lab.call("bench_kernel", source=self.candidate)
        missing = self.lab.call("unknown_tool")
        self.assertNotEqual(first["evaluation"], second["evaluation"])
        self.assertNotIn("evaluation", missing)
        self.assertIsNone(self.lab._evidence(first)["benchmark"])
        self.assertEqual(self.lab._evidence(second)["benchmark"]["samplesMs"], [0.8, 0.7, 0.9, 0.8, 0.8])
        self.assertEqual(first["sourceSha256"], agentlab._hash(Path(first["evaluation"], "candidate.py").read_bytes()))

    def test_pending_handoff_must_use_identical_supplied_source(self):
        source = self.candidate.rstrip("\n")
        bench = self.lab.step("bench_kernel", source=source)
        self.assertTrue(self.lab.state["pendingCandidate"])
        self.assertEqual(self.lab.state["candidateSha256"], bench["sourceSha256"])
        before = self.lab.best_path.read_bytes()
        wrong = self.lab.step("accept_candidate", source=source + "\n", change="different bytes")
        self.assertFalse(wrong["ok"])
        self.assertEqual(wrong["stage"], "pending_accept")
        self.assertNotIn("evaluation", wrong)
        self.assertEqual(self.lab.best_path.read_bytes(), before)
        self.assertEqual(self.lab.finish()["state"], "incomplete")
        accepted = self.lab.step("accept_candidate", source=source, change="same source")
        self.assertTrue(accepted["accepted"])
        self.assertFalse(self.lab.state["pendingCandidate"])
        self.assertEqual(self.lab.finish()["state"], "complete")

    def test_rejection_completes_judgement_without_claiming_improvement(self):
        wrong = self.candidate.replace("output = x + y", "output = x - y")
        before = self.lab.best_path.read_bytes()
        result = self.lab.step("accept_candidate", source=wrong, change="deliberately wrong")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "wrong_answer")
        self.assertEqual(self.lab.best_path.read_bytes(), before)
        self.assertEqual(self.lab.finish()["state"], "complete")

    def test_benching_incumbent_does_not_create_pending_for_previous_activity(self):
        self.lab.step("compile_kernel", source=self.candidate)
        self.lab.step("bench_kernel", source=self.lab.baseline_source)
        self.assertFalse(self.lab.state["pendingCandidate"])
        # The earlier candidate has not been judged, so finish remains incomplete.
        self.assertEqual(self.lab.finish()["state"], "incomplete")

    def test_scripted_uses_real_loop_with_no_model_and_truthful_event_states(self):
        self.lab.preflight()
        wrong = self.candidate.replace("output = x + y", "output = x - y")
        llm = importlib.import_module("kernel_optimize.llm")
        with mock.patch.object(llm, "chat", side_effect=AssertionError("model forbidden")):
            status = self.lab.run([{"source": wrong, "change": "wrong arithmetic"},
                                   {"source": self.candidate, "change": "larger block"}], live=False)
        self.assertEqual(status["state"], "complete")
        self.assertEqual(status["llmCalls"], 0)
        rows = agentlab._rows(self.lab.root / "trajectory.jsonl")
        self.assertEqual(rows[0]["status"], "wrong_answer")
        self.assertEqual(len(rows), 2)
        events = agentlab._rows(self.lab.root / "notebook-events.jsonl")
        bench = [event for event in events if event["tool"] == "bench_kernel" and event["state"]["planner"] == "scripted"]
        self.assertTrue(bench[0]["state"]["pendingCandidate"])
        judged = [event for event in events if event["tool"] == "accept_candidate"]
        self.assertTrue(all(not event["state"]["pendingCandidate"] for event in judged))
        self.assertNotIn("convert_kernel", self.lab.executor.tools)
        self.assertNotIn("setup_task", self.lab.executor.tools)

    def test_online_is_explicit_and_counts_actual_planner_calls(self):
        self.lab.preflight()
        llm = importlib.import_module("kernel_optimize.llm")
        messages = [SimpleNamespace(content="candidate", tool_calls=[SimpleNamespace(id="1", function=SimpleNamespace(
                    name="accept_candidate", arguments=json.dumps({"source": self.candidate, "change": "fixture"})))]),
                    SimpleNamespace(content="Finished.", tool_calls=None)]
        def chat(history, *, tools):
            names = {item["function"]["name"] for item in tools}
            self.assertFalse(names & {"setup_task", "convert_kernel", "run_code", "ask_user"})
            return messages.pop(0)
        with mock.patch.object(llm, "_model_config", return_value=("fixture/model", None)), mock.patch.object(llm, "chat", side_effect=chat):
            status = self.lab.run_online(max_steps=2, live=False)
        self.assertEqual(status["state"], "complete")
        self.assertEqual(status["llmCalls"], 2)
        self.assertEqual(status["planner"], "online")

    def test_preflight_is_bound_to_reference_and_baseline(self):
        self.lab.preflight()
        self.lab.workspace.reference_path.write_text("# changed reference\n")
        with self.assertRaisesRegex(RuntimeError, "正确性参考"):
            self.lab._require_preflight()

    def test_independent_retest_freezes_files_and_rejects_stale_report(self):
        self.lab.preflight()
        self.lab.call("accept_candidate", source=self.candidate)
        retest = {"status": "ok", "runs": 3, "incumbentMediansMs": [1.0] * 3,
                  "candidateMediansMs": [0.8] * 3, "pairedImprovementFractions": [0.2] * 3,
                  "medianPairedImprovementFraction": 0.2}
        def run(command, **kwargs):
            self.assertEqual(Path(command[1]).name, "compare.py")
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[command.index("--runs") + 1], "3")
            baseline = Path(command[command.index("--incumbent") + 1])
            self.assertEqual(baseline.read_text(), self.lab.baseline_source)
            return SimpleNamespace(stdout=json.dumps(retest), stderr="", returncode=0)
        with mock.patch.object(agentlab.subprocess, "run", side_effect=run):
            report = self.lab.final_retest(runs=3)
        with mock.patch("builtins.print"):
            self.assertEqual(self.lab.report(report)["retest"]["status"], "ok")
        self.lab.best_path.write_text(self.lab.baseline_source)
        with self.assertRaisesRegex(ValueError, "旧复测"):
            self.lab.report(report)

    def test_display_escapes_source_and_error_html(self):
        with mock.patch.object(agentlab, "_display_html") as display:
            agentlab.show({"ok": False, "error": "<script>alert(1)</script>"})
            agentlab.source_diff("safe", "<img onerror=alert(1)>")
        for call in display.call_args_list:
            self.assertNotIn("<script>", call.args[0])
            self.assertNotIn("<img onerror", call.args[0])

    def test_failed_samples_never_become_zero_latency(self):
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("optional plotting dependency missing")
        os.environ.setdefault("MPLBACKEND", "Agg")
        wrong = self.candidate.replace("output = x + y", "output = x - y")
        failed = self.lab.call("bench_kernel", source=wrong)
        with self.assertRaisesRegex(ValueError, "没有成功计时"):
            self.lab.sample_plot(failed)
        result = self.lab.call("accept_candidate", source=self.candidate)
        figure = self.lab.pair_plot(result)
        self.assertEqual(list(figure.axes[0].lines[0].get_ydata()), [1.0] * 5)
        self.assertEqual(list(figure.axes[0].lines[1].get_ydata()), [0.8, 0.7, 0.9, 0.8, 0.8])
        import matplotlib.pyplot as plt
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
