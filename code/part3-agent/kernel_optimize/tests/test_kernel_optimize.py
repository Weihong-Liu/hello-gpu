"""kernel_optimize 教学 agent 的单测（本地无 GPU 可跑：逻辑/编排/工具注册）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from kernel_optimize import llm as llm_module
from kernel_optimize import tools as tools_module
from kernel_optimize.agent import run_agent
from kernel_optimize.measure import _has_peaks, ensure_peak
from kernel_optimize.tools import ToolExecutor, Workspace, _decide, build_tools


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------


class ToolExecutorTest(unittest.TestCase):
    def test_register_call_schema(self) -> None:
        ex = ToolExecutor()
        ex.register("echo", "回声", lambda text="": f"echo:{text}",
                    {"type": "object", "properties": {"text": {"type": "string"}}})
        self.assertEqual(ex.call("echo", {"text": "hi"}), "echo:hi")
        self.assertIn("未知工具", ex.call("nope", {}))
        schema = ex.schema()
        self.assertEqual(schema[0]["function"]["name"], "echo")

    def test_call_error_is_caught(self) -> None:
        ex = ToolExecutor()

        def boom():
            raise ValueError("x")

        ex.register("boom", "会炸", boom)
        self.assertIn("执行出错", ex.call("boom", {}))


# ---------------------------------------------------------------------------
# 确定性裁决 _decide
# ---------------------------------------------------------------------------


class DecideTest(unittest.TestCase):
    def test_not_ok(self) -> None:
        ok, reason = _decide({"status": "compile_error"}, 0.01)
        self.assertFalse(ok)
        self.assertEqual(reason, "compile_error")

    def test_missing_comparison(self) -> None:
        ok, reason = _decide({"status": "ok"}, 0.01)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_paired_comparison")

    def test_insufficient_pairs(self) -> None:
        evaluation = {"status": "ok", "comparison": {"pairedImprovements": [0.5] * 4, "improvementFraction": 0.5}}
        ok, reason = _decide(evaluation, 0.01)
        self.assertFalse(ok)
        self.assertEqual(reason, "insufficient_benchmark_evidence")

    def test_below_threshold(self) -> None:
        evaluation = {"status": "ok", "comparison": {"pairedImprovements": [0.001] * 5, "improvementFraction": 0.001}}
        ok, reason = _decide(evaluation, 0.01)
        self.assertFalse(ok)
        self.assertIn("below_threshold", reason)

    def test_improved(self) -> None:
        evaluation = {"status": "ok", "comparison": {"pairedImprovements": [0.2] * 5, "improvementFraction": 0.2}}
        ok, reason = _decide(evaluation, 0.01)
        self.assertTrue(ok)
        self.assertIn("improved", reason)


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class WorkspaceTest(unittest.TestCase):
    def test_write_candidate_and_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            path = ws.write_candidate("print('hi')")
            self.assertTrue(path.exists())
            self.assertIn("print", path.read_text())
            path.unlink()
            ws.record({"iteration": 1, "accepted": True})
            ws.record({"iteration": 2, "accepted": False})
            lines = (ws.root / "trajectory.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["iteration"], 1)


# ---------------------------------------------------------------------------
# measure（本地无 GPU：优雅降级）
# ---------------------------------------------------------------------------


class MeasureTest(unittest.TestCase):
    def test_has_peaks(self) -> None:
        self.assertTrue(_has_peaks({"bandwidthGbS": 400, "fp32Tflops": 8}))
        self.assertFalse(_has_peaks({"bandwidthGbS": 400}))  # 没算力
        self.assertFalse(_has_peaks(None))

    def test_ensure_peak_no_gpu_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(ensure_peak(Path(tmp)))


# ---------------------------------------------------------------------------
# build_tools：工具注册齐全
# ---------------------------------------------------------------------------


class BuildToolsTest(unittest.TestCase):
    def test_registers_closed_loop_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            executor, schema = build_tools(ws)
            names = {s["function"]["name"] for s in schema}
            for expected in (
                "ask_user", "get_environment", "measure_peak",
                "compile_kernel", "bench_kernel", "profile_kernel", "accept_candidate",
                "convert_kernel", "run_code", "read_reference", "setup_task",
            ):
                self.assertIn(expected, names)
            for removed in ("evaluate_candidate", "benchmark", "profile", "verify_correct"):
                self.assertNotIn(removed, names)
            reply = executor.call("read_reference", {"name": "optimization-patterns"})
            self.assertTrue("memory-bound" in reply or "没有参考资料" in reply or "可用" in reply)


class Chapter15ToolContractTest(unittest.TestCase):
    """对齐线上 chapter15：三工具 JSON 字段契约（mock _run_eval）。"""

    def test_compile_kernel_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            executor, _ = build_tools(ws)
            with mock.patch.object(
                tools_module,
                "_run_eval",
                return_value={"status": "ok", "correctness": {"maxAbsError": 0.0}},
            ):
                payload = json.loads(executor.call("compile_kernel", {"source": "code"}))
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["stage"], "run")
            self.assertEqual(payload["errors"], [])

    def test_compile_kernel_error_parses_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            executor, _ = build_tools(ws)
            with mock.patch.object(
                tools_module,
                "_run_eval",
                return_value={
                    "status": "compile_error",
                    "details": "boom",
                    "capturedOutput": 'File "x.py", line 42: expected a type',
                },
            ):
                payload = json.loads(executor.call("compile_kernel", {"source": "code"}))
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["stage"], "compile")
            self.assertTrue(any(e.get("line") == 42 for e in payload["errors"]))

    def test_bench_kernel_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            (ws.root / "task.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 2,
                        "description": "vector add",
                        "language": "triton",
                        "shape": {"rows": 4, "cols": 4},
                        "optimization": {"minImprovementFraction": 0.01},
                        "costModel": {"flops": 1000, "bytes": 8000},
                    }
                ),
                encoding="utf-8",
            )
            executor, _ = build_tools(ws)
            samples = [0.10, 0.11, 0.12, 0.13, 0.14]
            with mock.patch.object(
                tools_module,
                "_run_eval",
                return_value={
                    "status": "ok",
                    "latencyMs": 0.12,
                    "benchmark": {"samplesMs": samples, "medianAbsoluteDeviationMs": 0.01},
                },
            ):
                payload = json.loads(executor.call("bench_kernel", {"source": "code"}))
            self.assertTrue(payload["ok"])
            self.assertAlmostEqual(payload["median_ms"], 0.12)
            self.assertAlmostEqual(payload["mean_ms"], sum(samples) / len(samples))
            self.assertIsNotNone(payload["p95_ms"])
            self.assertIsNotNone(payload["bandwidth_gbps"])
            self.assertIsNotNone(payload["tflops"])

    def test_accept_candidate_records_and_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            executor, _ = build_tools(ws)
            with mock.patch.object(
                tools_module,
                "_run_eval",
                return_value={
                    "status": "ok",
                    "latencyMs": 0.10,
                    "comparison": {
                        "pairedImprovements": [0.001] * 5,
                        "improvementFraction": 0.001,
                    },
                },
            ):
                payload = json.loads(
                    executor.call("accept_candidate", {"source": "new", "change": "noop"})
                )
            self.assertFalse(payload["accepted"])
            self.assertIn("below_threshold", payload["reason"])
            traj = (ws.root / "trajectory.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(traj), 1)
            self.assertFalse(json.loads(traj[0])["accepted"])


# ---------------------------------------------------------------------------
# agent loop：编排（mock LLM）
# ---------------------------------------------------------------------------


def _msg(content, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _call(name, arguments, call_id="c1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _make_workspace(tmp: str) -> Workspace:
    ws = Workspace(Path(tmp))
    (ws.root / "task.json").write_text(json.dumps({
        "schemaVersion": 2, "description": "vector add", "language": "triton",
        "shape": {"rows": 4096, "cols": 2048},
        "optimization": {"minImprovementFraction": 0.01},
    }), encoding="utf-8")
    (ws.root / "best.py").write_text("def launch(*a):\n    pass\n", encoding="utf-8")
    return ws


class AskUserTest(unittest.TestCase):
    def test_session_constructs(self) -> None:
        # 多行会话能正常构造（回车绑定不出错）
        self.assertIsNotNone(tools_module._make_ask_session())

    @staticmethod
    def _session_returning(answer):
        session = mock.Mock()
        session.prompt.return_value = answer
        return session

    def test_single_line_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = build_tools(_make_workspace(tmp))
            with mock.patch.object(tools_module, "_get_ask_session",
                                   return_value=self._session_returning("float32")):
                answer = executor.call("ask_user", {"question": "dtype?"})
        self.assertEqual(answer, "float32")

    def test_multiline_answer_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = build_tools(_make_workspace(tmp))
            with mock.patch.object(tools_module, "_get_ask_session",
                                   return_value=self._session_returning("line1\nline2\nline3")):
                answer = executor.call("ask_user", {"question": "贴 kernel"})
        self.assertEqual(answer, "line1\nline2\nline3")

    def test_eof_returns_no_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = build_tools(_make_workspace(tmp))
            session = mock.Mock()
            session.prompt.side_effect = EOFError
            with mock.patch.object(tools_module, "_get_ask_session", return_value=session):
                answer = executor.call("ask_user", {"question": "?"})
        self.assertEqual(answer, "（用户未回答）")


class AgentLoopTest(unittest.TestCase):
    def test_dispatch_tool_then_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            responses = [
                _msg(None, [_call("get_environment", "{}")]),  # 先调工具
                _msg("最终报告：baseline 0.10ms → best 0.06ms，加速 1.67x。"),  # 再给结论
            ]
            index = {"i": 0}

            def fake_chat(messages, tools=None, temperature=0.2):
                response = responses[index["i"]]
                index["i"] += 1
                return response

            actions = []
            with mock.patch.object(llm_module, "chat", fake_chat):
                report = run_agent(ws.root, on_step=lambda s, a, o: actions.append(a))

            self.assertIn("加速", report)
            self.assertIn("tool:get_environment", actions)
            self.assertIn("done", actions)

    def test_bad_tool_args_fed_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            responses = [
                _msg(None, [_call("get_environment", "{bad json")]),  # 参数非法
                _msg("收尾"),
            ]
            index = {"i": 0}

            def fake_chat(messages, tools=None, temperature=0.2):
                response = responses[index["i"]]
                index["i"] += 1
                return response

            with mock.patch.object(llm_module, "chat", fake_chat):
                report = run_agent(ws.root)
            self.assertEqual(report, "收尾")

    def test_text_only_opening_is_nudged_then_backstop(self) -> None:
        # 模型一直只用文本、从不调工具：被拦 _MAX_NUDGES 次后兜底返回。
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            responses = [_msg(f"文本{i}") for i in range(6)]
            index = {"i": 0}

            def fake_chat(messages, tools=None, temperature=0.2):
                response = responses[index["i"]]
                index["i"] += 1
                return response

            actions = []
            with mock.patch.object(llm_module, "chat", fake_chat):
                report = run_agent(ws.root, on_step=lambda s, a, o: actions.append(a))
            self.assertEqual(report, "文本3")  # 拦 3 次后第 4 条兜底返回
            self.assertEqual(actions.count("nudge"), 3)

    def test_question_after_tool_is_nudged(self) -> None:
        # 调过工具后，把提问写进最终回答 → 拦下；随后给出真报告 → 返回。
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            responses = [
                _msg(None, [_call("get_environment", "{}")]),  # 先调一个工具
                _msg("请确认：size 定多少？"),                  # 提问当最终回答 → 拦
                _msg("最终报告：baseline→best 加速 2x。"),       # 真报告 → 返回
            ]
            index = {"i": 0}

            def fake_chat(messages, tools=None, temperature=0.2):
                response = responses[index["i"]]
                index["i"] += 1
                return response

            actions = []
            with mock.patch.object(llm_module, "chat", fake_chat):
                report = run_agent(ws.root, on_step=lambda s, a, o: actions.append(a))
            self.assertIn("加速", report)
            self.assertNotIn("请确认", report)
            self.assertIn("nudge", actions)


# ---------------------------------------------------------------------------
# batch / fixtures
# ---------------------------------------------------------------------------


class BatchFixtureTest(unittest.TestCase):
    def test_seed_and_batch_ask_user(self) -> None:
        from kernel_optimize.__main__ import _resolve_fixture, _seed_workspace_from_fixture

        fixture = Path(__file__).resolve().parents[2] / "chapter15" / "fixtures" / "vector_add"
        self.assertTrue(fixture.is_dir())
        resolved = _resolve_fixture(fixture)
        with tempfile.TemporaryDirectory() as tmp:
            ws_path = Path(tmp) / "ws"
            _seed_workspace_from_fixture(resolved, ws_path)
            self.assertTrue((ws_path / "task.json").is_file())
            self.assertTrue((ws_path / "reference.py").is_file())
            self.assertTrue((ws_path / "best.py").is_file())
            ws = Workspace(ws_path)
            ex, _ = build_tools(ws, batch=True)
            reply = ex.call("ask_user", {"question": "shape?"})
            self.assertIn("非交互", reply)


if __name__ == "__main__":
    unittest.main()
