"""Notebook views and single steps over Part 3's authoritative evaluator.

The learner owns the kernel and the loop. This module owns file bookkeeping,
small views, and replayable observations. It never invents timings or provides
an alternative acceptance rule. Online models are used only by ``run_online``.
"""

from __future__ import annotations

import difflib
import hashlib
import html
import importlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from .paths import artifact_root


def _part_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        part = parent / "code" / "part3-agent"
        if (part / "kernel_optimize" / "tools.py").is_file():
            return part
    raise RuntimeError("找不到 Part 3 评测代码。请保留仓库的 notebooks/ 和 code/part3-agent/ 目录。")


def _backend():
    part = str(_part_root())
    if part not in sys.path:
        sys.path.insert(0, part)
    return importlib.import_module("kernel_optimize.tools")


def __getattr__(name: str):
    if name == "ToolExecutor":
        return _backend().ToolExecutor
    raise AttributeError(name)


def _hash(source: str | bytes) -> str:
    return hashlib.sha256(source.encode() if isinstance(source, str) else source).hexdigest()


def _source_identity(source: str) -> str:
    # The evaluator stores Python sources with a final newline. The guard still
    # compares the original supplied strings, without normalizing its protocol.
    return _hash(source if source.endswith("\n") else source + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse(observation: str) -> dict:
    try:
        value = json.loads(observation)
    except (TypeError, json.JSONDecodeError):
        value = None
    if isinstance(value, dict):
        return value
    return {"ok": False, "stage": "tool", "error": observation}


def task_spec(cfg: dict) -> dict:
    """Adapt the checked-in FP16 vector-add contract without allocating tensors."""
    settings = {"n": 1 << 20, "block": 256, "warmup": 5, "repeat": 30, "seed": 0} | dict(cfg)
    for key in ("n", "block", "warmup", "repeat", "seed"):
        value = settings[key]
        minimum = 0 if key in {"warmup", "seed"} else (5 if key == "repeat" else 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} 必须是 >= {minimum} 的整数。repeat 至少为 5，才能形成配对判定证据。")
    if settings["block"] & (settings["block"] - 1):
        raise ValueError("Triton block 必须是 2 的幂。")
    n = settings["n"]
    cols = n if n <= 1_000_000 else math.gcd(n, 2048)
    rows = n // cols
    if rows > 1_000_000:
        raise ValueError("n 无法在评测器的维度限制内表达；请选择可拆成两维的元素数。")
    fixture = _part_root() / "chapter15" / "fixtures" / "vector_add" / "task.json"
    task = _read_json(fixture)
    task["description"] = "Notebook vector add: output = x + y，固定 FP16 任务。"
    task["shape"] = {"rows": rows, "cols": cols}
    task["reference"]["selfCheckPassed"] = False
    task["correctness"]["seeds"] = [settings["seed"], settings["seed"] + 1]
    task["benchmark"].update(warmup=settings["warmup"], samples=settings["repeat"])
    for tensor in task["tensors"]:
        if tensor["name"] == "n_elements":
            tensor["value"] = n
    task["costModel"].update(flops=n, bytes=3 * n * 2)
    task["costModel"]["label"] = "FP16 add: 1 FLOP/element; logical read x+y and write output (6 bytes/element)"
    return task


def _display_html(markup: str):
    from IPython.display import HTML, display

    display(HTML(markup))


def show(result: dict, title: str | None = None) -> None:
    """Show a small conclusion first and HTML-escaped full evidence on demand."""
    okay = result.get("ok", result.get("status") == "ok")
    if "accepted" in result:
        label = "接受：更新当前实现" if result["accepted"] else "拒绝：保留当前实现"
    elif okay:
        label = "通过"
    else:
        label = "未通过"
    fields = []
    for key, caption in (("status", "结果"), ("stage", "阶段"), ("median_ms", "中位时间 / ms"),
                         ("latencyMs", "候选中位时间 / ms"), ("bandwidth_gbps", "有效带宽 / GB/s"),
                         ("reason", "原因"), ("error", "错误")):
        value = result.get(key)
        if value is not None:
            fields.append(f"<div><b>{caption}</b>：{html.escape(format(value, '.3g') if isinstance(value, float) else str(value))}</div>")
    if result.get("errors"):
        fields.append("<div><b>错误</b>：" + html.escape("；".join(str(item.get("msg", item)) for item in result["errors"])) + "</div>")
    evaluation = result.get("evidence", {})
    correctness = evaluation.get("correctness") or {}
    if correctness and not correctness.get("passed", True):
        fields.append("<div><b>正确性</b>：" + html.escape(str(correctness.get("reason", correctness))) + "</div>")
    if result.get("improvementFraction") is not None:
        fields.append(f"<div><b>本轮配对改善</b>：{result['improvementFraction']:.1%}</div>")
    environment = evaluation.get("environment") or {}
    if environment:
        fields.append("<div><b>本次环境</b>：" + html.escape(json.dumps(environment, ensure_ascii=False)) + "</div>")
    raw = html.escape(json.dumps(result, ensure_ascii=False, indent=2))
    heading = html.escape(title or result.get("tool", "工具结果"))
    _display_html(f'<div style="border:1px solid #aaa;border-radius:8px;padding:12px;margin:8px 0">'
                  f'<b>{heading} · {label}</b>{"".join(fields)}'
                  f'<details style="margin-top:8px"><summary>完整结果与证据</summary>'
                  f'<pre style="white-space:pre-wrap;max-height:360px;overflow:auto">{raw}</pre></details></div>')


def source_diff(before: str, after: str) -> None:
    """Display an exact source diff; source text never becomes executable HTML."""
    diff = "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                          fromfile="当前实现", tofile="候选实现", lineterm=""))
    _display_html('<pre style="white-space:pre-wrap;border:1px solid #aaa;padding:12px">'
                  + html.escape(diff or "源码相同。") + "</pre>")


def _gpu_ready() -> None:
    # Importing/constructing a Lab is CPU-only. Only an actual GPU tool reaches here.
    try:
        import torch
        import triton  # noqa: F401
    except ImportError as error:
        raise RuntimeError("GPU 实验需要当前 notebook Python 中的 ROCm PyTorch 和 Triton。") from error
    if not getattr(torch.version, "hip", None):
        raise RuntimeError("当前 Python 的 PyTorch 不是 ROCm 版本；请切换到已配置 ROCm 的 notebook kernel。")
    if not torch.cuda.is_available():
        raise RuntimeError("当前 notebook 看不到 AMD GPU。请检查设备分配、驱动以及 /dev/kfd、/dev/dri 的访问权限。")


_GPU_TOOLS = frozenset({"compile_kernel", "bench_kernel", "accept_candidate", "profile_kernel", "measure_peak"})
_NOTEBOOK_TOOLS = _GPU_TOOLS | {"get_environment", "read_reference"}


class Lab:
    """One fresh task workspace, never an alternative evaluator or optimizer."""

    def __init__(self, name: str, cfg: dict, *, baseline_source: str | None = None):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name):
            raise ValueError("实验名称只能包含字母、数字、下划线和连字符。")
        self.cfg = dict(cfg)
        task = task_spec(cfg)
        fixture = _part_root() / "chapter15" / "fixtures" / "vector_add"
        source = baseline_source if baseline_source is not None else (fixture / "baseline.py").read_text(encoding="utf-8")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("baseline_source 必须是完整的非空源码字符串。")
        if baseline_source is None:
            source = re.sub(r"block_size = 256[^\n]*", f"block_size = {cfg.get('block', 256)}", source)
        source = source if source.endswith("\n") else source + "\n"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:10]
        self.root = artifact_root() / "agent" / name / stamp
        self.workspace = _backend().Workspace(self.root)
        _write_json(self.workspace.task_path, task)
        self.workspace.reference_path.write_bytes((fixture / "reference.py").read_bytes())
        self.baseline_path = self.root / "baseline.py"
        self.baseline_path.write_text(source, encoding="utf-8")
        self.workspace.best_path.write_text(source, encoding="utf-8")
        self.executor, _ = _backend().build_tools(self.workspace, batch=True)
        # No implicit model calls, task regeneration, terminal questions or run_code.
        self.executor.tools = {name: tool for name, tool in self.executor.tools.items() if name in _NOTEBOOK_TOOLS}
        self._pending: str | None = None
        self._candidate: str | None = None
        self._activity = False
        self._stopped: str | None = None
        self._planner = "manual"
        self._llm_calls = 0
        self._step = 0
        _write_json(self.root / "notebook-run.json", {"planner": "manual", "llmCalls": 0})

    @property
    def task(self) -> dict:
        return self.workspace.task()

    @property
    def baseline_source(self) -> str:
        return self.baseline_path.read_text(encoding="utf-8")

    @property
    def best_path(self) -> Path:
        return self.workspace.best_path

    def task_summary(self) -> dict:
        task = self.task
        return {"任务": "output = x + y", "入口": task["entrypoint"], "shape": task["shape"],
                "元素数": task["costModel"]["flops"], "dtype": "float16",
                "逻辑流量 / bytes": task["costModel"]["bytes"],
                "warmup": task["benchmark"]["warmup"], "samples": task["benchmark"]["samples"],
                "innerRepeats": task["benchmark"]["innerRepeats"],
                "接受阈值": task["optimization"]["minImprovementFraction"]}

    def _decorate(self, result: dict, name: str, artifact: Path | None) -> dict:
        result = dict(result)
        result["tool"] = name
        if artifact is not None:
            result["evaluation"] = str(artifact)
            result["result_file"] = str(artifact / "result.json")
            if (artifact / "result.json").is_file():
                result["evidence"] = _read_json(artifact / "result.json")
            if (artifact / "candidate.py").is_file():
                result["sourceSha256"] = _hash((artifact / "candidate.py").read_bytes())
        return result

    @property
    def state(self) -> dict:
        pending = self.root / "pending_candidate.py"
        phase = self._stopped or ("pending_accept" if pending.is_file() else "candidate" if self._activity else "ready")
        return {"state": phase, "pendingCandidate": pending.is_file(),
                "candidateSha256": _source_identity(pending.read_text(encoding="utf-8")) if pending.is_file() else _source_identity(self._candidate) if self._candidate else None,
                "pendingSourceSha256": _hash(pending.read_bytes()) if pending.is_file() else None,
                "bestSha256": _hash(self.best_path.read_bytes()), "baselineSha256": _hash(self.baseline_path.read_bytes()),
                "trajectoryRows": len(_rows(self.root / "trajectory.jsonl")),
                "planner": self._planner, "llmCalls": self._llm_calls}

    def _event(self, name: str, result: dict, before: str) -> None:
        event = {"step": self._step, "recordedAt": datetime.now(timezone.utc).isoformat(),
                 "tool": name, "result": result, "state": self.state,
                 "phase": self.state["state"],
                 "bestChanged": before != _hash(self.best_path.read_bytes())}
        with (self.root / "notebook-events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _invoke(self, name: str, arguments: dict, *, guard: bool) -> dict:
        if not isinstance(arguments, dict):
            raise TypeError("工具参数必须是 dict。")
        self._step += 1
        before = _hash(self.best_path.read_bytes())
        error = None
        if guard:
            protocol = importlib.import_module("kernel_optimize.agent").pending_protocol_error
            error = protocol(name, arguments, self._pending)
            self._stopped = None
        if error:
            result = {"ok": False, "stage": "pending_accept", "error": error, "tool": name}
        else:
            if name in self.executor.tools and name in _GPU_TOOLS:
                _gpu_ready()
            self.workspace.last_evaluation = None
            rows_before = len(_rows(self.root / "trajectory.jsonl"))
            result = self._decorate(_parse(self.executor.call(name, arguments)), name, self.workspace.last_evaluation)
            if guard:
                source = arguments.get("source")
                if name in {"compile_kernel", "bench_kernel", "accept_candidate"} and isinstance(source, str):
                    self._candidate = source
                    if source != self.best_path.read_text(encoding="utf-8"):
                        self._activity = True
                if (name == "bench_kernel" and result.get("ok") is True
                        and isinstance(source, str)
                        and source != self.best_path.read_text(encoding="utf-8")):
                    self._pending = source
                    (self.root / "pending_candidate.py").write_text(source, encoding="utf-8")
                if name == "accept_candidate" and len(_rows(self.root / "trajectory.jsonl")) > rows_before:
                    self._pending = None
                    self._activity = False
                    (self.root / "pending_candidate.py").unlink(missing_ok=True)
        self._event(name, result, before)
        return result

    @staticmethod
    def _arguments(arguments: dict | None, kwargs: dict) -> dict:
        if arguments is not None and kwargs:
            raise TypeError("使用一个参数 dict，或使用关键字参数，不能同时使用。")
        return arguments if arguments is not None else kwargs

    def call(self, name: str, arguments: dict | None = None, **kwargs) -> dict:
        """Call an isolated authoritative tool, as in the Chapter 15 lesson."""
        return self._invoke(name, self._arguments(arguments, kwargs), guard=False)

    def step(self, name: str, arguments: dict | None = None, **kwargs) -> dict:
        """One visible loop step, sharing the CLI's pending-source guard."""
        return self._invoke(name, self._arguments(arguments, kwargs), guard=True)

    def finish(self) -> dict:
        self._stopped = "complete" if self.state["trajectoryRows"] and not self.state["pendingCandidate"] and not self._activity else "incomplete"
        state = self.state | {"evidenceReady": self._stopped == "complete", "report": "手动教学循环已停止；结果见实际轨迹。"}
        _write_json(self.root / "agent-status.json", state)
        return state

    def state_view(self) -> None:
        state = self.state
        labels = {"ready": "尚未提交候选", "candidate": "候选尚未完成裁决", "pending_accept": "候选等待裁决",
                  "complete": "已完成裁决", "complete_with_warning": "已有裁决，末尾仍有未完成尝试", "incomplete": "已停止，证据未完成"}
        _display_html('<div style="border:1px solid #aaa;padding:10px;margin:8px 0">'
                      f'<b>{labels.get(state["state"], state["state"])}</b> · 已裁决 {state["trajectoryRows"]} 轮'
                      f' · 待裁决：{"是" if state["pendingCandidate"] else "否"} · 模型调用 {state["llmCalls"]}'
                      f'<div>当前实现：{state["bestSha256"][:12]} · 候选：{(state["candidateSha256"] or "未提交")[:12]}</div></div>')

    def show(self, result: dict, title: str | None = None) -> None:
        show(result, title)

    def preflight(self) -> dict:
        result = self.call("bench_kernel", source=self.baseline_source)
        if result.get("ok"):
            _write_json(self.root / "preflight.json", result | {"taskSha256": _hash(self.workspace.task_path.read_bytes()),
                                                               "referenceSha256": _hash(self.workspace.reference_path.read_bytes()),
                                                               "baselineSha256": _hash(self.baseline_path.read_bytes())})
        return result

    def _require_preflight(self) -> None:
        path = self.root / "preflight.json"
        if not path.is_file():
            raise RuntimeError("请先运行 lab.preflight()，确认固定基线正确且可计时。")
        preflight = _read_json(path)
        expected = {"taskSha256": _hash(self.workspace.task_path.read_bytes()),
                    "referenceSha256": _hash(self.workspace.reference_path.read_bytes()),
                    "baselineSha256": _hash(self.baseline_path.read_bytes())}
        if any(preflight.get(key) != value for key, value in expected.items()):
            raise RuntimeError("任务、正确性参考或基线已改变，请创建新的 Lab 并重新做 preflight。")

    def table(self):
        import pandas as pd

        rows = _rows(self.root / "trajectory.jsonl")
        columns = ["round", "change", "status", "latencyMs", "improvementFraction", "accepted", "reason"]
        return pd.DataFrame([{"round": index, **row} for index, row in enumerate(rows, 1)]).reindex(columns=columns)

    def events(self):
        import pandas as pd

        return pd.DataFrame(_rows(self.root / "notebook-events.jsonl"))

    def inspect_round(self, index: int | None = None, *, change: str | None = None) -> None:
        """Review the frozen source and paired evidence of one actual judgement."""
        rows = _rows(self.root / "trajectory.jsonl")
        if change is not None:
            selected = [row for row in rows if row.get("change") == change]
            if not selected:
                raise ValueError(f"没有名为 {change!r} 的裁决。")
            row = selected[-1]
        else:
            if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= len(rows):
                raise ValueError("index 是从 1 开始的实际裁决轮次。")
            row = rows[index - 1]
        artifact = self.root / row["evaluation"]
        candidate = (artifact / "candidate.py").read_text(encoding="utf-8")
        incumbent = (artifact / "incumbent.py").read_text(encoding="utf-8")
        source_diff(incumbent, candidate)
        result = self._decorate(row, "accept_candidate", artifact)
        show(result, row.get("change") or "实际裁决")
        if result["evidence"].get("status") == "ok" and result["evidence"].get("comparison"):
            from IPython.display import display

            figure = self.pair_plot(result)
            display(figure)
            import matplotlib.pyplot as plt
            plt.close(figure)

    @staticmethod
    def _evidence(result: dict) -> dict:
        # Always load the file attached to this observation, never "last evaluation".
        path = result.get("result_file")
        return _read_json(Path(path)) if path else result.get("evidence", {})

    def sample_plot(self, result: dict):
        import matplotlib.pyplot as plt

        evidence = self._evidence(result)
        samples = (evidence.get("benchmark") or {}).get("samplesMs") or []
        if evidence.get("status") != "ok" or not samples:
            raise ValueError("这次结果没有成功计时的样本，不能画性能图。")
        figure, ax = plt.subplots(figsize=(7, 3))
        ax.plot(range(1, len(samples) + 1), samples, "o-", label="Measured candidate")
        ax.set(xlabel="Sample", ylabel="GPU event time / ms", title="Samples from this evaluation")
        ax.legend()
        figure.tight_layout()
        return figure

    def pair_plot(self, result: dict):
        import matplotlib.pyplot as plt

        evidence = self._evidence(result)
        comparison = evidence.get("comparison") or {}
        candidate = (evidence.get("benchmark") or {}).get("samplesMs") or []
        incumbent = (comparison.get("incumbentBenchmark") or {}).get("samplesMs") or []
        improvements = comparison.get("pairedImprovements") or []
        if evidence.get("status") != "ok" or not candidate or len(candidate) != len(incumbent) or len(improvements) != len(candidate):
            raise ValueError("这次结果没有完整的配对样本，不能画配对图。")
        figure, axes = plt.subplots(1, 2, figsize=(10, 3))
        indexes = range(1, len(candidate) + 1)
        axes[0].plot(indexes, incumbent, "o-", label="Incumbent")
        axes[0].plot(indexes, candidate, "o-", label="Candidate")
        axes[0].set(xlabel="Paired sample", ylabel="GPU event time / ms")
        axes[0].legend()
        axes[1].plot(indexes, [100 * value for value in improvements], "o-")
        threshold = 100 * self.task["optimization"]["minImprovementFraction"]
        axes[1].axhline(threshold, ls="--", label="Acceptance threshold")
        axes[1].axhline(0, color="gray", lw=1)
        axes[1].set(xlabel="Paired sample", ylabel="Improvement / %")
        axes[1].legend()
        figure.tight_layout()
        return figure

    def plot(self):
        import matplotlib.pyplot as plt

        rows = _rows(self.root / "trajectory.jsonl")
        if not rows:
            raise ValueError("没有实际裁决轨迹。先完成至少一次 accept_candidate。")
        preflight = _read_json(self.root / "preflight.json") if (self.root / "preflight.json").exists() else {}
        current = (preflight.get("evidence", {}).get("latencyMs"))
        figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        current_points = []
        round_labels = []
        failed_rounds = []
        for index, row in enumerate(rows, 1):
            evidence = _read_json(self.root / row["evaluation"] / "result.json")
            latency = row.get("latencyMs")
            if latency is not None and evidence.get("status") == "ok":
                round_labels.append(str(index))
                axes[0].scatter(index, latency, marker="o" if row["accepted"] else "x", color="tab:blue")
                comparison = evidence.get("comparison") or {}
                current = latency if row["accepted"] else comparison.get("incumbentLatencyMs", current)
            else:
                # A failed round has no latency point. Put its status in the
                # round label, so it cannot overlap data or look like a zero.
                round_labels.append(f"{index}\n{row.get('status', 'failed')}")
                failed_rounds.append(index - 1)
            current_points.append(current)
            if row.get("improvementFraction") is not None:
                axes[1].scatter(index, 100 * row["improvementFraction"],
                                marker="o" if row["accepted"] else "x", color="tab:blue")
        axes[0].plot(range(1, len(rows) + 1), current_points, "--", color="tab:orange", label="Incumbent observed in this round")
        baseline = preflight.get("evidence", {}).get("latencyMs")
        if baseline is not None:
            axes[0].axhline(baseline, color="gray", ls=":", label="Initial baseline preflight")
        axes[0].set(xlabel="Round (o accepted, x rejected)", ylabel="GPU event time / ms")
        axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=8)
        axes[1].axhline(0, color="gray", lw=1)
        axes[1].axhline(100 * self.task["optimization"]["minImprovementFraction"], ls="--")
        axes[1].set(xlabel="Round", ylabel="Paired improvement / %")
        for ax in axes:
            ax.set_xticks(range(1, len(rows) + 1), round_labels, fontsize=8)
            for index in failed_rounds:
                ax.get_xticklabels()[index].set_color("#a33b22")
        figure.tight_layout()
        return figure

    def _run(self, *, chat, planner: str, max_steps: int, on_step, live: bool) -> dict:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 100:
            raise ValueError("max_steps 必须是 1 到 100 的整数。")
        self._require_preflight()
        _gpu_ready()
        self._planner = planner
        self._stopped = None
        observed = set((self.root / "evaluations").glob("evaluation-*"))
        before_hash = _hash(self.best_path.read_bytes())

        def callback(step, action, observation):
            nonlocal observed, before_hash
            if action.startswith("tool:"):
                before_hash = _hash(self.best_path.read_bytes())
                if live:
                    _display_html(f'<div><b>第 {step} 步</b> · 执行 {html.escape(action[5:])}</div>')
            elif action.startswith("result:"):
                self._step += 1
                fresh = set((self.root / "evaluations").glob("evaluation-*")) - observed
                observed |= fresh
                artifact = next(iter(fresh)) if len(fresh) == 1 else None
                result = self._decorate(_parse(observation), action[7:], artifact)
                if artifact is not None and (artifact / "candidate.py").exists():
                    self._candidate = (artifact / "candidate.py").read_text(encoding="utf-8")
                    if action[7:] in {"compile_kernel", "bench_kernel"}:
                        self._activity = self._candidate != self.best_path.read_text(encoding="utf-8")
                if action[7:] == "accept_candidate" and "accepted" in result:
                    self._activity = False
                self._event(action[7:], result, before_hash)
                before_hash = _hash(self.best_path.read_bytes())
                if live:
                    show(result, f"第 {step} 步 · {action[7:]}")
            if on_step:
                on_step(step, action, observation)

        runner = importlib.import_module("kernel_optimize.agent").run_agent
        try:
            report = runner(self.root, max_steps=max_steps, batch=True, chat=chat,
                            allowed_tools=set(_NOTEBOOK_TOOLS), on_step=callback)
        except BaseException:
            self._stopped = "incomplete"
            pending = self.root / "pending_candidate.py"
            self._pending = pending.read_text(encoding="utf-8") if pending.is_file() else None
            _write_json(self.root / "agent-status.json", self.state | {
                "evidenceReady": False, "report": "运行中断；请检查实际工具记录，未完成的候选仍需裁决。"})
            raise
        finally:
            _write_json(self.root / "notebook-run.json", {"planner": planner, "llmCalls": self._llm_calls})
        status = _read_json(self.root / "agent-status.json")
        self._stopped = status["state"]
        self._activity = status.get("pendingCandidate", False)
        pending = self.root / "pending_candidate.py"
        self._pending = pending.read_text(encoding="utf-8") if pending.exists() else None
        return status | {"planner": planner, "llmCalls": self._llm_calls, "report": report,
                         "workspace": str(self.root)}

    def run(self, candidates: list[dict], *, max_steps: int | None = None, on_step=None, live: bool = True) -> dict:
        """A disclosed deterministic planner over the SAME agent loop, no model calls."""
        actions = []
        for candidate in candidates:
            if not isinstance(candidate.get("source"), str):
                raise ValueError("每个候选需要完整的 source 字符串和可选的 change。")
            arguments = {"source": candidate["source"]}
            actions.extend([("compile_kernel", arguments), ("bench_kernel", arguments),
                            ("accept_candidate", arguments | {"change": candidate.get("change", "教学候选")})])
        queue = iter(actions)
        call_count = 0

        def scripted(history, *, tools):
            nonlocal call_count
            action = next(queue, None)
            if action and action[0] == "bench_kernel" and history[-1].get("role") == "tool" and _parse(history[-1]["content"]).get("ok") is False:
                action = next(queue, None)
            if action is None:
                return SimpleNamespace(content="预设策略已结束；结论以工具轨迹和独立复测为准。", tool_calls=None)
            call_count += 1
            name, arguments = action
            return SimpleNamespace(content="预设教学行动", tool_calls=[SimpleNamespace(
                id=f"scripted-{call_count}", function=SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))])

        return self._run(chat=scripted, planner="scripted", max_steps=max_steps if max_steps is not None else max(4, len(actions) + 1), on_step=on_step, live=live)

    def run_online(self, *, max_steps: int = 12, on_step=None, live: bool = True) -> dict:
        """Explicitly opt into the configured online model; never called by ``run``."""
        llm = importlib.import_module("kernel_optimize.llm")
        llm._model_config()  # Validate the existing configuration without displaying credentials.

        def online(history, *, tools):
            self._llm_calls += 1
            return llm.chat(history, tools=tools)

        return self._run(chat=online, planner="online", max_steps=max_steps, on_step=on_step, live=live)

    def final_retest(self, *, runs: int = 3) -> dict:
        """Use compare.py's independent fresh processes against the frozen baseline."""
        if isinstance(runs, bool) or not isinstance(runs, int) or runs < 3:
            raise ValueError("独立复测至少需要 3 次。")
        self._require_preflight()
        if (self.root / "pending_candidate.py").exists():
            raise RuntimeError("还有候选等待裁决；请先完成 accept_candidate。")
        _gpu_ready()
        directory = self.root / ("retest-" + uuid4().hex[:10])
        directory.mkdir()
        for name, path in (("task.json", self.workspace.task_path), ("reference.py", self.workspace.reference_path),
                           ("baseline.py", self.baseline_path), ("candidate.py", self.best_path)):
            (directory / name).write_bytes(path.read_bytes())
        command = [sys.executable, str(_part_root() / "chapter14" / "compare.py"),
                   "--task", str(directory / "task.json"), "--candidate", str(directory / "candidate.py"),
                   "--incumbent", str(directory / "baseline.py"), "--runs", str(runs)]
        _write_json(directory / "request.json", {"command": command, "runs": runs,
                                                 "taskSha256": _hash((directory / "task.json").read_bytes()),
                                                 "referenceSha256": _hash((directory / "reference.py").read_bytes()),
                                                 "baselineSha256": _hash((directory / "baseline.py").read_bytes()),
                                                 "candidateSha256": _hash((directory / "candidate.py").read_bytes())})
        timeout = runs * self.task["benchmark"]["timeoutSeconds"] + 60
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            (directory / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (directory / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            report = _parse(completed.stdout)
            if completed.returncode and report.get("status") != "failed":
                report = {"status": "failed", "error": completed.stderr[-2000:] or completed.stdout[-2000:]}
        except subprocess.TimeoutExpired:
            report = {"status": "failed", "error": "独立复测进程超时，没有可报告的性能结论。"}
        report.update(directory=str(directory), taskSha256=_hash((directory / "task.json").read_bytes()),
                      referenceSha256=_hash((directory / "reference.py").read_bytes()),
                      baselineSha256=_hash((directory / "baseline.py").read_bytes()),
                      candidateSha256=_hash((directory / "candidate.py").read_bytes()))
        _write_json(directory / "report.json", report)
        _write_json(self.root / "final-retest.json", report)
        return report

    def retest_plot(self, report: dict):
        import matplotlib.pyplot as plt

        if report.get("status") != "ok":
            raise ValueError("独立复测未成功，不能画性能结论。")
        baseline = report["incumbentMediansMs"]
        candidate = report["candidateMediansMs"]
        figure, axes = plt.subplots(1, 2, figsize=(10, 3))
        indexes = range(1, len(baseline) + 1)
        axes[0].plot(indexes, baseline, "o-", label="Frozen baseline")
        axes[0].plot(indexes, candidate, "o-", label="Final incumbent")
        axes[0].set(xlabel="Independent process", ylabel="Median GPU event time / ms")
        axes[0].legend()
        axes[1].plot(indexes, [100 * value for value in report["pairedImprovementFractions"]], "o-")
        axes[1].axhline(0, color="gray", lw=1)
        axes[1].set(xlabel="Independent process", ylabel="Paired improvement / %")
        figure.tight_layout()
        return figure

    def report(self, retest: dict | None = None) -> dict:
        rows = _rows(self.root / "trajectory.jsonl")
        retest = retest if retest is not None else _read_json(self.root / "final-retest.json") if (self.root / "final-retest.json").exists() else {}
        if retest.get("status") == "ok":
            expected = {"taskSha256": _hash(self.workspace.task_path.read_bytes()),
                        "referenceSha256": _hash(self.workspace.reference_path.read_bytes()),
                        "baselineSha256": _hash(self.baseline_path.read_bytes()),
                        "candidateSha256": _hash(self.best_path.read_bytes())}
            if any(retest.get(key) != value for key, value in expected.items()):
                raise ValueError("复测对应的任务、基线或最终实现已改变。请重新 final_retest()，不能把旧复测归到当前实现。")
        report = {"planner": self._planner, "llmCalls": self._llm_calls, "task": self.task_summary(),
                  "rounds": len(rows), "acceptedRounds": sum(bool(row.get("accepted")) for row in rows),
                  "state": self.state, "retest": retest,
                  "interpretation": "改善来自固定基线与最终实现的独立配对复测；不累乘各轮收益。"}
        if retest.get("status") == "ok":
            fraction = retest["medianPairedImprovementFraction"]
            print(f"完成 {len(rows)} 轮实际裁决，接受 {report['acceptedRounds']} 轮；规划器 {self._planner}，模型调用 {self._llm_calls} 次。")
            print(f"最终实现与固定基线独立复测 {retest['runs']} 次，配对改善中位数 {fraction:.1%}。")
            positive = sum(value > 0 for value in retest["pairedImprovementFractions"])
            print(f"其中 {positive}/{retest['runs']} 次配对改善为正；请结合接受阈值和各次波动判断稳定性。")
            if fraction <= 0:
                print("本次中位估计没有改善；保留这一结果，不把失败改写成加速。")
        else:
            print("尚无成功的独立复测，不能据此报告最终加速。")
        _write_json(self.root / "notebook-report.json", report)
        return report
