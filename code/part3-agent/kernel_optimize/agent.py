"""agent 主循环：一个简单可读的 Reflection/ReAct loop（本工件的教学主角）。

骨架（hello-agents 范式，看清每个字节）：

    for step in range(max_steps):          # ① 循环上限，防死循环
        message = llm.chat(history, tools)  # ② 思考（可能决定调工具）
        if 没有工具调用:                     # ③ 判终止（给出最终报告/收敛）
            return 最终回答
        for call in message.tool_calls:     # ④ 路由到工具
            observation = executor.call(...)  # ⑤ 执行（权威工具给 ground truth）
            history.append(observation)      # ⑥ 观察回灌

LLM 自由发挥地驱动闭环、选策略、多轮对话；权威工具（compile_kernel / bench_kernel /
profile_kernel / accept_candidate / measure_peak）的返回就是真假判据。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import llm
from .prompts import SYSTEM_SOP, build_goal
from .tools import Workspace, build_tools

# 完成护栏：模型很容易「用文本说话」而不是调工具，导致一输出最终文本就结束运行。
# 因此用「状态」而非「猜文本」来判断能不能结束：只要优化闭环还没真正跑起来
# （没调过 accept_candidate / bench_kernel / compile_kernel），就不许用文本收尾。
_MAX_NUDGES = 3
_NUDGE_QUESTION = (
    "⚠ 你把要问用户的话写进了最终回答——输出最终回答会立刻结束运行，用户没法回答你。"
    "如果还需要用户确认或补充信息，请改用 ask_user 工具把问题问出去。"
)
_NUDGE_NOT_STARTED = (
    "⚠ 你只用文本回复、没有调用任何工具——文本回复会直接结束运行。"
    "请立即调用 ask_user 工具向用户要到 kernel 代码并问清参数（shape、dtype），"
    "不要只是用文字邀请用户。"
)
_NUDGE_NOT_STARTED_BATCH = (
    "⚠ 你只用文本回复、没有调用任何工具——文本回复会直接结束运行。"
    "【非交互模式】请立即调用 get_environment / measure_peak / profile_kernel，"
    "再对候选分步调用 compile_kernel → bench_kernel → accept_candidate；"
    "不要调用 ask_user，也不要用文字收尾。"
)
_NUDGE_NO_ACCEPT = (
    "⚠ 你还没有进入优化闭环（compile_kernel / bench_kernel / accept_candidate）。"
    "请按 compile_kernel → bench_kernel →（必要时）profile_kernel → accept_candidate "
    "完成至少一轮，不要只用文本收尾。"
)
_QUESTION_HINTS = (
    "？", "?", "请确认", "请问", "你觉得", "哪个", "多少", "可以吗",
    "是否", "请告诉", "请选", "要不要", "好吗", "合适吗", "行吗",
    "请贴", "请提供", "粘贴给我", "告诉我", "请说", "请描述", "请直接把",
)
_LOOP_TOOLS = frozenset({"accept_candidate", "bench_kernel", "compile_kernel"})


def _looks_like_user_question(text: str) -> bool:
    return any(hint in text for hint in _QUESTION_HINTS)


def run_agent(
    workspace_dir: str | Path,
    *,
    max_steps: int = 30,
    goal: str | None = None,
    on_step: Any = None,
    batch: bool = False,
) -> str:
    """跑优化 agent，返回最终报告文本。

    workspace_dir：含 task.json + reference.py + best.py（baseline）的工作目录。
    goal：用户的优化目标描述；缺省时从 task.json 推导。
    on_step(step, action, observation)：可选回调，用于打印进度。
    batch：非交互模式（ask_user 拒答并引导继续闭环）。
    """
    workspace = Workspace(Path(workspace_dir))
    executor, schema = build_tools(workspace, batch=batch)

    if goal is None:
        goal = build_goal(workspace, batch=batch)

    history: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_SOP},
        {"role": "user", "content": goal},
    ]
    nudges = 0
    called_tools: set[str] = set()

    for step in range(1, max_steps + 1):
        message = llm.chat(history, tools=schema)
        history.append(_assistant_to_dict(message))

        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            final = message.content or ""
            asking_user = _looks_like_user_question(final)
            entered_loop = bool(called_tools & _LOOP_TOOLS)
            premature = (not called_tools) or asking_user or (batch and not entered_loop)
            if premature and nudges < _MAX_NUDGES:
                nudges += 1
                if not called_tools:
                    nudge = _NUDGE_NOT_STARTED_BATCH if batch else _NUDGE_NOT_STARTED
                elif batch and not entered_loop:
                    nudge = _NUDGE_NO_ACCEPT
                else:
                    nudge = _NUDGE_QUESTION
                history.append({"role": "user", "content": nudge})
                if on_step:
                    on_step(step, "nudge", None)
                continue
            if on_step:
                on_step(step, "done", final)
            return final or "（无最终回答）"

        for call in tool_calls:
            name = call.function.name
            called_tools.add(name)
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as error:
                observation = f"✗ 你给 {name} 的参数不是合法 JSON：{error}。请修正后重新调用。"
                args = None
            if args is not None:
                if on_step:
                    on_step(step, f"tool:{name}", None)
                observation = executor.call(name, args)
                if on_step:
                    on_step(step, f"result:{name}", observation)
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": observation[:6000],
                }
            )

    return f"达到最大步数（{max_steps}），停止。轨迹见 {workspace.root / 'trajectory.jsonl'}。"


def _assistant_to_dict(message: Any) -> dict[str, Any]:
    """把 LiteLLM 的 assistant message 转成可回传的 dict（保留 tool_calls）。"""
    item: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        item["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in tool_calls
        ]
    return item
