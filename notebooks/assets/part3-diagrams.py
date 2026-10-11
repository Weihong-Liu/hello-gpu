"""Generate the editable Part 3 concept figures (no measured results).

Run with Python's standard library. Notebook authoring converts the SVG files
to cached PNGs for JupyterLab's Markdown attachment renderer.
"""
from __future__ import annotations

from html import escape
from pathlib import Path


HERE = Path(__file__).resolve().parent
BG = "#f8faf9"
INK = "#173d4e"
MUTED = "#445a62"
BLUE = "#edf4f5"
BLUE_EDGE = "#497789"
ORANGE = "#fff0db"
ORANGE_EDGE = "#ad782e"
GRAY = "#edf0ef"
GRAY_EDGE = "#87989d"


class Figure:
    def __init__(self, title: str, height: int):
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1240 {height}" role="img">',
            f"<title>{escape(title)}</title>",
            '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
            f'<path d="M0 0 L8 4 L0 8 Z" fill="{BLUE_EDGE}"/></marker></defs>',
            f'<rect width="1240" height="{height}" fill="{BG}"/>',
        ]
        self.text(24, 40, title, size=25)

    def rect(self, x, y, width, height, fill=BLUE, stroke=BLUE_EDGE, dash=False):
        dashed = ' stroke-dasharray="6 4"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dashed}/>'
        )

    def text(self, x, y, value, size=19, color=INK, anchor="start", weight="normal"):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" '
            f'fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>'
        )

    def arrow(self, x1, y1, x2, y2):
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{BLUE_EDGE}" stroke-width="2" marker-end="url(#arrow)"/>'
        )

    def write(self, filename):
        path = HERE / filename
        path.write_text("\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")
        return path


def vector_add():
    fig = Figure("从一对输入到一个输出：FP16 向量加法", 480)
    fig.text(24, 72, "手算例子：每个位置执行 z[i] = x[i] + y[i]。下面的数组只用于解释任务。", size=17, color=MUTED)
    for y, name, values in [(110, "x", [1, 2, 3, 4]), (210, "y", [10, 20, 30, 40])]:
        fig.rect(24, y, 442, 76)
        fig.text(48, y + 47, name, size=24, weight="bold")
        for index, value in enumerate(values):
            x = 112 + index * 82
            fig.rect(x, y + 13, 64, 50, fill=BG)
            fig.text(x + 32, y + 47, value, size=23, anchor="middle")
    fig.rect(536, 123, 154, 150)
    fig.text(613, 165, "同一位置", size=21, anchor="middle")
    fig.text(613, 225, "+", size=48, anchor="middle")
    fig.arrow(474, 148, 529, 164)
    fig.arrow(474, 248, 529, 231)
    fig.arrow(698, 198, 757, 198)
    fig.rect(768, 123, 448, 150, fill=ORANGE, stroke=ORANGE_EDGE)
    fig.text(790, 161, "输出 z", size=22, weight="bold")
    for index, value in enumerate([11, 22, 33, 44]):
        x = 804 + index * 96
        fig.rect(x, 184, 76, 54, fill=BG, stroke=ORANGE_EDGE)
        fig.text(x + 38, 220, value, size=23, anchor="middle")
    cards = [
        (24, "读两份输入", "2 × 2 B = 4 B", BLUE, BLUE_EDGE),
        (434, "做一次加法", "1 FLOP / 元素", BLUE, BLUE_EDGE),
        (844, "写一份输出", "1 × 2 B = 2 B", ORANGE, ORANGE_EDGE),
    ]
    for x, title, detail, fill, stroke in cards:
        fig.rect(x, 312, 372, 88, fill=fill, stroke=stroke)
        fig.text(x + 20, 343, title, size=20, weight="bold")
        fig.text(x + 20, 377, detail, size=22)
    fig.text(24, 437, "逻辑算术强度 AI = 1 / (4 + 2) = 1/6 FLOP/B", size=21, weight="bold")
    fig.text(24, 467, "逻辑读写量不等于实测显存流量；缓存与物理事务需要另外测量。", size=17, color=MUTED)
    return fig.write("part3-vector-add-task.svg")


def tail_mask():
    fig = Figure("Triton 的尾部 mask：只读写有效的逻辑位置", 504)
    fig.text(24, 74, "手算例子：n = 10，BLOCK = 4，启动 ceil(10 / 4) = 3 个 program。", size=18, color=MUTED)
    for program in range(3):
        x = 24 + program * 410
        fig.rect(x, 104, 372, 258)
        fig.text(x + 20, 139, f"program {program}", size=23, weight="bold")
        fig.text(x + 20, 174, f"offsets = {program * 4} + [0, 1, 2, 3]", size=19)
        for local in range(4):
            index = program * 4 + local
            valid = index < 10
            cell_x = x + 20 + local * 84
            fig.rect(cell_x, 194, 70, 58, fill=BLUE if valid else GRAY,
                     stroke=BLUE_EDGE if valid else GRAY_EDGE, dash=not valid)
            fig.text(cell_x + 35, 233, index, size=27, color=INK if valid else MUTED, anchor="middle")
            fig.text(cell_x + 35, 282, "True" if valid else "False", size=19,
                     color=INK if valid else MUTED, anchor="middle")
        fig.text(x + 20, 324, "全部有效" if program < 2 else "只读写 8、9；跳过 10、11", size=19)
    fig.text(24, 400, "mask = offsets < n", size=23, weight="bold")
    fig.text(24, 434, "tl.load(..., mask=mask, other=0)；tl.store(..., mask=mask)", size=21)
    fig.text(24, 472, "BLOCK 描述逻辑数据片段；这些格子不表示物理线程或 wavefront lane。", size=18, color=MUTED)
    return fig.write("part3-triton-tail-mask.svg")


def pending_candidate():
    fig = Figure("一轮候选的交接：测过的源码才能进入本轮接受判定", 472)
    fig.text(24, 74, "协议示意：A、B 是两份不同源码；示意不预设 A 的性能，也不保证最终接受。", size=17, color=MUTED)
    steps = [
        ("1  检查 A", ["通过正确性检查", "才继续计时", "失败时先修复源码"], BLUE, BLUE_EDGE),
        ("2  计时 A", ["保存 A 的测量", "pending = A", "当前实现暂不更新"], BLUE, BLUE_EDGE),
        ("3  尝试接受 B", ["B 不是待判定的 A", "Agent 协议拒绝", "pending 仍是 A"], ORANGE, ORANGE_EDGE),
        ("4  对 A 完成判定", ["独立配对复测", "满足条件：更新当前", "未满足条件：保持当前"], BLUE, BLUE_EDGE),
    ]
    for index, (title, lines, fill, edge) in enumerate(steps):
        x = 24 + index * 305
        fig.rect(x, 108, 277, 230, fill=fill, stroke=edge)
        fig.text(x + 17, 147, title, size=21, weight="bold")
        for line_index, line in enumerate(lines):
            fig.text(x + 17, 198 + line_index * 47, line, size=19)
        if index < 3:
            fig.arrow(x + 282, 222, x + 299, 222)
    fig.text(24, 381, "有计时结果 ≠ 已接受；对 A 完成判定后，才结束这一轮交接。", size=21, weight="bold")
    fig.text(24, 421, "正确性失败、源码交接的协议拒绝、性能判定未接受，属于不同阶段。", size=18, color=MUTED)
    fig.text(24, 456, "学习者运行时看到的状态和接受结果，以实际记录为准。", size=17, color=MUTED)
    return fig.write("part3-pending-candidate.svg")


def action_loop():
    fig = Figure("一个行动的闭环：提出、执行、观察，再决定下一步", 424)
    fig.text(24, 74, "先手动提出一个行动，观察实际反馈；后续再让模型承担提出行动的工作。", size=18, color=MUTED)
    steps = [
        ("提出行动", ["工具名 + 参数", "手写或由模型提出"]),
        ("执行工具", ["ToolExecutor 派发", "实际运行对应函数"]),
        ("得到观察", ["成功／失败 + 执行结果", "写入历史记录"]),
        ("选择下一步", ["依据反馈修改方案", "继续调用或停止"]),
    ]
    for index, (title, lines) in enumerate(steps):
        x = 24 + index * 305
        fig.rect(x, 110, 277, 180)
        fig.text(x + 18, 151, title, size=23, weight="bold")
        for line_index, line in enumerate(lines):
            fig.text(x + 18, 208 + line_index * 43, line, size=19)
        if index < 3:
            fig.arrow(x + 282, 197, x + 299, 197)
    fig.parts.append(
        f'<path d="M 1078 300 V 338 H 163 V 300" fill="none" '
        f'stroke="{BLUE_EDGE}" stroke-width="2" marker-end="url(#arrow)"/>'
    )
    fig.text(620, 326, "把观察作为下一次提出行动的依据", size=19, anchor="middle")
    fig.text(24, 395, "模型负责提出行动；kernel 的正确性与耗时，来自工具的真实执行。", size=19, color=MUTED)
    return fig.write("part3-action-loop.svg")


def independent_retest():
    fig = Figure("结束搜索后：固定最终实现，再独立复测", 454)
    fig.text(24, 74, "搜索时每轮对照可能变化；最终结论始终比较初始基线与固定后的最终实现。", size=18, color=MUTED)
    steps = [
        ("初始基线 B0", ["整个搜索期保持固定", "用于最终对照"]),
        ("搜索中的当前版本", ["接受候选才更新", "每轮有自己的对照"]),
        ("固定最终版本 Cfinal", ["搜索结束后不再修改", "保留对应源码身份"]),
        ("新进程配对复测", ["B0 与 Cfinal 交替运行", "保留各次独立结果"]),
    ]
    for index, (title, lines) in enumerate(steps):
        x = 24 + index * 305
        fig.rect(x, 110, 277, 180, fill=ORANGE if index == 3 else BLUE,
                 stroke=ORANGE_EDGE if index == 3 else BLUE_EDGE)
        fig.text(x + 17, 151, title, size=21, weight="bold")
        for line_index, line in enumerate(lines):
            fig.text(x + 17, 208 + line_index * 43, line, size=19)
        if index < 3:
            fig.arrow(x + 282, 197, x + 299, 197)
    fig.parts.append(
        f'<path d="M 162 299 V 345 H 1078 V 299" fill="none" '
        f'stroke="{BLUE_EDGE}" stroke-width="2" marker-end="url(#arrow)"/>'
    )
    fig.text(620, 334, "同一份初始基线直接进入最终复测", size=19, anchor="middle")
    fig.text(24, 391, "最终收益来自新一组配对结果；不把逐轮改善率相乘，也不只挑最快的一次。", size=19, weight="bold")
    fig.text(24, 428, "复测若没有稳定改善，保留真实结果，并说明当前证据不足。", size=18, color=MUTED)
    return fig.write("part3-independent-retest.svg")


if __name__ == "__main__":
    for generate in (vector_add, tail_mask, pending_candidate, action_loop, independent_retest):
        print(generate())
