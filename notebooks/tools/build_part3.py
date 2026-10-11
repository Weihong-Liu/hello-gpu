"""Author Part 3 as small, editable experiments with actual evaluator feedback.

The notebook bridge owns file/process plumbing and presentation. The learner
owns SOURCE and visible action lists. Lab.call executes isolated tools; Lab.step
also enforces the same pending-source protocol as the CLI; Lab.run uses the real
Agent loop with a disclosed deterministic planner. Only run_online calls a model.
Plots read the evidence attached to each invocation. inspect_round follows a
trajectory record to its frozen source and samples, and final_retest freezes
the initial baseline and selected implementation before fresh-process comparison.
"""
from authoring import Notebook


SETUP = '''import json
import pandas as pd
from IPython.display import display
import hello_gpu as gpu
from hello_gpu import agentlab
cfg = gpu.settings()'''

# Keep the complete candidate in the teaching cell. The bridge manages files and
# evaluation; it does not generate the implementation learners are studying.
SOURCE = '''SOURCE = """import triton
import triton.language as tl

@triton.jit
def add_kernel(x, y, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    a = tl.load(x + offsets, mask=valid, other=0)
    b = tl.load(y + offsets, mask=valid, other=0)
    result = a + b
    tl.store(output + offsets, result, mask=valid)

def launch(x, y, output, n_elements):
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    add_kernel[grid](x, y, output, n_elements, BLOCK=block_size,
                     num_warps=4)
"""
SOURCE = SOURCE.replace("block_size = 256", f"block_size = {cfg['block']}")'''


def observation(nb, text):
    nb.md("**观察**：" + text)


def summary(nb, text, next_chapter=None):
    nb.md("## 本章小结\n\n" + text)
    links = ["- [本章正文](../../../docs/part3-agent/chapter%d/index.md)" % nb.chapter,
             "- [Triton 向量加法教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)"]
    if next_chapter:
        links.append(f"- [第{next_chapter}章 notebook](../chapter{next_chapter}/chapter{next_chapter}.ipynb)")
    nb.md("## 延伸阅读\n\n" + "\n".join(links))
    nb.write()


def chapter14():
    nb = Notebook("part3-agent", 14)
    nb.md('''# 第14章 Agent 入门

## 本章导读

第 8 章中，你修改 kernel、运行检查，再根据时间决定是否保留。现在把这件事交给 Agent：它怎样提出行动，怎样得到真实反馈？

本章先亲手调用一个小工具，观察错误与恢复。工具只计算任务的逻辑工作量；GPU 正确性与计时将在第 15 章接上。默认流程无需模型密钥。

前置：[第8章 notebook](../../part2-kernels/chapter8/chapter8.ipynb)。对照阅读：[本章正文](../../../docs/part3-agent/chapter14/index.md)。''')
    nb.code(SETUP + "\nenv = gpu.environment(require_gpu=False)\nenv")
    nb.md('''## 14.1 从一次手动优化说起

先核对向量加法的含义。每个输出只依赖同位置的两个输入；修改工作分配时，这个答案不能变化。运行后尝试修改一个输入，预测哪一个输出会改变。''')
    nb.code('''x = [1.0, -2.0, 0.5, 3.0]
y = [2.0, 4.0, -0.5, -1.0]
expected = [a + b for a, b in zip(x, y)]
display(pd.DataFrame({"x": x, "y": y, "正确输出 x+y": expected}))''')
    observation(nb, "这个例子回答了“要算什么”。它没有测量 GPU 时间。优化需要在同一数学任务下提出改动、运行工具，再根据返回结果决定下一步。")
    nb.figure("notebooks/assets/part3-action-loop.svg",
              "行动来自学习者或模型，观察来自实际工具执行；下一步应依据返回结果。",
              "提出行动、执行工具、读取观察和继续行动的闭环")
    nb.md('''## 14.2 把算子优化拆成一个闭环

四章都使用 FP16 向量加法。我们固定输入规模、正确性容差和计时方式，只改变候选源码。先查看将交给工具的任务；这里的数量来自配置，性能结果要等真正运行之后才产生。''')
    nb.code('''task = agentlab.task_spec(cfg)
display(pd.Series({
    "元素数": task["tensors"][-1]["value"],
    "数据类型": task["tensors"][0]["dtype"],
    "调用入口": task["entrypoint"],
    "预热次数": task["benchmark"]["warmup"],
    "时间样本数": task["benchmark"]["samples"],
    "逻辑字节数": task["costModel"]["bytes"],
}))''')
    nb.md('''规划者负责提出一个可检查的改动；工具负责对答案和测量；工作区负责保留任务、源码和记录。规划者说“应该更快”只是一个假设，不能替代工具输出。

## 14.3 理解 ReAct 与反思

ReAct 把推理（Reason）与行动（Act）连接起来。先写出一项具体假设，再决定需要调用什么工具。反思要说明观察支持了什么、还缺什么证据。''')
    nb.code('''hypothesis = {
    "改动": f"block_size 从 {cfg['block']} 改为 {cfg['block'] * 2}",
    "保持固定": "输入、数据类型、数学语义、计时方式",
    "先做什么": "检查答案，然后配对比较候选和当前实现",
    "尚未知道": "是否更快，以及变化是否由资源使用造成",
}
display(pd.Series(hypothesis))''')
    observation(nb, "“block 变大”是行动计划；“快了多少”必须由测量回答。没有硬件资源证据时，也不能从速度变化直接推断寄存器或占用率变化。")
    nb.md('''## 14.4 模型怎样调用工具

先让你代替模型发出一次行动。下面注册的工具与后续 GPU 工具使用同一个 `ToolExecutor`。函数返回 JSON 文本，调度器根据名字和参数执行函数；参数检查写在函数中。''')
    nb.code('''tools = agentlab.ToolExecutor()
def describe_vector_add(elements):
    if not isinstance(elements, int) or isinstance(elements, bool) or elements < 1:
        raise ValueError("elements 必须是正整数")
    return json.dumps({"elements": elements, "flops": elements,
                       "logical_bytes": 3 * elements * 2}, ensure_ascii=False)

tools.register("describe_vector_add", "计算 FP16 加法的逻辑工作量",
               describe_vector_add,
               {"type": "object", "properties": {
                   "elements": {"type": "integer", "minimum": 1}},
                "required": ["elements"]})''')
    nb.code('''action = {"name": "describe_vector_add", "arguments": {"elements": 8}}
observation_text = tools.call(action["name"], action["arguments"])
observed = json.loads(observation_text)
display(pd.Series(observed, name="实际工具返回"))
assert observed["logical_bytes"] == 3 * action["arguments"]["elements"] * 2''')
    nb.md('''把 `elements` 改成另一个正整数，再运行调用单元格。接着观察两种错误：工具名字不存在，以及参数不符合函数要求。错误也应成为可以读取的反馈。''')
    nb.code('''unknown_tool = tools.call("measure_speed_without_running", {})
bad_argument = tools.call("describe_vector_add", {"elements": -1})
display(pd.Series({"未知工具": unknown_tool, "非法参数": bad_argument}))
recovered = json.loads(tools.call("describe_vector_add", {"elements": 4097}))
assert recovered["elements"] == 4097
display(pd.Series(recovered, name="修复后的调用"))''')
    observation(nb, "我们实际执行了函数，也读到了失败原因。工具说明告诉规划者怎样调用；函数实现负责执行和检查。模型接入后生成同样的行动结构，真实 GPU 时间仍由工具采集。")
    nb.md('''## 14.5 明确这套 Agent 能做什么

本篇的优化对象是固定任务中的 Triton 候选。第 8 章的 HIP 路线仍可用来理解同一个算子；当前 Agent 的候选检查和调用入口采用 Triton 合同。

本章由你手写行动，第 16–17 章会用预设策略演示自动循环，第 17 章另有显式开启的在线模型实验。预设策略不产生模型调用费用，也不等同于模型自主生成改动。我们不会预先保证任何配置提速。

## 14.6 练习：判断一份报告是否可信

先判断下面两句话，再运行自检：

- “这次工具计算了 4097 个元素的逻辑读写量。”
- “这份 kernel 更快，因为修改了 block。”

第一句能用工具返回值核对；第二句还缺正确性、相同条件下的测量和接受判定。请把缺少的证据写进你的报告。''')
    nb.code('''claim = "工具返回的 FP16 逻辑工作量"
assert recovered["logical_bytes"] == 3 * recovered["elements"] * 2
reflection = "修改配置后，要先验证，再配对比较；当前还没有 GPU 计时证据。"
print(claim, "：自检通过")
print("我的下一步：", reflection)''')
    summary(nb, "你已亲手注册、调用和修复一个工具，并区分了计划与证据。下一章把同样的调用方式接到真实 kernel 检查和计时上。学完后，应能根据工具返回值修正行动，也能指出一份性能报告缺少什么。", 15)


def chapter15():
    nb = Notebook("part3-agent", 15)
    nb.md('''# 第15章 工具封装

## 本章导读

候选 kernel 能编译，但它算对了吗？一次时间更短，就值得保留吗？我们从可编辑的向量加法开始，故意写错一次，修复后观察时间样本，再完成接受判定。

本页图表来自当前运行，不预设哪一个 block 更快。默认流程使用真实 GPU 工具，无需模型密钥。前置：[第14章](../chapter14/chapter14.ipynb)，对照阅读：[本章正文](../../../docs/part3-agent/chapter15/index.md)。''')
    nb.code(SETUP + "\nenv = gpu.environment()\nenv")
    nb.md('''## 15.1 先固定任务，再设计工具

继续计算 `output = x + y`。FP16 每个元素占 2 Byte：读取两个输入、写一个输出，逻辑数据量为 $6N$ Byte。这个模型帮助解释工作量，实际时间要通过运行得到。''')
    nb.figure("notebooks/assets/part3-vector-add-task.svg",
              "每个位置读两个输入、做一次加法、写一个输出；逻辑读写量不等于实测流量。",
              "FP16 向量加法的手算例子与逻辑工作量")
    nb.md('''下面是完整候选源码。`launch` 接收评测器准备的输入、输出和元素数；`BLOCK` 表示一个 program 的逻辑片段长度。输出提前分配，尾部使用 `mask` 保护。先读懂 `result = a + b` 和 `valid`，随后亲手修改它。''')
    nb.code(SOURCE)
    nb.code('''lab = agentlab.Lab("chapter15", cfg, baseline_source=SOURCE)
display(pd.Series(lab.task_summary()))''')
    nb.md('''## 15.2 编译通过之后，还要对答案

先把加法改成减法。它仍是合法计算，因此“编译成功”不足以说明正确。工具会用固定的非均匀输入核对整个输出，并把失败阶段返回给我们。''')
    nb.code('''wrong_source = SOURCE.replace("result = a + b", "result = a - b")
agentlab.source_diff(SOURCE, wrong_source)
wrong_check = lab.call("compile_kernel", source=wrong_source)
agentlab.show(wrong_check, title="把加法改成减法之后")
assert not wrong_check["ok"], "数值错误应被正确性检查拒绝"''')
    observation(nb, "查看卡片中的失败阶段和原因。此次失败说明答案错误，不代表 kernel 太慢；没有通过检查的候选不应获得有效性能结论。")
    nb.code('''fixed_source = wrong_source.replace("result = a - b", "result = a + b")
fixed_check = lab.call("compile_kernel", source=fixed_source)
agentlab.show(fixed_check, title="修复后重新检查")
assert fixed_check["ok"], "检查失败时先阅读卡片中的原因"''')
    nb.md('''## 15.3 计时工具回答“这份源码有多快”

正确实现进入预热和正式采样。先看单个样本，再看中心位置；一次最快值不能代表稳定改善。这里的样本描述评测器的批平均时间，计时方式保持固定。''')
    nb.code('''baseline_measurement = lab.preflight()
agentlab.show(baseline_measurement, title="固定基线的实际测量")
assert baseline_measurement["ok"], "基线必须先正确运行"
_ = lab.sample_plot(baseline_measurement)''')
    observation(nb, "样本是否集中？是否有较慢的样本？记下本机观察，再继续比较。曲线与统计值的单位、设备和任务条件来自本次评测记录。")
    nb.md('''## 15.4 用 Roofline 建立假设

算法强度为加法次数除以逻辑字节数：FP16 向量加法为 $N/(6N)=1/6$ FLOP/Byte。低算法强度提示我们研究访存与工作分配，但缓存、输入规模和提交开销也会影响实际结果。

先计算任务模型；如果没有校准带宽和硬件流量证据，就不画一个假定的 Roofline 上限，也不把逻辑有效带宽当作物理显存带宽。''')
    nb.code('''cost = lab.task["costModel"]
display(pd.Series({
    "加法次数 FLOP": cost["flops"],
    "逻辑读写量 Byte": cost["bytes"],
    "算法强度 FLOP/Byte": cost["flops"] / cost["bytes"],
    "本轮假设": "只扩大 block，观察工作分配是否改善实际时间",
    "仍需证据": "硬件流量、资源使用和设备带宽校准",
}))''')
    nb.md('''## 15.5 接受候选需要重新配对比较

只把 `block_size` 扩大两倍，输入和数学语义保持固定。先看源码差异，再让接受工具重新检查候选，并与当前实现配对计时。这个决定不直接使用前面一次测量的最快值。''')
    nb.code('''next_source = SOURCE.replace(f"block_size = {cfg['block']}",
                             f"block_size = {cfg['block'] * 2}")
agentlab.source_diff(SOURCE, next_source)
before = lab.best_path.read_text()
decision = lab.call("accept_candidate", source=next_source,
                    change="只把 block_size 扩大两倍")
agentlab.show(decision, title="候选是否值得保留")
_ = lab.pair_plot(decision)
if not decision.get("accepted", False):
    assert lab.best_path.read_text() == before''')
    observation(nb, "接受或拒绝都可能发生。先核对接受理由与配对样本，再解释当前实现是否变化。源码改动更大或 block 更大都不是接受条件。")
    nb.md('''## 15.6 根据失败阶段决定下一步

再次提交减法版本，观察它不会替换当前实现。正确性失败要修复数学语义；编译失败要检查语法或配置；正确但没有稳定改善时，可以保持当前版本并调整下一项假设。''')
    nb.code('''before = lab.best_path.read_text()
rejected = lab.call("accept_candidate", source=wrong_source,
                    change="反例：把加法写成减法")
agentlab.show(rejected, title="错误候选的接受判定")
assert not rejected.get("accepted", False)
assert lab.best_path.read_text() == before
display(lab.table())''')
    nb.md('''## 15.7 练习：为结果找对解释

输入长度不是 block 的整数倍时，最后一个 program 只有一部分位置有效。先用小表检查 `offsets < n` 的含义，再对长度 4097 的真实 GPU 输入检查。''')
    nb.figure("notebooks/assets/part3-triton-tail-mask.svg",
              "n=10、BLOCK=4 的逻辑片段；最后一个 program 通过 mask 跳过 10、11。",
              "三个 Triton program 的逻辑下标与尾部 mask")
    nb.code('''n_small, block_small = 10, 4
offsets = list(range(12))
valid = [i < n_small for i in offsets]  # 练习：解释这个条件保护了什么
display(pd.DataFrame({"program": [i // block_small for i in offsets],
                      "offset": offsets, "允许读写": valid}))
assert [i for i, enabled in zip(offsets, valid) if enabled] == list(range(n_small))''')
    nb.code('''tail_lab = agentlab.Lab("chapter15-tail", cfg | {"n": 4097},
                        baseline_source=SOURCE)
tail_check = tail_lab.call("compile_kernel", source=SOURCE)
agentlab.show(tail_check, title="长度 4097 的尾部检查")
assert tail_check["ok"]''')
    nb.md('''**练习**：只修改一个合法的 block 配置，重复检查、计时和接受判定。写出“改了什么—实际结果—是否保留—还不能解释什么”。再改变输入规模复测，比较结果是否仍然成立；不要把一份任务的结论直接推广到其他规模。''')
    summary(nb, "你已读到实际 kernel，制造并修复数值错误，观察原始时间样本，并用配对比较完成接受判定。应能解释失败发生在哪一步、当前实现为什么变化或保持不变。下一章把这些工具连成可以跟踪状态的循环。", 16)


def chapter16():
    nb = Notebook("part3-agent", 16)
    nb.md('''# 第16章 算子优化 Agent 设计

## 本章导读

我们已有检查、计时和接受工具，怎样让程序接着完成下一步？先亲手走一轮，观察“已经测过、尚未判定”的候选，再把重复步骤写成短循环。

默认规划动作由代码预先指定，GPU 评测仍是真实执行。前置：[第15章](../chapter15/chapter15.ipynb)，对照阅读：[本章正文](../../../docs/part3-agent/chapter16/index.md)。''')
    nb.code(SETUP + "\nenv = gpu.environment()\nenv")
    nb.md('''## 16.1 把系统分成三个部分

规划者提出行动，工具执行检查和测量，工作区保存状态与记录。沿用同一个 kernel；核心计算仍在下面，你可以直接修改。公共桥接层处理文件、进程和展示。''')
    nb.figure("notebooks/assets/part3-action-loop.svg",
              "规划、工具与记录通过实际返回值连接；下一次行动应读取前一步的观察。",
              "规划者与实际工具的反馈循环")
    nb.code(SOURCE)
    nb.code('''lab = agentlab.Lab("chapter16", cfg, baseline_source=SOURCE)
baseline = lab.preflight()
agentlab.show(baseline, title="本轮固定基线")
assert baseline["ok"]''')
    nb.md('''## 16.2 先把问题整理成可检查的任务

执行工具之前，明确它要检查什么。任务记录固定数学语义、调用参数、输入和计时设置；源码记录当前实现。先核对下面的条件，再修改候选。''')
    nb.code('''display(pd.Series(lab.task_summary()))
assert lab.task["entrypoint"] == "launch"
assert lab.task["launchArguments"] == ["x", "y", "output", "n_elements"]
assert lab.task["tensors"][-1]["value"] == cfg["n"]''')
    nb.md('''## 16.3 让一项假设对应一份候选

候选 A 把 block 扩为两倍，候选 B 扩为四倍。先查看差异。即使两份源码只差一个配置值，它们也必须分别检查，不能把 A 的测量当成 B 的证据。''')
    nb.code('''source_a = SOURCE.replace(f"block_size = {cfg['block']}",
                          f"block_size = {cfg['block'] * 2}")
source_b = SOURCE.replace(f"block_size = {cfg['block']}",
                          f"block_size = {cfg['block'] * 4}")
agentlab.source_diff(SOURCE, source_a)
agentlab.source_diff(source_a, source_b)''')
    nb.code('''checked_a = lab.step("compile_kernel", source=source_a)
agentlab.show(checked_a, title="第一步：检查候选 A")
assert checked_a["ok"]
lab.state_view()''')
    nb.md('''## 16.4 管理已经测过、尚未裁决的候选

检查通过后，对 A 计时。此时还没有作出接受决定：程序必须记住哪份源码等待判定，防止测了 A 却把 B 写入当前实现。逐个运行下面的单元格，观察状态变化。''')
    nb.figure("notebooks/assets/part3-pending-candidate.svg",
              "A 计时后必须对同一份源码完成判定；接受 B 是协议拒绝，性能结果仍需实测。",
              "候选 A 的待判定状态与拒绝换源的状态流程")
    nb.code('''measured_a = lab.step("bench_kernel", source=source_a)
agentlab.show(measured_a, title="第二步：测量候选 A")
assert measured_a["ok"]
_ = lab.sample_plot(measured_a)
lab.state_view()
assert lab.state["pendingCandidate"]''')
    nb.code('''before = lab.best_path.read_text()
wrong_handoff = lab.step("accept_candidate", source=source_b,
                         change="故意把 A 的交接对象换成 B")
agentlab.show(wrong_handoff, title="第三步：观察换源被拒绝")
assert not wrong_handoff["ok"]
assert lab.best_path.read_text() == before
assert lab.state["pendingCandidate"]
lab.state_view()''')
    observation(nb, "这是状态协议拒绝，不是“B 太慢”。A 仍等待判定。程序要修复交接对象，不能跳过这一步直接结束或继续替换候选。")
    nb.code('''decision_a = lab.step("accept_candidate", source=source_a,
                      change="对同一份候选 A 完成接受判定")
agentlab.show(decision_a, title="第四步：判定候选 A")
_ = lab.pair_plot(decision_a)
assert not lab.state["pendingCandidate"]
lab.state_view()''')
    nb.md('''## 16.5 用反馈调整下一次尝试

现在把重复动作写成循环。`actions` 是可见的预设计划；工具返回值进入 `history`。读懂它之后，才能在第 17 章把“提出行动”交给模型。下面继续尝试 B，每个动作后显示结果。''')
    nb.code('''actions = [
    ("compile_kernel", {"source": source_b}),
    ("bench_kernel", {"source": source_b}),
    ("accept_candidate", {"source": source_b, "change": "继续检查 block 扩四倍"}),
]
history = []
for name, arguments in actions:
    result = lab.step(name, arguments)
    history.append({"action": name, "observation": result})
    agentlab.show(result, title=f"循环中的动作：{name}")
    lab.state_view()
    if not result.get("ok", False) and name != "accept_candidate":
        break''')
    nb.code('''display(lab.table())
reflection = "先根据本轮接受理由决定保留或回退，再只修改一个因素。"
print("我的反思：", reflection)
print("仍需证据：资源使用变化不能仅凭时间推断。")''')
    observation(nb, "循环只是组织执行。真正的反思要读取本轮结果；固定列表不会自行产生新的假设。请修改 `reflection`，引用你刚看到的实际接受理由。")
    nb.md('''## 16.6 区分停止、完成与加速

停止表示不再继续执行；完成要求没有尚待判定的候选，并留下实际接受判定；加速则还要由独立复测回答。候选被正确拒绝也可以完成一轮优化。''')
    nb.code('''finished = lab.finish()
lab.state_view()
assert finished["state"] == "complete"
assert not finished["pendingCandidate"]
events = lab.events()
events["阶段"] = events["state"].map(lambda value: value["state"])
display(events[["step", "tool", "bestChanged", "阶段"]])''')
    nb.md('''## 16.7 练习：沿着状态检查程序

故意在计时之后停下，观察为何尚未完成；随后补上同一候选的接受判定。这里的“停止”不代表已获得改善。''')
    nb.code('''exercise = agentlab.Lab("chapter16-state-exercise", cfg,
                        baseline_source=SOURCE)
exercise_result = exercise.step("bench_kernel", source=source_a)
agentlab.show(exercise_result, title="练习：测量后暂时停止")
assert exercise_result["ok"]
stopped = exercise.finish()
exercise.state_view()
assert stopped["state"] == "incomplete"
assert stopped["pendingCandidate"]''')
    nb.code('''resolved = exercise.step("accept_candidate", source=source_a,
                         change="补齐同一候选的判定")
agentlab.show(resolved, title="练习：补齐判定")
completed = exercise.finish()
exercise.state_view()
assert completed["state"] == "complete"
assert not completed["pendingCandidate"]''')
    summary(nb, "你已逐步走完检查、计时和判定，看见换源为何被拒绝，并把动作组合成短循环。应能沿着状态解释一次停止是否算完成。下一章运行多轮搜索，再把最终实现固定下来独立复测。", 17)


def chapter17():
    nb = Notebook("part3-agent", 17)
    nb.md('''# 第17章 多轮优化实战

## 本章导读

一轮改动已经能解释，多轮之后怎样核对最终收益？我们先手动完成一轮，再运行几项预设候选，阅读源码变化、失败记录与性能曲线，最后固定最终实现独立复测。

默认运行无需模型密钥，预设策略不会调用在线模型。在线模型实验在本章练习中显式开启。前置：[第16章](../chapter16/chapter16.ipynb)，对照阅读：[本章正文](../../../docs/part3-agent/chapter17/index.md)。''')
    nb.code(SETUP + "\nenv = gpu.environment()\nenv")
    nb.md('''## 17.1 选择一个能读懂结果的任务

继续用 FP16 向量加法，让每次改动都能解释。核心源码放在下面，接受的计算仍然是 `a + b`；我们先在相同任务下研究配置变化。''')
    nb.figure("notebooks/assets/part3-vector-add-task.svg",
              "本章固定数学语义与逻辑工作量；性能由当前设备的实际运行产生。",
              "FP16 向量加法的输入输出与逻辑工作量")
    nb.code(SOURCE)
    nb.md('''## 17.2 运行之前固定基线与记录范围

固定基线保存搜索开始前的实现，当前实现随接受判定更新。先测量基线并手动提交一项候选；后续轮次都保留成功与失败，不只记录最好的一次。''')
    nb.code('''lab = agentlab.Lab("chapter17", cfg, baseline_source=SOURCE)
display(pd.Series(lab.task_summary()))
baseline = lab.preflight()
agentlab.show(baseline, title="搜索前的固定基线")
assert baseline["ok"]''')
    nb.code('''manual_source = SOURCE.replace(f"block_size = {cfg['block']}",
                               f"block_size = {cfg['block'] * 2}")
agentlab.source_diff(SOURCE, manual_source)
manual_check = lab.step("bench_kernel", source=manual_source)
agentlab.show(manual_check, title="手动一轮：检查并计时")
assert manual_check["ok"]
manual_decision = lab.step("accept_candidate", source=manual_source,
                          change="手动一轮：block 扩两倍")
agentlab.show(manual_decision, title="手动一轮：接受判定")
_ = lab.pair_plot(manual_decision)''')
    nb.md('''接着让预设策略组织几轮工具调用。先提交数值错误的版本，再尝试两个合法配置。策略不会根据反馈生成新的源码；它用真实 Agent 循环执行这些已知候选，帮助我们观察记录与状态。在线模型路径稍后单独运行。''')
    nb.code('''candidates = [
    {"change": "教学负例：加法改减法",
     "source": SOURCE.replace("result = a + b", "result = a - b")},
    {"change": "block 扩四倍",
     "source": SOURCE.replace(f"block_size = {cfg['block']}",
                              f"block_size = {cfg['block'] * 4}")},
    {"change": "block 扩八倍",
     "source": SOURCE.replace(f"block_size = {cfg['block']}",
                              f"block_size = {cfg['block'] * 8}")},
]
status = lab.run(candidates, max_steps=12, live=True)
lab.state_view()''')
    observation(nb, "每轮先看实际行动与结果，再看下一轮。错误版本应留下失败记录；合法配置也可能没有改善。预设策略的模型调用次数为零，不把这个过程解释为模型自主推理。")
    nb.md('''## 17.3 先检查产物，再读性能数字

先核对搜索是否完成、是否还有待判定候选，以及每次行动是否留下记录。展开某轮评测卡可查看完整证据位置；失败也应保留，以便解释它为什么没有成为当前实现。''')
    nb.code('''assert status["state"] == "complete", "请先检查最后一步的状态和原因"
assert not status["pendingCandidate"]
trajectory = lab.table()
display(trajectory)
events = lab.events()
display(events[["step", "tool", "bestChanged"]])
assert (trajectory["status"] == "wrong_answer").any()''')
    nb.md('''## 17.4 分清搜索曲线里的三种量

候选时间描述当轮源码；当前时间描述该轮结束时保留的实现；固定基线来自搜索前。单轮配对改善则比较候选与当时的当前实现。它们的参照对象不同，不能把每轮比例直接相乘当作最终收益。''')
    nb.code('_ = lab.plot()')
    nb.md('''选一份候选，先看源码差异，再核对它对应的记录。修改下面的 `selected_round` 可以查看其他预设候选；是否接受应以表格中的实际判定为准。''')
    nb.code('''selected_round = 1  # 0 为错误示例；1、2 为两个合法配置
chosen = candidates[selected_round]
print("查看改动：", chosen["change"])
lab.inspect_round(change=chosen["change"])''')
    nb.md('''## 17.5 独立复测回答整体收益

搜索结束后，冻结最终源码，与固定基线重新配对运行。复测不再挑新配置，也不使用搜索过程中最有利的一次样本。即使搜索接受了改动，复测仍可能显示波动或没有稳定改善。''')
    nb.figure("notebooks/assets/part3-independent-retest.svg",
              "搜索保留当前实现；搜索结束后冻结最终源码，再与固定基线重新配对测量。",
              "固定基线、搜索当前实现与独立复测的关系")
    nb.code('''final = lab.final_retest(runs=3)
_ = lab.retest_plot(final)
_ = lab.report(final)''')
    observation(nb, "逐组比较复测样本，而不仅看一个比例。如果固定基线仍是最终实现，就如实报告没有获得稳定改善。结论只适用于本次任务、数据类型和设备。")
    nb.md('''## 17.6 写一份读者能核对的报告

报告应把最终结论连接到固定任务、源码、采样和接受记录。工具会保留完整证据；先用自己的话解释一项失败和一项合法尝试，再核对生成的报告。不要从计时直接推断未采集的硬件指标。''')
    nb.code('''my_explanation = {
    "一项失败": "减法版本改变了任务的数学语义，应由正确性检查拒绝。",
    "一项合法尝试": "扩大 block 的配置只按实际接受记录判断是否保留。",
    "结论边界": "当前任务、FP16 和本次 GPU；其他规模需重新测量。",
}
display(pd.Series(my_explanation))
agentlab.source_diff(lab.baseline_path.read_text(), lab.best_path.read_text())
_ = lab.report(final)''')
    nb.md('''## 17.7 练习：完成一次可复盘的优化

先只改变输入规模或一个配置，重新走完整流程，并引用失败、接受和独立复测记录解释结果。接着可以把提出行动交给在线模型。

### 开启在线实验：先准备模型服务

先完成前面的基线与手动实验，再让模型提出下一步行动。在线实验需要当前 Python 内核已经安装 `litellm`、可以访问模型 API，并有一个支持工具调用（tool calling）的模型。我们仍在当前 GPU 上编译、对答案和计时；模型服务负责提出行动。

到模型服务的控制台准备三项信息：**模型 ID、API Base URL 和 API key**。模型 ID 使用控制台给出的 API 名称；API Base URL 使用服务文档给出的接口根地址，常见以 `/v1` 结尾，不能填写聊天网页地址或完整的 `/chat/completions` 地址。API key 是该服务创建的调用密钥。

本例使用 OpenAI 兼容的聊天接口。`openai/` 是 LiteLLM 的协议路由前缀，后面接服务实际提供的模型 ID；它也可以连接其他厂商的兼容服务。接口格式可对照 [LiteLLM 官方说明](https://docs.litellm.ai/docs/providers/openai_compatible)。

| 环境变量 | 在本章填写什么 | 示例或作用 |
| --- | --- | --- |
| `KERNEL_AGENT_MODEL` | `openai/` 加模型 ID | `openai/your-model-id` 中的 ID 要替换 |
| `KERNEL_AGENT_API_BASE` | API 的根地址 | `https://your-api-host/v1` 要替换为服务地址 |
| `KERNEL_AGENT_API_KEY` | 服务提供的 API key | 运行时通过隐藏输入框填写 |
| `KERNEL_AGENT_EXTRA_BODY` | 服务扩展参数的 JSON 对象 | 通用兼容接口先用 `{}`；需要关闭思考模式时按服务文档调整 |

### 第一步：在当前 Notebook 配置环境变量

先修改下面的 `MODEL`、`API_BASE` 和 `EXTRA_BODY`，再把 **`RUN_ONLINE = False` 改为 `True`**，运行这一格，在隐藏输入框粘贴 API key 并按 Enter。此时只配置当前 Python 进程，不发送模型请求。密钥不会写入源代码或单元格输出；重启内核后，需要重新运行配置并输入密钥。

`MAX_STEPS` 是 Agent 的模型决策循环步数上限，取值为 `1` 到 `100` 的整数。每一步可能返回多个工具调用，所以它不等于候选数量或工具调用次数，也不是费用上限。首次保持 `12`，随后根据实际轨迹调整。''')
    nb.code('''import os
from getpass import getpass

RUN_ONLINE = False
MAX_STEPS = 12
MODEL = "openai/your-model-id"
API_BASE = "https://your-api-host/v1"
EXTRA_BODY = {}

if RUN_ONLINE:
    if MODEL == "openai/your-model-id" or API_BASE == "https://your-api-host/v1":
        raise ValueError("请先把 MODEL 和 API_BASE 替换为服务实际提供的配置。")
    if not MODEL.startswith("openai/") or not MODEL.removeprefix("openai/").strip():
        raise ValueError("本例的 MODEL 格式为 openai/服务模型ID。")
    if not API_BASE.strip().startswith(("https://", "http://")):
        raise ValueError("API_BASE 需要是服务提供的完整 http:// 或 https:// 根地址。")
    if not isinstance(EXTRA_BODY, dict):
        raise ValueError("EXTRA_BODY 需要是 Python 字典，例如 {}。")
    api_key = getpass("请输入该模型服务的 API key（输入不可见）：").strip()
    if not api_key:
        raise ValueError("API key 不能为空，请重新运行这一格。")
    os.environ["KERNEL_AGENT_MODEL"] = MODEL.strip()
    os.environ["KERNEL_AGENT_API_BASE"] = API_BASE.strip().rstrip("/")
    os.environ["KERNEL_AGENT_API_KEY"] = api_key
    os.environ["KERNEL_AGENT_EXTRA_BODY"] = json.dumps(EXTRA_BODY)
    del api_key
    print("当前内核的在线配置已设置；尚未发送模型请求。")
else:
    print("在线实验未开启；修改配置并设 RUN_ONLINE = True 后再运行这一格。")''')
    nb.md('''**配置示例**：如果使用 DeepSeek 官方的 OpenAI 兼容接口，可以将上一格的三个配置行改为下面的形式，再输入 DeepSeek API key。模型 ID 以服务当前提供的列表为准；此处是配置示例，不包含在线实测结果。

```python
MODEL = "openai/deepseek-flash"
API_BASE = "https://api.deepseek.com"
EXTRA_BODY = {"thinking": {"type": "disabled"}}
```

接口地址和模型名称见 [DeepSeek 官方接入说明](https://api-docs.deepseek.com/)。本章的循环没有维护思考模式的推理历史，因此示例关闭思考模式；相应参数见 [DeepSeek 思考模式说明](https://api-docs.deepseek.com/guides/thinking_mode)。其他服务使用自己的模型 ID、地址和扩展参数，不直接套用 DeepSeek 的 `thinking` 字段。

如果平台已经在 Python 内核中设置了这些环境变量，可以跳过上面的隐藏输入框配置，新增一个代码格，执行下面两行，再继续检查与运行。平台配置也需要按服务要求提供 `KERNEL_AGENT_EXTRA_BODY`：

```python
RUN_ONLINE = True
MAX_STEPS = 12
```

在 Notebook 中写 `os.environ["变量名"] = "值"` 会立即影响当前内核；在 Jupyter 终端执行 `export` 不会改变已经运行的 Python 内核。使用平台环境变量设置时，通常需要重启内核才能读取新值。

### 第二步：检查配置，暂不发送请求

运行下面这一格，确认 LiteLLM 已安装、模型名和 API 地址正确。密钥只显示是否已配置。已准备好的学习环境应提供这项依赖；默认的预设策略不需要它。

如果提示缺少 LiteLLM，可以新增一个代码格，执行下面的可选安装，再通过 **Kernel → Restart Kernel** 重启内核，从头运行本章并重新输入密钥。`%pip` 会给当前 Notebook 使用的 Python 安装依赖；正常运行无需重复安装。

```python
%pip install "litellm>=1.95"
```

如果提示 `No module named pip`，且环境提供了 `uv`，使用下面的替代安装格。`sys.executable` 指定当前内核的 Python，避免装到另一个环境；安装后同样重启内核并从头运行。

```python
import sys
!uv pip install --python "{sys.executable}" "litellm>=1.95"
```''')
    nb.code('''import os
from importlib.util import find_spec

if RUN_ONLINE:
    if find_spec("litellm") is None:
        raise RuntimeError("当前内核缺少 litellm，请先补齐在线实验依赖。")
    from kernel_optimize import llm
    configured_model, configured_base = llm._model_config()
    display(pd.Series({
        "模型": configured_model,
        "API 根地址": configured_base or "服务默认地址",
        "统一 API key 已配置": bool(os.environ.get("KERNEL_AGENT_API_KEY")),
        "模型决策步数上限": MAX_STEPS,
    }))
else:
    print("在线实验未开启，配置检查已跳过。")''')
    nb.md('''### 第三步：启动在线循环并核对结果

运行下面这一格，先测量在线实验自己的 GPU 基线，再向配置的模型服务发送请求，可能产生服务费用。`live=True` 会逐步显示工具名与结果卡；每一步的模型消息、工具记录和评测结果保存在本次工作区中。

完成后查看 `online_status` 的状态、模型调用尝试次数和记录目录；调用尝试次数不包含 SDK 内部的 HTTP 重试统计。只有状态为 `complete`，才执行三次独立复测，它也不保证最终实现更快。`complete_with_warning` 表示已有有效裁决，但末尾仍有需要解释的尝试；先阅读报告中的限制。`incomplete` 表示实验尚未形成完整结论；先看状态与工具反馈，再决定是否增加 `MAX_STEPS`，重新运行这一格会创建新的实验工作区。''')
    nb.code('''if RUN_ONLINE:
    online_lab = agentlab.Lab("chapter17-online", cfg, baseline_source=SOURCE)
    online_baseline = online_lab.preflight()
    agentlab.show(online_baseline, title="在线实验的独立基线")
    assert online_baseline["ok"]
    online_status = online_lab.run_online(max_steps=MAX_STEPS, live=True)
    online_lab.state_view()
    display(pd.Series({key: online_status[key]
                       for key in ("state", "llmCalls", "workspace")}))
    if online_status["state"] == "complete":
        online_final = online_lab.final_retest(runs=3)
        _ = online_lab.retest_plot(online_final)
        _ = online_lab.report(online_final)
    else:
        print(online_status["report"])
        print("在线循环尚未完成，请先检查状态与工具反馈；本次不执行最终复测。")
else:
    print("在线实验未开启；本页默认流程的在线模型调用次数为 0。")''')
    nb.md('''**遇到错误时**：先确定失败发生在模型接口还是 GPU 工具。下面列出接口配置的排查方向，具体原因以实际返回为准。

| 现象 | 先检查什么 |
| --- | --- |
| 缺少模型配置或格式错误 | 是否先运行了配置格；模型是否采用 `openai/模型ID` 格式 |
| `401` / `403` | 密钥是否属于当前服务，是否有该模型的调用权限 |
| `404` 或模型不存在 | API 根地址和模型 ID 是否与服务文档一致；不要把 `/chat/completions` 填进根地址 |
| `400`，涉及 `tools` 或 `tool_choice` | 模型和接口是否支持工具调用 |
| `400`，涉及扩展字段或思考历史 | `EXTRA_BODY` 是否符合当前服务要求；本章示例关闭思考模式 |
| `429`、超时或连接失败 | 服务限流、额度和当前内核到 API 的网络连通性 |

**观察**：对照预设策略与在线策略，记录模型实际提出的改动、工具返回的原因和独立复测结果。模型回复“完成”或“更快”仍需要工具证据支持。''')
    summary(nb, "你已完成手动一轮、预设策略的多轮工具调用、轨迹核对和独立复测。应能区分候选、当前实现与固定基线，解释没有改善的结果，并让报告中的每一项结论对应实际证据。在线模型只是替换提出行动的方式，正确性与接受标准保持一致。")


def build():
    for author in (chapter14, chapter15, chapter16, chapter17):
        author()


if __name__ == "__main__":
    build()
