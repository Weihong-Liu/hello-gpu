<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, useId, watch } from 'vue'

type Scenario = 'atomic' | 'program' | 'lds' | 'local' | 'wave' | 'staged'
type VisualState = 'hidden' | 'context' | 'visible' | 'active' | 'complete' | 'hot'
type NodeShape = 'rect' | 'circle' | 'gate'
type Tone = 'input' | 'worker' | 'local' | 'buffer' | 'result' | 'hot'

interface Metric {
  label: string
  value: string
}

interface DiagramNode {
  id: string
  x: number
  y: number
  value: string
  caption: string
  tone: Tone
  shape?: NodeShape
  width?: number
  height?: number
  captionY?: number
  valueY?: number
}

interface DiagramEdge {
  id: string
  d: string
}

interface DiagramRegion {
  id: string
  x: number
  y: number
  width: number
  height: number
  label: string
  tone: Tone
}

interface Guide {
  label: string
  y: number
  x?: number
  textAnchor?: 'start' | 'middle' | 'end'
}

interface StoryFrame {
  id: string
  eyebrow: string
  title: string
  narration: string
  takeaway: string
  metrics: Metric[]
  contextNodes?: string[]
  visibleNodes?: string[]
  activeNodes?: string[]
  completeNodes?: string[]
  hotNodes?: string[]
  contextEdges?: string[]
  activeEdges?: string[]
  completeEdges?: string[]
  hotEdges?: string[]
  contextRegions?: string[]
  activeRegions?: string[]
  values?: Record<string, string>
  duration?: number
}

interface ScenarioStory {
  label: string
  title: string
  kicker: string
  viewBox: string
  compactViewBox: string
  guides: Guide[]
  nodes: DiagramNode[]
  edges: DiagramEdge[]
  regions: DiagramRegion[]
  frames: StoryFrame[]
}

const props = withDefaults(defineProps<{ scenario?: Scenario }>(), {
  scenario: 'atomic',
})

const inputValues = [3, 1, 4, 2, 5, 2, 1, 3]
const inputXs = [110, 180, 250, 320, 390, 460, 530, 600]

function inputsAt(y: number): DiagramNode[] {
  return inputValues.map((value, index) => ({
    id: `x${index}`,
    x: inputXs[index],
    y,
    value: String(value),
    caption: `x[${index}]`,
    tone: 'input',
  }))
}

const atomicInputNodeIds = inputValues.map((_, index) => `x${index}`)
const atomicThreadNodeIds = inputValues.map((_, index) => `thread-${index}`)
const atomicQueueNodeIds = inputValues.map((_, index) => `queue-${index}`)
const atomicLoadEdgeIds = inputValues.map((_, index) => `load-${index}`)
const atomicEnqueueEdgeIds = inputValues.map((_, index) => `enqueue-${index}`)
const atomicWriteEdgeIds = inputValues.map((_, index) => `atomic-${index}`)

const atomicNodes: DiagramNode[] = [
  ...inputsAt(82),
  ...inputXs.map((x, index) => ({
    id: `thread-${index}`,
    x,
    y: 188,
    value: String(index),
    caption: '',
    tone: 'worker' as const,
    shape: 'circle' as const,
    width: 42,
    height: 42,
  })),
  ...inputXs.map((x, index) => ({
    id: `queue-${index}`,
    x,
    y: 292,
    value: String(index),
    caption: '',
    tone: 'hot' as const,
    width: 44,
    height: 36,
  })),
  { id: 'gate', x: 355, y: 385, value: '', caption: '', tone: 'hot', shape: 'gate', width: 34, height: 54 },
  { id: 'output', x: 355, y: 490, value: '0', caption: 'output[0]', tone: 'result', width: 132, height: 60, captionY: -8, valueY: 19 },
]

const atomicEdges: DiagramEdge[] = [
  ...inputXs.map((x, index) => ({ id: `load-${index}`, d: `M ${x} 105 L ${x} 164` })),
  ...inputXs.map((x, index) => ({ id: `enqueue-${index}`, d: `M ${x} 212 L ${x} 270` })),
  ...inputXs.map((x, index) => ({
    id: `atomic-${index}`,
    d: `M ${x} 312 C ${x} 334, ${355 + (index - 3.5) * 4} 340, 355 356`,
  })),
  { id: 'gate-output', d: 'M 355 414 L 355 458' },
]

const atomicWriteFrames: StoryFrame[] = inputValues.map((value, index) => {
  const previousTotal = inputValues
    .slice(0, index)
    .reduce((total, item) => total + item, 0)
  const nextTotal = previousTotal + value
  const completedIndices = Array.from({ length: index }, (_, item) => item)
  const pendingIndices = Array.from(
    { length: inputValues.length - index - 1 },
    (_, item) => index + item + 1,
  )
  const waiting = inputValues.length - index - 1

  return {
    id: `atomic-write-${index}`,
    eyebrow: `排队写入 ${index + 1} / ${inputValues.length}`,
    title: `线程 ${index} 写入：${previousTotal} + ${value} = ${nextTotal}`,
    narration: waiting > 0
      ? `线程 ${index} 获得写入权：先读到 output[0] = ${previousTotal}，加上 input[${index}] = ${value}，再写回 ${nextTotal}；其余 ${waiting} 个请求继续等待。`
      : `线程 ${index} 最后写入：先读到 output[0] = ${previousTotal}，加上 input[${index}] = ${value}，再写回 ${nextTotal}；队列现在清空。`,
    takeaway: waiting > 0
      ? `同一时刻只有线程 ${index} 能更新 output[0]，下一个请求必须等它完成。`
      : '8 次更新全部完成；结果正确，但同一地址上的写入没有并行执行。',
    metrics: [
      { label: 'output 变化', value: `${previousTotal} → ${nextTotal}` },
      { label: '等待请求', value: String(waiting) },
    ],
    contextNodes: [
      ...atomicInputNodeIds,
      ...pendingIndices.map(item => `thread-${item}`),
      ...pendingIndices.map(item => `queue-${item}`),
    ],
    activeNodes: [`thread-${index}`, `queue-${index}`, 'output'],
    completeNodes: [
      ...completedIndices.map(item => `thread-${item}`),
      ...completedIndices.map(item => `queue-${item}`),
    ],
    hotNodes: ['gate'],
    contextEdges: [
      ...atomicLoadEdgeIds,
      ...pendingIndices.map(item => `enqueue-${item}`),
    ],
    activeEdges: [`atomic-${index}`, 'gate-output'],
    completeEdges: [
      ...completedIndices.map(item => `enqueue-${item}`),
      `enqueue-${index}`,
    ],
    activeRegions: ['atomic-queue'],
    values: { output: String(nextTotal) },
    duration: 2400,
  }
})

const atomicStory: ScenarioStory = {
  label: '全局 atomic',
  title: '八个请求，共用一个写入地址',
  kicker: '场景 / 全局原子更新热点',
  viewBox: '0 0 720 550',
  compactViewBox: '0 20 665 520',
  guides: [
    { label: '输入值', y: 116, x: 75, textAnchor: 'end' },
    { label: '线程编号', y: 224, x: 75, textAnchor: 'end' },
    { label: '等待队列', y: 340, x: 75, textAnchor: 'end' },
    { label: '全局结果', y: 445, x: 75, textAnchor: 'end' },
  ],
  nodes: atomicNodes,
  edges: atomicEdges,
  regions: [
    { id: 'atomic-queue', x: 82, y: 248, width: 556, height: 82, label: '', tone: 'worker' },
  ],
  frames: [
    {
      id: 'thread-loads',
      eyebrow: '线程读取',
      title: '每个输入交给一个线程',
      narration: '8 个线程可以并行读取 8 个值；此时还没有发生写入争用。',
      takeaway: '并行读取很自然，问题出在下一步的共享写入。',
      metrics: [{ label: '读取线程', value: '8' }, { label: '输入值', value: '8' }],
      contextNodes: ['output'],
      activeNodes: atomicThreadNodeIds,
      visibleNodes: atomicInputNodeIds,
      activeEdges: atomicLoadEdgeIds,
      values: { output: '0' },
      duration: 4200,
    },
    {
      id: 'atomic-queue',
      eyebrow: '建立写入队列',
      title: '八个请求排队，一次只放行一个',
      narration: '队列方框中的 0～7 是线程编号。实际硬件不保证它们的先后顺序；为了看清过程，图中固定按线程 0 → 7 演示。atomic 只保证同一时刻有一个请求完整更新 output[0]。',
      takeaway: '排队顺序可以变化，但对同一地址的原子更新不能重叠执行。',
      metrics: [{ label: '排队请求', value: '8' }, { label: '每次放行', value: '1' }],
      contextNodes: [...atomicInputNodeIds, ...atomicThreadNodeIds, 'output'],
      activeNodes: atomicQueueNodeIds,
      hotNodes: ['gate'],
      contextEdges: [...atomicLoadEdgeIds, ...atomicWriteEdgeIds, 'gate-output'],
      activeEdges: atomicEnqueueEdgeIds,
      activeRegions: ['atomic-queue'],
      values: { output: '0' },
      duration: 3600,
    },
    ...atomicWriteFrames,
    {
      id: 'atomic-result',
      eyebrow: '结构计数',
      title: '结果是 21，源码结构中有 8 次 atomic',
      narration: '线程 0 到线程 7 的 8 次写入已经逐步完成；这里的“8 次”由示意输入和 kernel 结构直接数出，不是性能测量。',
      takeaway: 'baseline 的优化目标是减少对同一 output 的 atomic 次数。',
      metrics: [{ label: 'atomic 结构计数', value: '8 次' }, { label: '结果', value: '21' }],
      contextNodes: [...atomicInputNodeIds, 'gate'],
      completeNodes: [...atomicThreadNodeIds, ...atomicQueueNodeIds, 'output'],
      completeEdges: [
        ...atomicLoadEdgeIds,
        ...atomicEnqueueEdgeIds,
        'gate-output',
      ],
      contextRegions: ['atomic-queue'],
      values: { output: '21' },
      duration: 3800,
    },
  ],
}

const programNodes: DiagramNode[] = [
  ...inputsAt(76),
  { id: 'program-0', x: 230, y: 215, value: '0', caption: 'Triton program 0', tone: 'worker', width: 132, height: 58 },
  { id: 'program-1', x: 490, y: 215, value: '1', caption: 'Triton program 1', tone: 'worker', width: 132, height: 58 },
  { id: 'program-partial-0', x: 230, y: 310, value: '10', caption: 'tl.sum partial', tone: 'local', width: 138, height: 60 },
  { id: 'program-partial-1', x: 490, y: 310, value: '11', caption: 'tl.sum partial', tone: 'local', width: 138, height: 60 },
  { id: 'program-gate', x: 360, y: 400, value: '', caption: '同一 output', tone: 'hot', shape: 'gate', width: 34, height: 54 },
  { id: 'program-output', x: 360, y: 485, value: '?', caption: 'output', tone: 'result', width: 132, height: 60, captionY: -8, valueY: 19 },
]

const programEdges: DiagramEdge[] = [
  ...inputXs.map((x, index) => {
    const targetX = index < 4 ? 230 : 490
    return {
      id: `program-load-${index}`,
      d: `M ${x} 100 C ${x} 140, ${targetX} 150, ${targetX} 184`,
    }
  }),
  { id: 'program-sum-0', d: 'M 230 246 L 230 278' },
  { id: 'program-sum-1', d: 'M 490 246 L 490 278' },
  { id: 'program-atomic-0', d: 'M 230 342 C 230 368, 330 360, 348 372' },
  { id: 'program-atomic-1', d: 'M 490 342 C 490 368, 390 360, 372 372' },
  { id: 'program-gate-output', d: 'M 360 429 L 360 452' },
]

const programStory: ScenarioStory = {
  label: 'Triton program atomic',
  title: '每个 program 先归约一个 chunk，再 atomic 合并',
  kicker: '场景 / TRITON PROGRAM 原子合并',
  viewBox: '0 0 720 540',
  compactViewBox: '45 15 630 515',
  guides: [
    { label: '输入', y: 112 },
    { label: 'program', y: 258 },
    { label: 'tl.sum', y: 354 },
    { label: 'atomic output', y: 444 },
  ],
  nodes: programNodes,
  edges: programEdges,
  regions: [
    { id: 'program-region-0', x: 80, y: 36, width: 280, height: 330, label: 'Triton program 0', tone: 'worker' },
    { id: 'program-region-1', x: 360, y: 36, width: 280, height: 330, label: 'Triton program 1', tone: 'buffer' },
  ],
  frames: [
    {
      id: 'program-loads',
      eyebrow: '两个 program',
      title: '每个 Triton program 加载四个值',
      narration: 'program 0 加载前四个输入，program 1 加载后四个输入；每个 program 负责一个连续数据块。',
      takeaway: 'Triton atomic baseline 不是每个元素做一次 atomic。',
      metrics: [{ label: 'program', value: '2' }, { label: '每个 chunk', value: '4 个值' }],
      visibleNodes: inputValues.map((_, index) => `x${index}`),
      activeNodes: ['program-0', 'program-1'],
      activeEdges: inputValues.map((_, index) => `program-load-${index}`),
      activeRegions: ['program-region-0', 'program-region-1'],
      duration: 6000,
    },
    {
      id: 'program-sums',
      eyebrow: 'program 内归约',
      title: '两次 tl.sum 得到 partial 10 和 11',
      narration: 'program 0 把 3、1、4、2 合成 10；program 1 把 5、2、1、3 合成 11。',
      takeaway: '每个 program 先把自己的 chunk 压成一个值。',
      metrics: [{ label: '输入', value: '8' }, { label: 'program partial', value: '2' }],
      contextNodes: [
        ...inputValues.map((_, index) => `x${index}`),
        'program-0',
        'program-1',
      ],
      activeNodes: ['program-partial-0', 'program-partial-1'],
      contextEdges: inputValues.map((_, index) => `program-load-${index}`),
      activeEdges: ['program-sum-0', 'program-sum-1'],
      contextRegions: ['program-region-0', 'program-region-1'],
      duration: 6500,
    },
    {
      id: 'program-atomics',
      eyebrow: '跨 program 合并',
      title: '两个 partial 对同一 output 做 atomic',
      narration: '10 和 11 都指向同一个 output；热点请求已经从 8 个元素降到 2 个 program partial。',
      takeaway: 'tl.sum 减少 atomic 次数，但最终标量仍是共享写入目标。',
      metrics: [{ label: 'atomic 请求', value: '2' }, { label: '写入地址', value: '1' }],
      contextNodes: ['program-partial-0', 'program-partial-1', 'program-output'],
      hotNodes: ['program-gate'],
      hotEdges: ['program-atomic-0', 'program-atomic-1', 'program-gate-output'],
      contextRegions: ['program-region-0', 'program-region-1'],
      values: { 'program-output': '...' },
      duration: 6500,
    },
    {
      id: 'program-result',
      eyebrow: '结构计数',
      title: '结果是 21，源码结构中有 2 次 atomic',
      narration: '每个 program 只发出一次 atomic；这里的两次来自示意图中的两个 program。',
      takeaway: 'Triton atomic baseline 的 atomic 数等于 program 数。',
      metrics: [{ label: 'atomic 结构计数', value: '2 次' }, { label: '结果', value: '21' }],
      contextNodes: ['program-partial-0', 'program-partial-1', 'program-gate'],
      completeNodes: ['program-output'],
      completeEdges: ['program-atomic-0', 'program-atomic-1', 'program-gate-output'],
      contextRegions: ['program-region-0', 'program-region-1'],
      values: { 'program-output': '21' },
      duration: 7000,
    },
  ],
}

const ldsPairNodes: DiagramNode[] = [
  { id: 'pair-0', x: 165, y: 230, value: '4', caption: '3 + 1', tone: 'local', width: 72, height: 54 },
  { id: 'pair-1', x: 295, y: 230, value: '6', caption: '4 + 2', tone: 'local', width: 72, height: 54 },
  { id: 'pair-2', x: 425, y: 230, value: '7', caption: '5 + 2', tone: 'local', width: 72, height: 54 },
  { id: 'pair-3', x: 555, y: 230, value: '4', caption: '1 + 3', tone: 'local', width: 72, height: 54 },
]

const ldsGroupNodes: DiagramNode[] = [
  { id: 'group-0', x: 230, y: 342, value: '10', caption: '组 0 局部和', tone: 'local', width: 104, height: 56 },
  { id: 'group-1', x: 490, y: 342, value: '11', caption: '组 1 局部和', tone: 'local', width: 104, height: 56 },
]

const ldsNodes: DiagramNode[] = [
  ...inputsAt(78),
  ...ldsPairNodes,
  ...ldsGroupNodes,
  { id: 'leader-gate', x: 360, y: 426, value: '', caption: '同一 output', tone: 'hot', shape: 'gate', width: 34, height: 52 },
  { id: 'lds-output', x: 360, y: 505, value: '21', caption: 'output', tone: 'result', width: 126, height: 58, captionY: -8, valueY: 18 },
]

const ldsEdges: DiagramEdge[] = [
  ...inputXs.map((x, index) => {
    const pair = ldsPairNodes[Math.floor(index / 2)]
    return { id: `lds-pair-${index}`, d: `M ${x} 102 C ${x} 158, ${pair.x} 170, ${pair.x} 201` }
  }),
  ...ldsPairNodes.map((pair, index) => {
    const group = ldsGroupNodes[Math.floor(index / 2)]
    return { id: `lds-group-${index}`, d: `M ${pair.x} 260 C ${pair.x} 286, ${group.x} 294, ${group.x} 312` }
  }),
  { id: 'leader-0', d: 'M 230 372 C 230 397, 330 386, 350 400' },
  { id: 'leader-1', d: 'M 490 372 C 490 397, 390 386, 370 400' },
  { id: 'leader-output', d: 'M 360 454 L 360 474' },
]

const ldsStory: ScenarioStory = {
  label: '局部数据共享（LDS）组内归约',
  title: '先在组内合并，再让 leader 写一次',
  kicker: '场景 / 局部数据共享归约树',
  viewBox: '0 0 720 550',
  compactViewBox: '45 25 630 510',
  guides: [{ label: '输入', y: 112 }, { label: '局部共享存储', y: 282 }, { label: '组局部和', y: 384 }, { label: '全局', y: 464 }],
  nodes: ldsNodes,
  edges: ldsEdges,
  regions: [
    { id: 'g0', x: 80, y: 38, width: 280, height: 355, label: '工作组 0', tone: 'local' },
    { id: 'g1', x: 360, y: 38, width: 280, height: 355, label: '工作组 1', tone: 'buffer' },
  ],
  frames: [
    {
      id: 'lds-groups',
      eyebrow: '建立协作边界',
      title: '八个输入先分给两个工作组',
      narration: '工作组 0 处理前四个值，工作组 1 处理后四个值；两组暂时互不通信。',
      takeaway: 'block 是组内协作的边界。',
      metrics: [{ label: '工作组', value: '2' }, { label: '每组输入', value: '4' }],
      activeNodes: inputValues.map((_, index) => `x${index}`),
      contextNodes: ['lds-output'],
      activeRegions: ['g0', 'g1'],
      duration: 5200,
    },
    {
      id: 'lds-eight-to-four',
      eyebrow: '第一轮',
      title: '8 个值在局部共享存储中缩成 4 个局部和',
      narration: '箭头保留输入来源；4、6、7、4 都能追溯到各自的两个输入。',
      takeaway: '组内树每一轮都减少需要继续传递的值。',
      metrics: [{ label: '组内值', value: '8 → 4' }, { label: '全局 atomic', value: '0' }],
      contextNodes: [...inputValues.map((_, index) => `x${index}`), 'lds-output'],
      activeNodes: ldsPairNodes.map(node => node.id),
      activeEdges: inputValues.map((_, index) => `lds-pair-${index}`),
      contextRegions: ['g0', 'g1'],
      duration: 6000,
    },
    {
      id: 'lds-group-partials',
      eyebrow: '第二轮',
      title: '每个工作组只留下一个 partial',
      narration: '工作组 0 把 4 和 6 合成 10，工作组 1 把 7 和 4 合成 11。',
      takeaway: '跨出 block 的值从 8 个减少到 2 个。',
      metrics: [{ label: 'partial 数', value: '2' }, { label: '值', value: '10, 11' }],
      contextNodes: [...ldsPairNodes.map(node => node.id), 'lds-output'],
      activeNodes: ldsGroupNodes.map(node => node.id),
      contextEdges: inputValues.map((_, index) => `lds-pair-${index}`),
      activeEdges: ldsPairNodes.map((_, index) => `lds-group-${index}`),
      contextRegions: ['g0', 'g1'],
      duration: 6200,
    },
    {
      id: 'lds-leader-atomic',
      eyebrow: '跨组收尾',
      title: '两个 leader 各做一次 atomic',
      narration: '热点仍然存在，但结构计数已经从 8 次降到 2 次。',
      takeaway: '局部数据共享（LDS）归约减少全局争用，没有彻底消除 atomic。',
      metrics: [{ label: 'leader atomic', value: '2 次' }, { label: '结果', value: '21' }],
      contextNodes: ldsGroupNodes.map(node => node.id),
      hotNodes: ['leader-gate'],
      completeNodes: ['lds-output'],
      completeEdges: ['leader-0', 'leader-1', 'leader-output'],
      contextRegions: ['g0', 'g1'],
      duration: 6500,
    },
  ],
}

const localThreadXs = [170, 300, 430, 560]
const localInputNodes: DiagramNode[] = localThreadXs.flatMap((x, lane) => [
  { id: `local-x${lane}`, x: x - 30, y: 62, value: String(inputValues[lane]), caption: `x[${lane}]`, tone: 'input' },
  { id: `local-x${lane + 4}`, x: x + 30, y: 130, value: String(inputValues[lane + 4]), caption: `x[${lane + 4}]`, tone: 'input' },
])
const localSums = [8, 3, 5, 5]
const localNodes: DiagramNode[] = [
  ...localInputNodes,
  ...localThreadXs.map((x, index) => ({
    id: `local-thread-${index}`,
    x,
    y: 220,
    value: String(index),
    caption: '2 个输入',
    tone: 'worker' as const,
    shape: 'circle' as const,
  })),
  ...localThreadXs.map((x, index) => ({
    id: `register-${index}`,
    x,
    y: 320,
    value: String(localSums[index]),
    caption: `${inputValues[index]} + ${inputValues[index + 4]}`,
    tone: 'local' as const,
    width: 82,
    height: 56,
  })),
  { id: 'lds-13', x: 275, y: 420, value: '13', caption: '8 + 5', tone: 'local', width: 92, height: 56 },
  { id: 'lds-8', x: 455, y: 420, value: '8', caption: '3 + 5', tone: 'local', width: 92, height: 56 },
  { id: 'local-output', x: 365, y: 515, value: '21', caption: 'block sum', tone: 'result', width: 132, height: 60, captionY: -8, valueY: 19 },
]

const localEdges: DiagramEdge[] = [
  ...localThreadXs.flatMap((x, lane) => [
    { id: `assign-a-${lane}`, d: `M ${x - 30} 86 C ${x - 30} 132, ${x - 18} 166, ${x - 9} 196` },
    { id: `assign-b-${lane}`, d: `M ${x + 30} 154 C ${x + 27} 174, ${x + 18} 184, ${x + 9} 196` },
    { id: `register-edge-${lane}`, d: `M ${x} 244 L ${x} 290` },
  ]),
  { id: 'local-lds-0', d: 'M 170 350 C 170 382, 250 374, 268 392' },
  { id: 'local-lds-2', d: 'M 430 350 C 430 382, 300 374, 282 392' },
  { id: 'local-lds-1', d: 'M 300 350 C 300 382, 430 374, 448 392' },
  { id: 'local-lds-3', d: 'M 560 350 C 560 382, 480 374, 462 392' },
  { id: 'local-final-0', d: 'M 275 450 C 275 472, 335 472, 350 485' },
  { id: 'local-final-1', d: 'M 455 450 C 455 472, 395 472, 380 485' },
]

const localStory: ScenarioStory = {
  label: '寄存器局部累加',
  title: '每个示意线程先处理两个输入',
  kicker: '场景 / 线程局部求和',
  viewBox: '60 0 610 560',
  compactViewBox: '45 10 630 550',
  guides: [{ label: '两轮输入', y: 166 }, { label: '线程', y: 256 }, { label: '寄存器', y: 360 }, { label: '局部共享存储', y: 460 }],
  nodes: localNodes,
  edges: localEdges,
  regions: [
    { id: 'register-band', x: 105, y: 280, width: 500, height: 82, label: '线程私有寄存器', tone: 'local' },
    { id: 'local-lds-band', x: 205, y: 380, width: 320, height: 82, label: '组内合并', tone: 'buffer' },
  ],
  frames: [
    {
      id: 'four-threads-two-inputs',
      eyebrow: 'grid-stride 缩放示意',
      title: '4 个示意线程各处理 2 个输入',
      narration: '线程 0 读取 x[0] 和 x[4]，其余线程按相同方式读取自己的两个位置。',
      takeaway: '线程数量不必和输入数量相等。',
      metrics: [{ label: '示意线程', value: '4' }, { label: '每线程输入', value: '2' }],
      visibleNodes: localInputNodes.map(node => node.id),
      activeNodes: localThreadXs.map((_, index) => `local-thread-${index}`),
      activeEdges: localThreadXs.flatMap((_, index) => [`assign-a-${index}`, `assign-b-${index}`]),
      duration: 6000,
    },
    {
      id: 'register-locals',
      eyebrow: '寄存器局部和',
      title: '先得到 8、3、5、5',
      narration: '每个线程只在自己的寄存器里累加，不需要为这一步和其他线程通信。',
      takeaway: '8 个输入先缩成 4 个线程局部值。',
      metrics: [{ label: '输入', value: '8' }, { label: '寄存器局部和', value: '4' }],
      contextNodes: [...localInputNodes.map(node => node.id), ...localThreadXs.map((_, index) => `local-thread-${index}`)],
      activeNodes: localThreadXs.map((_, index) => `register-${index}`),
      contextEdges: localThreadXs.flatMap((_, index) => [`assign-a-${index}`, `assign-b-${index}`]),
      activeEdges: localThreadXs.map((_, index) => `register-edge-${index}`),
      activeRegions: ['register-band'],
      duration: 6500,
    },
    {
      id: 'local-lds',
      eyebrow: '组内合并',
      title: '局部共享存储中把 4 个值缩成 13 和 8',
      narration: '8 与 5 合成 13，3 与 5 合成 8；这时才发生线程间协作。',
      takeaway: '先局部累加，再让更少的值进入组内归约。',
      metrics: [{ label: '组内值', value: '4 → 2' }, { label: '结果', value: '13, 8' }],
      contextNodes: localThreadXs.map((_, index) => `register-${index}`),
      activeNodes: ['lds-13', 'lds-8'],
      activeEdges: ['local-lds-0', 'local-lds-1', 'local-lds-2', 'local-lds-3'],
      contextRegions: ['register-band'],
      activeRegions: ['local-lds-band'],
      duration: 6500,
    },
    {
      id: 'local-result',
      eyebrow: 'block 结果',
      title: '13 + 8 = 21',
      narration: '示意图中的 block 最终留下一个和；跨 block 怎样收尾由 atomic 或 partial buffer 决定。',
      takeaway: '局部累加优化的是组内路径，不自动解决跨 block 合并。',
      metrics: [{ label: '组内汇聚', value: '8 → 4 → 2 → 1' }, { label: 'block sum', value: '21' }],
      contextNodes: ['lds-13', 'lds-8'],
      completeNodes: ['local-output'],
      completeEdges: ['local-final-0', 'local-final-1'],
      contextRegions: ['local-lds-band'],
      duration: 7000,
    },
  ],
}

const waveLaneXs = [180, 300, 420, 540]
const waveNodes: DiagramNode[] = [
  ...waveLaneXs.map((x, index) => ({
    id: `lane-${index}`,
    x,
    y: 150,
    value: String(localSums[index]),
    caption: `lane ${index}`,
    tone: 'worker' as const,
    width: 76,
    height: 58,
  })),
  { id: 'offset2-0', x: 270, y: 290, value: '13', caption: 'lane 0: 8 + 5', tone: 'local', width: 136, height: 62 },
  { id: 'offset2-1', x: 450, y: 290, value: '8', caption: 'lane 1: 3 + 5', tone: 'local', width: 136, height: 62 },
  { id: 'wave-result', x: 360, y: 420, value: '21', caption: 'lane 0 / wave sum', tone: 'result', width: 164, height: 64, captionY: -9, valueY: 21 },
]

const waveEdges: DiagramEdge[] = [
  { id: 'offset2-a', d: 'M 180 181 C 180 220, 245 230, 258 258' },
  { id: 'offset2-b', d: 'M 420 181 C 420 220, 295 230, 282 258' },
  { id: 'offset2-c', d: 'M 300 181 C 300 220, 425 230, 438 258' },
  { id: 'offset2-d', d: 'M 540 181 C 540 220, 475 230, 462 258' },
  { id: 'offset1-a', d: 'M 270 323 C 270 350, 330 362, 344 387' },
  { id: 'offset1-b', d: 'M 450 323 C 450 350, 390 362, 376 387' },
]

const waveStory: ScenarioStory = {
  label: 'Wave shuffle',
  title: '用 4 个 lane 的缩放图解释 Wave32 的寄存器读取',
  kicker: '场景 / WAVE 内寄存器读取 · 真实硬件 WAVE32',
  viewBox: '60 45 600 440',
  compactViewBox: '45 55 630 430',
  guides: [{ label: 'lane 值', y: 196 }, { label: 'offset = 2', y: 338 }, { label: 'offset = 1', y: 470 }],
  nodes: waveNodes,
  edges: waveEdges,
  regions: [
    { id: 'example-wave', x: 125, y: 92, width: 470, height: 116, label: '4 个 lane 的缩放示意（真实 wavefront 有 32 个 lane）', tone: 'worker' },
  ],
  frames: [
    {
      id: 'wave-four-lanes',
      eyebrow: '缩放模型',
      title: '先看 4 个 lane 中的 8、3、5、5',
      narration: '9070XT 的真实 wavefront 是 Wave32；这里缩成 4 lane，只为看清数据交换。',
      takeaway: '图中的 4 lane 是教学比例，不是硬件宽度。',
      metrics: [{ label: '图中', value: '4 lane' }, { label: '真实硬件', value: 'Wave32' }],
      activeNodes: waveLaneXs.map((_, index) => `lane-${index}`),
      activeRegions: ['example-wave'],
      duration: 6500,
    },
    {
      id: 'wave-offset-two',
      eyebrow: 'shuffle 编号偏移量（offset）= 2',
      title: 'lane 0/1 分别接收 lane 2/3 的值',
      narration: 'lane 0 得到 8 + 5 = 13，lane 1 得到 3 + 5 = 8。',
      takeaway: 'shuffle 让 lane 在 wave 内读取其他 lane 的寄存器值。',
      metrics: [{ label: '活跃结果', value: '2' }, { label: '值', value: '13, 8' }],
      contextNodes: waveLaneXs.map((_, index) => `lane-${index}`),
      activeNodes: ['offset2-0', 'offset2-1'],
      activeEdges: ['offset2-a', 'offset2-b', 'offset2-c', 'offset2-d'],
      contextRegions: ['example-wave'],
      duration: 7000,
    },
    {
      id: 'wave-offset-one',
      eyebrow: 'shuffle 编号偏移量（offset）= 1',
      title: 'lane 0 得到 13 + 8 = 21',
      narration: '缩放图只剩一个 wave sum；真实 Wave32 会从 offset 16 开始逐轮减半。',
      takeaway: 'offset 逐轮减半，最终只有 lane 0 的值有效。',
      metrics: [{ label: '缩放路径', value: '4 → 2 → 1' }, { label: 'wave sum', value: '21' }],
      contextNodes: ['offset2-0', 'offset2-1'],
      completeNodes: ['wave-result'],
      contextEdges: ['offset2-a', 'offset2-b', 'offset2-c', 'offset2-d'],
      completeEdges: ['offset1-a', 'offset1-b'],
      contextRegions: ['example-wave'],
      duration: 7500,
    },
  ],
}

const stagedNodes: DiagramNode[] = [
  { id: 'staged-group-0', x: 250, y: 105, value: '10', caption: '组 0 局部和', tone: 'local', width: 116, height: 62 },
  { id: 'staged-group-1', x: 470, y: 105, value: '11', caption: '组 1 局部和', tone: 'local', width: 116, height: 62 },
  { id: 'staged-gate', x: 360, y: 225, value: '', caption: '同一 output', tone: 'hot', shape: 'gate', width: 34, height: 54 },
  { id: 'partial-0', x: 250, y: 275, value: '10', caption: 'partial[0]', tone: 'buffer', width: 126, height: 62 },
  { id: 'partial-1', x: 470, y: 275, value: '11', caption: 'partial[1]', tone: 'buffer', width: 126, height: 62 },
  { id: 'next-kernel', x: 360, y: 385, value: 'Kernel 2', caption: '归约 [10, 11]', tone: 'local', width: 174, height: 66 },
  { id: 'staged-output', x: 360, y: 490, value: '?', caption: 'output', tone: 'result', width: 132, height: 60, captionY: -8, valueY: 19 },
]

const stagedEdges: DiagramEdge[] = [
  { id: 'staged-atomic-0', d: 'M 250 138 C 250 172, 330 178, 348 197' },
  { id: 'staged-atomic-1', d: 'M 470 138 C 470 172, 390 178, 372 197' },
  { id: 'staged-gate-output', d: 'M 360 254 C 360 330, 360 405, 360 458' },
  { id: 'store-partial-0', d: 'M 250 138 L 250 242' },
  { id: 'store-partial-1', d: 'M 470 138 L 470 242' },
  { id: 'kernel-read-0', d: 'M 250 308 C 250 340, 320 340, 338 352' },
  { id: 'kernel-read-1', d: 'M 470 308 C 470 340, 400 340, 382 352' },
  { id: 'kernel-output', d: 'M 360 420 L 360 458' },
]

const stagedStory: ScenarioStory = {
  label: '多阶段归约',
  title: '把同址 atomic 换成独立 partial 槽位',
  kicker: '场景 / 多阶段归约',
  viewBox: '60 40 600 500',
  compactViewBox: '45 45 630 495',
  guides: [{ label: 'group partial', y: 154 }, { label: '中间结果', y: 324 }, { label: '下一 kernel', y: 430 }],
  nodes: stagedNodes,
  edges: stagedEdges,
  regions: [
    { id: 'partial-buffer', x: 155, y: 225, width: 410, height: 100, label: 'partial buffer / 独立地址', tone: 'buffer' },
    { id: 'second-stage', x: 235, y: 342, width: 250, height: 104, label: '下一阶段', tone: 'result' },
  ],
  frames: [
    {
      id: 'partials-still-converge',
      eyebrow: '原子收尾仍有热点',
      title: '10 和 11 仍然汇聚到同一 output',
      narration: '组内归约已经完成，但两个 group leader 仍要更新同一个地址。',
      takeaway: '减少 atomic 次数不等于消除跨组热点。',
      metrics: [{ label: 'group partial', value: '2' }, { label: '热点地址', value: '1' }],
      activeNodes: ['staged-group-0', 'staged-group-1'],
      hotNodes: ['staged-gate'],
      contextNodes: ['staged-output'],
      hotEdges: ['staged-atomic-0', 'staged-atomic-1', 'staged-gate-output'],
      values: { 'staged-output': '...' },
      duration: 6000,
    },
    {
      id: 'independent-partials',
      eyebrow: '阶段 1',
      title: '分别写 partial[0] 和 partial[1]',
      narration: '工作组 0 和工作组 1 写入不同地址，不再通过同一扇 atomic 门。',
      takeaway: '独立槽位把同址争用变成普通并行 store。',
      metrics: [{ label: '独立槽位', value: '2' }, { label: '热点 atomic', value: '0' }],
      contextNodes: ['staged-group-0', 'staged-group-1'],
      activeNodes: ['partial-0', 'partial-1'],
      activeEdges: ['store-partial-0', 'store-partial-1'],
      activeRegions: ['partial-buffer'],
      duration: 6500,
    },
    {
      id: 'next-kernel',
      eyebrow: '阶段 2',
      title: '下一 kernel 读取 [10, 11]',
      narration: 'partial buffer 现在就是一个更小的新输入数组。',
      takeaway: '多阶段归约会对更小数组重复相同算法。',
      metrics: [{ label: '新输入长度', value: '2' }, { label: 'kernel', value: '第 2 次' }],
      contextNodes: ['partial-0', 'partial-1', 'staged-output'],
      activeNodes: ['next-kernel'],
      activeEdges: ['kernel-read-0', 'kernel-read-1'],
      contextRegions: ['partial-buffer'],
      activeRegions: ['second-stage'],
      duration: 6500,
    },
    {
      id: 'staged-result',
      eyebrow: '最终结果',
      title: '第二阶段得到 10 + 11 = 21',
      narration: '完整计时必须包含写 partial 和再次归约两个阶段。',
      takeaway: '多阶段消除 atomic，但会增加中间流量和 kernel launch。',
      metrics: [{ label: '数据路径', value: '2 → 1' }, { label: '结果', value: '21' }],
      contextNodes: ['partial-0', 'partial-1', 'next-kernel'],
      completeNodes: ['staged-output'],
      contextEdges: ['kernel-read-0', 'kernel-read-1'],
      completeEdges: ['kernel-output'],
      contextRegions: ['partial-buffer', 'second-stage'],
      values: { 'staged-output': '21' },
      duration: 7000,
    },
  ],
}

const stories: Record<Scenario, ScenarioStory> = {
  atomic: atomicStory,
  program: programStory,
  lds: ldsStory,
  local: localStory,
  wave: waveStory,
  staged: stagedStory,
}

const root = ref<HTMLElement | null>(null)
const stepIndex = ref(0)
const playing = ref(false)
const visible = ref(true)
const reduceMotion = ref(false)
const compactLayout = ref(false)
const markerPrefix = useId().replace(/:/g, '')
const boardId = `${markerPrefix}-board`
const autoplayRate = 0.45

let timer: number | undefined
let observer: IntersectionObserver | undefined
let motionQuery: MediaQueryList | undefined
let compactQuery: MediaQueryList | undefined

const story = computed(() => stories[props.scenario] ?? stories.atomic)
const currentFrame = computed(() => story.value.frames[stepIndex.value] ?? story.value.frames[0])
const canvasViewBox = computed(() => compactLayout.value ? story.value.compactViewBox : story.value.viewBox)
const canvasAspectRatio = computed(() => {
  const [, , width, height] = canvasViewBox.value.split(/\s+/).map(Number)
  return `${width} / ${height}`
})

function has(items: string[] | undefined, id: string) {
  return items?.includes(id) ?? false
}

function visualState(
  id: string,
  hot: string[] | undefined,
  active: string[] | undefined,
  complete: string[] | undefined,
  visibleItems: string[] | undefined,
  context: string[] | undefined,
): VisualState {
  if (has(hot, id)) return 'hot'
  if (has(active, id)) return 'active'
  if (has(complete, id)) return 'complete'
  if (has(visibleItems, id)) return 'visible'
  if (has(context, id)) return 'context'
  return 'hidden'
}

function nodeState(id: string) {
  const frame = currentFrame.value
  return visualState(id, frame.hotNodes, frame.activeNodes, frame.completeNodes, frame.visibleNodes, frame.contextNodes)
}

function edgeState(id: string) {
  const frame = currentFrame.value
  return visualState(id, frame.hotEdges, frame.activeEdges, frame.completeEdges, undefined, frame.contextEdges)
}

function regionState(id: string) {
  const frame = currentFrame.value
  return visualState(id, undefined, frame.activeRegions, undefined, undefined, frame.contextRegions)
}

function nodeValue(node: DiagramNode) {
  return currentFrame.value.values?.[node.id] ?? node.value
}

function markerFor(id: string) {
  const state = edgeState(id)
  if (state === 'hot') return `url(#${markerPrefix}-arrow-hot)`
  if (state === 'complete') return `url(#${markerPrefix}-arrow-complete)`
  if (state === 'active') return `url(#${markerPrefix}-arrow-active)`
  return `url(#${markerPrefix}-arrow-context)`
}

function clearTimer() {
  if (timer !== undefined) {
    window.clearTimeout(timer)
    timer = undefined
  }
}

function scheduleNextStep() {
  clearTimer()
  if (!playing.value || !visible.value || reduceMotion.value) return
  if (stepIndex.value >= story.value.frames.length - 1) {
    playing.value = false
    return
  }

  timer = window.setTimeout(() => {
    stepIndex.value += 1
  }, (currentFrame.value.duration ?? 6000) * autoplayRate)
}

function goToStep(index: number) {
  playing.value = false
  stepIndex.value = Math.min(Math.max(index, 0), story.value.frames.length - 1)
}

function previous() {
  goToStep(stepIndex.value - 1)
}

function next() {
  goToStep(stepIndex.value + 1)
}

function togglePlayback() {
  if (reduceMotion.value) return
  if (playing.value) {
    playing.value = false
    return
  }

  stepIndex.value = stepIndex.value >= story.value.frames.length - 1
    ? 0
    : stepIndex.value + 1
  playing.value = true
}

function reset() {
  goToStep(0)
}

function handleKeydown(event: KeyboardEvent) {
  if (event.altKey || event.ctrlKey || event.metaKey) return
  if (event.key === 'ArrowLeft') {
    event.preventDefault()
    previous()
  } else if (event.key === 'ArrowRight') {
    event.preventDefault()
    next()
  }
}

function handleMotionChange(event: MediaQueryListEvent) {
  reduceMotion.value = event.matches
  if (event.matches) playing.value = false
}

function handleCompactChange(event: MediaQueryListEvent) {
  compactLayout.value = event.matches
}

function handleVisibilityChange() {
  if (document.hidden) playing.value = false
}

watch([playing, visible, stepIndex], scheduleNextStep)
watch(() => props.scenario, () => {
  clearTimer()
  playing.value = false
  stepIndex.value = 0
})

onMounted(() => {
  motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
  reduceMotion.value = motionQuery.matches
  motionQuery.addEventListener('change', handleMotionChange)

  compactQuery = window.matchMedia('(max-width: 640px)')
  compactLayout.value = compactQuery.matches
  compactQuery.addEventListener('change', handleCompactChange)
  document.addEventListener('visibilitychange', handleVisibilityChange)

  if ('IntersectionObserver' in window && root.value) {
    observer = new IntersectionObserver(([entry]) => {
      visible.value = entry.intersectionRatio >= 0.15
    }, { threshold: 0.15 })
    observer.observe(root.value)
  }
})

onBeforeUnmount(() => {
  clearTimer()
  observer?.disconnect()
  motionQuery?.removeEventListener('change', handleMotionChange)
  compactQuery?.removeEventListener('change', handleCompactChange)
  document.removeEventListener('visibilitychange', handleVisibilityChange)
})
</script>

<template>
  <section
    ref="root"
    class="reduction-film"
    :data-scenario="scenario"
    :data-step="currentFrame.id"
    tabindex="0"
    :aria-label="`${story.label} Reduction 交互动画`"
    aria-keyshortcuts="ArrowLeft ArrowRight"
    @keydown="handleKeydown"
  >
    <header class="film-header">
      <div>
        <p class="film-kicker">{{ story.kicker }}</p>
        <h3>{{ story.title }}</h3>
      </div>
      <span class="step-counter">{{ String(stepIndex + 1).padStart(2, '0') }} / {{ story.frames.length }}</span>
    </header>

    <div class="blackboard">
      <div class="board-heading">
        <div>
          <span>{{ currentFrame.eyebrow }}</span>
          <strong>{{ currentFrame.title }}</strong>
        </div>
        <dl class="board-metrics">
          <div v-for="metric in currentFrame.metrics" :key="metric.label">
            <dt>{{ metric.label }}</dt>
            <dd>{{ metric.value }}</dd>
          </div>
        </dl>
      </div>

      <svg
        :id="boardId"
        class="story-canvas"
        :viewBox="canvasViewBox"
        :style="{ aspectRatio: canvasAspectRatio }"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        :aria-label="`${story.label}。${currentFrame.title}。${currentFrame.narration}`"
      >
        <defs>
          <marker :id="`${markerPrefix}-arrow-context`" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L7,3 z" fill="#777d83" />
          </marker>
          <marker :id="`${markerPrefix}-arrow-active`" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L7,3 z" fill="#f7cf68" />
          </marker>
          <marker :id="`${markerPrefix}-arrow-complete`" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L7,3 z" fill="#8fd694" />
          </marker>
          <marker :id="`${markerPrefix}-arrow-hot`" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L7,3 z" fill="#ff766f" />
          </marker>
        </defs>

        <g class="guide-layer" aria-hidden="true">
          <g v-for="guide in story.guides" :key="`${guide.label}-${guide.y}`">
            <text
              :x="guide.x ?? 78"
              :y="guide.y - 7"
              :text-anchor="guide.textAnchor ?? 'start'"
            >{{ guide.label }}</text>
            <line x1="76" :y1="guide.y" x2="650" :y2="guide.y" />
          </g>
        </g>

        <g class="region-layer" aria-hidden="true">
          <g
            v-for="region in story.regions"
            :key="region.id"
            class="diagram-region"
            :class="[`tone-${region.tone}`, `state-${regionState(region.id)}`]"
          >
            <rect :x="region.x" :y="region.y" :width="region.width" :height="region.height" rx="5" />
            <text v-if="region.label" :x="region.x + 12" :y="region.y - 8">{{ region.label }}</text>
          </g>
        </g>

        <g class="edge-layer" aria-hidden="true">
          <path
            v-for="edge in story.edges"
            :key="edge.id"
            :d="edge.d"
            pathLength="1"
            class="diagram-edge"
            :class="`state-${edgeState(edge.id)}`"
            :marker-end="markerFor(edge.id)"
            vector-effect="non-scaling-stroke"
          />
        </g>

        <g
          v-for="node in story.nodes"
          :key="node.id"
          class="diagram-node"
          :class="[`tone-${node.tone}`, `state-${nodeState(node.id)}`]"
          :transform="`translate(${node.x} ${node.y})`"
          aria-hidden="true"
        >
          <template v-if="node.shape === 'gate'">
            <text class="node-caption" x="-28" y="-34">{{ node.caption }}</text>
            <rect :x="-(node.width ?? 34) / 2" :y="-(node.height ?? 54) / 2" :width="node.width ?? 34" :height="node.height ?? 54" rx="3" />
            <path class="gate-lines" d="M -8 -12 L 8 -12 M -8 0 L 8 0 M -8 12 L 8 12" />
          </template>
          <template v-else>
            <text
              class="node-caption"
              x="0"
              :y="node.captionY ?? -((node.height ?? node.width ?? 46) / 2 + 9)"
            >{{ node.caption }}</text>
            <circle v-if="node.shape === 'circle'" :r="(node.width ?? 42) / 2" />
            <rect
              v-else
              :x="-(node.width ?? 46) / 2"
              :y="-(node.height ?? 46) / 2"
              :width="node.width ?? 46"
              :height="node.height ?? 46"
              rx="4"
            />
            <text class="node-value" x="0" :y="node.valueY ?? 8">{{ nodeValue(node) }}</text>
          </template>
        </g>
      </svg>

      <p class="board-subtitle" role="status" aria-live="polite" aria-atomic="true">
        {{ currentFrame.narration }}
      </p>
    </div>

    <div class="lesson-band">
      <p class="takeaway"><span>这一步只记住</span>{{ currentFrame.takeaway }}</p>
    </div>

    <footer class="film-footer">
      <div class="film-controls" role="group" :aria-label="`${story.label} 动画控制`">
        <button type="button" :disabled="stepIndex === 0" title="上一步" aria-label="上一步" @click="previous">←</button>
        <button
          type="button"
          class="play-button"
          :disabled="reduceMotion"
          :aria-pressed="playing"
          :aria-controls="boardId"
          :title="playing ? '暂停' : '播放'"
          :aria-label="playing ? '暂停动画' : '播放动画'"
          @click="togglePlayback"
        >
          {{ playing ? 'Ⅱ' : '▶' }}
        </button>
        <button type="button" :disabled="stepIndex === story.frames.length - 1" title="下一步" aria-label="下一步" @click="next">→</button>
        <button type="button" title="回到开头" aria-label="回到开头" @click="reset">↺</button>
      </div>

      <div class="frame-dots" role="group" aria-label="选择当前动画步骤">
        <button
          v-for="(frame, index) in story.frames"
          :key="frame.id"
          type="button"
          :class="{ current: index === stepIndex }"
          :aria-current="index === stepIndex ? 'step' : undefined"
          :aria-label="`第 ${index + 1} 步：${frame.title}`"
          @click="goToStep(index)"
        />
      </div>
    </footer>

    <p v-if="reduceMotion" class="motion-note">系统已启用“减少动态效果”，自动播放已关闭。</p>
  </section>
</template>

<style scoped>
.reduction-film {
  --board: #07090b;
  --board-ink: #f3f1ea;
  --board-muted: #8f969d;
  --hot: #ff766f;
  --local: #f7cf68;
  --buffer: #76d1d8;
  --result: #8fd694;
  box-sizing: border-box;
  width: 100%;
  max-width: 920px;
  min-width: 0;
  margin: 22px auto 8px;
  scroll-margin-top: 72px;
  color: var(--vp-c-text-1);
  outline: none;
}

.reduction-film:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 4px;
}

.film-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 12px;
}

.film-header h3 {
  margin: 2px 0 0;
  font-family: ui-serif, "Songti SC", "Noto Serif CJK SC", serif;
  font-size: 1.08rem;
  line-height: 1.4;
  letter-spacing: 0;
}

.film-kicker,
.step-counter,
.motion-note {
  margin: 0;
  color: var(--vp-c-text-2);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.72rem;
  letter-spacing: 0;
}

.step-counter {
  flex: 0 0 auto;
  padding-bottom: 2px;
  font-variant-numeric: tabular-nums;
}

.blackboard {
  position: relative;
  overflow: hidden;
  border: 1px solid #30353a;
  border-radius: 6px;
  background: var(--board);
  color: var(--board-ink);
  box-shadow: 0 18px 42px rgb(0 0 0 / 16%);
}

.board-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  min-height: 78px;
  padding: 18px 20px 12px;
  border-bottom: 1px solid #34383c;
}

.board-heading span {
  display: block;
  margin-bottom: 3px;
  color: var(--local);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.72rem;
}

.board-heading strong {
  display: block;
  font-family: ui-serif, "Songti SC", "Noto Serif CJK SC", serif;
  font-size: 1rem;
  font-weight: 600;
  line-height: 1.4;
  letter-spacing: 0;
}

.board-metrics {
  display: flex;
  flex: 0 0 auto;
  gap: 18px;
  margin: 0;
}

.board-metrics div {
  min-width: 68px;
  text-align: right;
}

.board-metrics dt {
  color: var(--board-muted);
  font-size: 0.66rem;
}

.board-metrics dd {
  margin: 2px 0 0;
  color: var(--board-ink);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.82rem;
  font-variant-numeric: tabular-nums;
}

.story-canvas {
  display: block;
  width: 100%;
  height: auto;
  overflow: visible;
  background-color: var(--board);
  background-image:
    linear-gradient(rgb(255 255 255 / 2%) 1px, transparent 1px),
    linear-gradient(90deg, rgb(255 255 255 / 2%) 1px, transparent 1px);
  background-size: 28px 28px;
}

.guide-layer text,
.diagram-region text {
  fill: var(--board-muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
  letter-spacing: 0;
}

.guide-layer line {
  stroke: #292e32;
  stroke-dasharray: 4 8;
  stroke-width: 1;
  vector-effect: non-scaling-stroke;
}

.diagram-region,
.diagram-node,
.diagram-edge {
  transition: opacity 360ms ease, filter 360ms ease, stroke 360ms ease, fill 360ms ease;
}

.diagram-region rect {
  fill: rgb(255 255 255 / 1%);
  stroke-width: 1.5;
  stroke-dasharray: 5 6;
  vector-effect: non-scaling-stroke;
}

.diagram-region.tone-local rect { stroke: var(--local); }
.diagram-region.tone-buffer rect { stroke: var(--buffer); }
.diagram-region.tone-result rect { stroke: var(--result); }
.diagram-region.tone-worker rect { stroke: var(--hot); }

.diagram-edge {
  fill: none;
  stroke-linecap: round;
  stroke-width: 1.7;
}

.diagram-edge.state-context {
  stroke: #777d83;
  opacity: 0.28;
}

.diagram-edge.state-active {
  stroke: var(--local);
  stroke-dasharray: 0.07 0.045;
  animation: edge-flow 1.2s linear infinite;
}

.diagram-edge.state-complete {
  stroke: var(--result);
  opacity: 0.78;
}

.diagram-edge.state-hot {
  stroke: var(--hot);
  stroke-dasharray: 0.055 0.035;
  animation: edge-flow 0.9s linear infinite;
}

@keyframes edge-flow {
  to { stroke-dashoffset: -0.18; }
}

.diagram-node text {
  text-anchor: middle;
  user-select: none;
}

.diagram-node rect,
.diagram-node circle {
  fill: #0b0e11;
  stroke-width: 1.8;
  vector-effect: non-scaling-stroke;
}

.diagram-node.tone-input rect { stroke: var(--buffer); }
.diagram-node.tone-worker rect,
.diagram-node.tone-worker circle { stroke: var(--hot); }
.diagram-node.tone-local rect { stroke: var(--local); }
.diagram-node.tone-buffer rect { stroke: var(--buffer); }
.diagram-node.tone-result rect { stroke: var(--result); stroke-width: 2.2; }
.diagram-node.tone-hot rect { stroke: var(--hot); }

.node-caption {
  fill: var(--board-muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
}

.node-value {
  fill: var(--board-ink);
  font-family: ui-serif, Georgia, serif;
  font-size: 21px;
  font-variant-numeric: tabular-nums;
}

.tone-result .node-value {
  fill: var(--result);
  font-size: 24px;
  font-weight: 700;
}

.gate-lines {
  fill: none;
  stroke: var(--hot);
  stroke-width: 1.8;
  vector-effect: non-scaling-stroke;
}

.state-hidden {
  opacity: 0;
  pointer-events: none;
}

.state-context { opacity: 0.35; }
.state-visible { opacity: 0.78; }
.state-active,
.state-complete,
.state-hot { opacity: 1; }

.diagram-node.state-active {
  filter: drop-shadow(0 0 7px rgb(247 207 104 / 34%));
}

.diagram-node.state-complete {
  filter: drop-shadow(0 0 7px rgb(143 214 148 / 30%));
}

.diagram-node.state-hot {
  filter: drop-shadow(0 0 8px rgb(255 118 111 / 42%));
  animation: hotspot-pulse 1.3s ease-in-out infinite;
}

@keyframes hotspot-pulse {
  50% { opacity: 0.58; }
}

.board-subtitle {
  min-height: 58px;
  margin: 0;
  padding: 14px 18px;
  border-top: 1px solid #34383c;
  background: #f1b6c8;
  color: #2b1820;
  font-size: 0.86rem;
  font-weight: 650;
  line-height: 1.55;
  text-align: center;
}

.lesson-band {
  padding: 15px 2px 11px;
  border-bottom: 1px solid var(--vp-c-divider);
}

.takeaway {
  display: grid;
  grid-template-columns: 98px minmax(0, 1fr);
  gap: 12px;
  margin: 0;
  color: var(--vp-c-text-1);
  font-size: 0.86rem;
  line-height: 1.55;
}

.takeaway span {
  color: var(--vp-c-text-3);
  font-size: 0.72rem;
  font-weight: 600;
}

.film-footer {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding-top: 12px;
}

.film-controls {
  display: flex;
  gap: 7px;
}

.film-controls button {
  display: inline-grid;
  place-items: center;
  width: 44px;
  height: 44px;
  padding: 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 5px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  cursor: pointer;
  font-size: 1rem;
  line-height: 1;
}

.film-controls button:hover:not(:disabled),
.film-controls button:focus-visible {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.film-controls button:focus-visible,
.frame-dots button:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
}

.film-controls .play-button {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-1);
  color: var(--vp-c-bg);
}

.film-controls button:disabled {
  cursor: not-allowed;
  opacity: 0.38;
}

.frame-dots {
  display: flex;
  flex: 1 1 280px;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  min-width: 0;
  gap: 4px;
}

.frame-dots button {
  position: relative;
  width: 24px;
  height: 36px;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: pointer;
}

.frame-dots button::before {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 10px;
  height: 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 50%;
  background: var(--vp-c-bg);
  content: "";
  transform: translate(-50%, -50%);
}

.frame-dots button.current::before {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-1);
}

.motion-note {
  margin-top: 9px;
}

@media (max-width: 640px) {
  .film-header {
    align-items: flex-start;
  }

  .film-header h3 {
    font-size: 0.98rem;
  }

  .film-kicker {
    font-size: 0.62rem;
  }

  .board-heading {
    display: grid;
    min-height: 112px;
    padding: 14px;
  }

  .board-heading strong {
    font-size: 0.9rem;
  }

  .board-metrics {
    justify-content: flex-start;
    gap: 24px;
  }

  .board-metrics div {
    min-width: 0;
    text-align: left;
  }

  .guide-layer text,
  .diagram-region text {
    font-size: 15px;
  }

  .node-caption {
    font-size: 14px;
  }

  .board-subtitle {
    min-height: 80px;
    padding: 12px 14px;
    font-size: 0.78rem;
    text-align: left;
  }

  .takeaway {
    grid-template-columns: 1fr;
    gap: 4px;
    font-size: 0.8rem;
  }

  .lesson-band {
    min-height: 89px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .diagram-region,
  .diagram-node,
  .diagram-edge {
    transition: none;
  }

  .diagram-edge.state-active,
  .diagram-edge.state-hot,
  .diagram-node.state-hot {
    animation: none;
  }
}
</style>
