<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'

type Scenario = 'paradigm' | 'dependency' | 'memory' | 'vector' | 'triton'

const props = withDefaults(defineProps<{ scenario?: Scenario }>(), {
  scenario: 'dependency'
})

const scenarioMeta: Record<Scenario, { eyebrow: string; title: string; steps: number }> = {
  paradigm: {
    eyebrow: '源码视角',
    title: '同一个 Vector Add，两种编程范式',
    steps: 4
  },
  dependency: {
    eyebrow: '语义层',
    title: '位置 i 独立计算 C[i] = A[i] + B[i]',
    steps: 8
  },
  memory: {
    eyebrow: 'HIP · 8-lane 教学缩略',
    title: '同样 8 个 lane，地址排法决定触及几组',
    steps: 4
  },
  vector: {
    eyebrow: 'HIP · float4',
    title: '四个 FP32 组成一组，尾部单独处理',
    steps: 3
  },
  triton: {
    eyebrow: 'Triton · Tile',
    title: '生成 offsets，再用 mask 关掉越界位置',
    steps: 8
  }
}

const step = ref(0)
const demoRoot = ref<HTMLElement | null>(null)
const playbackRequested = ref(true)
const visible = ref(false)
const playing = computed(() => playbackRequested.value && visible.value)
let timer: ReturnType<typeof setTimeout> | undefined
let observer: IntersectionObserver | undefined
let hasBeenVisible = false

const meta = computed(() => scenarioMeta[props.scenario])

// ---------- paradigm ----------
const paradigmSize = 6
const paradigmIndices = Array.from({ length: 8 }, (_, index) => index)
const paradigmPrograms = [
  { id: 0, offsets: [0, 1, 2, 3] },
  { id: 1, offsets: [4, 5, 6, 7] }
]
const paradigmStepTitles = [
  '先固定问题：只允许写回有效位置 0–5',
  'HIP：每个线程先得到一个标量下标 i',
  'Triton：每个 program 一次生成一块 offsets',
  '逐项对照：数学不变，但概念不是等价替换'
]
const paradigmStepTitle = computed(() => paradigmStepTitles[step.value])

// ---------- dependency ----------
const inputA = [2, -1, 4, 3, 0, 5, -2, 1]
const inputB = [7, 3, -1, 2, 6, -2, 4, 8]
const output = inputA.map((value, index) => value + inputB[index])
const dependencyGridScroll = ref<HTMLElement | null>(null)

// ---------- memory ----------
const lanes = Array.from({ length: 8 }, (_, index) => index)
// 教学分桶：每组 8 个连续 FP32（32B），不代表实测物理显存事务。
const ADDRESS_GROUP_SPAN = 8

const contiguousAddresses = computed(() =>
  lanes.map(lane => step.value * 8 + lane)
)
const stridedAddresses = computed(() =>
  lanes.map(lane => lane * 4 + step.value)
)

const contiguousOwner = computed(() => {
  const owners = new Map<number, number>()
  contiguousAddresses.value.forEach((address, lane) => owners.set(address, lane))
  return owners
})

const stridedOwner = computed(() => {
  const owners = new Map<number, number>()
  stridedAddresses.value.forEach((address, lane) => owners.set(address, lane))
  return owners
})

function touchedAddressGroups(owner: Map<number, number>): Set<number> {
  const blocks = new Set<number>()
  owner.forEach((_lane, index) => blocks.add(Math.floor(index / ADDRESS_GROUP_SPAN)))
  return blocks
}

const contiguousTouchedGroups = computed(() => touchedAddressGroups(contiguousOwner.value))
const stridedTouchedGroups = computed(() => touchedAddressGroups(stridedOwner.value))
const contiguousGroupsThisRound = computed(() => contiguousTouchedGroups.value.size)
const stridedGroupsThisRound = computed(() => stridedTouchedGroups.value.size)

function cumulativeGroupVisits(kind: 'contiguous' | 'strided', uptoRound: number): number {
  let visits = 0
  for (let round = 0; round <= uptoRound; round++) {
    const groups = new Set<number>()
    lanes.forEach(lane => {
      const index = kind === 'contiguous' ? round * 8 + lane : lane * 4 + round
      groups.add(Math.floor(index / ADDRESS_GROUP_SPAN))
    })
    visits += groups.size
  }
  return visits
}

const contiguousGroupVisits = computed(() => cumulativeGroupVisits('contiguous', step.value))
const stridedGroupVisits = computed(() => cumulativeGroupVisits('strided', step.value))

// ---------- vector ----------
const vectorIndices = Array.from({ length: 18 }, (_, index) => index)
const vectorTrackScroll = ref<HTMLElement | null>(null)
const vectorGroups = Array.from({ length: 4 }, (_, index) => ({
  id: `组 ${index}`,
  vectorIndex: index,
  start: index * 4,
  end: index * 4 + 3
}))
const vectorPhases = ['标量布局', '4 个 float4 组', '标量尾部']
const vectorStatuses = [
  '先把 A 看成 18 个独立的 FP32 元素',
  'A[0–15] 在源码中重解释为 4 个 float4 组',
  'A[16–17] 不足四个元素，回到标量路径'
]
const vectorStatus = computed(() =>
  vectorStatuses[Math.min(step.value, vectorStatuses.length - 1)]
)
const vectorSourceSummary = computed(() => [
  '输入 A：18 个标量值 × 4B',
  '输入 A：4 个 float4 值覆盖前 16 个元素',
  '输入 A：4 个 float4 值 + 2 个标量值'
][step.value])

watch(step, () => {
  requestAnimationFrame(() => {
    const track = props.scenario === 'dependency'
      ? dependencyGridScroll.value
      : props.scenario === 'vector'
        ? vectorTrackScroll.value
        : null
    if (!track) return
    const maxScroll = track.scrollWidth - track.clientWidth
    const position = props.scenario === 'dependency'
      ? (step.value / (inputA.length - 1)) * maxScroll
      : [0, 0.5, 1][step.value] * maxScroll
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    track.scrollTo({ left: position, behavior: reducedMotion ? 'auto' : 'smooth' })
  })
})

// ---------- triton ----------
const tritonPositions = Array.from({ length: 8 }, (_, index) => index)
const tritonProgram = computed(() => Math.floor(step.value / 4))
const tritonPhase = computed(() => step.value % 4)
const tritonOffsets = computed(() =>
  tritonPositions.map(position => tritonProgram.value * 8 + position)
)
const phaseAction = computed(() => ['读取 A', '读取 B', '执行 A + B', '写回 C'][tritonPhase.value])
const maskedPhaseLabel = computed(() => ['other=0', 'other=0', '不参与', '不写回'][tritonPhase.value])
const tritonValidCount = computed(() =>
  tritonOffsets.value.filter(offset => offset < 13).length
)
const tritonSubstitution = computed(() =>
  `${tritonProgram.value} × 8 + [0…7] = [${tritonOffsets.value[0]}…${tritonOffsets.value[7]}]`
)
const tritonBoundarySummary = computed(() => {
  const valid = tritonOffsets.value.filter(offset => offset < 13)
  const masked = tritonOffsets.value.filter(offset => offset >= 13)
  if (masked.length === 0) return `mask=true：${valid[0]}–${valid[valid.length - 1]}（全部有效）`
  return `mask=true：${valid[0]}–${valid[valid.length - 1]} · mask=false：${masked.join(', ')}`
})

const explanation = computed(() => {
  if (props.scenario === 'paradigm') {
    return [
      'N=6，但启动范围覆盖位置 0–7；两种写法都只能写回 C[0] 到 C[5]。',
      'HIP kernel 描述一个线程的工作：当前线程得到标量下标 i，线程 6 和 7 由 if 关闭。',
      'Triton kernel 描述一个 program 的工作：当前 program 生成一块 offsets，超出 N 的位置由 mask 关闭。',
      'HIP 从 thread 生成标量 i，Triton 从 program 生成一块 offsets；两条路径最终都只写回有效位置 0–5。'
    ][step.value]
  }
  if (props.scenario === 'dependency') {
    const index = step.value
    return `位置 ${index} 只读取 A[${index}]=${inputA[index]} 与 B[${index}]=${inputB[index]}，得到 C[${index}]=${output[index]}。`
  }
  if (props.scenario === 'memory') {
    const contiguous = contiguousAddresses.value
    return `第 ${step.value + 1} 轮：连续下标 [${contiguous[0]}…${contiguous[7]}] 触及 ${contiguousGroupsThisRound.value} 个地址组；跨步下标 [${stridedAddresses.value.join(', ')}] 触及 ${stridedGroupsThisRound.value} 个地址组。`
  }
  if (props.scenario === 'vector') {
    return [
      '这一步只画输入 A 的源码布局：18 个 FP32 元素分别占 4 Byte。',
      '前 16 个元素组成 4 个 float4 源码组；四组同时成立，动画展开顺序仅用于讲解。',
      '最后两个元素不足一组，走标量尾部。输入 B 的读取与输出 C 的写回采用同样分组。'
    ][step.value]
  }
  const offsets = tritonOffsets.value
  return `program ${tritonProgram.value} 在 offsets=[${offsets[0]}…${offsets[7]}] 上${phaseAction.value}；${tritonValidCount.value} 个 tile 位置 mask=true，其余位置不会产生越界访问。`
})

function stopTimer() {
  if (timer !== undefined) {
    clearInterval(timer)
    timer = undefined
  }
}

function startTimer() {
  stopTimer()
  if (!playing.value) return
  const delay = step.value === meta.value.steps - 1 ? 3600 : 2600
  timer = setTimeout(() => {
    step.value = (step.value + 1) % meta.value.steps
    startTimer()
  }, delay)
}

function previous() {
  playbackRequested.value = false
  stopTimer()
  step.value = (step.value - 1 + meta.value.steps) % meta.value.steps
}

function next() {
  playbackRequested.value = false
  stopTimer()
  step.value = (step.value + 1) % meta.value.steps
}

function togglePlaying() {
  playbackRequested.value = !playbackRequested.value
  startTimer()
}

watch(
  () => props.scenario,
  () => {
    step.value = 0
    playbackRequested.value = true
    startTimer()
  }
)

onMounted(() => {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    playbackRequested.value = false
  }

  if (typeof IntersectionObserver === 'undefined' || !demoRoot.value) {
    visible.value = true
    startTimer()
    return
  }

  observer = new IntersectionObserver(([entry]) => {
    const nextVisible = entry.isIntersecting
    if (nextVisible === visible.value) return
    visible.value = nextVisible
    if (nextVisible && !hasBeenVisible) {
      step.value = 0
      hasBeenVisible = true
    }
    startTimer()
  }, { threshold: 0.35 })
  observer.observe(demoRoot.value)
})

onUnmounted(() => {
  observer?.disconnect()
  stopTimer()
})
</script>

<template>
  <div
    ref="demoRoot"
    class="ew-demo"
    :data-scenario="scenario"
    :style="{ '--step-progress': `${((step + 1) / meta.steps) * 100}%` }"
  >
    <header class="demo-header">
      <div>
        <span class="eyebrow">{{ meta.eyebrow }}</span>
        <strong>{{ meta.title }}</strong>
      </div>
      <div class="controls" role="group" aria-label="动画控制">
        <button class="control-previous" type="button" title="上一步" aria-label="上一步" @click="previous">←</button>
        <button
          class="control-play"
          type="button"
          :title="playing ? '暂停' : '播放'"
          :aria-label="playing ? '暂停' : '播放'"
          :aria-pressed="playing"
          @click="togglePlaying"
        >
          {{ playing ? 'Ⅱ' : '▶' }}
        </button>
        <button class="control-next" type="button" title="下一步" aria-label="下一步" @click="next">→</button>
      </div>
    </header>

    <div class="stage" role="img" :aria-label="explanation">
      <!-- ============ paradigm ============ -->
      <div v-if="scenario === 'paradigm'" class="paradigm-board">
        <div class="paradigm-shared">
          <span>同一道题 · N = 6</span>
          <code>C[i] = A[i] + B[i]</code>
          <strong>{{ paradigmStepTitle }}</strong>
          <div class="paradigm-output-strip" aria-hidden="true">
            <i
              v-for="index in paradigmIndices"
              :key="`shared-${index}`"
              :class="{ invalid: index >= paradigmSize }"
            >{{ index }}{{ index >= paradigmSize ? ' ×' : '' }}</i>
          </div>
        </div>

        <div class="paradigm-panels">
          <section
            class="paradigm-panel paradigm-hip"
            :class="{
              active: step === 1 || step === 3,
              concealed: step === 0,
              subdued: step === 2
            }"
          >
            <header>
              <span>HIP / SIMT</span>
              <strong>每个 thread 执行标量代码</strong>
              <small>Scalar Program · Blocked Threads</small>
            </header>
            <div class="paradigm-subject">
              <span>图中启动配置</span>
              <b>blockIdx = 0 · blockDim = 8</b>
            </div>
            <code class="paradigm-formula">i = blockIdx.x × blockDim.x + threadIdx.x</code>
            <div class="hip-thread-grid">
              <span
                v-for="index in paradigmIndices"
                :key="`hip-thread-${index}`"
                :class="{ invalid: index >= paradigmSize }"
              >
                <small>thread {{ index }}</small>
                <b>{{ index < paradigmSize ? `i = ${index}` : 'if = false' }}</b>
              </span>
            </div>
            <div class="paradigm-boundary">
              <code>if (i &lt; N)</code>
              <span>一个线程保护一个标量位置</span>
            </div>
          </section>

          <section
            class="paradigm-panel paradigm-triton"
            :class="{
              active: step === 2 || step === 3,
              concealed: step < 2
            }"
          >
            <header>
              <span>TRITON</span>
              <strong>分块 program · 编译器安排线程</strong>
              <small>Blocked Program · Scalar Threads</small>
            </header>
            <div class="paradigm-subject">
              <span>图中启动配置</span>
              <b>grid = (2,) · BLOCK_SIZE = 4</b>
            </div>
            <code class="paradigm-formula">offsets = pid × 4 + tl.arange(0, 4)</code>
            <div class="triton-program-grid">
              <span v-for="program in paradigmPrograms" :key="`program-${program.id}`">
                <small>program {{ program.id }}</small>
                <b>[{{ program.offsets.join(', ') }}]</b>
                <em>
                  <i
                    v-for="offset in program.offsets"
                    :key="`offset-${offset}`"
                    :class="{ invalid: offset >= paradigmSize }"
                  >{{ offset }}{{ offset >= paradigmSize ? ' ×' : '' }}</i>
                </em>
              </span>
            </div>
            <div class="paradigm-boundary">
              <code>mask = offsets &lt; N</code>
              <span>一个 mask 逐位置保护整块数据</span>
            </div>
          </section>
        </div>

        <div class="paradigm-translation" :class="{ active: step === 3 }">
          <span><small>源码主语</small><b>HIP · thread</b><i>vs</i><b>Triton · program</b></span>
          <span><small>索引对象</small><b>HIP · 标量 i</b><i>vs</i><b>Triton · 块 offsets</b></span>
          <span><small>边界保护</small><b>HIP · if</b><i>vs</i><b>Triton · mask</b></span>
          <span><small>启动网格</small><b>HIP · block 组成 grid</b><i>vs</i><b>Triton · program 组成 grid</b></span>
        </div>
        <p class="paradigm-note" :class="{ active: step === 3 }">
          HIP 的 <code>blockDim.x</code> 显式给出每个 block 的 thread 数；Triton 的 <code>BLOCK_SIZE</code> 是逻辑 tile 宽度。
          <code>tl.arange</code> 不创建线程，编译参数与编译器布局继续安排整块计算的执行。
        </p>
      </div>

      <!-- ============ dependency ============ -->
      <div v-else-if="scenario === 'dependency'" class="dependency-board">
        <div class="dep-legend">
          <span><i class="chip chip-thread"></i>当前计算位置</span>
          <span><i class="chip chip-a"></i>输入 A</span>
          <span><i class="chip chip-b"></i>输入 B</span>
          <span><i class="chip chip-c"></i>输出 C</span>
        </div>
        <div class="dep-equation" aria-hidden="true">
          <span class="dep-equation-index">位置 {{ step }}</span>
          <span class="dep-equation-term input-a">
            <small>A[{{ step }}]</small><b>{{ inputA[step] }}</b>
          </span>
          <i>+</i>
          <span class="dep-equation-term input-b">
            <small>B[{{ step }}]</small><b>{{ inputB[step] }}</b>
          </span>
          <i>=</i>
          <span class="dep-equation-term output">
            <small>C[{{ step }}]</small><b>{{ output[step] }}</b>
          </span>
        </div>
        <div ref="dependencyGridScroll" class="dep-grid-scroll">
          <div class="dep-grid">
            <span class="dep-corner"></span>
            <span
              v-for="index in inputA.length"
              :key="`thread-${index}`"
              class="dep-thread"
              :class="{ current: index - 1 === step }"
            >位置 {{ index - 1 }}</span>

            <span class="row-label input-a-label">A</span>
            <span
              v-for="(value, index) in inputA"
              :key="`a-${index}`"
              class="data-cell input-a"
              :class="{ current: index === step }"
            >
              <small>{{ index }}</small>{{ value }}
            </span>

            <span class="row-op" aria-hidden="true">+</span>
            <span
              v-for="index in inputA.length"
              :key="`plus-${index}`"
              class="flow-arrow"
              :class="{ current: index - 1 === step }"
            >↓</span>

            <span class="row-label input-b-label">B</span>
            <span
              v-for="(value, index) in inputB"
              :key="`b-${index}`"
              class="data-cell input-b"
              :class="{ current: index === step }"
            >
              <small>{{ index }}</small>{{ value }}
            </span>

            <span class="row-op" aria-hidden="true">=</span>
            <span
              v-for="index in inputA.length"
              :key="`eq-${index}`"
              class="flow-arrow"
              :class="{ current: index - 1 === step }"
            >↓</span>

            <span class="row-label output-label">C</span>
            <span
              v-for="(value, index) in output"
              :key="`c-${index}`"
              class="data-cell output"
              :class="{ current: index === step, done: index < step }"
            >
              <small>{{ index }}</small>{{ index <= step ? value : '?' }}
            </span>
          </div>
        </div>
        <p class="dep-note">逐列播放只是为了讲解；每一列彼此独立，真实执行时可以并行完成。</p>
      </div>

      <!-- ============ memory ============ -->
      <div v-else-if="scenario === 'memory'" class="memory-comparison">
        <div class="memory-round">
          <span>第 {{ step + 1 }} / 4 轮</span>
          <strong>同样 8 个 lane：连续排列触及 {{ contiguousGroupsThisRound }} 组，跨步排列触及 {{ stridedGroupsThisRound }} 组</strong>
          <small>32B 地址组是教学分桶，不是硬件事务计数</small>
        </div>
        <section class="memory-panel contiguous">
          <div class="panel-heading">
            <div>
              <span class="panel-kind">合并</span>
              <strong>连续顺序</strong>
            </div>
            <code>index = round × 8 + lane</code>
          </div>
          <div class="address-sequence">
            <span>lane 0–7 得到的下标</span>
            <code>[{{ contiguousAddresses.join(', ') }}]</code>
          </div>
          <div class="memory-stack">
            <div
              v-for="block in 4"
              :key="`contiguous-block-${block}`"
              class="memory-tx-row"
              :class="{ touched: contiguousTouchedGroups.has(block - 1) }"
            >
              <span class="tx-row-label"><b>组 {{ block - 1 }}</b><small>32B</small></span>
              <span
                v-for="offset in ADDRESS_GROUP_SPAN"
                :key="`contiguous-${(block - 1) * ADDRESS_GROUP_SPAN + offset - 1}`"
                class="memory-cell"
                :class="{ active: contiguousOwner.has((block - 1) * ADDRESS_GROUP_SPAN + offset - 1) }"
              >
                <small v-if="contiguousOwner.has((block - 1) * ADDRESS_GROUP_SPAN + offset - 1)">
                  L{{ contiguousOwner.get((block - 1) * ADDRESS_GROUP_SPAN + offset - 1) }}
                </small>
                <b>{{ (block - 1) * ADDRESS_GROUP_SPAN + offset - 1 }}</b>
              </span>
            </div>
          </div>
          <div class="tx-counter">
            <span><small>本轮触及组数</small><b>{{ contiguousGroupsThisRound }}</b></span>
            <span><small>累计组访问</small><b>{{ contiguousGroupVisits }}</b></span>
          </div>
        </section>
        <section class="memory-panel strided">
          <div class="panel-heading">
            <div>
              <span class="panel-kind">分散</span>
              <strong>跨步顺序</strong>
            </div>
            <code>index = lane × 4 + round</code>
          </div>
          <div class="address-sequence">
            <span>lane 0–7 得到的下标</span>
            <code>[{{ stridedAddresses.join(', ') }}]</code>
          </div>
          <div class="memory-stack">
            <div
              v-for="block in 4"
              :key="`strided-block-${block}`"
              class="memory-tx-row"
              :class="{ touched: stridedTouchedGroups.has(block - 1) }"
            >
              <span class="tx-row-label"><b>组 {{ block - 1 }}</b><small>32B</small></span>
              <span
                v-for="offset in ADDRESS_GROUP_SPAN"
                :key="`strided-${(block - 1) * ADDRESS_GROUP_SPAN + offset - 1}`"
                class="memory-cell"
                :class="{ active: stridedOwner.has((block - 1) * ADDRESS_GROUP_SPAN + offset - 1) }"
              >
                <small v-if="stridedOwner.has((block - 1) * ADDRESS_GROUP_SPAN + offset - 1)">
                  L{{ stridedOwner.get((block - 1) * ADDRESS_GROUP_SPAN + offset - 1) }}
                </small>
                <b>{{ (block - 1) * ADDRESS_GROUP_SPAN + offset - 1 }}</b>
              </span>
            </div>
          </div>
          <div class="tx-counter">
            <span><small>本轮触及组数</small><b>{{ stridedGroupsThisRound }}</b></span>
            <span><small>累计组访问</small><b>{{ stridedGroupVisits }}</b></span>
          </div>
        </section>
      </div>

      <!-- ============ vector ============ -->
      <div v-else-if="scenario === 'vector'" class="vectorization-board">
        <div class="vector-legend">
          <span><i class="legend-scalar"></i>FP32 元素 · 4B</span>
          <span><i class="legend-vector"></i>float4 源码组 · 16B</span>
          <span><i class="legend-tail"></i>不足四个的尾部</span>
        </div>

        <div class="vector-summary">
          <span><small>当前视角</small><strong>{{ vectorStatus }}</strong></span>
          <code>{{ vectorSourceSummary }}</code>
        </div>

        <div ref="vectorTrackScroll" class="vector-track-scroll">
          <div class="vector-track">
            <span
              v-for="index in vectorIndices"
              :key="`element-${index}`"
              class="vector-element"
              :class="{
                'is-grouped': index < 16 && step >= 1,
                'is-current-group': index < 16 && step === 1,
                'is-tail-pending': index >= 16 && step === 1,
                'is-tail-active': index >= 16 && step === 2
              }"
              :style="{ gridColumn: String(index + 1) }"
            >
              <small>FP32</small>
              <b>A{{ index }}</b>
              <em>4B</em>
            </span>

            <span
              v-for="group in vectorGroups"
              :key="group.id"
              class="vector-group-frame"
              :class="{
                'is-active': step >= 1,
                'is-current': step === 1
              }"
              :style="{ gridColumn: `${group.start + 1} / span 4` }"
              aria-hidden="true"
            >
              <b>{{ group.id }} · A[{{ group.start }}–{{ group.end }}]</b>
              <em>float4[{{ group.vectorIndex }}] · 16B</em>
            </span>
          </div>
        </div>

        <div class="vector-progress" aria-hidden="true">
          <span
            v-for="(phase, index) in vectorPhases"
            :key="phase"
            :class="{ 'is-done': index < step, 'is-current': index === step }"
          >{{ phase }}</span>
        </div>

        <div class="alignment-line">
          <span class="alignment-start">地址 0（16 Byte 对齐）</span>
          <span class="alignment-size">N = 18</span>
        </div>
        <p class="vector-note">仅以输入 A 的读取为例；B 的读取与 C 的写回分组同形。源码分组不等于物理显存事务一定减少。</p>
      </div>

      <!-- ============ triton ============ -->
      <div v-else class="triton-board">
        <div class="program-strip">
          <span :class="{ active: tritonProgram === 0, done: tritonProgram > 0 }">
            <small>program 0</small><b>offsets 0–7</b>
          </span>
          <span :class="{ active: tritonProgram === 1 }">
            <small>program 1</small><b>offsets 8–15</b>
          </span>
        </div>
        <div class="triton-formula">
          <span>索引生成</span>
          <div>
            <code>offsets = pid × BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)</code>
            <strong>{{ tritonSubstitution }}</strong>
          </div>
        </div>
        <div class="offset-heading">
          <span>边界判断：offset &lt; N（N = 13）</span>
          <strong>{{ tritonBoundarySummary }}</strong>
        </div>
        <div class="offset-grid">
          <span
            v-for="(offset, position) in tritonOffsets"
            :key="`offset-${tritonProgram}-${position}`"
            class="offset-cell"
            :class="[
              offset >= 13 ? 'masked' : '',
              offset < 13 && tritonPhase === 0 ? 'phase-a' : '',
              offset < 13 && tritonPhase === 1 ? 'phase-b' : '',
              offset < 13 && tritonPhase === 2 ? 'phase-add' : '',
              offset < 13 && tritonPhase === 3 ? 'phase-c' : ''
            ]"
          >
            <small>tile 位置 {{ position }}</small>
            <b>{{ offset }}</b>
            <em>{{ offset < 13 ? phaseAction : maskedPhaseLabel }}</em>
          </span>
        </div>
        <div class="pipeline" aria-label="Triton 数据路径">
          <span :class="{ active: tritonPhase === 0, done: tritonPhase > 0 }">
            <small>tl.load</small><b>读取 A</b>
          </span>
          <i>→</i>
          <span :class="{ active: tritonPhase === 1, done: tritonPhase > 1 }">
            <small>tl.load</small><b>读取 B</b>
          </span>
          <i>→</i>
          <span :class="{ active: tritonPhase === 2, done: tritonPhase > 2 }">
            <small>element-wise</small><b>A + B</b>
          </span>
          <i>→</i>
          <span :class="{ active: tritonPhase === 3 }">
            <small>tl.store</small><b>写回 C</b>
          </span>
        </div>
        <p class="triton-note">program 依次出现只是讲解顺序；tile 位置到硬件线程的映射由 Triton 编译器决定。</p>
      </div>
    </div>

    <footer class="demo-footer">
      <div class="footer-copy">
        <span>本步</span>
        <p :aria-live="playing ? 'off' : 'polite'">{{ explanation }}</p>
      </div>
      <span class="step-count"><b>{{ step + 1 }}</b><i>/</i>{{ meta.steps }}</span>
    </footer>
  </div>
</template>

<style scoped>
.ew-demo {
  --ink: #172231;
  --muted: #647287;
  --subtle: #8a98aa;
  --line: #cdd6e0;
  --line-strong: #aebbc9;
  --surface: #ffffff;
  --surface-soft: #f4f7fa;
  --surface-raised: #fafbfd;
  --input-a: #138796;
  --input-a-soft: #dcf2f3;
  --input-b: #b47708;
  --input-b-soft: #fff1ca;
  --output: #c6535d;
  --output-soft: #ffe4e6;
  --focus: #18745b;
  --focus-soft: color-mix(in srgb, var(--focus) 13%, var(--surface));
  width: 100%;
  overflow: hidden;
  scroll-margin-top: 72px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  box-shadow: 0 10px 28px rgb(28 43 61 / 7%);
  color: var(--ink);
  font-size: 14px;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0;
}

.ew-demo[data-scenario='paradigm'] { --focus: #3f6784; }
.ew-demo[data-scenario='dependency'] { --focus: #18745b; }
.ew-demo[data-scenario='memory'] { --focus: #356a91; }
.ew-demo[data-scenario='vector'] { --focus: #0e8797; }
.ew-demo[data-scenario='triton'] { --focus: #5269b8; }

:global(.dark .ew-demo) {
  --ink: #edf2f7;
  --muted: #aab6c5;
  --subtle: #7f8b9b;
  --line: #394655;
  --line-strong: #536173;
  --surface: #161b22;
  --surface-soft: #20262f;
  --surface-raised: #1a2028;
  --input-a: #54c5cc;
  --input-a-soft: #173d42;
  --input-b: #e3b04b;
  --input-b-soft: #493b1d;
  --output: #ef8589;
  --output-soft: #4b292d;
  box-shadow: 0 10px 30px rgb(0 0 0 / 20%);
}

:global(.dark .ew-demo[data-scenario='paradigm']) { --focus: #8bb8d5; }
:global(.dark .ew-demo[data-scenario='dependency']) { --focus: #66c5a3; }
:global(.dark .ew-demo[data-scenario='memory']) { --focus: #78b4dc; }
:global(.dark .ew-demo[data-scenario='vector']) { --focus: #62c8d2; }
:global(.dark .ew-demo[data-scenario='triton']) { --focus: #a7b5ff; }

.demo-header {
  position: relative;
  display: flex;
  min-height: 68px;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 13px 16px 14px;
  border-bottom: 1px solid var(--line);
  background: var(--surface-raised);
}

.demo-header::after {
  position: absolute;
  right: auto;
  bottom: -1px;
  left: 0;
  width: var(--step-progress);
  height: 3px;
  background: var(--focus);
  content: '';
  transition: width 420ms cubic-bezier(0.22, 1, 0.36, 1);
}

.demo-header > div:first-child {
  display: grid;
  min-width: 0;
  gap: 2px;
}

.demo-header strong {
  overflow-wrap: anywhere;
  font-size: 15.5px;
  line-height: 1.4;
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: var(--focus);
  font-size: 10px;
  font-weight: 800;
  line-height: 1.3;
  text-transform: uppercase;
}

.eyebrow::before {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--focus);
  box-shadow: 0 0 0 3px var(--focus-soft);
  content: '';
}

.controls {
  display: flex;
  flex: 0 0 auto;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--surface);
}

.controls button {
  display: grid;
  width: 35px;
  height: 34px;
  place-items: center;
  padding: 0;
  border: 0;
  border-radius: 0;
  background: var(--surface);
  color: var(--ink);
  font: inherit;
  font-size: 16px;
  cursor: pointer;
  transition: background 160ms ease, color 160ms ease;
}

.controls button + button {
  border-left: 1px solid var(--line);
}

.controls .control-play {
  background: var(--focus-soft);
  color: var(--focus);
  font-size: 14px;
  font-weight: 800;
}

.controls button:hover,
.controls button:focus-visible {
  background: var(--focus-soft);
  color: var(--focus);
}

.controls button:focus-visible {
  z-index: 1;
  outline: 2px solid var(--focus);
  outline-offset: -3px;
}

.stage {
  display: grid;
  min-height: 330px;
  grid-template-columns: minmax(0, 1fr);
  place-items: center;
  padding: 24px 18px;
  background: var(--surface);
}

/* ---------- paradigm ---------- */
.paradigm-board {
  --hip-accent: #18745b;
  --hip-soft: color-mix(in srgb, var(--hip-accent) 11%, var(--surface));
  --triton-accent: #5269b8;
  --triton-soft: color-mix(in srgb, var(--triton-accent) 11%, var(--surface));
  display: grid;
  min-width: 0;
  width: min(100%, 900px);
  gap: 12px;
}

:global(.dark .paradigm-board) {
  --hip-accent: #66c5a3;
  --triton-accent: #a7b5ff;
}

.paradigm-shared {
  display: grid;
  grid-template-columns: auto auto minmax(0, 1fr);
  align-items: center;
  gap: 8px 14px;
  padding: 10px 12px;
  border-block: 1px solid var(--line);
  background: var(--surface-raised);
}

.paradigm-shared > span {
  color: var(--focus);
  font-size: 10px;
  font-weight: 800;
  white-space: nowrap;
}

.paradigm-shared > code {
  padding: 4px 7px;
  border-left: 2px solid var(--focus);
  background: var(--surface);
  color: var(--ink);
  font-size: 11px;
  white-space: nowrap;
}

.paradigm-shared > strong {
  min-width: 0;
  font-size: 12px;
  line-height: 1.45;
  text-align: right;
}

.paradigm-output-strip {
  display: grid;
  grid-column: 1 / -1;
  grid-template-columns: repeat(8, minmax(0, 1fr));
  gap: 4px;
}

.paradigm-output-strip i {
  display: grid;
  height: 22px;
  place-items: center;
  border: 1px solid color-mix(in srgb, var(--focus) 35%, var(--line));
  border-radius: 2px;
  background: var(--focus-soft);
  color: var(--focus);
  font-size: 10px;
  font-style: normal;
  font-weight: 750;
}

.paradigm-output-strip i.invalid,
.hip-thread-grid span.invalid,
.triton-program-grid i.invalid {
  border-style: dashed;
  border-color: var(--line-strong);
  background: var(--surface-soft);
  color: var(--subtle);
}

.paradigm-panels {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  border: 1px solid var(--line);
  background: var(--surface);
}

.paradigm-panel {
  --panel-accent: var(--focus);
  --panel-soft: var(--focus-soft);
  position: relative;
  display: grid;
  min-width: 0;
  min-height: 280px;
  grid-template-rows: auto auto auto 1fr auto;
  gap: 10px;
  padding: 14px;
  transition: background 320ms ease, box-shadow 320ms ease, opacity 320ms ease;
}

.paradigm-panel + .paradigm-panel {
  border-left: 1px solid var(--line);
}

.paradigm-panel.paradigm-hip {
  --panel-accent: var(--hip-accent);
  --panel-soft: var(--hip-soft);
}

.paradigm-panel.paradigm-triton {
  --panel-accent: var(--triton-accent);
  --panel-soft: var(--triton-soft);
}

.paradigm-panel.active {
  background: var(--panel-soft);
  box-shadow: inset 0 3px 0 var(--panel-accent);
}

.paradigm-panel > :not(header) {
  transition: opacity 320ms ease;
}

.paradigm-panel.concealed > :not(header) {
  opacity: 0;
}

.paradigm-panel.subdued > :not(header) {
  opacity: 0.28;
}

.paradigm-panel.concealed::after {
  position: absolute;
  top: 58%;
  left: 50%;
  width: calc(100% - 48px);
  color: var(--panel-accent);
  content: '';
  font-size: 10px;
  font-weight: 750;
  line-height: 1.5;
  text-align: center;
  transform: translate(-50%, -50%);
}

.paradigm-hip.concealed::after {
  content: '第 2 步展开：thread → 标量 index';
}

.paradigm-triton.concealed::after {
  content: '第 3 步展开：program → 一块 offsets';
}

.paradigm-panel > header {
  display: grid;
  gap: 2px;
  padding-bottom: 9px;
  border-bottom: 1px solid var(--line);
}

.paradigm-panel > header span {
  color: var(--panel-accent);
  font-size: 9px;
  font-weight: 850;
}

.paradigm-panel > header strong {
  font-size: 13px;
  line-height: 1.35;
}

.paradigm-panel > header small {
  color: var(--muted);
  font-size: 10px;
}

.paradigm-subject {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  color: var(--muted);
  font-size: 10px;
}

.paradigm-subject b {
  color: var(--panel-accent);
  font-size: 12px;
}

.paradigm-formula {
  overflow-x: auto;
  padding: 7px 8px;
  border-left: 2px solid var(--panel-accent);
  background: var(--surface-soft);
  color: var(--ink);
  font-size: 10px;
  white-space: nowrap;
}

.hip-thread-grid {
  display: grid;
  align-content: center;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 5px;
}

.hip-thread-grid span {
  display: grid;
  min-width: 0;
  gap: 2px;
  padding: 7px 3px;
  border: 1px solid color-mix(in srgb, var(--panel-accent) 35%, var(--line));
  border-radius: 2px;
  background: var(--surface);
  text-align: center;
}

.hip-thread-grid small,
.triton-program-grid small {
  color: var(--muted);
  font-size: 9px;
}

.hip-thread-grid b,
.triton-program-grid b {
  color: var(--panel-accent);
  font-size: 10px;
}

.triton-program-grid {
  display: grid;
  align-content: center;
  gap: 6px;
}

.triton-program-grid > span {
  display: grid;
  min-width: 0;
  grid-template-columns: auto auto minmax(0, 1fr);
  align-items: center;
  gap: 8px;
  padding: 6px 7px;
  border: 1px solid color-mix(in srgb, var(--panel-accent) 35%, var(--line));
  border-radius: 2px;
  background: var(--surface);
}

.triton-program-grid em {
  display: grid;
  min-width: 0;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 3px;
  font-style: normal;
}

.triton-program-grid i {
  display: grid;
  height: 24px;
  place-items: center;
  border: 1px solid color-mix(in srgb, var(--panel-accent) 42%, var(--line));
  background: var(--panel-soft);
  color: var(--panel-accent);
  font-size: 9px;
  font-style: normal;
  font-weight: 750;
}

.paradigm-boundary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding-top: 9px;
  border-top: 1px dashed var(--line);
  color: var(--muted);
  font-size: 9px;
}

.paradigm-boundary code {
  color: var(--panel-accent);
  font-size: 10px;
  font-weight: 750;
  white-space: nowrap;
}

.paradigm-translation {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  border: 1px solid var(--line);
  background: var(--surface-raised);
  transition: background 320ms ease, border-color 320ms ease;
}

.paradigm-translation.active {
  border-color: color-mix(in srgb, var(--focus) 55%, var(--line));
  background: var(--focus-soft);
}

.paradigm-translation span {
  display: grid;
  min-width: 0;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 4px;
  padding: 8px;
  text-align: center;
}

.paradigm-translation span + span {
  border-left: 1px solid var(--line);
}

.paradigm-translation small {
  grid-column: 1 / -1;
  color: var(--muted);
  font-size: 9px;
}

.paradigm-translation b {
  overflow-wrap: anywhere;
  font-size: 9px;
  line-height: 1.35;
  transition: opacity 320ms ease;
}

.paradigm-translation i {
  color: var(--focus);
  font-size: 11px;
  font-style: normal;
  transition: opacity 320ms ease;
}

.paradigm-translation:not(.active) b,
.paradigm-translation:not(.active) i {
  opacity: 0;
}

.paradigm-note {
  margin: 0;
  color: var(--muted);
  font-size: 10px;
  line-height: 1.55;
  opacity: 0;
  text-align: center;
  transition: opacity 320ms ease;
}

.paradigm-note.active {
  opacity: 1;
}

.paradigm-note code {
  color: var(--ink);
  font-size: 10px;
}

/* ---------- dependency ---------- */
.dependency-board {
  display: grid;
  min-width: 0;
  width: min(100%, 780px);
  gap: 14px;
}

.dep-legend,
.vector-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 16px;
  color: var(--muted);
  font-size: 11px;
}

.dep-legend span,
.vector-legend span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.chip {
  box-sizing: border-box;
  width: 11px;
  height: 11px;
  border-radius: 2px;
  border: 2px solid var(--line);
}

.chip-thread { border-color: var(--focus); background: var(--focus-soft); }
.chip-a { border-color: var(--input-a); background: var(--input-a-soft); }
.chip-b { border-color: var(--input-b); background: var(--input-b-soft); }
.chip-c { border-color: var(--output); background: var(--output-soft); }

.dep-equation {
  display: grid;
  grid-template-columns: auto minmax(48px, 1fr) auto minmax(48px, 1fr) auto minmax(48px, 1fr);
  align-items: stretch;
  gap: 6px;
  padding: 8px;
  border-block: 1px solid var(--line);
  background: var(--surface-raised);
}

.dep-equation-index {
  display: grid;
  place-items: center;
  padding-inline: 10px;
  border-left: 3px solid var(--focus);
  background: var(--focus-soft);
  color: var(--focus);
  font-size: 11px;
  font-weight: 800;
  white-space: nowrap;
}

.dep-equation-term {
  display: flex;
  min-width: 0;
  align-items: baseline;
  justify-content: space-between;
  gap: 6px;
  padding: 7px 9px;
  border: 1px solid var(--line);
  border-bottom-width: 3px;
  background: var(--surface);
}

.dep-equation-term small {
  color: var(--muted);
  font-size: 9px;
}

.dep-equation-term b {
  font-size: 14px;
}

.dep-equation-term.input-a { border-bottom-color: var(--input-a); }
.dep-equation-term.input-b { border-bottom-color: var(--input-b); }
.dep-equation-term.output { border-bottom-color: var(--output); }

.dep-equation > i {
  display: grid;
  place-items: center;
  color: var(--muted);
  font-size: 13px;
  font-style: normal;
  font-weight: 800;
}

.dep-grid-scroll {
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 3px 2px 7px;
  scrollbar-color: var(--line) transparent;
  scrollbar-width: thin;
}

.dep-grid {
  display: grid;
  min-width: 640px;
  grid-template-columns: 48px repeat(8, minmax(52px, 1fr));
  column-gap: 6px;
  row-gap: 5px;
}

.dep-corner { display: block; }

.dep-corner,
.row-label,
.row-op {
  position: sticky;
  left: 0;
  z-index: 3;
}

.dep-corner,
.row-op {
  background: var(--surface);
}

.dep-thread {
  display: grid;
  box-sizing: border-box;
  height: 29px;
  place-items: center;
  min-width: 0;
  border: 1px dashed var(--line-strong);
  border-radius: 4px;
  color: var(--muted);
  font-size: 10px;
  font-weight: 700;
  transition:
    background 260ms ease,
    border-color 260ms ease,
    box-shadow 260ms ease,
    color 260ms ease;
}

.dep-thread.current {
  border-color: var(--focus);
  background: var(--focus-soft);
  box-shadow: 0 0 0 2px var(--focus-soft);
  color: var(--focus);
}

.row-label {
  display: grid;
  box-sizing: border-box;
  place-items: center;
  border-radius: 4px;
  color: #ffffff;
  font-size: 15px;
  font-weight: 800;
}

.input-a-label { background: var(--input-a); }
.input-b-label { background: var(--input-b); }
.output-label { background: var(--output); }

.row-op {
  display: grid;
  height: 17px;
  place-items: center;
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
}

.flow-arrow {
  display: grid;
  height: 17px;
  place-items: center;
  color: var(--line);
  font-size: 14px;
  line-height: 1;
  transition: color 260ms ease, transform 260ms ease;
}

.flow-arrow.current {
  color: var(--focus);
  font-weight: 800;
  transform: scale(1.18);
}

.data-cell {
  position: relative;
  display: grid;
  box-sizing: border-box;
  min-width: 0;
  height: 48px;
  place-items: center;
  border: 1px solid var(--line);
  border-radius: 4px;
  font-weight: 750;
  transition:
    border-color 280ms ease,
    box-shadow 280ms ease,
    color 280ms ease,
    transform 280ms ease;
}

.data-cell small {
  position: absolute;
  top: 2px;
  left: 5px;
  color: var(--muted);
  font-size: 9px;
  font-weight: 600;
}

.data-cell.input-a { background: var(--input-a-soft); }
.data-cell.input-b { background: var(--input-b-soft); }
.data-cell.output { background: var(--output-soft); }
.data-cell.output.done {
  border-color: color-mix(in srgb, var(--output) 45%, var(--line));
  color: var(--output);
}

.data-cell.current {
  z-index: 1;
  border-color: var(--focus);
  box-shadow: 0 0 0 2px var(--focus-soft), inset 0 -3px 0 var(--focus);
  color: var(--ink);
  transform: translateY(-1px);
}

.dep-note {
  margin: 0 2px;
  padding-top: 10px;
  border-top: 1px dashed var(--line);
  color: var(--muted);
  font-size: 11px;
  line-height: 1.5;
  text-align: center;
}

/* ---------- memory ---------- */
.memory-comparison {
  display: grid;
  width: min(100%, 920px);
  grid-template-columns: repeat(2, minmax(0, 1fr));
  border-block: 1px solid var(--line);
  background: var(--surface-raised);
}

.memory-round {
  display: grid;
  grid-column: 1 / -1;
  grid-template-columns: auto minmax(0, 1fr);
  column-gap: 12px;
  padding: 9px 16px;
  border-bottom: 1px solid var(--line);
  background: var(--surface-soft);
}

.memory-round > span {
  display: grid;
  grid-row: 1 / 3;
  place-items: center;
  padding-inline: 10px;
  border-left: 3px solid var(--focus);
  background: var(--focus-soft);
  color: var(--focus);
  font-size: 10px;
  font-weight: 800;
  white-space: nowrap;
}

.memory-round strong {
  font-size: 11px;
  line-height: 1.45;
}

.memory-round small {
  color: var(--muted);
  font-size: 9px;
  line-height: 1.4;
}

.memory-panel {
  --panel-accent: var(--focus);
  --panel-soft: var(--focus-soft);
  min-width: 0;
  padding: 16px;
}

.memory-panel.contiguous {
  --panel-accent: #18775d;
  --panel-soft: color-mix(in srgb, var(--panel-accent) 12%, var(--surface));
}

.memory-panel.strided {
  --panel-accent: #b45f36;
  --panel-soft: color-mix(in srgb, var(--panel-accent) 12%, var(--surface));
}

:global(.dark .memory-panel.contiguous) { --panel-accent: #67c7a5; }
:global(.dark .memory-panel.strided) { --panel-accent: #ef9a70; }

.memory-panel + .memory-panel {
  border-left: 1px solid var(--line);
}

.panel-heading {
  display: flex;
  min-height: 43px;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 13px;
  padding-bottom: 11px;
  border-bottom: 1px solid var(--line);
}

.panel-heading > div {
  display: grid;
  gap: 2px;
}

.panel-kind {
  color: var(--panel-accent);
  font-size: 10px;
  font-weight: 800;
  line-height: 1.2;
}

.panel-heading strong {
  font-size: 13px;
  line-height: 1.3;
}

.panel-heading code {
  padding: 5px 7px;
  border-left: 2px solid var(--panel-accent);
  background: var(--surface-soft);
  color: var(--muted);
  font-size: 10px;
  white-space: nowrap;
}

.address-sequence {
  display: grid;
  min-width: 0;
  gap: 4px;
  margin-bottom: 10px;
  padding-left: 7px;
  border-left: 2px solid var(--panel-accent);
}

.address-sequence span {
  color: var(--muted);
  font-size: 9px;
}

.address-sequence code {
  overflow-x: auto;
  color: var(--panel-accent);
  font-size: 10px;
  font-weight: 700;
  white-space: nowrap;
}

.memory-stack {
  display: grid;
  gap: 5px;
}

.memory-tx-row {
  display: grid;
  box-sizing: border-box;
  grid-template-columns: 38px repeat(8, minmax(22px, 1fr));
  gap: 3px;
  padding: 3px;
  border: 1px solid transparent;
  background: var(--surface-soft);
  transition: background 260ms ease, border-color 260ms ease;
}

.memory-tx-row.touched {
  border-color: color-mix(in srgb, var(--panel-accent) 45%, var(--line));
  background: var(--panel-soft);
}

.tx-row-label {
  display: grid;
  align-content: center;
  justify-items: center;
  border-right: 1px solid var(--line);
  color: var(--muted);
  line-height: 1.1;
}

.tx-row-label b {
  font-size: 10px;
}

.tx-row-label small {
  margin-top: 3px;
  font-size: 8px;
}

.memory-tx-row.touched .tx-row-label {
  color: var(--panel-accent);
}

.memory-cell {
  position: relative;
  display: grid;
  box-sizing: border-box;
  height: 31px;
  min-width: 0;
  place-items: center;
  border: 1px solid var(--line);
  border-radius: 2px;
  background: var(--surface);
  color: var(--muted);
  transition:
    background 260ms ease,
    border-color 260ms ease,
    box-shadow 260ms ease,
    color 260ms ease;
}

.memory-cell small {
  position: absolute;
  top: 1px;
  left: 2px;
  font-size: 8px;
  font-weight: 700;
}

.memory-cell b {
  font-size: 10px;
}

.memory-cell.active {
  border-color: var(--panel-accent);
  background: var(--panel-soft);
  color: var(--panel-accent);
  box-shadow: inset 0 -3px 0 var(--panel-accent);
}

.tx-counter {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--line);
  color: var(--muted);
}

.tx-counter span {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}

.tx-counter span + span {
  padding-left: 12px;
  border-left: 1px solid var(--line);
}

.tx-counter small {
  font-size: 10px;
}

.tx-counter b {
  color: var(--panel-accent);
  font-size: 18px;
  line-height: 1;
}

/* ---------- vector ---------- */
.vectorization-board {
  min-width: 0;
  width: min(100%, 880px);
}

.vector-legend {
  margin-bottom: 16px;
}

.vector-legend i {
  box-sizing: border-box;
  width: 11px;
  height: 11px;
  border-radius: 2px;
  border: 2px solid var(--line);
}

.vector-legend .legend-scalar { border-color: var(--muted); }
.vector-legend .legend-vector { border-color: var(--input-a); }
.vector-legend .legend-tail {
  border-color: var(--focus);
  border-style: dashed;
}

.vector-summary {
  display: flex;
  min-width: 0;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 9px 10px;
  border-block: 1px solid var(--line);
  background: var(--surface-raised);
}

.vector-summary > span {
  display: grid;
  min-width: 0;
  gap: 2px;
}

.vector-summary small {
  color: var(--focus);
  font-size: 9px;
  font-weight: 800;
}

.vector-summary strong {
  overflow-wrap: anywhere;
  font-size: 11px;
  line-height: 1.4;
}

.vector-summary code {
  flex: 0 0 auto;
  padding-left: 10px;
  border-left: 2px solid var(--focus);
  color: var(--focus);
  font-size: 10px;
  white-space: nowrap;
}

.vector-track-scroll {
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 24px 4px 23px;
  scrollbar-color: var(--line) transparent;
  scrollbar-width: thin;
}

.vector-track {
  position: relative;
  display: grid;
  min-width: 638px;
  grid-template-columns: repeat(18, minmax(28px, 1fr));
  gap: 4px;
  isolation: isolate;
}

.vector-element {
  position: relative;
  display: grid;
  z-index: 1;
  box-sizing: border-box;
  grid-row: 1;
  height: 64px;
  min-width: 0;
  place-items: center;
  border: 1px solid var(--line);
  background: var(--surface-soft);
  transition:
    background 440ms cubic-bezier(0.22, 1, 0.36, 1),
    border-color 440ms cubic-bezier(0.22, 1, 0.36, 1),
    box-shadow 440ms cubic-bezier(0.22, 1, 0.36, 1),
    color 440ms cubic-bezier(0.22, 1, 0.36, 1),
    opacity 440ms cubic-bezier(0.22, 1, 0.36, 1);
}

.vector-element small {
  position: absolute;
  top: 3px;
  color: var(--muted);
  font-size: 9px;
}

.vector-element b {
  font-size: 11px;
}

.vector-element em {
  color: var(--muted);
  font-size: 9px;
  font-style: normal;
  transition: opacity 320ms ease;
}

.vector-element.is-grouped {
  border-color: color-mix(in srgb, var(--input-a) 62%, var(--line));
  background: var(--input-a-soft);
  color: var(--input-a);
}

.vector-element.is-grouped em {
  opacity: 0;
}

.vector-element.is-current-group {
  border-color: var(--input-a);
  box-shadow: inset 0 -3px 0 var(--input-a);
}

.vector-element.is-tail-pending {
  border-style: dashed;
  opacity: 0.34;
}

.vector-element.is-tail-active {
  border: 2px dashed var(--focus);
  background: var(--focus-soft);
  color: var(--focus);
  box-shadow: inset 0 -3px 0 var(--focus);
}

.vector-group-frame {
  position: relative;
  z-index: 2;
  box-sizing: border-box;
  grid-row: 1;
  margin: -5px -2px;
  border: 2px solid transparent;
  border-radius: 4px;
  opacity: 0;
  pointer-events: none;
  transform: scaleX(0.72);
  transform-origin: left center;
  transition:
    border-color 480ms cubic-bezier(0.22, 1, 0.36, 1),
    box-shadow 480ms cubic-bezier(0.22, 1, 0.36, 1),
    opacity 360ms ease,
    transform 480ms cubic-bezier(0.22, 1, 0.36, 1);
}

.vector-group-frame b,
.vector-group-frame em {
  position: absolute;
  left: 50%;
  width: max-content;
  padding-inline: 4px;
  background: var(--surface);
  font-size: 9px;
  font-style: normal;
  line-height: 1.4;
  transform: translateX(-50%);
}

.vector-group-frame b {
  top: -19px;
  color: var(--input-a);
  font-weight: 750;
}

.vector-group-frame em {
  bottom: -19px;
  color: var(--muted);
}

.vector-group-frame.is-active {
  border-color: var(--input-a);
  opacity: 0.72;
  transform: scaleX(1);
}

.vector-group-frame.is-current {
  box-shadow: 0 0 0 3px var(--input-a-soft);
  opacity: 1;
}

.vector-progress {
  position: relative;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin: 2px 10px 0;
}

.vector-progress::before {
  position: absolute;
  top: 5px;
  right: 16.667%;
  left: 16.667%;
  border-top: 1px solid var(--line);
  content: '';
}

.vector-progress span {
  position: relative;
  display: grid;
  justify-items: center;
  gap: 5px;
  color: var(--muted);
  font-size: 10px;
  transition: color 320ms ease;
}

.vector-progress span::before {
  z-index: 1;
  width: 9px;
  height: 9px;
  border: 1px solid var(--line);
  border-radius: 50%;
  background: var(--surface);
  content: '';
  transition:
    background 320ms ease,
    border-color 320ms ease,
    box-shadow 320ms ease,
    transform 320ms ease;
}

.vector-progress span.is-done::before {
  border-color: var(--input-a);
  background: var(--input-a);
}

.vector-progress span.is-current {
  color: var(--ink);
  font-weight: 750;
}

.vector-progress span.is-current::before {
  border-color: var(--focus);
  background: var(--focus);
  box-shadow: 0 0 0 3px var(--focus-soft);
  transform: scale(1.18);
}

.alignment-line {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 8px;
  min-height: 18px;
  margin-top: 10px;
  color: var(--muted);
  font-size: 11px;
}

.alignment-size { text-align: right; }

.vector-note,
.triton-note {
  margin: 10px 0 0;
  padding-top: 9px;
  border-top: 1px dashed var(--line);
  color: var(--muted);
  font-size: 10px;
  line-height: 1.5;
  text-align: center;
}

/* ---------- triton ---------- */
.triton-board {
  display: grid;
  min-width: 0;
  width: min(100%, 820px);
  gap: 14px;
}

.program-strip {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  border: 1px solid var(--line);
  background: var(--surface-raised);
}

.program-strip span {
  display: grid;
  min-width: 0;
  gap: 2px;
  padding: 9px 12px 10px;
  color: var(--muted);
  text-align: center;
  transition: background 280ms ease, box-shadow 280ms ease, color 280ms ease;
}

.program-strip small {
  font-size: 9px;
  font-weight: 750;
}

.program-strip b {
  font-size: 11px;
}

.program-strip span + span { border-left: 1px solid var(--line); }
.program-strip span.active {
  background: var(--focus-soft);
  box-shadow: inset 0 -3px 0 var(--focus);
  color: var(--focus);
}

.program-strip span.done {
  color: var(--focus);
}

.triton-formula {
  display: grid;
  min-width: 0;
  grid-template-columns: auto minmax(0, 1fr);
  align-items: stretch;
  overflow: hidden;
  border: 1px solid var(--line);
  background: var(--surface-soft);
  color: var(--muted);
  white-space: nowrap;
}

.triton-formula span {
  display: grid;
  place-items: center;
  padding: 8px 10px;
  background: var(--focus-soft);
  color: var(--focus);
  font-size: 9px;
  font-weight: 800;
}

.triton-formula > div {
  display: grid;
  min-width: 0;
}

.triton-formula code {
  overflow-x: auto;
  padding: 7px 10px 3px;
  color: var(--muted);
  font-size: 10px;
  text-align: center;
}

.triton-formula strong {
  padding: 2px 10px 7px;
  color: var(--focus);
  font-size: 11px;
  text-align: center;
}

.offset-heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  color: var(--muted);
  font-size: 10px;
}

.offset-heading strong {
  color: var(--focus);
  font-size: 11px;
}

.offset-grid {
  display: grid;
  grid-template-columns: repeat(8, minmax(48px, 1fr));
  gap: 5px;
}

.offset-cell {
  position: relative;
  display: grid;
  box-sizing: border-box;
  min-width: 0;
  height: 72px;
  place-items: center;
  border: 1px solid var(--line);
  border-radius: 3px;
  background: var(--surface-soft);
  transition:
    background 280ms ease,
    border-color 280ms ease,
    box-shadow 280ms ease,
    color 280ms ease,
    opacity 280ms ease;
}

.offset-cell small,
.offset-cell em {
  color: var(--muted);
  font-size: 10px;
  font-style: normal;
}

.offset-cell b { font-size: 17px; }

.offset-cell.phase-a {
  border-color: var(--input-a);
  background: var(--input-a-soft);
  box-shadow: inset 0 3px 0 var(--input-a);
}

.offset-cell.phase-b {
  border-color: var(--input-b);
  background: var(--input-b-soft);
  box-shadow: inset 0 3px 0 var(--input-b);
}

.offset-cell.phase-add {
  border-color: var(--focus);
  background: var(--focus-soft);
  box-shadow: inset 0 3px 0 var(--focus);
}

.offset-cell.phase-c {
  border-color: var(--output);
  background: var(--output-soft);
  box-shadow: inset 0 3px 0 var(--output);
}

.offset-cell.masked {
  border-style: dashed;
  border-color: var(--line-strong);
  color: var(--subtle);
  opacity: 0.7;
  background: var(--surface-soft);
  box-shadow: none;
}

.pipeline {
  display: grid;
  grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr;
  align-items: center;
  gap: 8px;
}

.pipeline span {
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: center;
  gap: 8px;
  padding: 9px 10px;
  border: 1px solid var(--line);
  border-radius: 3px;
  background: var(--surface-raised);
  color: var(--muted);
  transition:
    background 280ms ease,
    border-color 280ms ease,
    box-shadow 280ms ease,
    color 280ms ease;
}

.pipeline span.active {
  border-color: var(--focus);
  background: var(--focus-soft);
  box-shadow: inset 0 -3px 0 var(--focus);
  color: var(--focus);
}

.pipeline span.done {
  border-color: color-mix(in srgb, var(--focus) 45%, var(--line));
  color: var(--focus);
}

.pipeline small {
  font-size: 9px;
  font-weight: 800;
}

.pipeline b {
  font-size: 11px;
  text-align: center;
}

.pipeline i {
  color: var(--muted);
  font-style: normal;
}

/* ---------- footer ---------- */
.demo-footer {
  display: flex;
  min-height: 74px;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  padding: 12px 16px;
  border-top: 1px solid var(--line);
  background: var(--surface-soft);
}

.footer-copy {
  display: grid;
  min-width: 0;
  grid-template-columns: 30px minmax(0, 1fr);
  align-items: start;
  gap: 9px;
}

.footer-copy > span {
  padding-top: 3px;
  color: var(--focus);
  font-size: 9px;
  font-weight: 800;
}

.footer-copy p {
  margin: 0;
  color: var(--ink);
  font-size: 12px;
  line-height: 1.6;
}

.step-count {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: baseline;
  gap: 4px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  padding-top: 2px;
}

.step-count b {
  color: var(--focus);
  font-size: 19px;
  line-height: 1;
}

.step-count i {
  color: var(--line-strong);
  font-style: normal;
}

@media (max-width: 720px) {
  .stage {
    min-height: 310px;
    padding: 20px 10px;
  }

  .dep-grid-scroll { margin-inline: -2px; }

  .paradigm-panels { grid-template-columns: 1fr; }
  .paradigm-panel { min-height: 250px; }
  .paradigm-panel + .paradigm-panel {
    border-top: 1px solid var(--line);
    border-left: 0;
  }

  .paradigm-translation { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .paradigm-translation span:nth-child(3) {
    border-top: 1px solid var(--line);
    border-left: 0;
  }
  .paradigm-translation span:nth-child(4) { border-top: 1px solid var(--line); }

  .memory-comparison { grid-template-columns: 1fr; }
  .memory-panel + .memory-panel {
    border-top: 1px solid var(--line);
    border-left: 0;
  }

  .vector-track-scroll { margin-inline: -2px; }

  .alignment-line {
    grid-template-columns: 1fr auto;
    row-gap: 4px;
  }

  .offset-grid { grid-template-columns: repeat(4, minmax(48px, 1fr)); }

  .demo-footer {
    gap: 10px;
    padding-inline: 12px;
  }
}

@media (max-width: 440px) {
  .demo-header { align-items: flex-start; }
  .controls button { width: 40px; height: 40px; }
  .demo-footer { min-height: 84px; }

  .dep-grid { min-width: 620px; }

  .paradigm-shared { grid-template-columns: auto minmax(0, 1fr); }
  .paradigm-shared > strong {
    grid-column: 1 / -1;
    text-align: left;
  }

  .paradigm-panel { padding: 12px 9px; }
  .paradigm-formula {
    overflow-wrap: anywhere;
    white-space: normal;
  }

  .paradigm-boundary {
    align-items: flex-start;
    flex-direction: column;
  }

  .dep-equation {
    grid-template-columns: auto minmax(44px, 1fr) auto minmax(44px, 1fr) auto minmax(44px, 1fr);
    gap: 4px;
    padding-inline: 5px;
  }

  .dep-equation-index { padding-inline: 6px; }
  .dep-equation-term { padding-inline: 6px; }

  .memory-panel { padding: 12px 8px; }

  .memory-round {
    column-gap: 8px;
    padding-inline: 8px;
  }

  .memory-tx-row {
    grid-template-columns: 34px repeat(8, minmax(20px, 1fr));
    gap: 2px;
    padding: 2px;
  }

  .triton-formula { grid-template-columns: 1fr; }
  .triton-formula code {
    overflow-wrap: anywhere;
    white-space: normal;
  }

  .triton-formula strong {
    overflow-wrap: anywhere;
    white-space: normal;
  }

  .vector-summary {
    align-items: stretch;
    flex-direction: column;
    gap: 7px;
  }

  .vector-summary code {
    padding-top: 5px;
    padding-left: 0;
    border-top: 1px solid var(--line);
    border-left: 0;
    white-space: normal;
  }

  .offset-heading {
    align-items: flex-start;
    flex-direction: column;
    gap: 3px;
  }

  .pipeline { gap: 4px; }
  .pipeline span {
    grid-template-columns: 1fr;
    gap: 2px;
    padding: 7px 3px;
  }

  .pipeline small,
  .pipeline b { text-align: center; }

}

@media (prefers-reduced-motion: reduce) {
  .ew-demo * { transition: none !important; }
}
</style>
