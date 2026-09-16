<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import {
  Activity,
  AlertTriangle,
  Ban,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Bell,
  BrainCircuit,
  Check,
  CircleDollarSign,
  Clock3,
  Database,
  FileText,
  Gauge,
  LayoutDashboard,
  LineChart,
  Pause,
  Pencil,
  Play,
  RefreshCw,
  Settings2,
  ShieldAlert,
  SlidersHorizontal,
  WalletCards,
  X,
  Zap,
} from 'lucide-vue-next'

type ViewName = 'overview' | 'signals' | 'orders' | 'reports' | 'diagnostics'
type RunMode = 'paper_auto' | 'paper_manual' | 'manual_record'

interface Status {
  runner_status: string
  mode: RunMode
  model_version: string
  model_name?: string
  model_metrics?: Record<string, number>
  model_artifact_path?: string | null
  factor_coverage?: number
  data: {
    mode: string
    provider: string
    last_update: string | null
    symbol_count: number
    latency_seconds: number | null
    message: string
  }
  last_heartbeat: string | null
  clock: string
  last_error: string | null
  notifications_enabled: boolean
  paper_only: boolean
}

interface Account {
  initial_capital: number
  cash: number
  financing_debt: number
  financing_interest: number
  short_borrow_value: number
  long_market_value: number
  short_market_value: number
  total_equity: number
  collateral_value: number
  maintenance_ratio: number | null
  available_margin: number | null
  gross_exposure: number
  net_exposure: number
  as_of: string
}

interface Benchmark {
  symbol: string
  name: string
  price: number | null
  change_pct: number | null
  source: string
  as_of: string | null
}

interface MarketOverview {
  as_of: string | null
  status: string
  provider: string
  message: string
  primary_benchmark: string
  primary_benchmark_name: string
  benchmark_return: number | null
  strategy_day_pnl: number | null
  strategy_day_return: number | null
  excess_return: number | null
  benchmarks: Benchmark[]
}

interface Position {
  symbol: string
  name: string
  board: string
  sector: string
  last_price: number
  long_qty: number
  long_available_qty: number
  long_pending_qty: number
  long_avg_cost: number
  short_qty: number
  short_avg_cost: number
  market_value: number
}

interface Signal {
  signal_id: string
  symbol: string
  name: string
  board: string
  sector: string
  score: number
  action: string
  target_weight: number
  current_weight: number
  reason: string
  model_version: string
}

interface Order {
  order_id: string
  signal_id: string | null
  symbol: string
  name: string
  board: string
  side: string
  quantity: number
  filled_quantity: number
  average_fill_price: number | null
  limit_price: number | null
  status: string
  reason: string
  created_at: string
  eligible_at: string
  reject_reason: string | null
  is_margin_order: boolean
}

interface RiskEvent {
  event_id: string
  severity: string
  code: string
  message: string
  created_at: string
  acknowledged: boolean
}

interface Report {
  report_date: string
  status: string
  model_version: string
  model_name?: string
  data_status?: string
  factor_coverage?: number
  model_metrics?: Record<string, number>
  research_metrics?: ResearchMetrics
  summary: string
  turnover: number
  fills_count: number
  generated_at: string
}

interface EquityHistoryPoint {
  as_of: string
  equity: number
}

interface ResearchMetrics {
  status: string
  source?: string
  data_start?: string | null
  data_end?: string | null
  fold_count?: number | null
  last_model_version?: string | null
  test_ic_mean?: number | null
  test_ic_positive_ratio?: number | null
  test_top_bottom_spread_mean?: number | null
  factor_coverage_mean?: number | null
  total_return?: number | null
  annualized_return?: number | null
  sharpe?: number | null
  max_drawdown?: number | null
  turnover?: number | null
  trade_count?: number | null
  message?: string
}

interface Dashboard {
  status: Status
  account: Account
  market: MarketOverview
  positions: Position[]
  signals: Signal[]
  orders: Order[]
  fills: unknown[]
  risk_events: RiskEvent[]
  manual_records: unknown[]
  reports: Report[]
  equity_history?: EquityHistoryPoint[]
}

const apiBase = '/api/v1'
const activeView = ref<ViewName>('overview')
const dashboard = ref<Dashboard | null>(null)
const loading = ref(true)
const apiOnline = ref(true)
const actionMessage = ref('')
const actionError = ref('')
const refreshing = ref(false)
const editingOrder = ref<Order | null>(null)
const editQuantity = ref('')
const editQuantityError = ref('')
const editQuantitySubmitting = ref(false)
const editQuantityInput = ref<HTMLInputElement | null>(null)
let pollingTimer: number | undefined

const navItems: { id: ViewName; label: string; hint: string; icon: typeof LayoutDashboard }[] = [
  { id: 'overview', label: '总览', hint: '运行状态和组合', icon: LayoutDashboard },
  { id: 'signals', label: '信号中心', hint: '模型候选和解释', icon: BrainCircuit },
  { id: 'orders', label: '订单队列', hint: '确认和模拟成交', icon: SlidersHorizontal },
  { id: 'reports', label: '收盘报告', hint: '每日研究结果', icon: FileText },
  { id: 'diagnostics', label: '诊断', hint: '数据和服务健康', icon: Settings2 },
]

const account = computed<Account>(() => dashboard.value?.account ?? {
  initial_capital: 1_000_000,
  cash: 0,
  financing_debt: 0,
  financing_interest: 0,
  short_borrow_value: 0,
  long_market_value: 0,
  short_market_value: 0,
  total_equity: 0,
  collateral_value: 0,
  maintenance_ratio: null,
  available_margin: null,
  gross_exposure: 0,
  net_exposure: 0,
  as_of: '',
})
const market = computed<MarketOverview>(() => dashboard.value?.market ?? {
  as_of: null,
  status: 'UNAVAILABLE',
  provider: '--',
  message: '尚未连接后台 Runner',
  primary_benchmark: '000300',
  primary_benchmark_name: '沪深300',
  benchmark_return: null,
  strategy_day_pnl: null,
  strategy_day_return: null,
  excess_return: null,
  benchmarks: [
    { symbol: '000001', name: '上证指数', price: null, change_pct: null, source: '', as_of: null },
    { symbol: '399001', name: '深证成指', price: null, change_pct: null, source: '', as_of: null },
    { symbol: '000300', name: '沪深300', price: null, change_pct: null, source: '', as_of: null },
  ],
})
const status = computed<Status>(() => dashboard.value?.status ?? {
  runner_status: 'OFFLINE',
  mode: 'paper_auto',
  model_version: '--',
  model_name: 'factor_baseline',
  model_metrics: {},
  model_artifact_path: null,
  factor_coverage: 0,
  data: { mode: 'UNAVAILABLE', provider: '--', last_update: null, symbol_count: 0, latency_seconds: null, message: 'API 尚未连接' },
  last_heartbeat: null,
  clock: '',
  last_error: null,
  notifications_enabled: false,
  paper_only: true,
})
const signals = computed(() => dashboard.value?.signals ?? [])
const orders = computed(() => dashboard.value?.orders ?? [])
const positions = computed(() => dashboard.value?.positions ?? [])
const risks = computed(() => dashboard.value?.risk_events ?? [])
const reports = computed(() => dashboard.value?.reports ?? [])
const pendingOrders = computed(() => orders.value.filter((item) => ['DRAFT', 'CONFIRMED', 'SUBMITTED', 'PARTIAL_FILLED'].includes(item.status)))
const latestRisks = computed(() => risks.value.slice(0, 4))
const equityChange = computed(() => account.value.total_equity - account.value.initial_capital)
const equityChangePct = computed(() => account.value.initial_capital ? equityChange.value / account.value.initial_capital : 0)
const equitySparklinePoints = computed(() => {
  if (!dashboard.value) return '0,18 120,18'

  const history = (dashboard.value.equity_history ?? [])
    .map((point) => Number(point.equity))
    .filter((value) => Number.isFinite(value))
  const values = history.length >= 2
    ? history
    : [account.value.initial_capital, account.value.total_equity]
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  const padding = Math.max((maximum - minimum) * 0.2, Math.max(Math.abs(maximum), Math.abs(minimum), 1) * 0.0005)
  const low = minimum - padding
  const span = Math.max(maximum + padding - low, 1)

  return values.map((value, index) => {
    const x = values.length === 1 ? 60 : (index / (values.length - 1)) * 120
    const y = 32 - ((value - low) / span) * 26
    return `${x.toFixed(1)},${Math.min(32, Math.max(4, y)).toFixed(1)}`
  }).join(' ')
})
const equitySparklineLabel = computed(() =>
  (dashboard.value?.equity_history?.length ?? 0) >= 2
    ? '净值历史趋势'
    : '净值累计方向示意图，历史数据不足'
)
const primaryBenchmark = computed(() => market.value.benchmarks.find((item) => item.symbol === market.value.primary_benchmark) ?? market.value.benchmarks[0])

function money(value: number | null | undefined): string {
  if (value === null || value === undefined) return '--'
  return new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)
}

function pct(value: number | null | undefined): string {
  if (value === null || value === undefined) return '--'
  return `${(value * 100).toFixed(2)}%`
}

function decimal(value: number | null | undefined): string {
  if (value === null || value === undefined) return '--'
  return value.toFixed(2)
}

function signedMoney(value: number | null | undefined): string {
  if (value === null || value === undefined) return '--'
  return `${value >= 0 ? '+' : '-'}¥${money(Math.abs(value))}`
}

function movementClass(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'neutral'
  return value >= 0 ? 'up' : 'down'
}

function researchMetricClass(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'neutral'
  return value > 0 ? 'positive' : value < 0 ? 'negative' : 'neutral'
}

function ratio(value: number | null | undefined): string {
  if (value === null || value === undefined) return '--'
  return `${value.toFixed(2)}x`
}

function time(value: string | null | undefined): string {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function statusLabel(value: string): string {
  const labels: Record<string, string> = {
    RUNNING: '运行中', STOPPED: '已停止', DEGRADED: '降级', ERROR: '异常', OFFLINE: '离线',
    REPLAY: '本地回放', LIVE: '实时', STALE: '数据陈旧', UNAVAILABLE: '不可用',
    DRAFT: '待确认', CONFIRMED: '已确认', SUBMITTED: '撮合中', PARTIAL_FILLED: '部分成交', FILLED: '已成交',
    REJECTED: '已拒绝', CANCELLED: '已撤销', EXPIRED: '已过期',
  }
  return labels[value] ?? value
}

function boardLabel(board: string): string {
  return ({ MAIN: '主板', CHINEXT_300: '创业板', STAR_688: '科创板' } as Record<string, string>)[board] ?? board
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, { headers: { 'Content-Type': 'application/json' }, ...options })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `HTTP ${response.status}`)
  }
  return response.json() as Promise<T>
}

async function loadDashboard(showSpinner = false): Promise<void> {
  if (showSpinner) refreshing.value = true
  try {
    dashboard.value = await request<Dashboard>('/dashboard')
    apiOnline.value = true
  } catch (error) {
    apiOnline.value = false
    actionError.value = '后台 API 尚未连接，请先启动 quant-paper serve。'
  } finally {
    loading.value = false
    refreshing.value = false
  }
}

async function runAction(action: () => Promise<unknown>, success: string): Promise<void> {
  actionError.value = ''
  actionMessage.value = ''
  try {
    await action()
    actionMessage.value = success
    await loadDashboard()
    window.setTimeout(() => { actionMessage.value = '' }, 3200)
  } catch (error) {
    actionError.value = error instanceof Error ? error.message : '操作失败'
  }
}

function startRunner(): Promise<void> {
  return runAction(() => request('/runner/start', { method: 'POST' }), 'Runner 已启动')
}

function stopRunner(): Promise<void> {
  return runAction(() => request('/runner/stop', { method: 'POST' }), 'Runner 已停止')
}

function tickRunner(): Promise<void> {
  return runAction(() => request('/runner/tick', { method: 'POST' }), '已完成一次行情回放')
}

function changeMode(event: Event): Promise<void> {
  const value = (event.target as HTMLSelectElement).value as RunMode
  return runAction(() => request('/runner/mode', { method: 'POST', body: JSON.stringify({ mode: value }) }), `已切换到 ${statusLabel(value)}`)
}

function confirmOrder(orderId: string): Promise<void> {
  return runAction(() => request(`/orders/${orderId}/confirm`, { method: 'POST' }), '订单已确认，等待模拟成交')
}

function rejectOrder(orderId: string): Promise<void> {
  return runAction(() => request(`/orders/${orderId}/reject`, { method: 'POST' }), '订单已拒绝')
}

function cancelOrder(orderId: string): Promise<void> {
  return runAction(() => request(`/orders/${orderId}/cancel`, { method: 'POST' }), '订单已撤销')
}

function openModifyOrder(order: Order): void {
  editingOrder.value = order
  editQuantity.value = String(order.quantity)
  editQuantityError.value = ''
}

function closeModifyOrder(): void {
  if (editQuantitySubmitting.value) return
  editingOrder.value = null
  editQuantityError.value = ''
}

async function submitModifyOrder(): Promise<void> {
  if (!editingOrder.value) return
  const raw = String(editQuantity.value).trim()
  const quantity = Number.parseInt(raw, 10)
  if (!/^\d+$/.test(raw) || !Number.isSafeInteger(quantity) || quantity <= 0) {
    editQuantityError.value = '委托数量必须是正整数'
    await nextTick()
    editQuantityInput.value?.focus()
    return
  }
  editQuantityError.value = ''
  editQuantitySubmitting.value = true
  await runAction(() => request(`/orders/${editingOrder.value!.order_id}/modify`, {
    method: 'POST',
    body: JSON.stringify({ quantity }),
  }), '订单数量已修改')
  editQuantitySubmitting.value = false
  if (!actionError.value) closeModifyOrder()
}

watch(editingOrder, async (order) => {
  if (!order) return
  await nextTick()
  editQuantityInput.value?.focus()
  editQuantityInput.value?.select()
})

function rebalance(): Promise<void> {
  return runAction(async () => {
    const result = await request<{ blocked_reason?: string }>('/rebalance', { method: 'POST' })
    if (result.blocked_reason) throw new Error(result.blocked_reason)
    return result
  }, '已完成一次因子重平衡')
}

function trainModel(): Promise<void> {
  return runAction(() => request('/model/train', { method: 'POST' }), '本地 LightGBM 训练完成')
}

function generateReport(): Promise<void> {
  return runAction(() => request('/reports/run', { method: 'POST' }), '收盘报告已生成')
}

onMounted(() => {
  loadDashboard()
  pollingTimer = window.setInterval(() => loadDashboard(), 3000)
})

onUnmounted(() => {
  if (pollingTimer) window.clearInterval(pollingTimer)
})
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar" aria-label="主导航">
      <div class="brand-block">
        <div class="brand-mark" aria-hidden="true"><Activity :size="20" /></div>
        <div>
          <p class="eyebrow">QUANTPAPER</p>
          <h1>量化纸盘</h1>
        </div>
      </div>
      <div class="paper-badge"><span class="pulse-dot"></span> PAPER ONLY</div>
      <nav class="nav-list">
        <button
          v-for="item in navItems"
          :key="item.id"
          class="nav-item"
          :class="{ active: activeView === item.id }"
          type="button"
          :aria-current="activeView === item.id ? 'page' : undefined"
          @click="activeView = item.id"
        >
          <component :is="item.icon" :size="18" stroke-width="1.8" aria-hidden="true" />
          <span><strong>{{ item.label }}</strong><small>{{ item.hint }}</small></span>
        </button>
      </nav>
      <div class="sidebar-footer">
        <div class="account-mini"><WalletCards :size="16" aria-hidden="true" /><span>安全纸盘账户</span><strong>¥1M</strong></div>
        <p>数据与交易均为本地模拟</p>
      </div>
    </aside>

    <main class="main-content">
      <header class="topbar">
        <div class="page-heading">
          <p class="eyebrow">LOCAL OPERATIONS CONSOLE</p>
          <h2>{{ navItems.find((item) => item.id === activeView)?.label }}</h2>
        </div>
        <div class="topbar-actions">
          <div class="connection-status" :class="{ offline: !apiOnline }">
            <span class="status-dot"></span>
            <span>{{ apiOnline ? 'API 已连接' : 'API 离线' }}</span>
          </div>
          <label class="mode-select">
            <span class="sr-only">运行模式</span>
            <select :value="status.mode" aria-label="运行模式" @change="changeMode">
              <option value="paper_auto">自动纸盘</option>
              <option value="paper_manual">人工确认</option>
              <option value="manual_record">手工记录</option>
            </select>
          </label>
          <button class="icon-button" type="button" aria-label="刷新数据" :disabled="refreshing" @click="loadDashboard(true)">
            <RefreshCw :size="17" :class="{ spinning: refreshing }" aria-hidden="true" />
          </button>
        </div>
      </header>

      <div v-if="actionMessage || actionError" class="toast-stack" aria-live="polite">
        <div v-if="actionMessage" class="toast success"><Check :size="16" aria-hidden="true" />{{ actionMessage }}</div>
        <div v-if="actionError" class="toast error"><AlertTriangle :size="16" aria-hidden="true" />{{ actionError }}</div>
      </div>

      <div v-if="editingOrder" class="modal-backdrop" @click.self="closeModifyOrder">
        <section
          class="modal"
          role="dialog"
          aria-modal="true"
          aria-labelledby="modify-order-title"
          tabindex="-1"
          @keydown.esc="closeModifyOrder"
        >
          <div class="modal-header">
            <div>
              <p class="eyebrow">PAPER BROKER</p>
              <h2 id="modify-order-title">修改订单数量</h2>
            </div>
            <button class="icon-button" type="button" aria-label="关闭修改数量窗口" @click="closeModifyOrder">
              <X :size="17" aria-hidden="true" />
            </button>
          </div>
          <p class="modal-description">
            {{ editingOrder.symbol }} {{ editingOrder.name }} · {{ editingOrder.side === 'BUY' ? '买入' : '卖出' }}
          </p>
          <form class="modal-form" novalidate @submit.prevent="submitModifyOrder">
            <label for="edit-order-quantity">委托数量（股）</label>
            <input
              id="edit-order-quantity"
              ref="editQuantityInput"
              v-model="editQuantity"
              type="number"
              step="1"
              inputmode="numeric"
              autocomplete="off"
              :disabled="editQuantitySubmitting"
              :aria-invalid="Boolean(editQuantityError)"
              aria-describedby="edit-order-hint edit-order-error"
            />
            <p id="edit-order-hint" class="field-hint">原数量 {{ editingOrder.quantity.toLocaleString() }} 股；系统会按板块交易单位校正。</p>
            <p v-if="editQuantityError" id="edit-order-error" class="field-error" role="alert">{{ editQuantityError }}</p>
            <div class="modal-actions">
              <button class="button ghost" type="button" :disabled="editQuantitySubmitting" @click="closeModifyOrder">取消</button>
              <button class="button primary" type="submit" :disabled="editQuantitySubmitting">
                <RefreshCw v-if="editQuantitySubmitting" :size="15" class="spinning" aria-hidden="true" />
                <Check v-else :size="15" aria-hidden="true" />
                {{ editQuantitySubmitting ? '保存中…' : '保存数量' }}
              </button>
            </div>
          </form>
        </section>
      </div>

      <section class="status-strip" aria-label="运行状态">
        <div class="status-item"><span class="status-label">运行状态</span><strong :class="`status-text ${status.runner_status.toLowerCase()}`">{{ statusLabel(status.runner_status) }}</strong></div>
        <div class="status-item"><span class="status-label">行情</span><strong>{{ statusLabel(status.data.mode) }}</strong><small>{{ status.data.provider }}</small></div>
        <div class="status-item"><span class="status-label">更新时间</span><strong>{{ time(status.data.last_update) }}</strong><small>{{ status.data.symbol_count }} 个标的</small></div>
        <div class="status-item"><span class="status-label">模型</span><strong class="mono">{{ status.model_version }}</strong><small>{{ status.model_name === 'lightgbm' ? `验证 IC ${((status.model_metrics?.validation_ic_mean ?? 0) * 100).toFixed(2)}%` : '因子基线' }}</small></div>
        <div class="status-controls">
          <button v-if="status.runner_status !== 'RUNNING'" class="button primary compact" type="button" @click="startRunner"><Play :size="15" aria-hidden="true" />启动 Runner</button>
          <button v-else class="button ghost compact" type="button" @click="stopRunner"><Pause :size="15" aria-hidden="true" />暂停</button>
          <button class="button ghost compact" type="button" @click="tickRunner"><Zap :size="15" aria-hidden="true" />单步回放</button>
        </div>
      </section>

      <div v-if="loading" class="loading-state"><RefreshCw :size="18" class="spinning" />正在连接本地 Runner…</div>

      <template v-else>
        <section v-if="activeView === 'overview'" class="page-view">
          <div class="metrics-grid">
            <article class="metric-card featured">
              <div class="metric-icon green"><CircleDollarSign :size="19" aria-hidden="true" /></div>
              <div><span class="metric-label">账户净值</span><strong>¥{{ money(account.total_equity) }}</strong><small :class="movementClass(equityChange)"><ArrowUpRight v-if="equityChange >= 0" :size="14" aria-hidden="true" /><ArrowDownRight v-else :size="14" aria-hidden="true" />累计 {{ signedMoney(equityChange) }} / {{ pct(equityChangePct) }}</small></div>
              <div class="mini-sparkline" :class="movementClass(equityChange)" :aria-label="equitySparklineLabel"><svg viewBox="0 0 120 36" role="img"><polyline :points="equitySparklinePoints" fill="none" stroke="currentColor" stroke-width="2" /></svg></div>
            </article>
            <article class="metric-card">
              <div class="metric-icon red"><Activity :size="19" aria-hidden="true" /></div>
              <div><span class="metric-label">今日浮动收益</span><strong :class="movementClass(market.strategy_day_pnl)">{{ signedMoney(market.strategy_day_pnl) }}</strong><small :class="movementClass(market.strategy_day_return)">{{ pct(market.strategy_day_return) }} · 按今日净值基准</small></div>
            </article>
            <article class="metric-card">
              <div class="metric-icon blue"><LineChart :size="19" aria-hidden="true" /></div>
              <div><span class="metric-label">大盘 · {{ market.primary_benchmark_name }}</span><strong :class="movementClass(market.benchmark_return)">{{ pct(market.benchmark_return) }}</strong><small>{{ primaryBenchmark?.price ? primaryBenchmark.price.toFixed(2) : '--' }} · {{ market.status === 'REPLAY' ? '回放代理' : '指数行情' }}</small></div>
            </article>
            <article class="metric-card">
              <div class="metric-icon purple"><BarChart3 :size="19" aria-hidden="true" /></div>
              <div><span class="metric-label">策略超额</span><strong :class="movementClass(market.excess_return)">{{ pct(market.excess_return) }}</strong><small>策略 {{ pct(market.strategy_day_return) }} − 大盘 {{ pct(market.benchmark_return) }}</small></div>
            </article>
            <article class="metric-card"><div class="metric-icon blue"><WalletCards :size="19" aria-hidden="true" /></div><div><span class="metric-label">可用现金</span><strong>¥{{ money(account.cash) }}</strong><small>现金底线 {{ pct(0.1) }}</small></div></article>
            <article class="metric-card"><div class="metric-icon amber"><Gauge :size="19" aria-hidden="true" /></div><div><span class="metric-label">总敞口</span><strong>{{ pct(account.gross_exposure) }}</strong><small>上限 {{ pct(1) }} · 净敞口 {{ pct(account.net_exposure) }}</small></div></article>
            <article class="metric-card"><div class="metric-icon purple"><ShieldAlert :size="19" aria-hidden="true" /></div><div><span class="metric-label">维持担保比例</span><strong>{{ ratio(account.maintenance_ratio) }}</strong><small>{{ account.financing_debt > 0 ? `负债 ¥${money(account.financing_debt)}` : '当前无融资负债' }}</small></div></article>
            <article class="metric-card"><div class="metric-icon amber"><BrainCircuit :size="19" aria-hidden="true" /></div><div><span class="metric-label">模型验证 IC</span><strong>{{ status.model_name === 'lightgbm' ? `${((status.model_metrics?.validation_ic_mean ?? 0) * 100).toFixed(2)}%` : '基线' }}</strong><small>因子覆盖 {{ pct(status.factor_coverage ?? 0) }} · {{ status.model_version }}</small></div></article>
          </div>

          <article class="panel market-panel">
            <div class="panel-header"><div><p class="eyebrow">MARKET CONTEXT</p><h3>市场基准与组合表现</h3><p class="panel-description">{{ market.message }} · 更新时间 {{ time(market.as_of) }}</p></div><span class="table-state" :class="market.status === 'LIVE' ? 'stable' : 'submitted'">{{ market.status === 'LIVE' ? '实时' : market.status === 'REPLAY' ? '回放' : '待更新' }}</span></div>
            <div class="benchmark-grid"><div v-for="benchmark in market.benchmarks" :key="benchmark.symbol" class="benchmark-item"><span>{{ benchmark.name }}</span><strong :class="movementClass(benchmark.change_pct)">{{ benchmark.price === null ? '--' : benchmark.price.toFixed(2) }}</strong><small :class="movementClass(benchmark.change_pct)">{{ pct(benchmark.change_pct) }}</small></div><div class="benchmark-item strategy"><span>策略今日</span><strong :class="movementClass(market.strategy_day_return)">{{ pct(market.strategy_day_return) }}</strong><small :class="movementClass(market.excess_return)">超额 {{ pct(market.excess_return) }}</small></div></div>
          </article>

          <div class="content-grid top-grid">
            <article class="panel signal-panel">
              <div class="panel-header"><div><p class="eyebrow">MODEL OUTPUT</p><h3>今日信号</h3></div><button class="text-button" type="button" @click="activeView = 'signals'">查看全部 <ArrowUpRight :size="15" aria-hidden="true" /></button></div>
              <div class="signal-list">
                <div v-for="signal in signals.slice(0, 6)" :key="signal.signal_id" class="signal-row">
                  <div class="symbol-avatar" :class="signal.action === 'BUY' ? 'buy' : 'hold'"><ArrowUpRight v-if="signal.action === 'BUY'" :size="15" aria-hidden="true" /><LineChart v-else :size="15" aria-hidden="true" /></div>
                  <div class="signal-main"><strong>{{ signal.symbol }} <span>{{ signal.name }}</span></strong><small>{{ boardLabel(signal.board) }} · {{ signal.sector }}</small></div>
                  <div class="signal-score"><span>评分</span><strong>{{ signal.score.toFixed(3) }}</strong></div>
                  <div class="action-pill" :class="signal.action.toLowerCase()">{{ signal.action === 'BUY' ? '买入' : '观察' }}</div>
                </div>
                <div v-if="!signals.length" class="empty-state">暂无信号，先运行一次因子重平衡。</div>
              </div>
              <div class="panel-footer"><button class="button secondary full" type="button" @click="rebalance"><BrainCircuit :size="15" aria-hidden="true" />运行一次因子重平衡</button></div>
            </article>

            <article class="panel order-panel">
              <div class="panel-header"><div><p class="eyebrow">ACTION QUEUE</p><h3>待处理订单 <span class="count-badge">{{ pendingOrders.length }}</span></h3></div><button class="text-button" type="button" @click="activeView = 'orders'">订单中心 <ArrowUpRight :size="15" aria-hidden="true" /></button></div>
              <div class="order-list">
                <div v-for="order in pendingOrders.slice(0, 5)" :key="order.order_id" class="order-row">
                  <div class="order-direction" :class="order.side.toLowerCase()"><ArrowUpRight v-if="order.side === 'BUY'" :size="15" aria-hidden="true" /><ArrowDownRight v-else :size="15" aria-hidden="true" /></div>
                  <div class="order-main"><strong>{{ order.symbol }} <span>{{ order.name }}</span></strong><small>{{ order.quantity.toLocaleString() }} 股 · {{ statusLabel(order.status) }}</small></div>
                  <div class="order-price"><strong>{{ order.limit_price ? `¥${order.limit_price.toFixed(2)}` : '--' }}</strong><small>{{ order.is_margin_order ? '融资候选' : '现金' }}</small></div>
                  <button v-if="status.mode === 'paper_manual' && order.status === 'DRAFT'" class="icon-button tiny approve" type="button" aria-label="确认订单" @click="confirmOrder(order.order_id)"><Check :size="15" /></button>
                  <button v-if="status.mode === 'paper_manual' && order.status === 'DRAFT'" class="icon-button tiny reject" type="button" aria-label="拒绝订单" @click="rejectOrder(order.order_id)"><X :size="15" /></button>
                </div>
                <div v-if="!pendingOrders.length" class="empty-state">订单队列为空。</div>
              </div>
              <div class="panel-footer"><span class="subtle"><Clock3 :size="14" aria-hidden="true" />人工确认延迟默认 60 秒</span></div>
            </article>
          </div>

          <div class="content-grid bottom-grid">
            <article class="panel positions-panel">
              <div class="panel-header"><div><p class="eyebrow">PORTFOLIO</p><h3>当前持仓 <span class="count-badge">{{ positions.length }}</span></h3></div><button class="text-button" type="button" @click="activeView = 'diagnostics'">风险详情 <ArrowUpRight :size="15" aria-hidden="true" /></button></div>
              <div class="table-wrap"><table><thead><tr><th>标的</th><th>板块</th><th>现价</th><th>持仓</th><th>成本</th><th>市值</th><th>状态</th></tr></thead><tbody><tr v-for="position in positions" :key="position.symbol"><td><strong>{{ position.symbol }}</strong><small>{{ position.name }}</small></td><td><span class="board-tag">{{ boardLabel(position.board) }}</span></td><td class="mono">¥{{ position.last_price.toFixed(2) }}</td><td class="mono">{{ position.long_qty.toLocaleString() }}</td><td class="mono">¥{{ position.long_avg_cost.toFixed(2) }}</td><td class="mono">¥{{ money(position.market_value) }}</td><td><span class="table-state stable">正常</span></td></tr></tbody></table></div>
            </article>

            <article class="panel risk-panel">
              <div class="panel-header"><div><p class="eyebrow">RISK WATCH</p><h3>风险监控</h3></div><ShieldAlert :size="18" class="panel-icon" aria-hidden="true" /></div>
              <div class="risk-list"><div v-for="risk in latestRisks" :key="risk.event_id" class="risk-row" :class="risk.severity.toLowerCase()"><div class="risk-icon"><AlertTriangle :size="15" aria-hidden="true" /></div><div><strong>{{ risk.code }}</strong><p>{{ risk.message }}</p><small>{{ time(risk.created_at) }}</small></div></div><div v-if="!latestRisks.length" class="healthy-state"><Check :size="20" aria-hidden="true" /><strong>当前没有高优先级风险</strong><span>Runner 会持续检查敞口、现金和两融指标。</span></div></div>
            </article>
          </div>
        </section>

        <section v-else-if="activeView === 'signals'" class="page-view single-page">
           <article class="panel full-panel"><div class="panel-header"><div><p class="eyebrow">MODEL OUTPUT</p><h3>信号中心</h3><p class="panel-description">当前模型版本：{{ status.model_version }}。信号只生成建议，不发送真实委托。</p></div><button class="button primary" type="button" @click="rebalance"><RefreshCw :size="15" aria-hidden="true" />重新评分</button></div><div class="table-wrap"><table><thead><tr><th>排名</th><th>标的</th><th>板块</th><th>评分</th><th>目标仓位</th><th>动作</th><th>解释</th></tr></thead><tbody><tr v-for="(signal, index) in signals" :key="signal.signal_id"><td class="muted-number">{{ String(index + 1).padStart(2, '0') }}</td><td><strong>{{ signal.symbol }}</strong><small>{{ signal.name }}</small></td><td><span class="board-tag">{{ boardLabel(signal.board) }}</span> {{ signal.sector }}</td><td class="score-cell">{{ signal.score.toFixed(3) }}</td><td>{{ pct(signal.target_weight) }}</td><td><span class="action-pill" :class="signal.action.toLowerCase()">{{ signal.action === 'BUY' ? '买入' : signal.action === 'SELL' ? '卖出' : '观察' }}</span></td><td class="reason-cell">{{ signal.reason }}</td></tr></tbody></table></div></article>
          <article class="panel action-card"><div class="action-card-icon"><BrainCircuit :size="20" aria-hidden="true" /></div><div><h3>训练本地模型</h3><p>使用当前回放数据生成 Point-in-Time 训练集，保存 LightGBM 模型版本和特征清单。</p></div><button class="button secondary" type="button" @click="trainModel">开始训练</button></article>
        </section>

        <section v-else-if="activeView === 'orders'" class="page-view single-page"><article class="panel full-panel"><div class="panel-header"><div><p class="eyebrow">PAPER BROKER</p><h3>订单队列</h3><p class="panel-description">{{ status.mode === 'paper_manual' ? '人工确认模式：只有确认后才进入模拟撮合。' : '自动纸盘模式：订单会自动确认并进入模拟撮合。' }}</p></div><div class="order-summary"><span>{{ pendingOrders.length }} 待处理</span><span>{{ dashboard?.fills.length ?? 0 }} 成交</span></div></div><div class="table-wrap"><table><thead><tr><th>订单</th><th>方向</th><th>数量</th><th>限价</th><th>状态</th><th>原因</th><th>操作</th></tr></thead><tbody><tr v-for="order in orders" :key="order.order_id"><td><strong>{{ order.symbol }}</strong><small>{{ order.order_id }}</small></td><td><span class="direction-text" :class="order.side.toLowerCase()">{{ order.side === 'BUY' ? '买入' : order.side === 'SELL' ? '卖出' : order.side }}</span></td><td class="mono">{{ order.quantity.toLocaleString() }}<small> / {{ order.filled_quantity.toLocaleString() }}</small></td><td class="mono">{{ order.limit_price ? `¥${order.limit_price.toFixed(2)}` : '--' }}</td><td><span class="table-state" :class="order.status.toLowerCase()">{{ statusLabel(order.status) }}</span></td><td class="reason-cell">{{ order.reject_reason || order.reason }}</td><td><div class="row-actions" v-if="order.status === 'DRAFT'"><button class="icon-button tiny" type="button" aria-label="修改数量" @click="openModifyOrder(order)"><Pencil :size="15" /></button><button class="icon-button tiny approve" type="button" aria-label="确认订单" @click="confirmOrder(order.order_id)"><Check :size="15" /></button><button class="icon-button tiny reject" type="button" aria-label="拒绝订单" @click="rejectOrder(order.order_id)"><X :size="15" /></button></div><div class="row-actions" v-else-if="['SUBMITTED', 'PARTIAL_FILLED'].includes(order.status)"><button class="icon-button tiny reject" type="button" aria-label="撤销订单" @click="cancelOrder(order.order_id)"><Ban :size="15" /></button></div><span v-else class="muted-number">—</span></td></tr></tbody></table></div></article></section>

        <section v-else-if="activeView === 'reports'" class="page-view single-page"><article class="panel full-panel"><div class="panel-header"><div><p class="eyebrow">END OF DAY</p><h3>收盘报告</h3><p class="panel-description">报告同时保存为 HTML、Markdown 和 JSON，所有结论关联模型与数据版本。</p></div><button class="button primary" type="button" @click="generateReport"><FileText :size="15" aria-hidden="true" />生成今日报告</button></div><div class="report-list"><div v-for="report in reports" :key="report.report_date" class="report-card"><div class="report-date"><strong>{{ report.report_date }}</strong><span :class="report.status === 'OK' ? 'stable' : 'error-state'">{{ report.status }}</span></div><div class="report-summary">{{ report.summary }}</div><div class="report-meta"><span>模型 {{ report.model_version }}</span><span>换手 {{ pct(report.turnover) }}</span><span>{{ report.fills_count }} 笔成交</span><span>数据 {{ statusLabel(report.data_status ?? 'UNKNOWN') }}</span></div><div v-if="report.research_metrics" class="report-research"><div class="report-research-header"><strong>样本外 Walk-forward</strong><span :class="report.research_metrics?.status === 'OK' ? 'stable' : 'error-state'">{{ report.research_metrics?.status ?? 'UNKNOWN' }}</span></div><div class="report-research-grid"><div><span>测试 IC</span><strong :class="researchMetricClass(report.research_metrics?.test_ic_mean)">{{ pct(report.research_metrics?.test_ic_mean) }}</strong></div><div><span>Top/Bottom</span><strong :class="researchMetricClass(report.research_metrics?.test_top_bottom_spread_mean)">{{ pct(report.research_metrics?.test_top_bottom_spread_mean) }}</strong></div><div><span>正 IC 折数</span><strong :class="researchMetricClass(report.research_metrics?.test_ic_positive_ratio)">{{ pct(report.research_metrics?.test_ic_positive_ratio) }}</strong></div><div><span>Sharpe</span><strong :class="researchMetricClass(report.research_metrics?.sharpe)">{{ decimal(report.research_metrics?.sharpe) }}</strong></div><div><span>因子覆盖</span><strong :class="researchMetricClass(report.research_metrics?.factor_coverage_mean)">{{ pct(report.research_metrics?.factor_coverage_mean) }}</strong></div></div><small>{{ report.research_metrics?.data_start ?? '--' }} → {{ report.research_metrics?.data_end ?? '--' }} · {{ report.research_metrics?.fold_count ?? '--' }} 折 · {{ report.research_metrics?.message ?? '仅用于研究评估，不自动晋级纸盘模型' }}</small></div></div><div v-if="!reports.length" class="empty-state">尚未生成报告。收盘后点击生成，或由 Runner 调度。</div></div></article></section>

        <section v-else class="page-view single-page"><div class="content-grid diagnostics-grid"><article class="panel full-panel"><div class="panel-header"><div><p class="eyebrow">DATA HEALTH</p><h3>数据与服务诊断</h3></div><Database :size="18" class="panel-icon" aria-hidden="true" /></div><div class="diagnostic-list"><div class="diagnostic-row"><span>行情提供者</span><strong>{{ status.data.provider }}</strong><span class="table-state stable">{{ statusLabel(status.data.mode) }}</span></div><div class="diagnostic-row"><span>标的数量</span><strong>{{ status.data.symbol_count }}</strong><span>股票池与回放源</span></div><div class="diagnostic-row"><span>最后行情时间</span><strong>{{ time(status.data.last_update) }}</strong><span>{{ status.data.message }}</span></div><div class="diagnostic-row"><span>Runner 心跳</span><strong>{{ time(status.last_heartbeat) }}</strong><span>{{ status.last_error || '没有错误' }}</span></div><div class="diagnostic-row"><span>基本面覆盖</span><strong>{{ pct(status.factor_coverage ?? 0) }}</strong><span>估值、质量、成长和规模字段的可用比例</span></div><div class="diagnostic-row"><span>模型验证</span><strong>{{ status.model_name === 'lightgbm' ? ((status.model_metrics?.validation_rank_corr ?? 0) * 100).toFixed(2) + '%' : '未训练' }}</strong><span>{{ status.model_name === 'lightgbm' ? `时间外验证 ${status.model_metrics?.validation_start ?? '--'} → ${status.model_metrics?.validation_end ?? '--'}` : '当前使用透明因子基线' }}</span></div><div class="diagnostic-row"><span>通知</span><strong>{{ status.notifications_enabled ? '已启用' : '已关闭' }}</strong><span>自动纸盘默认关闭弹窗</span></div><div class="diagnostic-row"><span>交易边界</span><strong>Paper Only</strong><span>不会连接券商发送委托</span></div></div></article><article class="panel full-panel"><div class="panel-header"><div><p class="eyebrow">MARGIN ACCOUNT</p><h3>两融账户概览</h3></div><CircleDollarSign :size="18" class="panel-icon" aria-hidden="true" /></div><div class="margin-grid"><div><span>融资负债</span><strong>¥{{ money(account.financing_debt) }}</strong></div><div><span>融资利息</span><strong>¥{{ money(account.financing_interest) }}</strong></div><div><span>融券费用</span><strong>¥{{ money(account.short_borrow_value) }}</strong></div><div><span>可用保证金</span><strong>¥{{ money(account.available_margin) }}</strong></div></div></article></div></section>
      </template>
    </main>
  </div>
</template>
