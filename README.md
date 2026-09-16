# A 股量化纸盘控制台

一个本地运行的 A 股量化研究、实时纸盘模拟和人工决策辅助系统。

完整的阶段计划、当前进度和后续执行顺序见：[项目总计划](docs/project-plan.md)。

## 边界

- 只生成信号、订单建议和纸盘成交，不连接券商发送真实委托。
- `paper_auto` 用于自动纸盘测试，默认关闭 Windows 弹窗。
- `paper_manual` 用于人工确认后模拟成交。
- `manual_record` 用于录入用户在外部软件完成的实际成交。
- 300/301/688 和融资融券在纸盘账户内独立建模，默认安全账户不超过 1 倍总敞口。

## 当前状态

第一版提供：

- FastAPI 本地 API。
- Vue 3 + TypeScript 操作面板。
- 东方财富掘金只读行情适配器；默认股票价格从真实 API 获取，不使用 CSV 作为行情输入。
- SQLite 账户/订单/成交账本。
- 日频可审计因子基线：5/20/60/120 日动量、5 日反转、20 日低波、20 日流动性、估值、质量、成长和规模。
- 本地 LightGBM：按时间切分验证，保存训练区间、验证区间、特征清单、指标和模型文件；API 与后台 Runner 会从同一模型文件恢复。
- 东方财富基础面因子桥接：日 PE/PB、市值、换手率，以及按公告日期回填的 ROE、利润增速；缺失时显示覆盖率，不用虚构数据补齐。
- 订单确认、人工延迟、T+1、板块交易单位、涨跌停和部分成交模拟。
- 收盘报告、健康状态和 Windows Toast 可选适配。
- Runner 与 UI 进程通过 SQLite WAL 账本同步，UI 关闭不会删除纸盘状态。
- 订单有完整状态链路：草稿、确认、提交、部分成交、成交、拒绝、撤销、过期。
- API 返回的历史日线可以经过质量检查后落盘为 Parquet，并用 DuckDB 按日期和股票读取。
- 因子研究回测严格采用“日 T 收盘生成信号，日 T+1 开盘模拟成交”，输出 JSON 和权益曲线 CSV。

股票价格和研究因子不从 CSV 导入。当前默认数据源为东方财富掘金的只读 API；数据源不可用或基础面因子接口不完整时，系统会显示 `UNAVAILABLE`/`STALE` 与具体原因，不会退回演示数据，也不会继续生成看似正常的开盘调仓。

`ReplayProvider` 和 `--source demo` 只用于自动化测试与撮合引擎验收，不是研究或纸盘的数据来源。

## 环境

项目主环境选用 Python 3.11.9。当前机器的 Python 3.15 alpha 不作为项目运行环境。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,windows]"
```

东方财富掘金 SDK 使用独立环境 `.venv-eastmoney`，版本固定为 `gm 3.0.183`。这是因为该 SDK 目前要求 `pandas<2` 和 `numpy<2`，而项目主环境使用较新的数据科学依赖，不能把两者安装到同一个环境。SDK 只用于只读行情桥接；项目仍由 `PaperBroker` 负责纸盘撮合，绝不调用真实委托接口。

```powershell
$eastmoneyPython = ".\\.venv-eastmoney\\Scripts\\python.exe"
& $eastmoneyPython -c "import gm; print('gm 3.0.183')"
```

配置东财量化 Token 时，不要把 Token 发到聊天或写入代码。请在本机 PowerShell 执行：

```powershell
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File .\\scripts\\configure_eastmoney_token.ps1
```

脚本会打开本地密码输入窗口，并保存到当前 Windows 用户的 `QUANTPAPER_EASTMONEY_TOKEN` 环境变量；重新打开终端后才会对新进程生效。

同花顺 iFinD 适配器已保留为可选实现，但它不是当前默认方案。若后续确认接受其账号和额度，再额外安装：

```powershell
python -m pip install -e ".[ifind]"
$env:QUANTPAPER_IFIND_ACCOUNT = "你的 iFinD 接口账号"
$env:QUANTPAPER_IFIND_PASSWORD = "你的 iFinD 接口密码"
```

这里的账号是 iFinD 数据接口账号，不是券商交易账号；目前不建议为了本项目直接申请或付费。

当前配置把研究日线池设为全量合格 A 股（`research_universe_limit = 0`），把默认盘中行情池设为最多 500 只（`realtime_universe_limit = 500`），并在模型评分后只为排名靠前的候选股和当前持仓请求行情。股票范围包含普通 A 股、创业板 300/301 和科创板 688。后续更换数据源时，只需要替换 provider，不需要修改策略和 PaperBroker。

这两个范围有意分开：历史研究需要尽量避免股票池选择偏差，盘中则不需要为全市场持续拉取 1 分钟行情。`candidate_pool_size = 500` 仍表示组合引擎每天从模型排名中考虑的候选数量，不代表研究数据只有 500 只股票。

系统只把账号和密码从环境变量读取，不写入配置文件、数据库或报告。

东方财富量化终端适配器位于 `scripts/eastmoney_terminal_adapter.py`。监督进程在交易时段以只读轮询模式启动它，把终端行情写入 `data/runtime/eastmoney-terminal-feed.json`；`EastmoneyProvider` 优先使用新鲜的适配器行情，缺失时回退到原有只读桥接。适配器不导入任何下单函数，PaperBroker 仍是唯一模拟成交引擎。需要在量化终端的研究策略中运行时，可使用同一文件的 `init`/`on_bar` 回调；`--strategy` 模式同样只转发 60 秒行情，不发送委托。

接口说明可参考[同花顺免费版权限说明](https://quantapi.10jqka.com.cn/gwstatic/static/ds_web/quantapi-web/help-center/permission.html)、[Python 应用示例](https://quantapi.10jqka.com.cn/gwstatic/static/ds_web/quantapi-web/example.html)和 [iFinDAPI 安装包](https://pypi.org/project/iFinDAPI/0.0.8/)。

## 启动

日常启动和验收统一使用独立的打开流程，不要再分别手动启动 API 和 Runner。这样可以复用东方财富登录会话、写入手工测试会话标记，并避免旧的后台实例抢占端口。

正式实时纸盘入口：

```powershell
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File .\scripts\open_quantpaper.ps1 -Mode live
```

视觉验收和自动化 UI 测试入口：

```powershell
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File .\scripts\open_quantpaper.ps1 -Mode replay -FreshReplay -ForceRestart -NoBrowser
```

回放模式使用独立的 `data/runtime/visual-paper.db`，明确标记为 `REPLAY`，不会污染正式纸盘账本，也不会访问真实委托接口。需要关闭项目后台时执行：

```powershell
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File .\scripts\close_quantpaper.ps1
```

桌面的 `QuantPaper One-Click.lnk` 已指向上述 `open_quantpaper.ps1 -Mode live` 流程。开发时仍可直接使用下面的底层命令，但它们不作为验收入口：

终端一，启动后台 Runner：

终端一，启动后台 Runner：

```powershell
python -m quantpaper.cli runner
```

终端二，启动 API 和内置网页面板：

```powershell
python -m quantpaper.cli serve
```

打开：`http://127.0.0.1:8000/`

如需开机后自动后台运行（不打开浏览器窗口），在当前用户 PowerShell 执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_quantpaper_autostart.ps1
```

它会创建当前用户的 Windows 计划任务，登录后启动一个轻量监督进程。监督进程只在交易日 09:10–15:30（Asia/Shanghai）启动东方财富经典客户端，再按经典客户端左下角“量化”按钮实际使用的 `gmstarter.exe --token=<会话令牌>` 方式打开掘金量化终端、API 和 Runner；令牌只从当前用户已配置的环境变量读取，不写入日志。令牌缺失时不会打开未登录的量化终端。其它时间会停止本项目的重型后台进程，并关闭由监督进程启动的东方财富客户端。电脑 09:10 之后才开机也不影响，只要仍在 15:30 前，登录后会立即补启动；超过 15:30 则等待下一个交易日。东方财富经典客户端必须保持登录，因为量化终端需要复用它的登录会话；脚本不会自动填写账号、密码或验证码。删除自动启动任务：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_quantpaper_autostart.ps1 -Remove
```

如果正在开发前端，才需要另开终端启动 Vite：

```powershell
cd frontend
npm install
npm run dev
```

开发地址：`http://127.0.0.1:5173`。API 和打包后的网页面板地址：`http://127.0.0.1:8000`。

启动后，Runner 只访问配置的数据 API，不会访问券商，也不会发送真实委托。运行模式可以在面板切换：

- `paper_auto`：后台自动确认和撮合。
- `paper_manual`：生成订单草稿，面板确认后撮合。
- `manual_record`：只记录外部手动成交，不进入券商接口。

当前测试配置由 Runner 常驻轮询真实行情，并在上海时间下午盘开盘后的
13:00–13:15 只生成一轮调仓计划。`paper_auto` 会自动确认这轮订单并交给
`PaperBroker` 模拟撮合；行情状态为 `STALE` 或 `UNAVAILABLE` 时会暂停新增订单。
该窗口可在 `config/default.toml` 的 `session_rebalance` 和
`session_rebalance_window_minutes` 中调整。

模型训练入口在“信号中心”的“开始训练”。训练目标是未来 5 个交易日的横截面超额收益，验证集永远位于训练区间之后。验证 IC、排序相关系数、Top-Bottom spread 和基本面覆盖率会写入模型元数据、诊断页和收盘报告。首次训练前使用透明的 `factor-baseline-v0.1`，训练成功后才切换到带版本号的 LightGBM。

默认数据和数据库目录为 `data/`，模型和报告不会提交到代码仓库。启动时如果没有可用行情源，系统会显示 `UNAVAILABLE` 并停止新增订单，不会退回演示数据。

运行测试：

```powershell
python -m pytest
```

## 研究数据和回测

研究数据必须由数据 API 获取。可以先同步到本地 Parquet 缓存，质量检查仍然会检查重复交易日、无效日期和非正价格：

```powershell
python -m quantpaper.cli sync-data --dataset daily
```

默认从配置的真实 API 获取历史行情进行回测：

```powershell
python -m quantpaper.cli backtest
```

在首次补齐市值和行业字段前，先在东方财富量化终端已登录且处于运行状态时执行只读能力探针：

```powershell
& .\.venv-eastmoney\Scripts\python.exe .\scripts\probe_eastmoney_capabilities.py
```

探针只查询 4 只股票的最近 7 天数据，不会下单。量化终端关闭时接口会返回“无法连接到终端服务”，这不等同于权限不足；探针通过后才会开启行业全量请求和新缓存同步。

如果已经用 API 同步过 Parquet，可以使用内部缓存回测：

```powershell
python -m quantpaper.cli backtest --source store --dataset daily
```

如需把东方财富量化客户端当前账号能访问的历史数据完整归档到本地，使用：

```powershell
python .\scripts\sync_eastmoney_history.py --dataset all --root data\eastmoney
```

归档脚本会保存当前股票主表、交易日历、各股票上市以来的日线，以及有权限的
历史估值、市值、换手和财务衍生字段。每只股票单独保存为压缩 Parquet，并写入
`data/eastmoney/manifest.json`，可中断后重复执行继续下载。归档数据先保持在
独立目录，不会自动覆盖当前纸盘使用的 `data/market/daily.parquet`；完成质量检查
后再决定是否切换研究数据源。行业分类接口若账号没有权限，会记录为不可用，
不会用当前分类回填历史。

只有在测试撮合引擎时才使用确定性演示数据：

```powershell
python -m quantpaper.cli backtest --source demo
```

默认输出：

- `data/reports/backtest-demo.json`：回测统计、数据质量、交易明细和权益曲线。
- `data/reports/backtest-demo.equity.csv`：回测结果导出文件，不是行情输入文件。

回测当前是多因子基线，不代表实盘收益承诺。因子、模型版本、数据快照和成交边界会在后续研究迭代中继续补充审计字段。
