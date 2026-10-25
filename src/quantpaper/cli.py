from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path


def research_backtest_config(
    app_config,
    *,
    rebalance_every: int | None = None,
    hold_buffer: float | None = None,
):
    """Build the research backtest config from application settings.

    ``backtest``, ``walkforward`` and ``period-backtest`` previously assembled
    this by hand, and the plain ``backtest`` command skipped it entirely and
    silently used dataclass defaults.  One helper keeps all three commands on
    the same cost, position and turnover rules.
    """

    from .backtest import BacktestConfig

    portfolio = app_config.portfolio
    execution = app_config.execution
    return BacktestConfig(
        initial_capital=app_config.account.initial_capital,
        min_positions=portfolio.min_positions,
        max_positions=portfolio.max_positions,
        cash_floor=app_config.account.cash_floor,
        single_weight_cap=portfolio.single_weight_cap,
        industry_weight_cap=portfolio.industry_weight_cap,
        max_daily_turnover=portfolio.max_daily_turnover,
        max_daily_orders=portfolio.max_daily_action_orders,
        commission_rate=execution.commission_rate,
        stamp_duty_rate=execution.stamp_duty_rate,
        slippage_bps=execution.slippage_bps,
        rebalance_every_n_days=(
            portfolio.rebalance_every_n_days if rebalance_every is None else rebalance_every
        ),
        hold_buffer_ratio=(
            portfolio.hold_buffer_ratio if hold_buffer is None else hold_buffer
        ),
    )


def add_turnover_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rebalance-every",
        type=int,
        default=None,
        help="每 N 个交易日再平衡一次；省略时取 config 的 portfolio.rebalance_every_n_days",
    )
    parser.add_argument(
        "--hold-buffer",
        type=float,
        default=None,
        help="持仓缓冲带比例，0 表示严格 top-N；省略时取 config 的 portfolio.hold_buffer_ratio",
    )


def build_parser() -> argparse.ArgumentParser:
    from .features import FeatureEngine
    from .model import LightGBMModel

    parser = argparse.ArgumentParser(description="A股量化纸盘控制台")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="启动 FastAPI API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    runner = subparsers.add_parser("runner", help="启动常驻纸盘 Runner")
    runner.add_argument("--log-level", default="INFO")
    subparsers.add_parser("tick", help="执行一次纸盘回放 tick")
    backtest = subparsers.add_parser(
        "backtest", help="运行因子回测；默认从配置的数据 API 获取历史行情"
    )
    backtest.add_argument("--output", default="data/reports/backtest-demo.json")
    backtest.add_argument("--days", type=int, default=0, help="只使用最近 N 个交易日，0 表示全部")
    backtest.add_argument("--seed", type=int, default=7)
    backtest.add_argument(
        "--source",
        choices=("live", "store", "demo"),
        default="live",
        help="live=数据 API，store=API 已同步的 Parquet，demo=仅用于测试引擎",
    )
    backtest.add_argument(
        "--score",
        choices=("factor_score", "f1_factor_score", "research_factor_score"),
        default="factor_score",
        help="回测评分列；F1/研究分只用于候选比较，不会改变纸盘生产模型",
    )
    backtest.add_argument("--market-root", default="data/eastmoney")
    backtest.add_argument("--dataset", default="daily")
    backtest.add_argument(
        "--quality-policy",
        choices=("exclude", "include"),
        default="exclude",
        help="store 研究默认排除已标记的极端价格事件窗口；include 仅用于诊断",
    )
    add_turnover_arguments(backtest)
    factor_research = subparsers.add_parser(
        "factor-research", help="拆分评估单个因子的 Rank IC 和分组收益"
    )
    factor_research.add_argument(
        "--output", default="data/reports/factor-research-f1-3y.json"
    )
    factor_research.add_argument(
        "--source",
        choices=("live", "store", "demo"),
        default="store",
        help="live=数据 API，store=API 已同步的 Parquet，demo=仅用于测试引擎",
    )
    factor_research.add_argument("--days", type=int, default=0)
    factor_research.add_argument("--horizon-days", type=int, default=5)
    factor_research.add_argument(
        "--factor-set",
        choices=("f1", "baseline", "research"),
        default="f1",
        help="f1=8个候选，baseline=当前11因子，research=全部研究因子",
    )
    factor_research.add_argument(
        "--neutralization",
        choices=("none", "sector", "sector_size_beta"),
        default="none",
        help="横截面中性化方式",
    )
    factor_research.add_argument("--market-root", default="data/eastmoney")
    factor_research.add_argument("--dataset", default="daily")
    factor_research.add_argument(
        "--quality-policy",
        choices=("exclude", "include"),
        default="exclude",
        help="store 研究默认排除已标记的极端价格事件窗口；include 仅用于诊断",
    )
    walkforward = subparsers.add_parser(
        "walkforward", help="运行按时间滚动训练的本地 LightGBM 回测"
    )
    walkforward.add_argument("--output", default="data/reports/lightgbm-walkforward.json")
    walkforward.add_argument(
        "--source",
        choices=("live", "store"),
        default="store",
        help="live=数据 API，store=API 已同步的 Parquet",
    )
    walkforward.add_argument("--market-root", default="data/eastmoney")
    walkforward.add_argument("--dataset", default="daily")
    walkforward.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行拟合的 fold 数；1 为历史顺序行为。LightGBM 训练释放 GIL，"
        "线程池在多 fold 时有效（每个 fit 自动钉在单线程）",
    )
    walkforward.add_argument(
        "--threads-per-fit",
        type=int,
        default=None,
        help="每个 LightGBM fit 的线程数；缺省按 workers 推导。显式指定可让"
        "结果跨 worker 数逐位可复现",
    )
    walkforward.add_argument("--train-days", type=int, default=504)
    walkforward.add_argument("--test-days", type=int, default=21)
    walkforward.add_argument("--step-days", type=int, default=21)
    walkforward.add_argument("--horizon-days", type=int, default=5)
    walkforward.add_argument("--min-training-rows", type=int, default=500)
    walkforward.add_argument(
        "--feature-profile",
        choices=sorted(FeatureEngine.FEATURE_PROFILES),
        default="baseline-v1",
        help="模型特征画像；candidate-v3 只保留有横截面区分度的因子",
    )
    walkforward.add_argument(
        "--early-stopping-metric",
        choices=LightGBMModel.EARLY_STOPPING_METRICS,
        default="rank_ic",
        help="早停指标；rank_ic 按横截面排名能力选模型，l2 为旧行为",
    )
    walkforward.add_argument("--early-stopping-rounds", type=int, default=50)
    walkforward.add_argument(
        "--score-smoothing-days",
        type=int,
        default=1,
        help="对模型打分做 N 日滚动均值再排名；1 表示不平滑，用于抑制排名抖动",
    )
    walkforward.add_argument(
        "--hold-buffer",
        type=float,
        default=None,
        help="持仓缓冲带比例，0 表示严格 top-N；省略时取 config 的 portfolio.hold_buffer_ratio",
    )
    walkforward.add_argument(
        "--rebalance-every",
        type=int,
        default=None,
        help="每 N 个交易日再平衡一次；省略时取 config 的 portfolio.rebalance_every_n_days",
    )
    walkforward.add_argument(
        "--quality-policy",
        choices=("exclude", "include"),
        default="exclude",
        help="store 研究默认排除已标记的极端价格事件窗口；include 仅用于诊断",
    )
    period_backtest = subparsers.add_parser(
        "period-backtest", help="按指定起止日期运行多个因子回测区间并对比大盘"
    )
    period_backtest.add_argument(
        "--output", default="data/reports/period-backtest.json"
    )
    period_backtest.add_argument(
        "--period",
        dest="periods",
        action="append",
        help="区间格式 name:YYYY-MM-DD:YYYY-MM-DD；可重复，省略则使用默认年度区间",
    )
    period_backtest.add_argument("--benchmark", default="000300")
    period_backtest.add_argument(
        "--score",
        choices=("factor_score", "f1_factor_score", "research_factor_score"),
        default="factor_score",
    )
    period_backtest.add_argument("--warmup-trading-days", type=int, default=120)
    period_backtest.add_argument("--market-root", default="data/eastmoney_adjusted")
    period_backtest.add_argument(
        "--quality-policy", choices=("exclude", "include"), default="exclude"
    )
    add_turnover_arguments(period_backtest)
    sync_data = subparsers.add_parser("sync-data", help="从配置的数据 API 同步日线到 Parquet")
    sync_data.add_argument("--root", default="data/market")
    sync_data.add_argument("--dataset", default="daily")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, getattr(args, "log_level", "INFO").upper()))
    if args.command == "serve":
        import uvicorn

        uvicorn.run("quantpaper.api:app", host=args.host, port=args.port, reload=False)
    elif args.command == "runner":
        from .runtime import RuntimeService

        async def run() -> None:
            service = RuntimeService()
            await service.start()
            try:
                while not service.stop_requested:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                await service.stop()
                raise
            finally:
                if not service.stop_requested:
                    await service.stop()

        asyncio.run(run())
    elif args.command == "tick":
        from .runtime import RuntimeService

        async def one_tick() -> None:
            service = RuntimeService()
            await service.tick()
            print(service.dashboard()["status"])

        asyncio.run(one_tick())
    elif args.command == "backtest":
        import pandas as pd

        from .backtest import FactorBacktester
        from .config import load_config
        from .data import ProviderUnavailable, build_provider, make_demo_provider
        from .features import FeatureEngine
        from .storage import HistoricalStore

        if args.source == "store":
            try:
                frame = HistoricalStore(args.market_root).read_daily(
                    args.dataset,
                    enrich_fundamentals=True,
                    a_shares_only=True,
                    require_adjusted=True,
                    exclude_quality_events=args.quality_policy == "exclude",
                )
            except (FileNotFoundError, ValueError) as exc:
                raise SystemExit(f"研究归档不可用：{exc}") from exc
            securities = None
            parsed_dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
            calendar = sorted(parsed_dates.dt.date.unique())
        elif args.source == "demo":
            provider = make_demo_provider(seed=args.seed)
            frame = provider.get_daily_frame()
            securities = provider.get_security_master()
            calendar = provider.get_trading_calendar()
        else:
            provider = build_provider(load_config())
            try:
                frame = provider.get_daily_frame()
                securities = provider.get_security_master()
            except ProviderUnavailable as exc:
                raise SystemExit(f"真实数据 API 不可用：{exc}") from exc
            parsed_dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
            calendar = sorted(parsed_dates.dt.date.unique())
        if args.days < 0:
            raise SystemExit("--days must be zero or a positive integer")
        if args.days:
            if args.days < 2:
                raise SystemExit("--days 至少需要 2 个交易日")
            if not calendar:
                raise SystemExit("输入数据没有可用的 trade_date")
            cutoff = calendar[max(0, len(calendar) - args.days)]
            frame = frame[pd.to_datetime(frame["trade_date"]).dt.date >= cutoff].reset_index(
                drop=True
            )
        feature_engine = FeatureEngine()
        feature_frame = feature_engine.build_features(frame)
        result = FactorBacktester(securities, feature_engine=feature_engine).run(
            frame,
            config=research_backtest_config(
                load_config(),
                rebalance_every=args.rebalance_every,
                hold_buffer=args.hold_buffer,
            ),
            feature_frame=feature_frame,
            score_column=args.score,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result.equity_curve.to_csv(output.with_name(f"{output.stem}.equity.csv"), index=False)
        print(json.dumps(result.stats, ensure_ascii=False, indent=2))
        print(f"report: {output}")
    elif args.command == "factor-research":
        import pandas as pd

        from .config import load_config
        from .data import ProviderUnavailable, build_provider, make_demo_provider
        from .features import FeatureEngine
        from .research import FactorResearchEvaluator
        from .storage import HistoricalStore

        if args.source == "store":
            try:
                frame = HistoricalStore(args.market_root).read_daily(
                    args.dataset,
                    enrich_fundamentals=True,
                    a_shares_only=True,
                    require_adjusted=True,
                    exclude_quality_events=args.quality_policy == "exclude",
                )
            except (FileNotFoundError, ValueError) as exc:
                raise SystemExit(f"研究归档不可用：{exc}") from exc
        elif args.source == "demo":
            frame = make_demo_provider().get_daily_frame()
        else:
            try:
                frame = build_provider(load_config()).get_daily_frame()
            except ProviderUnavailable as exc:
                raise SystemExit(f"真实数据 API 不可用：{exc}") from exc

        if args.days < 0:
            raise SystemExit("--days must be zero or a positive integer")
        if args.days:
            if args.days < 2:
                raise SystemExit("--days 至少需要 2 个交易日")
            normalized = HistoricalStore.normalize_daily(frame)
            dates = sorted(pd.to_datetime(normalized["trade_date"]).dt.date.unique())
            if not dates:
                raise SystemExit("输入数据没有可用的 trade_date")
            cutoff = dates[max(0, len(dates) - args.days)]
            frame = normalized[
                pd.to_datetime(normalized["trade_date"]).dt.date >= cutoff
            ].reset_index(drop=True)

        engine = FeatureEngine()
        if args.factor_set == "f1":
            factors = list(engine.F1_FACTOR_COLUMNS)
        elif args.factor_set == "baseline":
            factors = list(engine.BASELINE_FACTOR_COLUMNS)
        else:
            factors = list(engine.RESEARCH_FACTOR_COLUMNS)
        result = FactorResearchEvaluator(engine).evaluate(
            frame,
            factors=factors,
            horizon_days=args.horizon_days,
            neutralization=args.neutralization,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        print(f"report: {output}")
    elif args.command == "walkforward":
        from .config import load_config
        from .data import ProviderUnavailable, build_provider
        from .features import FeatureEngine
        from .ml_backtest import LightGBMWalkForwardBacktester, WalkForwardConfig
        from .storage import HistoricalStore

        app_config = load_config()
        if args.source == "store":
            try:
                frame = HistoricalStore(args.market_root).read_daily(
                    args.dataset,
                    enrich_fundamentals=True,
                    a_shares_only=True,
                    require_adjusted=True,
                    exclude_quality_events=args.quality_policy == "exclude",
                )
            except (FileNotFoundError, ValueError) as exc:
                raise SystemExit(f"研究归档不可用：{exc}") from exc
        else:
            try:
                frame = build_provider(app_config).get_daily_frame()
            except ProviderUnavailable as exc:
                raise SystemExit(f"真实数据 API 不可用：{exc}") from exc

        feature_engine = FeatureEngine()
        walkforward_config = WalkForwardConfig(
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            horizon_days=args.horizon_days,
            min_training_rows=args.min_training_rows,
            validation_fraction=app_config.model.validation_fraction,
            min_validation_days=app_config.model.min_validation_days,
            validation_gap_days=app_config.model.validation_gap_days,
            early_stopping_rounds=args.early_stopping_rounds,
            early_stopping_metric=args.early_stopping_metric,
            score_smoothing_days=args.score_smoothing_days,
            artifact_dir=Path(app_config.model.artifact_dir) / "walkforward",
            workers=args.workers,
            threads_per_fit=args.threads_per_fit,
        )
        result = LightGBMWalkForwardBacktester(feature_engine=feature_engine).run(
            frame,
            walkforward=walkforward_config,
            backtest=research_backtest_config(
                app_config,
                rebalance_every=args.rebalance_every,
                hold_buffer=args.hold_buffer,
            ),
            feature_columns=feature_engine.feature_profile(args.feature_profile),
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result.equity_curve.to_csv(output.with_name(f"{output.stem}.equity.csv"), index=False)
        print(json.dumps(result.stats, ensure_ascii=False, indent=2))
        print(f"report: {output}")
    elif args.command == "period-backtest":
        from .config import load_config
        from .period_backtest import parse_periods, report_as_dict, run_period_backtests
        from .storage import HistoricalStore

        if args.warmup_trading_days < 0:
            raise SystemExit("--warmup-trading-days must be non-negative")
        try:
            periods = parse_periods(args.periods)
            app_config = load_config()
            store = HistoricalStore(args.market_root)
            backtest_config = research_backtest_config(
                app_config,
                rebalance_every=args.rebalance_every,
                hold_buffer=args.hold_buffer,
            )
            runs = run_period_backtests(
                store,
                periods,
                benchmark_symbol=args.benchmark,
                score_column=args.score,
                backtest_config=backtest_config,
                warmup_trading_days=args.warmup_trading_days,
                exclude_quality_events=args.quality_policy == "exclude",
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise SystemExit(f"多区间回测不可用：{exc}") from exc
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        report = report_as_dict(
            runs,
            market_root=args.market_root,
            benchmark_symbol=args.benchmark,
            score_column=args.score,
            warmup_trading_days=args.warmup_trading_days,
        )
        # Keep the run configuration visible in the report while reusing the
        # production account/risk defaults loaded by the application.
        report["backtest_initial_capital"] = app_config.account.initial_capital
        for run in runs:
            equity_path = output.with_name(f"{output.stem}.{run.period.name}.equity.csv")
            run.result.equity_curve.to_csv(equity_path, index=False)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = [
            {
                "period": run.period.name,
                "strategy_return": run.result.stats.get("total_return", 0.0),
                "benchmark_return": run.benchmark_return,
                "excess_return": run.excess_return,
                "trades": len(run.result.trades),
            }
            for run in runs
        ]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"report: {output}")
    elif args.command == "sync-data":
        from .config import load_config
        from .data import ProviderUnavailable, build_provider
        from .storage import HistoricalStore

        try:
            frame = build_provider(load_config()).get_daily_frame()
            report = HistoricalStore(args.root).write_daily(frame, args.dataset)
        except ProviderUnavailable as exc:
            raise SystemExit(f"真实数据 API 不可用：{exc}") from exc
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
