from __future__ import annotations

import json

from quantpaper.config import AppConfig, ModelConfig, NotificationConfig
from quantpaper.data import make_demo_provider
from quantpaper.ledger import Ledger
from quantpaper.runtime import RuntimeService


def test_walkforward_metrics_are_attached_to_close_report(tmp_path) -> None:
    walkforward_path = tmp_path / "walkforward.json"
    walkforward_path.write_text(
        json.dumps(
            {
                "stats": {"fold_count": 2, "total_return": 0.12, "sharpe": 1.1},
                "quality": {
                    "ok": True,
                    "start_date": "2023-01-01",
                    "end_date": "2026-01-01",
                },
                "folds": [
                    {
                        "model_version": "fold-a",
                        "metrics": {
                            "test_ic_mean": 0.04,
                            "test_top_bottom_spread": 0.01,
                            "factor_coverage_mean": 0.2,
                        },
                    },
                    {
                        "model_version": "fold-b",
                        "metrics": {
                            "test_ic_mean": -0.02,
                            "test_top_bottom_spread": 0.03,
                            "factor_coverage_mean": 0.4,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    service = RuntimeService(
        config=AppConfig(
            model=ModelConfig(walkforward_report_path=str(walkforward_path)),
            notifications=NotificationConfig(enabled=False),
        ),
        provider=make_demo_provider(quote_steps=2),
        ledger=Ledger(tmp_path / "paper.db"),
    )
    service.report_generator.output_dir = tmp_path / "reports"
    service.report_generator.output_dir.mkdir(parents=True, exist_ok=True)

    report = service.generate_report(notify=False, sync=False)

    metrics = report["research_metrics"]
    assert metrics["status"] == "OK"
    assert metrics["test_ic_mean"] == 0.01
    assert metrics["test_ic_positive_ratio"] == 0.5
    assert metrics["test_top_bottom_spread_mean"] == 0.02
    assert metrics["factor_coverage_mean"] == 0.3
    assert metrics["last_model_version"] == "fold-b"
    assert "样本外测试 IC" in (tmp_path / "reports" / f"{report['report_date']}.md").read_text(
        encoding="utf-8"
    )
