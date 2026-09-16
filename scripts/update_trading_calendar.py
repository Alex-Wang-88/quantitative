"""Refresh the cached A-share trading calendar from the configured real provider."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="更新本地交易日历缓存")
    parser.add_argument("--output", default="data/runtime/trading_calendar.json")
    parser.add_argument("--back-days", type=int, default=30)
    parser.add_argument("--forward-days", type=int, default=400)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.back_days < 0 or args.forward_days < 30:
        print(
            "back-days must be non-negative and forward-days must be at least 30",
            file=sys.stderr,
        )
        return 2

    # Imports are intentionally delayed so --help works even if the project
    # environment is being repaired.
    from quantpaper.config import load_config
    from quantpaper.data import ProviderUnavailable, build_provider

    now = datetime.now(SHANGHAI)
    start = now.date() - timedelta(days=args.back_days)
    end = now.date() + timedelta(days=args.forward_days)
    try:
        provider = build_provider(load_config())
        dates = sorted(
            {
                value.isoformat()
                for value in provider.get_trading_calendar(start=start, end=end)
                if isinstance(value, date)
            }
        )
    except ProviderUnavailable as exc:
        print(f"真实交易日历不可用：{exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"交易日历更新失败：{exc}", file=sys.stderr)
        return 4

    if not dates:
        print("真实交易日历返回为空，拒绝覆盖已有缓存", file=sys.stderr)
        return 5

    project_root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": now.isoformat(),
        "provider": getattr(provider, "name", "unknown"),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "dates": dates,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {"output": str(output), "count": len(dates), "updated_at": payload["updated_at"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
