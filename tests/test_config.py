from __future__ import annotations

from pathlib import Path

import pytest

from quantpaper.config import PortfolioConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_min_positions_may_not_exceed_max_positions() -> None:
    """A floor above the ceiling is unsatisfiable, not stricter.

    This was accepted silently for as long as the floor was unread, so the
    invariant the field implied was never actually a rule.
    """

    with pytest.raises(ValueError, match="must not exceed"):
        PortfolioConfig(min_positions=30, max_positions=10)


def test_min_positions_may_not_be_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PortfolioConfig(min_positions=-1)


def test_min_positions_defaults_to_no_floor() -> None:
    """Research configs run deliberately small books (``max_positions = 6``).

    A non-zero default would make those configs invalid on construction, which is
    why the live floor lives in ``default.toml`` instead.
    """

    assert PortfolioConfig().min_positions == 0
    assert PortfolioConfig(max_positions=6).min_positions == 0


def test_shipped_default_profile_declares_a_satisfiable_floor() -> None:
    """The live profile is the one that must carry a real floor."""

    config = load_config(REPO_ROOT / "config" / "default.toml")

    assert config.portfolio.min_positions == 20
    assert config.portfolio.min_positions <= config.portfolio.max_positions
