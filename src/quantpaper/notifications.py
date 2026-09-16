from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class NotificationSink(Protocol):
    def notify(self, title: str, message: str, severity: str = "info") -> None: ...


class NoopNotifier:
    def notify(self, title: str, message: str, severity: str = "info") -> None:
        return None


class ConsoleNotifier:
    def notify(self, title: str, message: str, severity: str = "info") -> None:
        logger.info("notification[%s] %s: %s", severity, title, message)


class WindowsToastNotifier:
    """Best-effort Windows Toast adapter with a safe logging fallback."""

    def notify(self, title: str, message: str, severity: str = "info") -> None:
        try:
            from winotify import Notification

            toast = Notification(app_id="A股量化纸盘", title=title, msg=message)
            toast.show()
        except Exception as exc:  # optional dependency or non-Windows environment
            logger.warning("Windows Toast unavailable: %s", exc)
            logger.info("notification[%s] %s: %s", severity, title, message)


def build_notifier(enabled: bool) -> NotificationSink:
    if not enabled:
        return NoopNotifier()
    return WindowsToastNotifier()
