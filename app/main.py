import logging
import logging.handlers
import sys
import json
from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

# Initialise locale before any UI imports that call t()
import i18n_setup

_cfg_path = Path(__file__).resolve().parent.parent / "config" / "app.json"
try:
    with open(_cfg_path, "r", encoding="utf-8") as _f:
        _locale = json.load(_f).get("locale", "en")
except (OSError, json.JSONDecodeError):
    _locale = "en"
i18n_setup.setup(_locale)

from app.main_window import MainWindow


def _setup_logging() -> None:
    """Configure root logger: console + optional rotating file."""
    _app_root = Path(__file__).resolve().parent.parent
    log_level_name = "INFO"
    log_file = ""
    log_file_level_name = ""
    try:
        with open(_cfg_path, "r", encoding="utf-8") as _f:
            cfg = json.load(_f)
            log_level_name = str(cfg.get("log_level", "INFO")).upper()
            log_file = cfg.get("log_file", "")
            log_file_level_name = str(cfg.get("log_file_level", "")).upper()
    except (OSError, json.JSONDecodeError):
        pass
    log_level = getattr(logging, log_level_name, logging.WARNING)
    log_file_level = getattr(logging, log_file_level_name, None) if log_file_level_name else None

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Root logger level = minimum of console and file levels
    root = logging.getLogger()
    effective_min = min(log_level, log_file_level) if log_file_level is not None else log_level
    root.setLevel(effective_min)

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(log_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler (rotating, 1 MB × 3 backups)
    if log_file:
        log_path = _app_root / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_h = logging.handlers.RotatingFileHandler(
            str(log_path), maxBytes=1_000_000, backupCount=3, encoding="utf-8",
        )
        file_h.setLevel(log_file_level if log_file_level is not None else log_level)
        file_h.setFormatter(fmt)
        root.addHandler(file_h)


class _StartupWarningCollector(logging.Handler):
    """Collect WARNING+ records emitted while MainWindow constructs
    (config / calibration / lamp DB / tube_params loads) so they can be
    surfaced in the status-bar warning indicator — per the failure-visibility
    rule, the log alone is not a user-facing channel."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def main() -> int:
    _setup_logging()
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    collector = _StartupWarningCollector()
    logging.getLogger().addHandler(collector)
    try:
        window = MainWindow()
    finally:
        logging.getLogger().removeHandler(collector)
    window.set_startup_warnings(collector.messages)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
