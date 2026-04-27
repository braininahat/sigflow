"""NodeGraphQt visual pipeline editor entry point."""
import logging
import os
import sys

from PySide6.QtWidgets import QApplication

from sigflow_editor.window import EditorWindow

log = logging.getLogger(__name__)


def main():
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("sigflow_editor starting (log_level=%s)", level_name)

    log.info("creating QApplication...")
    app = QApplication(sys.argv)
    log.info("QApplication ready (platform=%s)", app.platformName())

    log.info("constructing EditorWindow...")
    window = EditorWindow()
    log.info("EditorWindow constructed; showing...")
    window.show()
    log.info("window.show() called (visible=%s, geom=%s); entering Qt event loop",
             window.isVisible(), window.geometry().getRect())
    rc = app.exec()
    log.info("Qt event loop exited (rc=%d)", rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
