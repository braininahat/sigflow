"""Allow running as: uv run python -m sigflow_editor"""
import logging

log = logging.getLogger(__name__)

from sigflow_editor.app import main
main()
