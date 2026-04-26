"""LangGraph server entrypoints for local/dev deployment."""

import logging
import sys
from pathlib import Path

# Ensure src/ is importable when LangGraph loads this file directly.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from cognigraph.config import load_settings
from cognigraph.db import initialize_database
from cognigraph.graph import build_graph
from cognigraph.llm import get_llm
from cognigraph.logging_setup import configure_logging

configure_logging()
logging.info("Initializing LangGraph server graphs")

settings = load_settings()
initialize_database()

_llm = get_llm(settings)

# Primary graph to connect from Agent Chat UI.
graph = build_graph(_llm, obsidian_vault_path=settings.obsidian_vault_path)
