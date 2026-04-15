"""LangGraph server entrypoints for local/dev deployment."""

import logging

from cognigraph.config import load_settings
from cognigraph.db import initialize_database
from cognigraph.graph import build_graph, build_summary_graph
from cognigraph.llm import get_llm
from cognigraph.logging_setup import configure_logging

configure_logging()
logging.info("Initializing LangGraph server graphs")

settings = load_settings()
initialize_database()

_llm = get_llm(settings)

# Primary graph to connect from Agent Chat UI.
graph = build_graph(_llm)

# Optional secondary graph (useful for internal APIs/tests).
summary_graph = build_summary_graph(_llm)
