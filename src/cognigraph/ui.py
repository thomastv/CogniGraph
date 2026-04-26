import logging
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from cognigraph.config import load_settings
from cognigraph.db import initialize_database
from cognigraph.graph import build_graph
from cognigraph.llm import get_llm
from cognigraph.logging_setup import configure_logging


def render_app() -> None:
    """Render Streamlit UI and run the app workflow."""
    configure_logging()
    logging.info("Starting CogniGraph application")

    settings = load_settings()
    logging.info("Loaded environment variables")
    logging.info(
        f"LLM Provider: {settings.llm_provider}, Model: {settings.llm_model}"
    )

    llm = get_llm(settings)
    app = build_graph(
        llm,
        obsidian_vault_path=settings.obsidian_vault_path,
        use_inmemory_checkpointer=True,
    )

    st.title("CogniGraph 🧠")

    initialize_database()
    logging.info("Database initialized")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logging.info(f"New session started: {st.session_state.session_id}")

    if "graph_config" not in st.session_state:
        st.session_state.graph_config = {
            "configurable": {"thread_id": st.session_state.session_id}
        }

    if "awaiting_save_confirmation" not in st.session_state:
        st.session_state.awaiting_save_confirmation = False

    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="Hi, I'm CogniGraph. How can I help you learn something new today?")
        ]
        logging.info("Started a new conversation")

    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

    if prompt := st.chat_input():
        is_resume = st.session_state.awaiting_save_confirmation
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)
        logging.info(f"User input: {prompt}")

        logging.info("Invoking graph")
        if is_resume:
            response = app.invoke(
                Command(resume=prompt),
                config=st.session_state.graph_config,
            )
            st.session_state.awaiting_save_confirmation = False
        else:
            response = app.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=st.session_state.graph_config,
            )

        if "messages" in response and response["messages"]:
            st.session_state.messages = response["messages"]
            ai_response = st.session_state.messages[-1]
            if isinstance(ai_response, AIMessage):
                st.chat_message("assistant").write(ai_response.content)
                logging.info(f"AI response: {ai_response.content}")

        if response.get("__interrupt__"):
            st.session_state.awaiting_save_confirmation = True
            interrupt_prompt = (
                "Do you want me to save this summary to your Obsidian vault? "
                "Reply with yes or no."
            )
            st.session_state.messages.append(AIMessage(content=interrupt_prompt))
            st.chat_message("assistant").write(interrupt_prompt)
            logging.info("Waiting for user confirmation to save summary")

    if st.button("End Session & Save Notes"):
        logging.info("'End Session & Save Notes' button clicked")

        logging.info("Invoking unified graph summarization")
        summary_response = app.invoke(
            {"messages": [HumanMessage(content="/summarize")]},
            config=st.session_state.graph_config,
        )

        if "messages" in summary_response and summary_response["messages"]:
            st.session_state.messages = summary_response["messages"]
            ai_response = st.session_state.messages[-1]
            if isinstance(ai_response, AIMessage):
                st.chat_message("assistant").write(ai_response.content)
                logging.info(f"Summary response: {ai_response.content}")

        if summary_response.get("__interrupt__"):
            st.session_state.awaiting_save_confirmation = True
            interrupt_prompt = (
                "Do you want me to save this summary to your Obsidian vault? "
                "Reply with yes or no."
            )
            st.session_state.messages.append(AIMessage(content=interrupt_prompt))
            st.chat_message("assistant").write(interrupt_prompt)
            logging.info("Waiting for user confirmation to save summary")
