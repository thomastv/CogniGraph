import logging
import os
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from cognigraph.config import load_settings
from cognigraph.db import initialize_database
from cognigraph.graph import build_graph
from cognigraph.llm import get_llm
from cognigraph.logging_setup import configure_logging
from cognigraph.preferences import extract_and_save_preference


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
    app = build_graph(llm)

    st.title("CogniGraph 🧠")

    initialize_database()
    logging.info("Database initialized")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logging.info(f"New session started: {st.session_state.session_id}")

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
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)
        logging.info(f"User input: {prompt}")

        extract_and_save_preference(llm, prompt)

        logging.info("Invoking graph")
        response = app.invoke({"messages": [HumanMessage(content=prompt)]})
        ai_response = response["messages"][-1]
        st.session_state.messages.append(ai_response)
        st.chat_message("assistant").write(ai_response.content)
        logging.info(f"AI response: {ai_response.content}")

    if st.button("End Session & Save Notes"):
        logging.info("'End Session & Save Notes' button clicked")

        summarization_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that summarizes conversations. "
                    "Your summary should be concise and include key points. "
                    "Wrap related concepts in double brackets for Obsidian graph view, like [[this]].",
                ),
                ("human", "Please summarize the following conversation:\n\n{conversation_history}"),
            ]
        )

        summarization_chain = summarization_prompt | llm | StrOutputParser()
        logging.info("Invoking summarization chain")

        history_string = "\n".join(
            [
                f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in st.session_state.messages
            ]
        )
        summary = summarization_chain.invoke({"conversation_history": history_string})

        logging.info("Summarization complete")

        if settings.obsidian_vault_path and os.path.isdir(settings.obsidian_vault_path):
            notes_folder = os.path.join(settings.obsidian_vault_path, "AINotes")
            os.makedirs(notes_folder, exist_ok=True)
            file_name = f"CogniGraph_Summary_{st.session_state.session_id[:8]}.md"
            file_path = os.path.join(notes_folder, file_name)
            logging.info(f"Saving summary to: {file_path}")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(summary)
            st.success(f"Summary saved to {file_path}")
            logging.info("Summary saved successfully")
        else:
            st.error("OBSIDIAN_VAULT_PATH is not a valid directory. Please check your .env file.")
            logging.error(f"Invalid OBSIDIAN_VAULT_PATH: {settings.obsidian_vault_path}")

        logging.info("Ending session and clearing state")
        st.session_state.messages = [AIMessage(content="Session ended. Ready for a new topic!")]
        st.rerun()
