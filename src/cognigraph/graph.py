import logging
import os
from datetime import datetime
from typing import Annotated, Any, Callable, List, NotRequired, TypedDict

from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from cognigraph.db import save_preference


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    summary_text: NotRequired[str]
    save_summary_approved: NotRequired[bool]
    saved_summary_path: NotRequired[str]


class PreferenceExtraction(BaseModel):
    key: str | None = Field(default=None, description="Stable preference key in snake_case")
    value: str | None = Field(default=None, description="Preference value")


SavePreferenceFn = Callable[[str, str], None]
SaveSummaryFn = Callable[[str], str]


SUMMARY_COMMANDS = (
    "/summarize",
    "/summary",
    "summarize this conversation",
    "summarise this conversation",
    "summarize this chat",
    "summarise this chat",
)


def _message_content(message: Any) -> str:
    """Extract message content from dict-style or message-object inputs."""
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _normalize_messages(messages: List[Any]) -> List[AnyMessage]:
    """Convert incoming dict-style messages to LangChain message objects."""
    normalized: List[AnyMessage] = []
    for message in messages:
        if isinstance(message, dict):
            role = (message.get("role") or message.get("type") or "user").lower()
            content = str(message.get("content", ""))
            if role in ("assistant", "ai"):
                normalized.append(AIMessage(content=content))
            elif role == "system":
                normalized.append(SystemMessage(content=content))
            else:
                normalized.append(HumanMessage(content=content))
            continue
        if isinstance(message, BaseMessage):
            normalized.append(message)
            continue
        normalized.append(HumanMessage(content=str(message)))
    return normalized


def _wants_summary(message_text: str) -> bool:
    """Heuristic trigger for summarization requests from any chat client."""
    normalized = message_text.strip().lower()
    if not normalized:
        return False
    if normalized in SUMMARY_COMMANDS:
        return True
    if "summar" in normalized and any(
        token in normalized for token in ("conversation", "chat", "session", "so far")
    ):
        return True
    return False


def _conversation_history_from_messages(messages: List[AnyMessage]) -> str:
    """Build a plain-text transcript suitable for LLM summarization prompts."""
    working_messages = list(messages)
    if working_messages:
        latest = working_messages[-1]
        if isinstance(latest, HumanMessage) and _wants_summary(_message_content(latest)):
            working_messages = working_messages[:-1]

    history_lines: List[str] = []
    for message in working_messages:
        content = _message_content(message).strip()
        if not content:
            continue
        if isinstance(message, HumanMessage):
            history_lines.append(f"User: {content}")
        elif isinstance(message, AIMessage):
            history_lines.append(f"AI: {content}")

    return "\n".join(history_lines)


def make_obsidian_summary_saver(obsidian_vault_path: str | None) -> SaveSummaryFn:
    """Create a callback that saves summaries to an Obsidian vault."""

    def save_summary_to_obsidian(summary: str) -> str:
        if not obsidian_vault_path or not os.path.isdir(obsidian_vault_path):
            raise ValueError("OBSIDIAN_VAULT_PATH is not configured or is not a valid directory.")

        notes_folder = os.path.join(obsidian_vault_path, "AINotes")
        os.makedirs(notes_folder, exist_ok=True)
        file_name = f"CogniGraph_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        file_path = os.path.join(notes_folder, file_name)

        with open(file_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(summary)

        return file_path

    return save_summary_to_obsidian


def make_extract_preference_node(
    preference_llm,
    save_preference_fn: SavePreferenceFn,
):
    """Create a node function that extracts and stores durable user preferences."""

    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract stable user preferences/facts from the user message. "
                "Return a key/value pair only if the message includes durable preferences "
                "that will likely matter in future conversations. "
                "If nothing should be saved, return null fields.",
            ),
            ("human", "User message: {message}"),
        ]
    )

    chain = extraction_prompt | preference_llm

    def extract_preference_node(state):
        logging.info("Executing extract_preference node")
        if not state.get("messages"):
            logging.info("No messages found for preference extraction")
            return {}

        messages = _normalize_messages(state["messages"])
        latest_message = messages[-1]
        if not isinstance(latest_message, HumanMessage):
            logging.info("Latest message is not from the user; skipping extraction")
            return {}

        user_message = _message_content(latest_message).strip()
        if not user_message:
            logging.info("Latest user message is empty; skipping extraction")
            return {}

        try:
            extracted = chain.invoke({"message": user_message})
        except Exception as exc:
            logging.warning(f"Preference extraction failed: {exc}")
            return {}

        key = (extracted.key or "").strip() if extracted else ""
        value = (extracted.value or "").strip() if extracted else ""
        if key and value:
            save_preference_fn(key, value)
            logging.info(f"Saved user preference: {key}={value}")

        return {}

    return extract_preference_node


def make_assistant_node(llm_with_tools):
    """Create a node function that drives LLM responses with bound tools."""

    def assistant_node(state):
        logging.info("Executing assistant node")
        if not state.get("messages"):
            return {"messages": [AIMessage(content="Please send a message to begin.")]}

        messages = _normalize_messages(state["messages"])
        return {"messages": [llm_with_tools.invoke(messages)]}

    return assistant_node


def make_summarize_node(llm):
    """Create a node that summarizes the in-graph conversation history."""
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

    def summarize_node(state):
        logging.info("Executing summarize node")
        if not state.get("messages"):
            return {"messages": [AIMessage(content="No conversation found to summarize yet.")]}

        messages = _normalize_messages(state["messages"])
        conversation_history = _conversation_history_from_messages(messages)
        if not conversation_history:
            return {
                "messages": [
                    AIMessage(content="I need more conversation context before I can summarize.")
                ]
            }

        summary = summarization_chain.invoke(
            {"conversation_history": conversation_history}
        )
        return {
            "messages": [AIMessage(content=summary)],
            "summary_text": summary,
        }

    return summarize_node


def make_confirm_save_summary_node():
    """Create a HITL node asking whether summary should be saved to Obsidian."""

    def confirm_save_summary_node(state):
        summary_text = str(state.get("summary_text", "")).strip()
        if not summary_text:
            return {"save_summary_approved": False}

        decision = interrupt(
            {
                "kind": "confirm_save_summary",
                "question": "Do you want me to save this summary to your Obsidian vault?",
                "options": ["yes", "no"],
                "default": "no",
            }
        )
        approved = str(decision).strip().lower() in {"y", "yes", "save", "true", "1"}
        if approved:
            return {"save_summary_approved": True}

        return {
            "save_summary_approved": False,
            "messages": [AIMessage(content="Okay, I will not save this summary to Obsidian.")],
        }

    return confirm_save_summary_node


def make_save_summary_node(save_summary_fn: SaveSummaryFn):
    """Create a node that persists summary text after human approval."""

    def save_summary_node(state):
        summary_text = str(state.get("summary_text", "")).strip()
        if not summary_text:
            return {
                "messages": [AIMessage(content="No summary text available to save.")],
            }

        try:
            file_path = save_summary_fn(summary_text)
        except Exception as exc:
            logging.warning(f"Summary save failed: {exc}")
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I could not save the summary to Obsidian. "
                            "Please verify OBSIDIAN_VAULT_PATH and try again."
                        )
                    )
                ]
            }

        return {
            "saved_summary_path": file_path,
            "messages": [AIMessage(content=f"Summary saved to {file_path}")],
        }

    return save_summary_node


def build_graph(
    llm,
    save_preference_fn: SavePreferenceFn | None = None,
    save_summary_fn: SaveSummaryFn | None = None,
    obsidian_vault_path: str | None = None,
    use_inmemory_checkpointer: bool = False,
):
    """Build and compile the LangGraph workflow."""

    tavily_tool = TavilySearch()
    llm_with_tools = llm.bind_tools([tavily_tool])
    preference_llm = llm.with_structured_output(PreferenceExtraction)
    persistence_callback = save_preference_fn or save_preference
    summary_save_callback = save_summary_fn or make_obsidian_summary_saver(obsidian_vault_path)

    extract_preference_node = make_extract_preference_node(
        preference_llm,
        persistence_callback,
    )
    assistant_node = make_assistant_node(llm_with_tools)
    summarize_node = make_summarize_node(llm)
    confirm_save_summary_node = make_confirm_save_summary_node()
    save_summary_node = make_save_summary_node(summary_save_callback)

    def route_after_assistant(state):
        messages = _normalize_messages(state.get("messages", []))
        if not messages:
            return END

        latest = messages[-1]
        latest_human = next(
            (message for message in reversed(messages) if isinstance(message, HumanMessage)),
            None,
        )

        if latest_human and _wants_summary(_message_content(latest_human)):
            return "summarize"

        if isinstance(latest, AIMessage) and getattr(latest, "tool_calls", None):
            return "tools"

        return END

    def route_after_confirm_save(state):
        if state.get("save_summary_approved"):
            return "save_summary"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("extract_preference", extract_preference_node)
    workflow.add_node("assistant", assistant_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("confirm_save_summary", confirm_save_summary_node)
    workflow.add_node("save_summary", save_summary_node)
    workflow.add_node("tools", ToolNode([tavily_tool]))

    workflow.add_edge("extract_preference", "assistant")
    workflow.add_conditional_edges(
        "assistant",
        route_after_assistant,
        {"summarize": "summarize", "tools": "tools", END: END},
    )
    workflow.add_edge("tools", "assistant")
    workflow.add_edge("summarize", "confirm_save_summary")
    workflow.add_conditional_edges(
        "confirm_save_summary",
        route_after_confirm_save,
        {"save_summary": "save_summary", END: END},
    )
    workflow.add_edge("save_summary", END)
    workflow.set_entry_point("extract_preference")

    if use_inmemory_checkpointer:
        app = workflow.compile(checkpointer=MemorySaver())
    else:
        app = workflow.compile()
    logging.info("Graph compiled")
    return app
