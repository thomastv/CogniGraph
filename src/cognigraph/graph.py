import logging
from typing import Annotated, Any, Callable, List, TypedDict

from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from cognigraph.db import save_preference


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


class PreferenceExtraction(BaseModel):
    key: str | None = Field(default=None, description="Stable preference key in snake_case")
    value: str | None = Field(default=None, description="Preference value")


SavePreferenceFn = Callable[[str, str], None]


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
        return {"messages": [AIMessage(content=summary)]}

    return summarize_node


def build_graph(
    llm,
    save_preference_fn: SavePreferenceFn | None = None,
):
    """Build and compile the LangGraph workflow."""

    tavily_tool = TavilySearch()
    llm_with_tools = llm.bind_tools([tavily_tool])
    preference_llm = llm.with_structured_output(PreferenceExtraction)
    persistence_callback = save_preference_fn or save_preference

    extract_preference_node = make_extract_preference_node(
        preference_llm,
        persistence_callback,
    )
    assistant_node = make_assistant_node(llm_with_tools)
    summarize_node = make_summarize_node(llm)

    def route_after_preference(state):
        messages = _normalize_messages(state.get("messages", []))
        if not messages:
            return "assistant"
        latest = messages[-1]
        if isinstance(latest, HumanMessage) and _wants_summary(_message_content(latest)):
            return "summarize"
        return "assistant"

    workflow = StateGraph(AgentState)
    workflow.add_node("extract_preference", extract_preference_node)
    workflow.add_node("assistant", assistant_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("tools", ToolNode([tavily_tool]))

    workflow.add_conditional_edges(
        "extract_preference",
        route_after_preference,
        {"summarize": "summarize", "assistant": "assistant"},
    )
    workflow.add_conditional_edges(
        "assistant",
        tools_condition,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "assistant")
    workflow.add_edge("summarize", END)
    workflow.set_entry_point("extract_preference")

    app = workflow.compile()
    logging.info("Graph compiled")
    return app
