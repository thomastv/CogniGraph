import logging
from typing import Annotated, Any, List, TypedDict

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


def build_graph(llm):
    """Build and compile the LangGraph workflow."""

    tavily_tool = TavilySearch()
    llm_with_tools = llm.bind_tools([tavily_tool])
    preference_llm = llm.with_structured_output(PreferenceExtraction)

    def extract_preference_node(state):
        """Extract and persist stable user preferences from the latest user message."""
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
        try:
            extracted = chain.invoke({"message": user_message})
        except Exception as exc:
            logging.warning(f"Preference extraction failed: {exc}")
            return {}

        key = (extracted.key or "").strip() if extracted else ""
        value = (extracted.value or "").strip() if extracted else ""
        if key and value:
            save_preference(key, value)
            logging.info(f"Saved user preference: {key}={value}")

        return {}

    def assistant_node(state):
        logging.info("Executing assistant node")
        if not state.get("messages"):
            return {"messages": [AIMessage(content="Please send a message to begin.")]}

        messages = _normalize_messages(state["messages"])
        return {"messages": [llm_with_tools.invoke(messages)]}

    workflow = StateGraph(AgentState)
    workflow.add_node("extract_preference", extract_preference_node)
    workflow.add_node("assistant", assistant_node)
    workflow.add_node("tools", ToolNode([tavily_tool]))

    workflow.add_edge("extract_preference", "assistant")
    workflow.add_conditional_edges(
        "assistant",
        tools_condition,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "assistant")
    workflow.set_entry_point("extract_preference")

    app = workflow.compile()
    logging.info("Graph compiled")
    return app


class SummaryState(TypedDict):
    conversation_history: str
    summary: str


def build_summary_graph(llm):
    """Build and compile a graph dedicated to session summarization."""

    def summarize_node(state: SummaryState):
        logging.info("Executing summarize node")
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
        summary = summarization_chain.invoke(
            {"conversation_history": state["conversation_history"]}
        )
        return {"summary": summary}

    workflow = StateGraph(SummaryState)
    workflow.add_node("summarize", summarize_node)
    workflow.add_edge("summarize", END)
    workflow.set_entry_point("summarize")

    summary_app = workflow.compile()
    logging.info("Summary graph compiled")
    return summary_app
