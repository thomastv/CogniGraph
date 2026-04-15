import logging
import operator
from typing import Annotated, List, TypedDict
import json

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from cognigraph.db import save_preference


class AgentState(TypedDict):
    messages: Annotated[List[any], operator.add]


def build_graph(llm):
    """Build and compile the LangGraph workflow."""

    def extract_preference_node(state):
        """Extract and persist a stable user preference from the latest message."""
        logging.info("Executing extract_preference node")
        user_message = state["messages"][-1].content

        extraction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extract stable user preferences/facts from the user message. "
                    "If present, return ONLY JSON in this format: {\"key\":\"snake_case_key\",\"value\":\"value\"}. "
                    "If nothing is worth saving, return {}.",
                ),
                ("human", "User message: {message}"),
            ]
        )

        chain = extraction_prompt | llm | StrOutputParser()
        raw_output = chain.invoke({"message": user_message})

        cleaned_output = raw_output.strip()
        if "```json" in cleaned_output:
            cleaned_output = cleaned_output.split("```json", 1)[1].split("```", 1)[0].strip()
        elif cleaned_output.startswith("```"):
            cleaned_output = cleaned_output.split("```", 1)[1].rsplit("```", 1)[0].strip()

        try:
            extracted = json.loads(cleaned_output)
            if isinstance(extracted, dict) and extracted.get("key") and extracted.get("value"):
                key = str(extracted["key"]).strip()
                value = str(extracted["value"]).strip()
                if key and value:
                    save_preference(key, value)
                    logging.info(f"Saved user preference: {key}={value}")
        except json.JSONDecodeError:
            logging.info("No valid preference JSON extracted from this message")

        return {}

    def chat_node(state):
        logging.info("Executing chat node")
        return {"messages": [llm.invoke(state["messages"])]}

    def search_node(state):
        logging.info(f"Executing search node for query: {state['messages'][-1].content}")
        tavily_tool = TavilySearchResults()
        result = tavily_tool.invoke({"query": state["messages"][-1].content})
        logging.info("Search complete")
        return {"messages": [AIMessage(content=result)]}

    def router(state):
        """Route message to search or direct chat response."""
        logging.info("Executing router")

        router_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at routing user questions. "
                    "Based on the user's message, decide if you should search the web "
                    "for more information or if you can answer from your existing knowledge. "
                    "Respond with only 'search' or 'chat'.",
                ),
                ("human", "User message: {message}"),
            ]
        )

        chain = router_prompt | llm | StrOutputParser()
        result = chain.invoke({"message": state["messages"][-1].content})

        if "search" in result.lower():
            logging.info("Router decision: search")
            return "search"

        logging.info("Router decision: chat")
        return "chat"

    workflow = StateGraph(AgentState)
    workflow.add_node("extract_preference", extract_preference_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("search", search_node)

    workflow.add_edge("extract_preference", "chat")
    workflow.add_conditional_edges(
        "chat",
        router,
        {"search": "search", "chat": END},
    )
    workflow.add_edge("search", "chat")
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
