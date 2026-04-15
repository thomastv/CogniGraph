import logging
import operator
from typing import Annotated, List, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    messages: Annotated[List[any], operator.add]


def build_graph(llm):
    """Build and compile the LangGraph workflow."""

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
    workflow.add_node("chat", chat_node)
    workflow.add_node("search", search_node)

    workflow.add_conditional_edges(
        "chat",
        router,
        {"search": "search", "chat": END},
    )
    workflow.add_edge("search", "chat")
    workflow.set_entry_point("chat")

    app = workflow.compile()
    logging.info("Graph compiled")
    return app
