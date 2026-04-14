import os
import json
import logging
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_community.tools.tavily_search import TavilySearchResults
from database import save_conversation, load_conversation, initialize_database

# --- Logging Setup ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)

logging.info("Starting CogniGraph application")

# Load environment variables
load_dotenv()
logging.info("Loaded environment variables")

# Get configuration from environment variables
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH")
logging.info(f"LLM Provider: {LLM_PROVIDER}, Model: {LLM_MODEL}")

# --- LLM Abstraction ---
def get_llm():
    """Initializes and returns the appropriate LLM based on the provider."""
    logging.info(f"Initializing LLM with provider: {LLM_PROVIDER}")
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            logging.error("OPENAI_API_KEY is not set.")
            raise ValueError("OPENAI_API_KEY is not set in the .env file.")
        return ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL, temperature=0)
    else: # Default to ollama
        if not LLM_BASE_URL:
            logging.error("LLM_BASE_URL is not set for Ollama.")
            raise ValueError("LLM_BASE_URL is required for the 'ollama' provider.")
        logging.info(f"Using Ollama model: {LLM_MODEL} from {LLM_BASE_URL}")
        return ChatOllama(model=LLM_MODEL, base_url=LLM_BASE_URL, temperature=0)

llm = get_llm()

# --- LangGraph Agent Definition ---

class AgentState(TypedDict):
    messages: Annotated[List[any], operator.add]

# Nodes
def chat_node(state):
    logging.info("Executing chat node")
    return {"messages": [llm.invoke(state["messages"])]}

def search_node(state):
    logging.info(f"Executing search node for query: {state['messages'][-1].content}")
    tavily_tool = TavilySearchResults()
    result = tavily_tool.invoke({"query": state["messages"][-1].content})
    logging.info("Search complete")
    return {"messages": [AIMessage(content=result)]}

# Graph
workflow = StateGraph(AgentState)
workflow.add_node("chat", chat_node)
workflow.add_node("search", search_node)

# Edges
def router(state):
    """
    This function decides the next step based on the user's message.
    If the user is asking a question that requires up-to-date information or external knowledge,
    it routes to the 'search' node. Otherwise, it goes to the 'chat' node.
    """
    logging.info("Executing router")
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at routing user questions. Based on the user's message, decide if you should search the web for more information or if you can answer from your existing knowledge. Respond with only 'search' or 'chat'."),
        ("human", "User message: {message}")
    ])
    
    chain = router_prompt | llm | StrOutputParser()
    result = chain.invoke({"message": state['messages'][-1].content})

    if "search" in result.lower():
        logging.info("Router decision: search")
        return "search"
    else:
        logging.info("Router decision: chat")
        return "chat"


workflow.add_conditional_edges(
    "chat",
    router,
    {"search": "search", "chat": END} # End after chat or search
)
workflow.add_edge('search', 'chat') # After searching, go to chat to formulate response
workflow.set_entry_point("chat")
app = workflow.compile()
logging.info("Graph compiled")


# --- Streamlit Chat Interface ---
st.title("CogniGraph 🧠")

# Initialize database
initialize_database()
logging.info("Database initialized")

# Session state management
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
    logging.info(f"New session started: {st.session_state.session_id}")

if "messages" not in st.session_state:
    # Try to load from DB, otherwise start fresh
    logging.info(f"Loading conversation for session: {st.session_state.session_id}")
    history = load_conversation(st.session_state.session_id)
    if history:
        st.session_state.messages = [AIMessage(**msg) if msg.get('type') == 'ai' else HumanMessage(**msg) for msg in json.loads(history)]
        logging.info("Loaded conversation from database")
    else:
        st.session_state.messages = [AIMessage(content="Hi, I'm CogniGraph. How can I help you learn something new today?")]
        logging.info("Started a new conversation")

# Display chat messages
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Chat input
if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)
    logging.info(f"User input: {prompt}")

    # Invoke graph
    logging.info("Invoking graph")
    response = app.invoke({"messages": [HumanMessage(content=prompt)]})
    ai_response = response['messages'][-1]
    st.session_state.messages.append(ai_response)
    st.chat_message("assistant").write(ai_response.content)
    logging.info(f"AI response: {ai_response.content}")

    # Save conversation to DB
    logging.info("Saving conversation to database")
    save_conversation(st.session_state.session_id, json.dumps([m.dict() for m in st.session_state.messages]))
    logging.info("Conversation saved")

# Summarize and save button
if st.button("End Session & Save Notes"):
    logging.info("'End Session & Save Notes' button clicked")
    
    # Define the template with a placeholder
    summarization_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that summarizes conversations. Your summary should be concise and include key points. Wrap related concepts in double brackets for Obsidian graph view, like [[this]]."),
        ("human", "Please summarize the following conversation:\n\n{conversation_history}")
    ])
    
    summarization_chain = summarization_prompt | llm | StrOutputParser()
    logging.info("Invoking summarization chain")
    
    # Format the history and pass it during invocation
    history_string = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in st.session_state.messages])
    summary = summarization_chain.invoke({"conversation_history": history_string})
    
    logging.info("Summarization complete")

    if OBSIDIAN_VAULT_PATH and os.path.isdir(OBSIDIAN_VAULT_PATH):
        notes_folder = os.path.join(OBSIDIAN_VAULT_PATH, "AINotes")
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
        logging.error(f"Invalid OBSIDIAN_VAULT_PATH: {OBSIDIAN_VAULT_PATH}")

    # Clear session for next conversation
    logging.info("Ending session and clearing state")
    st.session_state.messages = [AIMessage(content="Session ended. Ready for a new topic!")]
    # Correctly serialize messages for JSON
    serializable_messages = [{'type': 'ai', 'content': m.content} if isinstance(m, AIMessage) else {'type': 'human', 'content': m.content} for m in st.session_state.messages]
    save_conversation(st.session_state.session_id, json.dumps(serializable_messages)) # Save initial message
    st.rerun()
