# CogniGraph 🧠

CogniGraph is an AI-powered research assistant designed to help you learn new concepts. It provides a conversational interface to explore topics, performs real-time web searches for up-to-date information, and summarizes key points at the end of your session.

A key feature of CogniGraph is its integration with Obsidian. It automatically saves summarized notes to your vault, wrapping related concepts in `[[double brackets]]` to leverage Obsidian's powerful graph view and create a connected knowledge base.

This project is built using LangGraph, Streamlit, and can be configured to use different Large Language Models (LLMs) like local models via Ollama or proprietary models from OpenAI.

## Architecture Diagram

The following diagram illustrates the flow of information within the CogniGraph agent:

```mermaid
graph TD
    subgraph "User Interaction (Streamlit UI)"
        A[User Input] --> B{LangGraph Agent};
        B --> C[Display Response];
    end

    subgraph "LangGraph Core Logic"
        B --> D(chat: Entry Point);
        D --> E{router};
        E -- Needs Web Search --> F[search: Tavily API];
        F --> G[chat: Formulate Response w/ LLM];
        E -- Direct Answer --> G;
        G --> B;
    end

    subgraph "Data & Persistence"
        B --> H[(SQLite DB<br>User Preferences)];
        I[End Session Button] --> J{Summarization Chain w/ LLM};
        J --> K([Obsidian Vault<br>Save as .md]);
    end

    style A fill:#cde4ff
    style C fill:#cde4ff
    style I fill:#ffcdd2
    style K fill:#d4edda
    style H fill:#fff2cc
    style E fill:#e0cffc

```

## Features

- **Conversational AI**: Engage in a natural conversation to ask questions and learn.
- **LLM Agnostic**: Easily switch between a locally hosted Ollama model (e.g., Gemma, Llama) and OpenAI's models (e.g., GPT-4o) via a simple configuration change.
- **Web Search**: Integrates with Tavily Search API to provide current information on any topic.
- **Intelligent Routing**: The agent decides whether to answer from its existing knowledge or perform a web search.
- **Automated Summarization**: At the end of a session, the agent summarizes the entire conversation, highlighting key takeaways.
- **Obsidian Integration**: Automatically saves summaries as Markdown files in a specified Obsidian vault, creating links between concepts for graph visualization.
- **Persistent Memory**: Stable user preferences are extracted and stored as key-value pairs in a local SQLite database.
- **Simple UI**: A clean and straightforward chat interface built with Streamlit.
- **Logging**: Detailed logs are generated in the `logs/` directory for easy debugging and monitoring.

## Project Structure

```
.
├── src/
│   └── cognigraph/
│       ├── ui.py           # Streamlit UI + app workflow
│       ├── graph.py        # LangGraph workflow
│       ├── llm.py          # LLM provider factory
│       ├── preferences.py  # Preference extraction logic
│       ├── db.py           # SQLite persistence layer
│       ├── config.py       # Environment config loader
│       └── logging_setup.py
├── app.py                  # Thin Streamlit entrypoint
├── pyproject.toml          # uv project configuration
├── .env / .env.example     # Environment variables
├── logs/                   # Log files
└── README.md               # This file
```

## Setup and Installation

1.  **Prerequisites**:
    *   Python 3.9+
    *   An active internet connection
    *   (Optional) [Ollama](https://ollama.com/) installed and running for local LLM usage.

2.  **Clone the Repository**:

3.  **Install uv**:
    Follow the official instructions: https://docs.astral.sh/uv/getting-started/installation/

4.  **Create the Environment and Install Dependencies**:

    ```bash
    uv sync
    ```

5.  **Configure Environment Variables**:
    Create a file named `.env` in the root of the project directory and populate it with your configuration. A template is provided below.

## Configuration (`.env` file)

Copy the following into your `.env` file and replace the placeholder values with your actual information.

```ini
# --- LLM Configuration ---
# Set the provider: "ollama", "openai", etc.
LLM_PROVIDER="ollama" 
# Set the model name for the selected provider (e.g., "gemma", "gpt-4o")
LLM_MODEL="gemma"
# Set the base URL for the LLM API (required for local models like Ollama)
LLM_BASE_URL="http://localhost:11434"

# --- API Keys and Paths ---
# Required if using LLM_PROVIDER="openai"
OPENAI_API_KEY="your-openai-api-key"
# Required for web search functionality
TAVILY_API_KEY="your-tavily-api-key"
# Absolute path to your Obsidian vault's root directory
OBSIDIAN_VAULT_PATH="C:/Users/YourUser/Documents/ObsidianVault"
```

**Important**:
- You can get a free Tavily API key from the [Tavily website](https://tavily.com/).
- Ensure the `OBSIDIAN_VAULT_PATH` is an absolute path to your vault's root directory.

## Usage

1.  **Install dependencies** (if not already done):
    ```bash
    uv sync
    ```

2.  **(Optional) Start Ollama**: If you are using `LLM_PROVIDER="ollama"`, make sure your Ollama application is running and the specified model (`gemma` by default) is downloaded.
    ```bash
    ollama run gemma
    ```

3.  **Run the Application**:
    Start the Streamlit application from your terminal.

    ```bash
    uv run streamlit run app.py
    ```

4.  **Interact with the Agent**:
    - Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.
    - Type your questions into the chat input at the bottom of the page.

5.  **Save Notes**:
    - When you are finished with a topic, click the **"End Session & Save Notes"** button.
    - The agent will summarize the conversation and save it as a new Markdown file in the `AINotes` folder inside your specified Obsidian vault.

## Logging

All application events, including API calls, node executions, and errors, are logged to `logs/app.log`. This is the first place to check if you encounter any issues.
