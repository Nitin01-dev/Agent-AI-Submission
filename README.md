# AI Research Agent (RAG + LangGraph)

This is a Python-based AI agent designed to answer complex questions by retrieving information from the web (Wikipedia) and processing it through a structured reasoning graph.

## 🚀 Key Features
* **RAG Pipeline**: Uses `ChromaDB` to store and retrieve real-time data, ensuring the AI doesn't hallucinate.
* **LangGraph Orchestration**: Instead of a simple prompt, this uses a state-driven graph with specialized nodes for *Searching* and *Answering*.
* **Observability**: Integrated with `LangSmith` for real-time tracing of every step the agent takes.

## 🛠️ Setup Instructions
1.  **Environment**: Ensure you are using Python 3.12 (on macOS).
2.  **Install Dependencies**:
    ```bash
    pip install langchain-openai langchain-chroma langgraph langsmith beautifulsoup4 python-dotenv
    ```
3.  **API Keys**: Set your `OPENAI_API_KEY` and `LANGCHAIN_API_KEY` at the top of the `main.py` file.
4.  **Run the Agent**:
    ```bash
    python main.py
    ```

## 📂 Project Structure
* `main.py`: The core application logic.
* `agent_db/`: Local folder where the Chroma vector database is stored.
* `.env`: (Optional) File for managing environment variables.