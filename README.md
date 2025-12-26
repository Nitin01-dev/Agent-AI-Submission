# AI Agent Workflow with LangGraph & LangSmith

This project implements a stateful AI agent architecture using **LangGraph** for production logic and **LangSmith** for automated performance benchmarking. This is a Python-based AI agent designed to answer complex questions by retrieving information from the web (Langchain Documentation) and processing it through a structured reasoning graph.

---------
## 🏗️ Core Components

### 1. Production Engine (`RAG_agent.py`)
This script handles the live user interaction.
* **Architecture:** Uses a `StateGraph` to manage the agent's lifecycle and decision-making process.
* **Capability:** Maintains conversation state and processes technical queries dynamically.
* **How to run:** ```bash
    python RAG_agent.py
    ```

### 2. QA & Evaluation Suite (`Evaluator_agent.py`)
This script acts as the automated testing laboratory to ensure the agent is performing accurately. By running the Evaluator_agent.py script, you can view detailed traces and scoring metrics in your LangSmith Dashboard. This allows for iterative improvement of the agent's prompt logic and retrieval accuracy.
* **Benchmarking:** Connects to **LangSmith** to run experiments against a "Ground Truth" dataset.
* **Dataset:** A curated set of 5–10 high-quality technical questions and reference answers. ( Thius dataset needs to be set manually)
* **Custom Logic Gates:**
    * **Context Utilization:** Validates if the agent is finding and using the correct source data.
    * **Correctness:** Measures the accuracy of the final answer against the reference ground truth.
* **How to run:** ```bash
    python evaluate_agent.py
    ```

---

## 🚀 Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone <your-repo-link>
   cd <your-repo-folder>

## 🛠️ Setup Instructions
1.  **Environment**: Ensure you are using Python 3.12 (on macOS).
2.  **Install Dependencies**:
    ```bash
    pip install langchain-openai langchain-chroma langgraph langsmith beautifulsoup4 python-dotenv
    ```
3. **Environment Variables**: Ensure you have your API keys configured: Set your `OPENAI_API_KEY` and `LANGCHAIN_API_KEY` in .env file.
Example: 
OPENAI_API_KEY='your_key_here'
LANGCHAIN_API_KEY='your_key_here'
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT= "your project name"
LANGCHAIN_ENDPOINT= your end point url
LANGSMITH_DISABLE_RUN_COMPRESSION=true

5.  ## 📂 Project Structure
* `RAG_agent.py`: The core application logic. This script handles the live user interaction.
* `Evaluator_agent.py`: This script acts as the automated testing laboratory to ensure the agent is performing accurately.
* `agent_db/`: Local folder where the Chroma vector database is stored.
* `.env`: File for managing environment variables.

