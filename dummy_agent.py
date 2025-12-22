import os
from typing import TypedDict, List
from dotenv import load_dotenv

# --- 1. ENVIRONMENT SETUP ---
# This loads variables from a .env file into the script's environment.
load_dotenv()

# Safety check: Ensuring keys are present before starting
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY not found.")
    print("Please ensure you have a .env file with your keys.")

# --- 2. IMPORTS ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph

# --- 3. RAG SETUP (Knowledge Base) ---
def setup_knowledge_base():
    print("--- SCRAPING DATA ---")
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    # Splitting long text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Creating the vector database
    print("--- INDEXING DATA ---")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings(),
        persist_directory="./agent_db"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize the retriever
retriever = setup_knowledge_base()

# --- 4. GRAPH STATE & NODES ---
class AgentState(TypedDict):
    question: str
    context: List
    answer: str

def retrieve_node(state: AgentState):
    """Search for relevant documents based on the user question."""
    print("--- RETRIEVING CONTEXT ---")
    return {"context": retriever.invoke(state["question"])}

def generate_node(state: AgentState):
    """Generate an answer using the LLM and retrieved context."""
    print("--- GENERATING ANSWER ---")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Format the context for the prompt
    context_text = "\n\n".join([doc.page_content for doc in state["context"]])
    
    prompt = ChatPromptTemplate.from_template(
        "Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
    )
    
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": state["question"]})
    return {"answer": response.content}

# --- 5. LANGGRAPH WORKFLOW ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# Connect Nodes (Edges)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the Agent
agent_app = workflow.compile()

# --- 6. EXECUTION ---
if __name__ == "__main__":
    user_query = "What are the main goals of AI research?"
    print(f"\nUSER QUESTION: {user_query}")
    
    # Run the agent
    result = agent_app.invoke({"question": user_query})
    
    print("\n" + "="*50)
    print(f"AI RESPONSE:\n{result['answer']}")
    print("="*50)