import os
from pathlib import Path
from dotenv import load_dotenv

# --- 1. ENVIRONMENT SETUP ---
# This loads variables from a .env file into the script's environment.
load_dotenv()

# Safety check: Ensuring keys are present before starting
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY not found.")
    print("Please ensure you have a .env file with your keys.")


# --- STEP 2: IMPORTS ---
from typing import TypedDict, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph

# --- STEP 3: KNOWLEDGE BASE (RAG) ---
def setup_knowledge_base():
    print("--- 1. SCRAPING WEBSOURCE ---")
    url= "https://python.langchain.com/docs/tutorials/multi_agent/"
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print("--- 2. BUILDING DATABASE ---")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings(),
        persist_directory="./interactive_db"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = setup_knowledge_base()

# --- STEP 4: AGENT LOGIC (LANGGRAPH) ---
class AgentState(TypedDict):
    question: str
    context: List
    answer: str

def retrieve_node(state: AgentState):
    print("--- 3. SEARCHING DATABASE ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"context": documents}

def generate_node(state: AgentState):
    print("--- 4. GENERATING ANSWER ---")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    context_text = "\n\n".join([doc.page_content for doc in state["context"]])
    
    prompt = ChatPromptTemplate.from_template(
        "You are a technical assistant. Use the following LangChain documentation"
        "to answer the question accurately.\n\nContext: {context}\n\nQuestion: {question}"
    )
    
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": state["question"]})
    return {"answer": response.content}

# --- STEP 5: BUILD THE GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
agent_app = workflow.compile()

# --- STEP 6: RUN INTERACTIVE LOOP ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("AI AGENT IS READY")
    print("="*50)
    
    while True:
        user_query = input("\nAsk me anything about 'Langchain' (or type 'exit'): ")
        
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
            
        if user_query.strip():
            result = agent_app.invoke({"question": user_query})
            print(f"\nAI ANSWER:\n{result['answer']}")
            print("-" * 30)
