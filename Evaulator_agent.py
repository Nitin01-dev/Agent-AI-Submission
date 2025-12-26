import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

# 1. Setup Environment
load_dotenv()
client = Client()

# 2. Import your working agent
try:
    from RAG_agent import agent_app
except ImportError:
    print("❌ Error: RAG_agent.py not found in this folder.")

# 3. Robust Prediction Function
def predict_result(inputs: dict):
    # This looks for 'question', 'Question', or 'input' automatically
    query = inputs.get("question") or inputs.get("Question") or inputs.get("input")
    
    if not query:
        print(f"⚠️ Dataset keys found: {list(inputs.keys())}. Trying first available key...")
        query = list(inputs.values())[0]  # Fallback: just take the first column

    # Run the agent
    print(f"🤖 Testing Question: {query}")
    result = agent_app.invoke({"question": query})
    
    # Safely extract the answer
    answer = result.get("answer") or "No answer returned by agent."
    return {"output": answer}

# 4. Safer Evaluator Judges
def correctness_judge(run, example):
    """Evaluation 1: Factual Accuracy"""
    # Use .get() with empty string default to avoid NoneType errors
    prediction = str(run.outputs.get("output") or "").lower()
    reference = str(example.outputs.get("output") or "").lower()
    
    if not reference:
        return {"key": "correctness", "score": 0, "comment": "No reference answer found in dataset."}

    # Simple check: is the ground truth contained in the answer?
    score = 1 if reference in prediction else 0
    return {"key": "correctness", "score": score}

def context_judge(run, example):
    """Evaluation 2: Context Utilization"""
    prediction = str(run.outputs.get("output") or "")
    # Does the agent provide a detailed response (more than 20 chars)?
    score = 1 if len(prediction) > 20 else 0
    return {"key": "context_utilization", "score": score}

# 5. Run the Evaluation
def run_evaluation():
    print("--- 🚀 STARTING FINAL EVALUATION ---")
    try:
        evaluate(
            predict_result,
            #data="AI_Knowledge_Evaluation", 
            data="LangChain_MultiAgent_Eval_Set",
            evaluators=[correctness_judge, context_judge],
            experiment_prefix="Final_Submission_Test"
        )
        print("\n--- ✅ DONE! Check your LangSmith Dashboard for the green bars. ---")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    run_evaluation()
