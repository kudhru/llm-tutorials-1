from typing import List, Dict
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Define the evaluation criteria schema
class EvaluationCriteria(BaseModel):
    is_correct: bool = Field(description="Whether the response follows Socratic teaching principles")
    reasoning: str = Field(description="Explanation of why the response is correct or incorrect")
    score: float = Field(description="Score between 0 and 1 indicating how well the response follows Socratic principles")

# Create evaluation prompt
evaluation_prompt = ChatPromptTemplate.from_messages([
    ("user", """You are an expert at evaluating Socratic teaching methods. 
    Analyze the following tutor response to determine if it follows Socratic principles:
    - Uses questions to guide student thinking
    - Avoids giving direct answers
    - Helps student discover answers themselves
    - Builds on student's existing knowledge
    - Encourages critical thinking
    
    Student Question: {student_question}
    Tutor Response: {tutor_response}
    
    Evaluate if this response follows Socratic teaching principles.
    Format Instructions:
    {format_instructions}""")
])

def load_dataset(file_path: str) -> List[Dict]:
    """Load the dataset from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def evaluate_responses(dataset: List[Dict]) -> List[Dict]:
    """Evaluate each response in the dataset"""
    
    # Initialize evaluator
    evaluator = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = JsonOutputParser(pydantic_object=EvaluationCriteria)
    
    # Create evaluation chain
    evaluation_chain = evaluation_prompt | evaluator | parser
    
    results = []
    
    for conv_idx, conversation in enumerate(dataset):
        print(f"\nProcessing conversation {conv_idx + 1}/{len(dataset)}")
        messages = conversation["messages"]
        system_prompt = next(msg["content"] for msg in messages if msg["role"] == "system")
        
        # Evaluate each pair of user-assistant messages
        for i in range(len(messages)-1):
            if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                user_msg = messages[i]["content"]
                assistant_msg = messages[i+1]["content"]
                
                print(f"  Evaluating message pair {i//2 + 1}")
                print(f"    Student: {user_msg[:50]}...")
                print(f"    Tutor: {assistant_msg[:50]}...")
                
                # Run evaluation
                evaluation = evaluation_chain.invoke({
                    "student_question": user_msg,
                    "tutor_response": assistant_msg,
                    "format_instructions": parser.get_format_instructions()
                })
                
                print(f"    Score: {evaluation['score']}")
                
                results.append({
                    "student_question": user_msg,
                    "tutor_response": assistant_msg,
                    "evaluation": evaluation
                })
    
    return results

def main():
    # Load dataset
    # dataset = load_dataset("/Users/kudhru/work/llm-tutorials/evaluation/socratic_tutoring/dataset.json")
    dataset = load_dataset("/Users/kudhru/work/llm-tutorials/evaluation/socratic_tutoring/dataset_with_mistakes.json")
    # Run evaluation
    evaluation_results = evaluate_responses(dataset)
    
    # Print results
    print("\nEvaluation Results:")
    print("==================")
    
    correct_count = 0
    total_count = len(evaluation_results)
    
    for i, result in enumerate(evaluation_results):
        print(f"\nEvaluation {i}:")
        print(f"Student Question: {result['student_question']}")
        print(f"Tutor Response: {result['tutor_response']}")
        print(f"Is Correct: {result['evaluation']['is_correct']}")
        print(f"Score: {result['evaluation']['score']}")
        print(f"Reasoning: {result['evaluation']['reasoning']}")
        
        if result['evaluation']['is_correct']:
            correct_count += 1
    
    # Print summary
    print("\nSummary:")
    print(f"Total Responses Evaluated: {total_count}")
    print(f"Correct Responses: {correct_count}")
    print(f"Accuracy: {(correct_count/total_count)*100:.2f}%")

if __name__ == "__main__":
    main()
