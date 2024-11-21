from typing import List, Dict
import json
from ragas import evaluate
from ragas.metrics import MultiTurnMetric
from ragas.metrics.base import MetricWithLLM
from datasets import Dataset
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ragas.metrics.base import MetricType
from dataclasses import dataclass, field
from ragas.messages import AIMessage, HumanMessage
import typing as t

@dataclass
class SocraticQuestioningMetric(MetricWithLLM, MultiTurnMetric):
    """Measures how well the response uses Socratic questioning techniques"""
    
    name: str = "socratic_questioning"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.MULTI_TURN: {"user_input"}
        }
    )
    
    def __post_init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are evaluating a tutor's response for Socratic questioning techniques.
            Score the response on a scale of 0-1 based on these criteria:
            - Uses questions to guide thinking (0.2)
            - Avoids giving direct answers (0.2)
            - Helps student discover answers themselves (0.2)
            - Builds on student's existing knowledge (0.2)
            - Encourages critical thinking (0.2)
            
            Return only the numerical score."""),
            ("human", """Student Question: {question}
            Tutor Response: {response}""")
        ])

    async def _multi_turn_ascore(self, sample, callbacks) -> float:
        conversations = sample.user_input
        scores = []
        
        # Group messages into pairs of human and AI messages
        for i in range(len(conversations)-1):
            if isinstance(conversations[i], HumanMessage) and isinstance(conversations[i+1], AIMessage):
                result = await self.llm.ainvoke(
                    self.prompt.format_messages(
                        question=conversations[i].content,
                        response=conversations[i+1].content
                    )
                )
                try:
                    score = float(result.content.strip())
                    scores.append(min(max(score, 0.0), 1.0))
                except ValueError:
                    scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0

@dataclass
class KnowledgeConstructionMetric(MetricWithLLM, MultiTurnMetric):
    """Measures how well the response helps build student's knowledge"""
    
    name: str = "knowledge_construction"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.MULTI_TURN: {"user_input"}
        }
    )
    
    def __post_init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Evaluate how well the tutor's response helps construct knowledge:
            - Connects to prior knowledge (0.25)
            - Uses analogies or examples (0.25)
            - Scaffolds learning progressively (0.25)
            - Encourages active reasoning (0.25)
            
            Return only the numerical score."""),
            ("human", """Student Question: {question}
            Tutor Response: {response}""")
        ])

    async def _multi_turn_ascore(self, sample, callbacks) -> float:
        conversations = sample.user_input
        scores = []
        
        # Group messages into pairs of human and AI messages
        for i in range(len(conversations)-1):
            if isinstance(conversations[i], HumanMessage) and isinstance(conversations[i+1], AIMessage):
                result = await self.llm.ainvoke(
                    self.prompt.format_messages(
                        question=conversations[i].content,
                        response=conversations[i+1].content
                    )
                )
                try:
                    score = float(result.content.strip())
                    scores.append(min(max(score, 0.0), 1.0))
                except ValueError:
                    scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0

def load_dataset(file_path: str) -> Dataset:
    """Load and format the dataset for Ragas evaluation"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    formatted_data = {
        "user_input": []
    }
    
    for conversation in data:
        messages = []
        for msg in conversation["messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        formatted_data["user_input"].append(messages)
    
    return Dataset.from_dict(formatted_data)

def evaluate_responses(dataset: Dataset) -> Dict:
    """Evaluate responses using custom Ragas metrics"""
    
    # Initialize custom metrics with LLM
    evaluator_llm = ChatOpenAI(model="gpt-4", temperature=0)
    metrics = [
        SocraticQuestioningMetric(llm=evaluator_llm),
        KnowledgeConstructionMetric(llm=evaluator_llm)
    ]
    
    # Run evaluation
    results = evaluate(
        dataset=dataset,
        metrics=metrics
    )
    
    return results

def main():
    # Load dataset
    dataset = load_dataset("evaluation/socratic_tutoring/test_dataset.json")
    
    # Run evaluation
    results = evaluate_responses(dataset)
    
    # Print results
    print("\nEvaluation Results:")
    print("==================")
    
    # Print metric scores
    print("\nMetric Scores:")
    for metric_name, score in results.items():
        print(f"{metric_name}: {score:.3f}")
    
    # Print detailed results for each conversation
    print("\nDetailed Results:")
    for i, conversation in enumerate(dataset["user_input"]):
        print(f"\nConversation {i+1}:")
        for j, msg in enumerate(conversation):
            if isinstance(msg, HumanMessage):
                print(f"Student: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"Tutor: {msg.content}")
        print("Scores:")
        for metric_name, scores in results.items():
            if isinstance(scores, np.ndarray):
                print(f"  {metric_name}: {scores[i]:.3f}")
    
    # Calculate and print average scores
    print("\nAverage Scores:")
    for metric_name, scores in results.items():
        if isinstance(scores, np.ndarray):
            avg_score = np.mean(scores)
            print(f"{metric_name}: {avg_score:.3f}")

if __name__ == "__main__":
    main() 