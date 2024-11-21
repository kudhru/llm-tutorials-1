from typing import List
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Pydantic models for output parsing
class RelevanceCheck(BaseModel):
    is_relevant: bool = Field(description="Whether the query is related to computer science")
    explanation: str = Field(description="Explanation for the relevance decision")

class ValidatorResponse(BaseModel):
    reveals_answer: bool = Field(description="Whether the tutor's response reveals the answer")
    explanation: str = Field(description="Explanation for why the response reveals/doesn't reveal the answer")

class TutorResponse(BaseModel):
    response: str = Field(description="The Socratic tutor's response")

# Initialize LLM
llm = ChatOpenAI(temperature=0.7)

# Prompts with format instructions
relevance_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluator that checks if student queries are related to computer science.
    Return a boolean indicating if the query is related to computer science and an explanation for your decision.
    
    {format_instructions}"""),
    ("human", "{query}")
])

tutor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Socratic tutor helping students learn computer science concepts. 
    Never reveal the direct answer to their questions. Instead, guide them towards the answer by:
    1. Asking probing questions that make them think
    2. Breaking down complex concepts into simpler parts
    3. Using analogies when appropriate
    4. Encouraging critical thinking
    
    Remember: Your goal is to help students discover the answer themselves through questioning.
    
    {format_instructions}"""),
    ("human", "{query}")
])

validator_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a validator checking if a tutor's response reveals too much information.
    Analyze the student's question and the tutor's response to determine if the answer is directly revealed.
    Return True if the answer is revealed, False if the response appropriately guides without revealing.
    
    {format_instructions}"""),
    ("human", """Student Question: {query}
    Tutor Response: {response}
    
    Does this response reveal the answer?""")
])

# Output parsers
relevance_parser = PydanticOutputParser(pydantic_object=RelevanceCheck)
validator_parser = PydanticOutputParser(pydantic_object=ValidatorResponse)
tutor_parser = PydanticOutputParser(pydantic_object=TutorResponse)

# Chain definitions with format instructions
relevance_chain = relevance_prompt.partial(format_instructions=relevance_parser.get_format_instructions()) | llm | relevance_parser
tutor_chain = tutor_prompt.partial(format_instructions=tutor_parser.get_format_instructions()) | llm | tutor_parser
validator_chain = validator_prompt.partial(format_instructions=validator_parser.get_format_instructions()) | llm | validator_parser

def get_socratic_response(query: str, max_attempts: int = 3) -> str:
    """Get a valid Socratic response for a query"""
    
    # First check relevance
    relevance_result = relevance_chain.invoke({"query": query})
    
    if not relevance_result.is_relevant:
        return f"I apologize, but your question doesn't appear to be related to computer science. {relevance_result.explanation}"
    
    # Get and validate tutor response
    for attempt in range(max_attempts):
        tutor_result = tutor_chain.invoke({"query": query})
        validator_result = validator_chain.invoke({
            "query": query,
            "response": tutor_result.response
        })
        
        if not validator_result.reveals_answer:
            return tutor_result.response
            
        print(f"Attempt {attempt + 1}: Response revealed too much, retrying...")
    
    return "I apologize, but I'm having trouble generating an appropriate response. Please try rephrasing your question."

def main():
    # Sample student queries
    sample_queries = [
        "How does a binary search algorithm work?",
        "What is the difference between HTTP and HTTPS?",
        "What is your favorite color?",
        "Can you explain recursion in programming?",
        "How do I implement a stack data structure?"
    ]
    
    print("Processing sample queries...\n")
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 50)
        response = get_socratic_response(query)
        print(f"Response: {response}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()
