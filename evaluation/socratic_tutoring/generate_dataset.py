import json
import random
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

def create_chat_model():
    """Create ChatOpenAI instance"""
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )

def get_topics() -> List[Dict]:
    """Return list of CS topics to generate conversations about"""
    return [
        {"topic": "quicksort", "category": "sorting algorithms"},
        {"topic": "binary search trees", "category": "data structures"},
        {"topic": "dynamic programming", "category": "algorithms"},
        {"topic": "hash tables", "category": "data structures"},
        {"topic": "depth-first search", "category": "graph algorithms"},
        {"topic": "heaps", "category": "data structures"},
        {"topic": "merge sort", "category": "sorting algorithms"},
        {"topic": "binary search", "category": "algorithms"},
        {"topic": "linked lists", "category": "data structures"},
        {"topic": "recursion", "category": "programming concepts"}
    ]

def create_conversation_prompt(make_mistake: bool) -> ChatPromptTemplate:
    """Create prompt template for generating conversations"""
    
    if make_mistake:
        template = """You are generating a conversation between a CS student and an AI tutor who makes the mistake of giving direct answers instead of using the Socratic method.

Topic: {topic}
Category: {category}

Generate a conversation with:
1. Student asking about the topic
2. Tutor giving a direct, complete answer (wrong approach)
3. Student asking a follow-up question
4. Tutor again giving a direct answer

The conversation should be in this JSON format:
{
    "messages": [
        {{"role": "system", "content": "You are an AI tutor guiding a first-year computer science student through problems in data structures and algorithms without directly providing answers. Your approach is Socratic, focusing on asking insightful questions that help the student reason through problems and build a deeper understanding."}},
        {{"role": "user", "content": "student question"}},
        {{"role": "assistant", "content": "direct answer"}},
        {{"role": "user", "content": "follow-up question"}},
        {{"role": "assistant", "content": "direct answer"}}
    ]
}

Generate only the JSON, no other text."""

    else:
        template = """You are generating a conversation between a CS student and an AI tutor who correctly uses the Socratic method.

Topic: {topic}
Category: {category}

Generate a conversation with:
1. Student asking about the topic
2. Tutor asking a guiding question to help student understand (correct Socratic approach)
3. Student attempting to answer
4. Tutor acknowledging and asking another guiding question

The conversation should be in this JSON format:
{
    "messages": [
        {{"role": "system", "content": "You are an AI tutor guiding a first-year computer science student through problems in data structures and algorithms without directly providing answers. Your approach is Socratic, focusing on asking insightful questions that help the student reason through problems and build a deeper understanding."}},
        {{"role": "user", "content": "student question"}},
        {{"role": "assistant", "content": "Socratic question"}},
        {{"role": "user", "content": "student attempt"}},
        {{"role": "assistant", "content": "acknowledgment and follow-up question"}}
    ]
}

Generate only the JSON, no other text."""

    return ChatPromptTemplate.from_template(template)

async def generate_conversation(
    chat: ChatOpenAI,
    topic: Dict,
    make_mistake: bool
) -> Dict:
    """Generate a single conversation"""
    
    prompt = create_conversation_prompt(make_mistake)
    
    response = await chat.ainvoke(
        prompt.format_messages(
            topic=topic["topic"],
            category=topic["category"]
        )
    )
    
    # Parse JSON response
    try:
        conversation = json.loads(response.content)
        return conversation
    except json.JSONDecodeError:
        print(f"Error parsing JSON for topic {topic['topic']}")
        return None

async def generate_dataset(
    num_conversations: int = 10,
    mistake_ratio: float = 0.75
) -> List[Dict]:
    """Generate dataset of conversations"""
    
    chat = create_chat_model()
    topics = get_topics()
    dataset = []
    
    # Calculate how many conversations should have mistakes
    mistakes_needed = int(num_conversations * mistake_ratio)
    
    for i in range(num_conversations):
        topic = random.choice(topics)
        make_mistake = i < mistakes_needed
        
        conversation = await generate_conversation(chat, topic, make_mistake)
        if conversation:
            dataset.append(conversation)
    
    return dataset

async def main():
    # Generate dataset
    dataset = await generate_dataset(num_conversations=10, mistake_ratio=0.75)
    
    # Save to file
    output_file = "evaluation/socratic_tutoring/generated_dataset_with_mistakes.json"
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Generated dataset saved to {output_file}")
    print(f"Total conversations: {len(dataset)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())