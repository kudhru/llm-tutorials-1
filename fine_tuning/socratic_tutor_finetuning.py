import os
import time
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

import json
import openai

import pandas as pd
from pprint import pprint

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    # organization="<org id>",
    # project="<project id>",
)

training_file_name = "/Users/kudhru/work/llm-tutorials/fine_tuning/data/socratic_tutor_dataset_training.jsonl"
validation_file_name = "/Users/kudhru/work/llm-tutorials/fine_tuning/data/socratic_tutor_dataset_validation.jsonl"
def upload_file(file_name: str, purpose: str) -> str:
    with open(file_name, "rb") as file_fd:
        response = client.files.create(file=file_fd, purpose=purpose)
    return response.id


training_file_id = upload_file(training_file_name, "fine-tune")
validation_file_id = upload_file(validation_file_name, "fine-tune")
# training_file_id = 'file-PTDpx7rk1lqxsSFjApXuBTZe'
# validation_file_id = 'file-QmkHpo5UahQLwZZ4hNd5gPwt'

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)


MODEL = "gpt-4o-mini-2024-07-18"

response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model=MODEL,
    suffix="socratic-tutor",
)

job_id = response.id

print("Job ID:", response.id)
print("Status:", response.status)

# job_id = "ftjob-BUuofV2rCHrSiyWpAs7So219"
while True:
    response = client.fine_tuning.jobs.retrieve(job_id)
    print("Job ID:", response.id)
    print("Status:", response.status)
    print("Trained Tokens:", response.trained_tokens)

    if response.status == "succeeded":
        break

    response = client.fine_tuning.jobs.list_events(job_id)
    events = response.data
    events.reverse()

    for event in events:
        print(event.message)

    # Optional: Add a sleep interval to avoid hitting rate limits
    time.sleep(10)
