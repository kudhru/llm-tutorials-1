import os
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

job_ids = [
    "ftjob-WPEcJjQoZSPogMZOEVPx37Op",
    "ftjob-nyM12MB8fSFW9oPXrv6i8zyx",
    "ftjob-Yd7cCFrqjrHZyQyKfZZK4Aft"
]

for job_id in job_ids:
    response = client.fine_tuning.jobs.retrieve(job_id)

    print("Job ID:", response.id)
    print("Status:", response.status)
    print("Trained Tokens:", response.trained_tokens)
    # print("Fine-tuned Model:", response.fine_tuned_model)
    
    # response = client.fine_tuning.jobs.list_events(job_id)
    # events = response.data
    # events.reverse()

    # print("\nEvents:")
    # for event in events:
    #     print(event.message)

    try:
        content = client.files.content(response.result_files[0]).decode('utf-8')
        print("\nResult file content:")
        print(content)
    except:
        print("\nNo result files available yet")

