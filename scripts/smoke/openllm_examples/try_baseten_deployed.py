# Example: Using Baseten deployed model with OpenAI client
# Baseten provides OpenAI-compatible API for deployed models

import os
from openai import OpenAI

# Load API key from environment
client = OpenAI(
    api_key=os.environ.get("BASETEN_API_KEY"),
    base_url="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
)

response = client.chat.completions.create(
    model="google/gemma-4-26B-A4B-it",
    messages=[{"role":"user","content":[{"text":"Describe this image in one sentence.","type":"text"},{"type":"image_url","image_url":{"url":"https://picsum.photos/id/237/200/300"}}]}],
)

print(response.choices[0].message.content)
