# You can use this model with any of the OpenAI clients in any language!
# Simply change the API Key to get started

from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get("BASETEN_API_KEY"),
    base_url="https://inference.baseten.co/v1"
)

response = client.chat.completions.create(
    model="zai-org/GLM-5",
    messages=[
        {
            "role": "user",
            "content": "Implement Hello World in Python"
        }
    ],
    stream=True,
    stream_options={
        "include_usage": True,
        "continuous_usage_stats": True
    },
    top_p=1,
    max_tokens=1000,
    temperature=1,
    presence_penalty=0,
    frequency_penalty=0
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
