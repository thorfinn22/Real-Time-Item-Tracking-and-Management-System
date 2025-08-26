import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="nebius",
    api_key="HF_Token"
)

completion = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    
)

print(completion.choices[0].message)