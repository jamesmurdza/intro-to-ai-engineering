In this workshop we will explore how large language models (LLMs) are created, trained, and fine-tuned to generate useful responses. You'll also learn how to interact with these models using APIs and apply them to tasks like chat, vision, tool use, and vector search.

# Prerequisites

<p align="center">
  <img src="https://github.com/user-attachments/assets/be39ac7f-538d-48bb-b3db-2a6f85cd4ed0" height="200" />
</p>

Required:
* A Google account for [Google Colab](https://colab.research.google.com/), or any Python development environment
* A free [OpenRouter API Key](https://openrouter.ai/settings/keys)

Optional:
* An [OpenAI API Key](https://platform.openai.com/account/api-keys)

# How LLMs are trained

<p align="center">
<img src="https://github.com/user-attachments/assets/db8885fe-75a6-45d6-aae3-01cd62fa2ca2" height="400" />
</p>

More information:
* [Tokenizer](https://platform.openai.com/tokenizer)
* [Transformers (how LLMs work) explained visually](https://www.youtube.com/watch?v=wjZofJX0v4M&t=246s&pp=ygUQM2JsdWUxYnJvd24gbGxtcw%3D%3D)

# Comparison of LLMs

![image](https://github.com/user-attachments/assets/76ccacf7-dd81-4951-a42f-d6dff874ce57)

* Crowdsourced evals: [ChatBot Arena](https://lmarena.ai/)
* Coding evals: [Aider](https://aider.chat/docs/leaderboards/)

# Using an LLM

<p align="center">
<img src="https://github.com/user-attachments/assets/89a11d99-9ec4-4501-8614-52dba7a76773" height="300" />
</p>

In this example, we initialize the OpenRouter client with an API key and send a chat message to the Llama model. The model processes the input and returns a response, which is then printed.

```python
from google.colab import userdata
from openai import OpenAI

# Initialize OpenRouter client with API key from Colab secrets
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=userdata.get('OPENROUTER_API_KEY')
)

# Send chat completion request to Llama model
completion = client.chat.completions.create(
  model="meta-llama/llama-4-maverick:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)

# Print the model's response
print(completion.choices[0].message.content)
```

# Vision

<p align="center">
<img src="https://github.com/user-attachments/assets/a0082f62-c641-4a29-9da8-4a1f76ebbb77" height="300" />
</p>

In this example, we fetch an image from a URL, encode it to base64, and send it along with a question to the Llama model using OpenRouter. The model analyzes the image and provides a descriptive response, which is printed.

```python
from google.colab import userdata
from openai import OpenAI

import base64
import requests

# Initialize OpenRouter client
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=userdata.get('OPENROUTER_API_KEY')
)

# Function to encode image from URL
def encode_image_from_url(image_url):
    response = requests.get(image_url)
    return base64.b64encode(response.content).decode("utf-8")

# Get and encode the image
image_url = "https://www.dogster.com/wp-content/uploads/2012/03/group-of-dachshund-dogs_4sally-scott-Shutterstock.jpg.webp"
base64_image = encode_image_from_url(image_url)

# Send image analysis request
completion = client.chat.completions.create(
  model="meta-llama/llama-4-maverick:free",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What's in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ]
)

print(completion.choices[0].message.content)
```

# Tool-use

<p align="center">
<img src="https://github.com/user-attachments/assets/654df9a6-4e3c-4094-bd70-a4b844c3260a" height="300" />
</p>

In this example, we define a tool schema for evaluating mathematical expressions, then send a user query along with the tool definition to the Llama model. The model responds by invoking the appropriate tool function, and we print the function call it generated.

```python
from google.colab import userdata
from openai import OpenAI

# Initialize OpenRouter client
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=userdata.get('OPENROUTER_API_KEY')
)

# Define function schema for mathematical calculations
tools = [{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform mathematical calculations including basic arithmetic, algebra, and more complex operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', 'sin(pi/2)')"
                }
            },
            "required": ["expression"],
            "additionalProperties": False
        }
    }
}]

# Make API call with function calling capability
completion = client.chat.completions.create(
    model="meta-llama/llama-4-maverick:free",
    messages=[{"role": "user", "content": "What is 25 * 47 + 156 / 12?"}],
    tools=tools
)

# Print the function call details
print(completion.choices[0].message.tool_calls[0].function)
```

# Vector search

<p align="center">
<img src="https://github.com/user-attachments/assets/6f08c496-1b5d-491e-9f15-bbe71a905394" width="300" />
</p>

In this example, we generate vector embeddings for a set of predefined customer service intents and a user query using OpenAI’s embedding model. We calculate cosine similarities between the query and each intent, rank them, and print the top 3 most similar intents based on the similarity scores.

```python
from google.colab import userdata

openai.api_key = userdata.get('OPENAI_API_KEY')

import openai
import numpy as np

# Define sample customer service intents
intents = [
    "How do I reset my password?",
    "What are your business hours?",
    "I need help with my order.",
    "Where is my package?",
    "Can I return a product?",
]

# Generate embeddings for each intent
intent_embeddings = [
    openai.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding
    for text in intents
]

# User query to match against intents
query = "Help me track my delivery"
query_embedding = openai.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding

# Calculate cosine similarity between query and each intent
similarities = [
    np.dot(query_embedding, intent_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(intent_emb))
    for intent_emb in intent_embeddings
]

# Find top 3 most similar intents
top_k = 3
top_indices = np.argsort(similarities)[-top_k:][::-1]  # Sorted descending
top_matches = [intents[i] for i in top_indices]

print("Query:", query)
print("\nTop matches:")
for i, match in enumerate(top_matches, 1):
    print(f"{i}. {match} (score: {similarities[top_indices[i-1]]:.4f})")
```
