---
sidebar_position: 2
---

# Hello Agent: Getting Started

The OpenAI Agents SDK provides a powerful framework for creating AI agents. This guide will walk you through creating your first agent using the SDK.

## Prerequisites

Before getting started, make sure you have:
- Python 3.10+
- An API key from your preferred model provider (OpenAI, Google Gemini, etc.)

## Installation

First, create a new project and install the required dependencies:

```bash
uv init hello_agent
cd hello_agent
uv add openai-agents python-dotenv
```

## Basic Agent Creation

Here's a simple example of creating an agent:

```python
import asyncio
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

# Load your API key from environment
import os
from dotenv import load_dotenv
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")  # or your preferred provider

# Reference: https://ai.google.dev/gemini-api/docs/openai
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

set_tracing_disabled(disabled=True)

async def main():
    # Create a simple agent
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    )

    result = await Runner.run(
        agent,
        "Hello, introduce yourself!",
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuring LLM Providers

The OpenAI Agents SDK is set up to use OpenAI as default providers. When using other providers, you can configure them at different levels:

### 1. Agent Level Configuration

This allows you to specify a different LLM provider for a specific agent:

```python
import asyncio
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Reference: https://ai.google.dev/gemini-api/docs/openai
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

set_tracing_disabled(disabled=True)

async def main():
    # This agent will use the custom LLM provider
    agent = Agent(
        name="Haiku Assistant",
        instructions="You only respond in haikus.",
        model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    )

    result = await Runner.run(
        agent,
        "Tell me about recursion in programming.",
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Run Level Configuration

This allows you to specify a different LLM provider for a specific run:

```python
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent: Agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Hello, how are you?", run_config=config)

print(result.final_output)
```

### 3. Global Configuration

This sets a default provider for all agents in the session:

```python
from agents import Agent, Runner, AsyncOpenAI, set_default_openai_client, set_tracing_disabled, set_default_openai_api

gemini_api_key = os.getenv("GEMINI_API_KEY")
set_tracing_disabled(True)
set_default_openai_api("chat_completions")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(external_client)

agent: Agent = Agent(name="Assistant", instructions="You are a helpful assistant", model="gemini-2.0-flash")

result = Runner.run_sync(agent, "Hello")

print(result.final_output)
```

## Next Steps

Once you have your basic agent running, you can explore:
- Adding tools to your agent
- Implementing memory capabilities
- Creating multi-agent systems
- Adding tracing and monitoring