---
sidebar_position: 4
---

# Streaming Agent Output

Streaming enables real-time responses from agents, providing immediate feedback to users as the agent processes their request. This is particularly useful for enhancing user experience during long-running tasks or complex interactions.

## Overview

Streaming provides:
- Real-time responses during agent processing
- Immediate feedback for user engagement
- Better user experience for long-running tasks
- Efficient resource utilization during processing

## Example 1: Streamed Agent with Tool Calls

This example demonstrates how an agent utilizes asynchronous tools to perform tasks dynamically while streaming responses:

```python
import asyncio
import random
from agents import Agent, Runner, ItemHelpers

async def how_many_jokes():
    """Tool that returns a random integer determining the number of jokes."""
    return random.randint(1, 5)

async def main():
    agent = Agent(
        instructions="You are a helpful assistant. First, determine how many jokes to tell, then provide jokes.",
        tools=[how_many_jokes],
    )

    result = Runner.run_streamed(agent, input="Hello")

    async for event in result.stream_events():
        if event.item.type == "tool_call_output_item":
            print(f"Tool output: {event.item.output}")
        elif event.item.type == "message_output_item":
            print(ItemHelpers.text_message_output(event.item))

asyncio.run(main())
```

### Expected Output Example
```
=== Run starting ===
-- Tool output: 4
-- Message output:
 Sure, here are four jokes for you:

1. **Why don't skeletons fight each other?**
   They don't have the guts!

2. **What do you call fake spaghetti?**
   An impasta!

3. **Why did the scarecrow win an award?**
   Because he was outstanding in his field!

4. **Why can't you give Elsa a balloon?**
   Because she will let it go!
```

## Event Types in Streaming

### Tool Call Output Items
- **tool_call_output_item**: Shows the tool's returned data.

### Message Output Items
- **message_output_item**: Contains the generated messages by the agent.

## Example 2: Raw Response Event Handling

```python
from agents import Agent, Runner, ItemHelpers
import asyncio

async def main():
    agent = Agent(
        name="Joker",
        instructions="You are a helpful assistant.",
    )

    result = Runner.run_streamed(agent, input="Please tell me 5 jokes.")

    async for event in result.stream_events():
        if event.item.type == "message_output_item":
            print(ItemHelpers.text_message_output(event.item))

asyncio.run(main())
```

## Key Concepts

- **Streaming Output**: Real-time responses from asynchronous agent executions.
- **Event Handling**: Filtering and processing streamed events for specific outputs.
- **Agent Tools**: Modular functions called by agents during tasks.

## Best Practices

- Clearly separate logic for handling different event types.
- Ensure asynchronous methods (`async`/`await`) are used properly.
- Provide user-friendly outputs by ignoring non-relevant event types (e.g., raw event deltas).
- Implement proper error handling for streaming operations.
- Consider buffering strategies for optimizing output display.

Streaming is essential for creating responsive AI agent applications where users can see progress and receive partial results before the complete response is ready.