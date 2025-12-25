---
sidebar_position: 7
---

# Guardrails for AI Agents

Guardrails are critical safety mechanisms that ensure AI agents operate within defined boundaries, maintaining appropriate behavior and preventing harmful or off-topic interactions. This guide implements a PIAIC (Presidential Initiative for Artificial Intelligence and Computing) agent using the OpenAI Agents SDK with custom input and output guardrails.

## Overview

- **Purpose**: Ensure the agent processes and responds only to approved topics.
- **Guardrails**:
  - **Input Guardrail**: Verifies that user input is relevant to approved topics.
  - **Output Guardrail**: Confirms the agent's response is appropriate and on-topic.
- **Implementation**: Uses dedicated guardrail agents to assess input/output relevance, with tripwires to stop off-topic or inappropriate content.

## Key Components

### PIAIC Agent
The core agent for answering approved-topic-related questions.

### Input Guardrail Agent
Checks input relevance to approved topics before processing.

### Output Guardrail Agent
Ensures response relevance and appropriateness before returning to the user.

## Implementation

Here's an example implementation using the OpenAI Agents SDK:

```python
from agents import Agent, Runner
from pydantic import BaseModel, Field

class PIAICRelevanceOutput(BaseModel):
    """Output model for guardrail assessment."""
    is_relevant: bool = Field(description="Whether the input/output is PIAIC-related")
    reasoning: str = Field(description="Reasoning for the relevance assessment")

async def input_guardrail(input_text: str) -> PIAICRelevanceOutput:
    """Checks if input is relevant to PIAIC topics."""
    # Implementation would use LLM to assess relevance
    # Return PIAICRelevanceOutput with is_relevant and reasoning
    pass

async def output_guardrail(output_text: str) -> PIAICRelevanceOutput:
    """Checks if output is relevant to PIAIC topics."""
    # Implementation would use LLM to assess relevance
    # Return PIAICRelevanceOutput with is_relevant and reasoning
    pass

# Create the main PIAIC agent
piaic_agent = Agent(
    name="PIAIC Assistant",
    instructions="You are an expert assistant for PIAIC-related topics including AI, Cloud Computing, Blockchain, and IoT.",
    # Add guardrail tools here
)

# Use the agent with guardrail checks
async def guarded_run(agent, input_text):
    # Check input
    input_check = await input_guardrail(input_text)
    if not input_check.is_relevant:
        return f"Input Guardrail tripped: {input_check.reasoning}"
    
    # Run the agent
    result = await Runner.run(agent, input_text)
    
    # Check output
    output_check = await output_guardrail(result.final_output)
    if not output_check.is_relevant:
        return f"Output Guardrail tripped: {output_check.reasoning}"
    
    return result.final_output
```

## Usage

1. **Define Guardrail Criteria**: Establish clear boundaries for what constitutes appropriate input/output.

2. **Implement Assessment Logic**: Create functions that can evaluate whether content meets your criteria.

3. **Integrate into Workflow**: Add guardrail checks before processing input and before returning output.

4. **Handle Violations**: Define appropriate responses when guardrails are triggered.

## Example Scenarios

- PIAIC-relevant input: "What is the curriculum for PIAIC's AI course?"
- Non-PIAIC input: "How do I bake a chocolate cake?"

In the second case, the input guardrail would trigger, preventing the main agent from processing the request.

## Benefits

- **Safety**: Prevents the agent from engaging with harmful or inappropriate content
- **Focus**: Maintains the agent's relevance to its intended purpose
- **Compliance**: Helps ensure the agent operates within regulatory or organizational guidelines
- **User Experience**: Provides clear feedback when requests are outside the agent's scope

Guardrails are essential for deploying responsible AI agents, particularly in enterprise or sensitive applications where maintaining control over the agent's behavior is crucial.