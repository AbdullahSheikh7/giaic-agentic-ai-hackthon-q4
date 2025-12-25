---
sidebar_position: 3
---

# Hello World Agent Project

This is a step-by-step guide for building a "Hello World" project in the **Agentia** ecosystem, where you will build two agents that communicate through natural language interfaces.

## Project Overview

In this project, you'll create a minimal multi-agent system with:

1. **Front-End Orchestration Agent**
   - Receives user requests
   - Delegates tasks to specialized agents
   - Consolidates responses for the user

2. **Greeting Agent**
   - Handles simple greeting requests
   - Returns appropriate greeting responses

## Architecture

### Front-End Orchestration Agent
- **Purpose**: User interface layer that the user interacts with directly
- **Responsibilities**:
  - Receive user input from CLI, web UI, or chatbot interface
  - Determine the type of task requested
  - Delegate to appropriate specialized agents
  - Aggregate responses from specialized agents
  - Return consolidated reply to the user

### Greeting Agent
- **Purpose**: Handle simple greeting requests
- **Functionality**:
  - Respond to greetings like "Hello," "Hi," "Good morning," or "How are you?"
  - Return a default message if the input doesn't match greeting intent

## Implementation Options

You have a choice of building these Agents for the Agentia World using:
- **CrewAI**: Framework for multi-agent collaboration
- **LangGraph**: Graph-based agent framework
- **Microsoft AutoGen**: Framework for building conversational agents
- **AG2**: (Agent General 2) or another agent framework

## Conversation Flow

1. **Receive User Input**: Wait for a text message from the user
2. **Determine Task**: Identify if the user's message is a greeting or other request
3. **Delegate to Greeting Agent**: Forward greeting messages to the Greeting Agent
4. **Aggregate Response**: Retrieve the greeting response
5. **Return Consolidated Reply**: Send the greeting back to the user

## Example Implementation with LangGraph

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List

class AgentState(TypedDict):
    user_input: str
    agent_response: str
    task_type: str

def front_end_agent(state: AgentState) -> AgentState:
    """Front-end orchestration agent"""
    user_input = state['user_input'].lower()
    
    if any(greeting in user_input for greeting in ['hello', 'hi', 'hey', 'good morning', 'how are you']):
        state['task_type'] = 'greeting'
    else:
        state['task_type'] = 'other'
    
    return state

def greeting_agent(state: AgentState) -> AgentState:
    """Greeting agent that handles greeting requests"""
    if state['task_type'] == 'greeting':
        response = "Hello! I'm your friendly greeting agent. How can I assist you today?"
    else:
        response = "I only handle greeting requests right now."
    
    state['agent_response'] = response
    return state

def main_agent(state: AgentState) -> AgentState:
    """Main agent orchestrating the workflow"""
    # Determine task type
    state = front_end_agent(state)
    
    # Process based on task type
    state = greeting_agent(state)
    
    return state

# Create the workflow
workflow = StateGraph(AgentState)
workflow.add_node("front_end", front_end_agent)
workflow.add_node("greeting", greeting_agent)
workflow.add_node("main", main_agent)

# Set entry point
workflow.set_entry_point("main")
workflow.add_edge("main", "__end__")

app = workflow.compile()

# Example usage
result = app.invoke({"user_input": "Hello, how are you?"})
print(result)
```

## Testing and Validation

1. **Manual Testing**: Send requests to the Front-End Agent endpoint
2. **Verification**: Ensure the Front-End Agent routes greeting messages to the Greeting Agent
3. **Response Validation**: Verify that greeting responses are properly returned to the user

## Benefits of This Approach

This minimal, end-to-end demonstration provides practical insights into:
- Multi-agent conversation patterns
- Natural language interface design
- Agent collaboration via orchestration
- Reusable, autonomous component architecture

By following standard practices such as microservices architecture, containerization, HTTP/gRPC communication, and simple Natural Language Understanding (NLU), you'll build a strong foundation for more advanced multi-agent systems where agents can become increasingly specialized, robust, and scalable.