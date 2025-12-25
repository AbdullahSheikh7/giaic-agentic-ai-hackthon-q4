---
sidebar_position: 5
---

# Implementing Agentic AI with Popular Libraries

This chapter demonstrates practical implementation of agentic AI systems using leading frameworks and libraries. We'll explore hands-on examples with the most popular tools in the ecosystem.

## Overview of Agentic AI Frameworks

### Key Considerations for Framework Selection
- **Ease of Use**: How quickly can you build and deploy agents?
- **Flexibility**: How much control do you have over agent behavior?
- **Integration**: How well does it connect with external tools and APIs?
- **Community Support**: Quality of documentation, tutorials, and community
- **Production Readiness**: Stability and performance for real-world deployment

## LangChain: Comprehensive Framework for Language Model Applications

LangChain provides a comprehensive toolkit for developing applications powered by language models, including sophisticated agentic systems.

### Core Components of LangChain Agentic Systems

#### 1. Agents
LangChain provides several pre-built agents:
- **ZeroShotReactDescription**: Uses ReAct prompting for reasoning
- **ConversationalReactDescription**: Maintains conversation history
- **ChatAgent**: Designed for chat model-based agents

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["llm-math", "wikipedia"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
```

#### 2. Memory Management
LangChain offers various memory types:
- **ConversationBufferMemory**: Stores entire conversation history
- **ConversationSummaryMemory**: Summarizes conversation history
- **Entity Memory**: Tracks specific entities mentioned in conversations

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
                        memory=memory, verbose=True)
```

#### 3. Tool Integration
LangChain makes it easy to integrate with various tools:

```python
from langchain.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# Create a custom tool
from langchain.tools import BaseTool
from langchain.llms import OpenAI

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "useful for doing mathematical calculations"

    def _run(self, query: str):
        # Implementation for calculator functionality
        return eval(query)
    
    async def _arun(self, query: str):
        raise NotImplementedError("CalculatorTool does not support async")

tools = [search_tool, CalculatorTool()]
```

### Building a Custom Agentic System with LangChain

Here's a complete example of a research agent:

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import LLMMathChain
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory

# Initialize components
llm = OpenAI(temperature=0)
search = DuckDuckGoSearchRun()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

# Define tools
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for searching the internet"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for mathematical calculations"
    )
]

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Create agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Use the agent
result = agent.run("What is the GDP of the USA and how has it changed in the last 5 years?")
```

## AutoGen: Multi-Agent Framework by Microsoft

AutoGen enables the development of complex multi-agent systems with minimal code.

### Core Concepts

#### Conversable Agents
Agents that can communicate with each other and take actions based on messages.

```python
import autogen

# Configuration for OpenAI API
config_list = [
    {
        'model': 'gpt-4',
        'api_key': 'YOUR_API_KEY',
    }
]

# Create agents
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)

planner = autogen.AssistantAgent(
    name="Planner",
    system_message="Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.",
    llm_config={"config_list": config_list},
)

# Start a group chat
groupchat = autogen.GroupChat(agents=[user_proxy, planner], messages=[], max_round=50)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

# Initiate chat
user_proxy.initiate_chat(manager, message="Plan a trip from Beijing to New York")
```

### AutoGen Multi-Agent Workflow

AutoGen supports several communication patterns:
- **Sequential**: Agents communicate in a predefined sequence
- **Round-robin**: Agents take turns communicating
- **Broadcast**: One agent sends message to all others
- **Selective**: Agents send messages to specific recipients

Example of a complex workflow:

```python
import autogen

# Define multiple specialized agents
researcher = autogen.AssistantAgent(
    name="Researcher",
    system_message="Researcher. You search for information online and provide facts.",
    llm_config={"config_list": config_list},
)

writer = autogen.AssistantAgent(
    name="Writer",
    system_message="Writer. You write detailed articles based on research findings.",
    llm_config={"config_list": config_list},
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    code_execution_config=False,
)

# Create a group chat with multiple agents
groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, writer], 
    messages=[],
    max_round=20,
    speaker_selection_method="round_robin"  # or "auto" for LLM-driven selection
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

# Research and write an article
user_proxy.initiate_chat(
    manager, 
    message="Research and write an article about the latest advances in agentic AI."
)
```

## LlamaIndex: Data Framework for LLM Applications

LlamaIndex specializes in connecting custom data sources to language models, making it ideal for agentic systems that need to work with private or domain-specific data.

### Key Components for Agentic Systems

#### 1. Data Connectors
LlamaIndex provides connectors for various data sources:
- Files (PDF, DOCX, TXT, etc.)
- Databases (SQL, NoSQL)
- APIs and web sources
- Cloud storage services

#### 2. Indexing and Retrieval
```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader('data/').load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query the data
query_engine = index.as_query_engine()
response = query_engine.query("What are the key points about agentic AI?")
```

#### 3. Agentic Patterns
LlamaIndex supports:
- **SubQuestion Query Engine**: Breaks complex queries into sub-queries
- **Router Query Engine**: Routes queries to the most appropriate data source
- **Context Augmentation**: Enhances agent context with relevant data

### Complete LlamaIndex Agentic Example

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import OpenAIAgent
from llama_index.prompts import PromptTemplate

# Load and index documents
documents = SimpleDirectoryReader('data/').load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

# Create a tool for the index
agentic_ai_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="agentic_ai_docs",
        description="Provides information about agentic AI concepts, implementation, and best practices. Use this for any questions related to agentic AI."
    )
)

# Create the agent
llm = OpenAI(model="gpt-4")
agent = OpenAIAgent.from_tools([agentic_ai_tool], llm=llm, verbose=True)

# Use the agent
response = agent.chat("Explain the key components of an agentic AI system based on the documentation.")
```

## CrewAI: Crew of AI Agents

CrewAI focuses on creating collaborative multi-agent systems that work together to solve complex tasks.

### Core Concepts

#### 1. Agents
```python
from crewai import Agent

researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in agentic AI',
    backstory='You are a senior research analyst with expertise in AI technologies.',
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role='Tech Content Writer',
    goal='Write comprehensive articles about technical topics',
    backstory='You are a skilled content writer with a background in AI and technology.',
    verbose=True,
    allow_delegation=True
)
```

#### 2. Tasks
```python
from crewai import Task

research_task = Task(
    description='Research the latest developments in agentic AI',
    agent=researcher,
    expected_output='A detailed report on recent advances in agentic AI'
)

writing_task = Task(
    description='Write an article based on research findings',
    agent=writer,
    expected_output='A well-structured article about agentic AI'
)
```

#### 3. Crew
```python
from crewai import Crew, Process

# Create the crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # or Process.hierarchical
    verbose=2
)

# Execute the crew
result = crew.kickoff()
print(result)
```

## Custom Implementation

For maximum flexibility, you can implement your own agentic systems using basic components.

### Basic Agent Loop Implementation

```python
class SimpleAgenticSystem:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []
    
    def perceive(self, observation):
        """Add new observation to memory"""
        self.memory.append({"type": "observation", "content": observation})
    
    def think(self, goal):
        """Generate a plan or decision based on current state"""
        # Use LLM to reason about current state and goal
        context = self._format_memory()
        prompt = f"Given the context: {context}\nGoal: {goal}\nWhat should be the next step?"
        
        response = self.llm.generate(prompt)
        return response
    
    def act(self, action_plan):
        """Execute the planned action"""
        # Parse and execute the planned action
        for action in action_plan:
            if action["type"] in self.tools:
                result = self.tools[action["type"]](**action["params"])
                self.perceive(result)
            else:
                # Handle unknown actions
                self.perceive(f"Unknown action: {action['type']}")
    
    def _format_memory(self):
        """Format memory for LLM consumption"""
        return "\n".join([f"{item['type']}: {item['content']}" for item in self.memory[-5:]])
    
    def run(self, goal):
        """Main execution loop"""
        while not self._is_goal_achieved(goal):
            plan = self.think(goal)
            self.act(plan)
        
        return self._get_final_result()
```

## Implementation Best Practices

### 1. Error Handling and Fallbacks
Always implement robust error handling for tool failures and unexpected LLM outputs.

### 2. Rate Limiting and Cost Management
Monitor API usage and implement rate limiting to manage costs effectively.

### 3. Logging and Observability
Maintain detailed logs of agent decisions and actions for debugging and improvement.

### 4. Testing and Evaluation
Create comprehensive test suites for your agentic systems using both synthetic and real-world scenarios.

### 5. Security Considerations
Implement proper input validation and output sanitization to prevent injection attacks.

## Performance Optimization

### Caching
Cache results of expensive operations to improve response times and reduce costs.

### Parallelization
Execute independent tasks in parallel when possible to improve throughput.

### Model Selection
Choose the right model for the task - more complex models for reasoning, simpler models for straightforward tasks.

## Next Steps

Now that we've explored implementation approaches, let's dive into advanced agentic AI concepts that push the boundaries of what agents can accomplish.