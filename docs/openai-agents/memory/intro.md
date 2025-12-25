---
sidebar_position: 5
---

# Memory Systems in AI Agents

Memory is a foundational element for creating truly intelligent and adaptive AI Agents. To effectively master the Memory Layer, adopting an AI Researcher's mindset is key.

## Learning Plan

We will focus on building conceptual understanding and then applying it through specific, powerful frameworks like LangMem and Zep.

### Phase 1: Foundational Understanding (Thinking Like a Researcher)
- Understand Core Memory Concepts:
  - Dive deep into the different types of memory crucial for agents: Episodic (experiences), Semantic (facts, knowledge), Procedural (how-to, rules), and Temporal (time-awareness).
  - Use human memory as an analogy to grasp the nuances and importance of each type.
  - Study memory mechanisms: Hot Path/Short-Term (immediate context) vs. Background/Long-Term (persistent storage and asynchronous updates).
- **Goal:** Build a strong theoretical framework for why and what an agent needs to remember.

### Phase 2: Practical Implementation with LangMem
- Implement Concepts with LangMem:
  - Why LangMem first? LangMem is designed with a strong conceptual alignment to human memory types and features a clear separation between its Core API (memory operations) and Storage, aiding foundational understanding.
  - Implement basic memory operations: Storing facts, recalling past interactions, and managing procedural rules using LangMem's tools (create_manage_memory_tool, create_search_memory_tool).
- **Goal:** Gain hands-on experience implementing different memory types and understand LangMem's specific approach to memory management.

### Phase 3: Exploring Advanced Concepts with Zep
- Explore Temporal and Graph-Based Memory with Zep:
  - Why Zep next? Zep offers a different, powerful approach centered around a temporal knowledge graph, known for strong performance (especially low latency) and features suited for complex, evolving contexts often seen in enterprise applications. Learning Zep provides exposure to alternative state-of-the-art memory architectures.

### Phase 4: Broaden Research, Synthesis and Mastery
- Investigate other frameworks like MemGPT and Mem0. Study how memory influences agent response generation.

## Memory Types in Agentic Frameworks

### Episodic Memory
- These are experiences that can help agents remember how to do tasks.
- Stores individual interaction "episodes" or past events.
- This stores specific, personal experiences or "episodes" of the agent's interaction with the environment or users.
- Think of it as a log of what the agent has done and observed.
- In an OpenAI Agents SDK context, this could involve storing:
  - User inputs and agent responses.
  - API call results.
  - Observations from tools the agent used.
  - Timestamps associated with each event.
- Example: "At 2:30 PM, the user asked for the weather in London, and I called the weather API, which returned 15 degrees Celsius."

### Semantic Memory
- Stores facts, user profiles, and external knowledge.
- This stores general knowledge and facts about the world.
- It's about understanding concepts, relationships, and meanings.
- In the Agents SDK, this could be:
  - Pre-trained knowledge embedded in the language model.
  - Knowledge retrieved from external knowledge bases (e.g., databases, APIs).
  - Information extracted and stored from previous interactions.
- Example: "London is the capital of the United Kingdom." or "The capital of France is Paris."
- Example: "Important Birthdates for Calendar Agent"

### Procedural Memory
- Rules for Agents to follow.
- Holds the agent's "how-to" information, such as system instructions and operational rules.
- This stores knowledge about how to do things, or "procedures."
- It's about skills, habits, and learned behaviors.
- It's used to update your system prompt dynamically.
- In the Agents SDK, this could involve:
  - Storing sequences of actions or tool calls that have been successful in the past.
  - Learning and refining strategies for achieving specific goals.
  - Storing the result of function calls, and the proper way to call those functions.
- Example: "To get the weather, first, call the weather API with the city name, then parse the temperature from the response."

### Temporal Memory
- Captures the order and timing of events or interactions.
- This is related to the agents ability to track and understand the passage of time, and the order of events.
- This is often tightly coupled with episodic memory, but can also be considered its own type.
- It can be updated using a dynamic, temporally-aware knowledge graph that continuously integrates new information while tracking how relationships and facts evolve over time.
- Bi-Temporal Modeling: Instead of a single timeline, the system employs dual time-tracking. One timeline captures the actual occurrence of events (when facts were valid), while a second timeline records the order in which data was ingested. This dual approach allows the memory layer to mark outdated information as invalid when new, conflicting data is received.
- Dynamic Knowledge Integration: As new interactions occur or business data changes, the system extracts facts and relationships from both unstructured and structured sources. These facts are enriched with temporal metadata—such as timestamps indicating when a fact became valid and when it was superseded—ensuring that only current and relevant information is used.
- Hybrid Retrieval: For efficient recall, the memory layer combines semantic similarity search, full-text retrieval, and graph-based queries. This hybrid approach ensures that the most pertinent, time-sensitive context is retrieved quickly, while also preserving the underlying relationships between entities.
- In the Agents SDK, this would include:
  - Timestamps on all events.
  - The ability to understand "before" and "after" relationships.
  - The ability to understand time-based context for actions.

## Memory Mechanisms

### Hot Path (Immediate/Short-Term) Memory
- This refers to the agent's immediate working memory.
- It's used for processing the current task and keeping track of recent interactions.
- In the Agents SDK, this would involve:
  - The context window of the language model.
  - Variables and data structures used during the execution of a function call.
  - The current conversation history.
- This is the memory that the agent uses to create its next response.

### Background (Long-Term) Memory
- This refers to the agent's persistent memory, which is stored and retrieved over longer periods.
- In the Agents SDK, this would involve:
  - External databases or vector stores.
  - File storage.
  - Knowledge graphs.
- This is the memory that the agent uses to inform its actions over multiple conversations.
- Memory updates are performed asynchronously. A separate process gathers and organizes new information after the primary response is generated, which decouples memory maintenance from the immediate conversation flow and reduces latency.

## Memory Implementation Strategies

### You can:

- **Configure Memory Stores:**
  Set up persistent memory stores (such as vector databases, graph databases or JSON-based stores) for each type of memory. Semantic and episodic memories are often stored for retrieval during future interactions, while procedural memory is used to update your system prompt dynamically.

- **Customize Update Strategies:**
  Choose between hot path updates for immediate context enrichment and background processes for less time-critical memory updates. This flexibility lets you balance performance and context preservation based on your application's needs.

- **Leverage Memory in Prompts:**
  When constructing your system prompt for an agent call, you can inject relevant memories—whether they are high-level summaries (from episodic memory) or key facts (from semantic memory)—to make responses more context-aware and personalized. Temporal memory helps ensure that the most recent interactions carry appropriate weight.

By combining these different memory types and update mechanisms, your agent becomes better equipped to maintain a coherent, contextually rich conversation over long interactions, adapt its behavior based on past events, and improve over time.

## Popular Memory Solutions

### LangMem
LangMem is engineered to equip AI agents with long-term memory, allowing them to store important details from conversations, refine their behavior based on experience, and maintain knowledge across different sessions. Key features include:

- Storage-agnostic memory API
- Active "hot-path" memory tools
- Background memory manager
- Native integration with LangGraph's long-term memory store

LangMem categorizes long-term memory into three primary types: semantic memory for facts and knowledge, episodic memory for past experiences, and procedural memory for system instructions and learned behaviors.

### Zep
Zep presents itself as a long-term memory service for agentic applications, utilizing a temporal knowledge graph to continuously learn from user interactions and evolving business data. Zep autonomously constructs a knowledge graph for each user, capturing entities, relationships, and facts, and importantly, tracking how these evolve over time.

### MemGPT
MemGPT is an innovative framework inspired by operating system architectures, specifically designed to enable LLMs to manage their own memory and overcome the limitations of restricted context windows. The core concept involves a memory hierarchy, comprising in-context (core) memory and out-of-context (archival) memory, managed by the agent itself through tool calls.

## Integration with OpenAI Agents SDK

The OpenAI Agents SDK provides a framework for building agentic AI applications. Several approaches can be employed to integrate memory solutions:

1. **Wrapping memory APIs as tools** within the OpenAI Agents SDK. This allows the agent to utilize memory's functionalities, such as storing and searching for information, as part of its available tools.

2. **Using persistent storage systems** that work with the memory solution to ensure that the agent's memories are retained across sessions.

3. **Implementing proper namespacing** to manage context, especially in applications involving multiple users, preventing data from different users or contexts from interfering with each other.

This layered memory design is one of the key innovations in building robust agentic systems with the OpenAI Agents SDK. It helps transform a stateless model into one that is dynamically adaptive and contextually aware, closely mimicking how human memory contributes to intelligent behavior.