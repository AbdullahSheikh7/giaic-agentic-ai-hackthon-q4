import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Main sidebar for the entire documentation
  tutorialSidebar: [
    {
      type: "category",
      label: "Agentic AI Guide",
      collapsed: false,
      items: [
        "agentic-ai/intro",
        "agentic-ai/fundamentals",
        "agentic-ai/building-blocks",
        "agentic-ai/design-patterns",
        "agentic-ai/implementation",
        "agentic-ai/advanced-concepts",
        "agentic-ai/applications",
        "agentic-ai/evaluation",
        "agentic-ai/deployment",
        "agentic-ai/ethics",
        "agentic-ai/future",
        "agentic-ai/troubleshooting",
      ],
    },
    {
      type: "category",
      label: "AI Protocols",
      collapsed: false,
      items: [
        "protocols/intro",
        "protocols/mcp-concepts/intro",
        "protocols/mcp/intro",
        "protocols/a2a/intro",
        "protocols/llms-txt/intro",
      ],
    },
    {
      type: "category",
      label: "OpenAI Agents SDK",
      collapsed: false,
      items: [
        "openai-agents/intro",
        "openai-agents/hello-agent/intro",
        "openai-agents/tools/intro",
        "openai-agents/streaming/intro",
        "openai-agents/memory/intro",
        "openai-agents/tracing/intro",
        "openai-agents/guardrails/intro",
      ],
    },
    {
      type: "category",
      label: "Vector Databases",
      collapsed: false,
      items: ["vector-databases/intro"],
    },
    {
      type: "category",
      label: "AI Projects",
      collapsed: false,
      items: ["projects/intro", "projects/hello-world/intro"],
    },
  ],
};

export default sidebars;
