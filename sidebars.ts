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
      type: 'category',
      label: 'Agentic AI Guide',
      collapsed: false,
      items: [
        'intro',
        {
          type: 'category',
          label: 'Mega Chapter',
          collapsed: true,
          items: [
            'agentic-ai/mega-chapter/intro',
            'agentic-ai/mega-chapter/fundamentals',
            'agentic-ai/mega-chapter/building-blocks',
            'agentic-ai/mega-chapter/design-patterns',
            'agentic-ai/mega-chapter/implementation',
            'agentic-ai/mega-chapter/advanced-concepts',
            'agentic-ai/mega-chapter/applications',
            'agentic-ai/mega-chapter/evaluation',
            'agentic-ai/mega-chapter/deployment',
            'agentic-ai/mega-chapter/ethics',
            'agentic-ai/mega-chapter/future',
            'agentic-ai/mega-chapter/troubleshooting',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'AI Protocols',
      collapsed: false,
      items: [
        'protocols/intro',
        {
          type: 'category',
          label: 'Foundation Concepts',
          collapsed: true,
          items: [
            'protocols/mcp-concepts/intro',
          ],
        },
        {
          type: 'category',
          label: 'Model Context Protocol (MCP)',
          collapsed: true,
          items: [
            'protocols/mcp/intro',
          ],
        },
        {
          type: 'category',
          label: 'Agent-to-Agent Protocol (A2A)',
          collapsed: true,
          items: [
            'protocols/a2a/intro',
          ],
        },
        {
          type: 'category',
          label: 'LLMs.txt Protocol',
          collapsed: true,
          items: [
            'protocols/llms-txt/intro',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'OpenAI Agents SDK',
      collapsed: false,
      items: [
        'openai-agents/intro',
        {
          type: 'category',
          label: 'Hello Agent',
          collapsed: true,
          items: [
            'openai-agents/hello-agent/intro',
          ],
        },
        {
          type: 'category',
          label: 'Tools Integration',
          collapsed: true,
          items: [
            'openai-agents/tools/intro',
          ],
        },
        {
          type: 'category',
          label: 'Streaming Output',
          collapsed: true,
          items: [
            'openai-agents/streaming/intro',
          ],
        },
        {
          type: 'category',
          label: 'Memory Systems',
          collapsed: true,
          items: [
            'openai-agents/memory/intro',
          ],
        },
        {
          type: 'category',
          label: 'Tracing and Monitoring',
          collapsed: true,
          items: [
            'openai-agents/tracing/intro',
          ],
        },
        {
          type: 'category',
          label: 'Guardrails and Safety',
          collapsed: true,
          items: [
            'openai-agents/guardrails/intro',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Vector Databases',
      collapsed: false,
      items: [
        'vector-databases/intro',
      ],
    },
    {
      type: 'category',
      label: 'AI Projects',
      collapsed: false,
      items: [
        'projects/intro',
        {
          type: 'category',
          label: 'Hello World Agent',
          collapsed: true,
          items: [
            'projects/hello-world/intro',
          ],
        },
      ],
    },
  ],
};

export default sidebars;
