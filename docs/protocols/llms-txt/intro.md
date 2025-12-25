---
sidebar_position: 5
---

# LLMs.txt Protocol

The `llms.txt` protocol (and its comprehensive counterpart `llms-full.txt`) establishes standards for how website owners can declare permissions and guidelines for how LLMs and other AI agents should interact with their content. Similar to `robots.txt` for web crawlers, `llms.txt` provides explicit instructions for AI systems.

## Purpose

The `llms.txt` protocol addresses several key challenges:
- Responsible AI interaction with web content
- Permission management for AI agents
- Data usage guidelines for LLM training
- Rate limiting and access control for AI systems

## File Structure

The `llms.txt` file is placed in a website's root directory and contains directives similar to `robots.txt` but specific to AI agents:

```
# LLMs.txt - Guidelines for AI agents
User-Agent: *
Allow: /
Disallow: /private/
Disallow: /sensitive-data/

# Suggested rate limits for AI agents
Crawl-delay: 10

# Data usage policy
Data-Usage: training-allowed
Data-Usage: commercial-use-prohibited
```

## Directives

### Basic Directives
- **User-Agent**: Specifies which AI agents the rules apply to
- **Allow**: Explicitly allows access to specific paths
- **Disallow**: Blocks access to specific paths
- **Crawl-delay**: Specifies minimum delay between requests

### Advanced Directives
- **Data-Usage**: Defines how scraped data can be used
- **AI-Agent**: Specific rules for different AI agent types
- **Training-Data**: Guidelines for including content in training datasets

## Benefits

- **For Website Owners**: Control how AI systems access and use their content
- **For AI Developers**: Clear guidelines for responsible content access
- **For End Users**: Better quality AI interactions based on proper data access