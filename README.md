---
title: Git Time Machine
emoji: ⏰
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Git Time Machine

Navigate any codebase through time with semantic understanding. Analyze git history using AI to understand code evolution patterns and decisions.

## Features

- **Semantic Search**: Ask natural language questions about code evolution
- **Deep Code Analysis**: Analyze actual code changes, not just commit messages
- **Pattern Detection**: Identify design patterns and refactoring trends
- **Business Context**: Link commits to PRs, issues, and business decisions
- **Visualizations**: See commit trends, contributor stats, and code complexity evolution
- **Time Navigation**: Explore how your codebase changed over time
- **AI-Powered Analysis**: Understand the "why" behind code changes

## What's New: Semantic Understanding

This enhanced version goes beyond simple commit message analysis:

### 1. Deep Code Analysis
- AST-based parsing of Python, JavaScript, and TypeScript code
- Tracks functions, classes, and methods with complexity metrics
- Detects design patterns (singleton, factory, observer, etc.)
- Monitors code complexity evolution over time

### 2. Business Context Integration
- Links commits to GitHub PRs and issues
- Extracts business rationale from commit messages and PR descriptions
- Maps code changes to feature areas
- Generates decision timelines

### 3. Semantic Indexing
- Multiple specialized indices for different aspects of code
- Enables queries about patterns, dependencies, and architecture
- Tracks entity relationships and evolution

## Setup

### Required API Keys

Set these as environment variables or Streamlit secrets:
- `ANTHROPIC_API_KEY` - For AI-powered analysis (required)
- `OPENAI_API_KEY` - For embeddings and semantic search (required)
- `GITHUB_TOKEN` - For PR/issue integration (optional but recommended)

### Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Example Questions

- "Why was this pattern introduced?"
- "How did the authentication system evolve?"
- "What major refactoring happened recently?"
- "Show me the evolution of the API design"
- "What decisions led to the current architecture?"
- "Which files have the highest complexity?"

## Advanced Features

### Semantic Analysis Tab
- Entity evolution (functions/classes added/removed)
- Code complexity trends
- Design pattern detection
- Dependency changes
- Most frequently refactored files

### Business Context Tab
- Key business decisions and rationale
- Feature impact analysis
- PR/issue linkage
- Decision timeline visualization

## How It Works

1. **Clone Repository**: Fetches the git repository locally
2. **Analyze Commits**: Performs deep code analysis on each commit
3. **Extract Context**: Links to PRs/issues for business context
4. **Build Indices**: Creates semantic indices for intelligent search
5. **Enable Queries**: Allows natural language questions about code evolution

## Limitations

- Currently supports Python, JavaScript, and TypeScript for deep analysis
- GitHub integration requires a personal access token
- Analysis depth depends on commit history quality
- Limited to configured maximum commits (default: 100)

## Contributing

This project demonstrates semantic code understanding capabilities. Feel free to extend with:
- Support for more programming languages
- Additional design pattern detection
- Integration with other version control systems
- More sophisticated complexity metrics