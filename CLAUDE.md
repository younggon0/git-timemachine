# Project Context

Building a Codebase Time Machine for 2-hour hackathon using Streamlit.io deployment.

## Core Functionality
- Analyze git history semantically using Anthropic Claude
- Answer questions like "Why was X introduced?" and "How did Y evolve?"
- Use LangChain for LLM orchestration
- FAISS for vector search of code changes

## Implementation Notes
- Limit to last 100 commits for speed
- Focus on commit messages and diffs
- Use embeddings for semantic search
- Pre-built prompts for common code evolution questions

## Environment
- Use ANTHROPIC_API_KEY from environment
- Deploy to Streamlit.io
- Use uv for dependency management