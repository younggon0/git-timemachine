# Codebase Time Machine - 2 Hour Plan

## Tech Stack
- **Package Manager**: uv
- **UI**: Streamlit
- **LLM**: Anthropic Claude via LangChain
- **Git**: GitPython
- **Vector Store**: FAISS
- **Viz**: Plotly

## MVP Features
1. Clone repo from URL
2. Index commits/diffs into vector store
3. Natural language Q&A about code evolution
4. Visualize commit trends & file ownership

## Timeline
- 0:00-0:15: Setup project, dependencies
- 0:15-0:45: Git parsing + indexing
- 0:45-1:15: LangChain + Claude integration
- 1:15-1:30: Visualizations
- 1:30-1:45: Polish UI
- 1:45-2:00: Deploy to Streamlit.io

## Key Decisions
- Use FAISS (no DB setup needed)
- Index only last 100 commits (speed)
- Pre-built prompts for common questions
- Focus on commit messages + diffs (skip full file analysis)