import streamlit as st
import git
import os
import shutil
import tempfile
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Import new semantic analysis components
try:
    from code_analyzer import CodeAnalyzer, CodeChange
    from semantic_indexer import SemanticIndexer
    from context_enricher import ContextEnricher
    SEMANTIC_ANALYSIS_AVAILABLE = True
except ImportError as e:
    SEMANTIC_ANALYSIS_AVAILABLE = False
    print(f"Semantic analysis modules not available: {e}")
# Try to load API keys from environment or Streamlit secrets
try:
    api_key = os.getenv('ANTHROPIC_API_KEY') or st.secrets.get('ANTHROPIC_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')
    github_token = os.getenv('GITHUB_TOKEN') or st.secrets.get('GITHUB_TOKEN', None)
except Exception as e:
    st.error(f"Error loading secrets: {e}")
    api_key = None
    openai_api_key = None
    github_token = None

st.set_page_config(
    page_title="Codebase Time Machine",
    page_icon="⏰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-username/git-timemachine',
        'Report a bug': 'https://github.com/your-username/git-timemachine/issues',
        'About': "# Git Time Machine\nAnalyze any codebase through time with AI-powered semantic understanding."
    }
)

st.title("⏰ Codebase Time Machine")
st.markdown("Navigate any codebase through time with semantic understanding")

# Add a demo section for public users
with st.expander("📖 How to Use This App", expanded=False):
    st.markdown("""
    1. **Enter a Repository URL**: Paste any public GitHub repository URL
    2. **Set API Keys**: Contact admin or use your own Anthropic & OpenAI API keys
    3. **Analyze**: Click "Analyze Repository" to process the git history
    4. **Ask Questions**: Use natural language to explore code evolution
    
    **Example Questions**:
    - "What major refactoring happened recently?"
    - "How did the authentication system evolve?"
    - "What patterns were introduced for error handling?"
    """)

# Add demo repository suggestions
with st.expander("🚀 Try These Demo Repositories", expanded=False):
    demo_repos = [
        "https://github.com/microsoft/vscode",
        "https://github.com/facebook/react", 
        "https://github.com/tensorflow/tensorflow",
        "https://github.com/rails/rails"
    ]
    for repo in demo_repos:
        if st.button(f"Load {repo.split('/')[-1]}", key=repo):
            st.session_state.demo_repo = repo
            st.rerun()

# Initialize session state
if 'repo' not in st.session_state:
    st.session_state.repo = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'commits_df' not in st.session_state:
    st.session_state.commits_df = None
if 'code_changes' not in st.session_state:
    st.session_state.code_changes = None
if 'semantic_indexer' not in st.session_state:
    st.session_state.semantic_indexer = None
if 'business_contexts' not in st.session_state:
    st.session_state.business_contexts = None
if 'evolution_summary' not in st.session_state:
    st.session_state.evolution_summary = None

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Auto-fill from demo selection
    default_url = st.session_state.get('demo_repo', '')
    repo_url = st.text_input("Repository URL", 
                            value=default_url,
                            placeholder="https://github.com/user/repo.git")
    
    # Show API key status
    if api_key and openai_api_key:
        st.success("✅ API keys loaded")
        if github_token:
            st.success("✅ GitHub integration enabled")
        else:
            st.info("ℹ️ GitHub token not provided (optional for PR/issue linking)")
    else:
        missing = []
        if not api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not openai_api_key:
            missing.append("OPENAI_API_KEY") 
        st.warning(f"⚠️ Missing: {', '.join(missing)}")
    
    max_commits = st.slider("Max commits to analyze", 10, 500, 50)
    
    if SEMANTIC_ANALYSIS_AVAILABLE:
        enable_deep_analysis = st.checkbox("Enable Deep Code Analysis", value=False, 
                                          help="⚠️ SLOW: Analyzes actual code changes with AST parsing (adds 1-2 min for 100 commits)")
        
        enable_pr_integration = st.checkbox("Enable GitHub PR/Issue Integration", 
                                           value=False,
                                           help="Requires GitHub token. Links commits to PRs and issues for business context")
    else:
        enable_deep_analysis = False
        enable_pr_integration = False
        st.info("Advanced semantic analysis features are disabled. Install additional dependencies to enable.")
    
    if st.button("🔄 Analyze Repository", type="primary"):
        if not repo_url:
            st.error("Please enter a repository URL")
        elif not api_key or not openai_api_key:
            st.error("Please set ANTHROPIC_API_KEY and OPENAI_API_KEY environment variables")
        else:
            with st.spinner("Cloning and analyzing repository..."):
                # Clone repository
                temp_dir = tempfile.mkdtemp()
                try:
                    repo = git.Repo.clone_from(repo_url, temp_dir)
                    st.session_state.repo = repo
                    st.success(f"✅ Cloned repository: {repo_url}")
                    
                    # Deep semantic analysis if enabled
                    if enable_deep_analysis:
                        st.warning("⚠️ Deep analysis enabled - this may take 1-2 minutes for large repositories")
                        progress_bar = st.progress(0, text="Starting deep code analysis...")
                        analyzer = CodeAnalyzer()
                        commits_list = list(repo.iter_commits(max_count=max_commits))
                        
                        # Analyze with progress updates
                        code_changes = []
                        for i, commit in enumerate(commits_list):
                            progress = (i + 1) / len(commits_list)
                            progress_bar.progress(progress, text=f"Analyzing commit {i+1}/{len(commits_list)}: {commit.hexsha[:8]}")
                            change = analyzer._analyze_commit(commit, max_files=5)  # Limit to 5 files per commit for speed
                            if change:
                                code_changes.append(change)
                        
                        st.session_state.code_changes = code_changes
                        
                        # Generate evolution summary
                        st.session_state.evolution_summary = analyzer.get_evolution_summary(code_changes)
                        progress_bar.empty()
                        st.success(f"✅ Analyzed {len(code_changes)} commits with semantic understanding")
                    
                    # Business context enrichment if enabled
                    if enable_pr_integration and github_token:
                        st.warning("⚠️ GitHub integration enabled - enriching first 20 commits to avoid rate limits")
                        progress_bar = st.progress(0, text="Starting GitHub PR/Issue enrichment...")
                        
                        enricher = ContextEnricher(github_token)
                        commits_list = list(repo.iter_commits(max_count=max_commits))
                        
                        # Create a custom progress callback
                        def progress_callback(current, total):
                            progress = current / total if total > 0 else 0
                            progress_bar.progress(progress, text=f"Enriching commit {current}/{total} with GitHub data...")
                        
                        # Enrich with progress updates (only first 20 commits)
                        business_contexts = {}
                        max_enriched = min(20, len(commits_list))  # Limit to 20 commits
                        
                        # First, get local context for all commits
                        for commit in commits_list:
                            business_contexts[commit.hexsha] = enricher._extract_local_context(commit)
                        
                        # Then enrich top commits with GitHub data
                        repo_info = enricher._parse_repo_url(repo_url)
                        if repo_info:
                            try:
                                github_repo = enricher.github_client.get_repo(f"{repo_info['owner']}/{repo_info['repo']}")
                                
                                for i, commit in enumerate(commits_list[:max_enriched]):
                                    progress_callback(i + 1, max_enriched)
                                    enricher._enrich_with_github_data_limited(
                                        business_contexts[commit.hexsha], 
                                        commit, 
                                        github_repo
                                    )
                            except Exception as e:
                                st.error(f"GitHub API error: {e}")
                        
                        st.session_state.business_contexts = business_contexts
                        progress_bar.empty()
                        st.success(f"✅ Enriched {max_enriched} commits with GitHub context, {len(commits_list) - max_enriched} with local analysis only")
                    
                    # Process commits for basic analysis
                    commits_data = []
                    documents = []
                    
                    for i, commit in enumerate(repo.iter_commits(max_count=max_commits)):
                        # Collect commit data
                        commit_data = {
                            'hash': commit.hexsha[:8],
                            'author': commit.author.name,
                            'date': datetime.fromtimestamp(commit.committed_date),
                            'message': commit.message.strip(),
                            'files_changed': len(commit.stats.files),
                            'insertions': commit.stats.total['insertions'],
                            'deletions': commit.stats.total['deletions']
                        }
                        
                        # Add semantic analysis data if available
                        if enable_deep_analysis and st.session_state.code_changes:
                            for change in st.session_state.code_changes:
                                if change.commit_hash == commit.hexsha:
                                    commit_data['change_type'] = change.change_type
                                    commit_data['patterns'] = ', '.join(change.patterns_detected)
                                    commit_data['complexity_delta'] = change.complexity_delta
                                    break
                        
                        commits_data.append(commit_data)
                        
                        # Create enhanced document for vector store
                        doc_content = f"""
                        Commit: {commit.hexsha}
                        Author: {commit.author.name}
                        Date: {datetime.fromtimestamp(commit.committed_date)}
                        Message: {commit.message}
                        """
                        
                        # Add semantic analysis to document
                        if enable_deep_analysis and st.session_state.code_changes:
                            for change in st.session_state.code_changes:
                                if change.commit_hash == commit.hexsha:
                                    doc_content += f"\nChange Type: {change.change_type}"
                                    doc_content += f"\nPatterns: {', '.join(change.patterns_detected)}"
                                    doc_content += f"\nComplexity Delta: {change.complexity_delta}"
                                    doc_content += f"\nEntities Added: {', '.join([e.name for e in change.entities_added])}"
                                    doc_content += f"\nEntities Modified: {', '.join([e.name for e in change.entities_modified])}"
                                    break
                        
                        # Add business context to document
                        if enable_pr_integration and st.session_state.business_contexts:
                            if commit.hexsha in st.session_state.business_contexts:
                                context = st.session_state.business_contexts[commit.hexsha]
                                if context.business_impact:
                                    doc_content += f"\nBusiness Impact: {context.business_impact}"
                                if context.decision_rationale:
                                    doc_content += f"\nDecision: {context.decision_rationale}"
                                if context.feature_tags:
                                    doc_content += f"\nFeatures: {', '.join(context.feature_tags)}"
                        
                        doc_content += "\nFiles changed:"
                        
                        # Add diff information
                        if commit.parents:
                            diffs = commit.diff(commit.parents[0])
                            for diff in diffs[:10]:  # Limit to 10 files per commit
                                doc_content += f"\n{diff.a_path or diff.b_path}: "
                                if diff.new_file:
                                    doc_content += "new file"
                                elif diff.deleted_file:
                                    doc_content += "deleted"
                                elif diff.renamed_file:
                                    doc_content += f"renamed from {diff.a_path}"
                                else:
                                    doc_content += "modified"
                        
                        documents.append(Document(
                            page_content=doc_content,
                            metadata={
                                'commit_hash': commit.hexsha,
                                'author': commit.author.name,
                                'date': str(datetime.fromtimestamp(commit.committed_date))
                            }
                        ))
                    
                    st.session_state.commits_df = pd.DataFrame(commits_data)
                    
                    # Create vector store
                    with st.spinner("Building semantic index..."):
                        embeddings = OpenAIEmbeddings(
                            api_key=openai_api_key
                        )
                        st.session_state.vector_store = FAISS.from_documents(
                            documents, embeddings
                        )
                    
                    st.success(f"✅ Indexed {len(documents)} commits")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    # Clean up temp directory
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

# Main content area
if st.session_state.commits_df is not None:
    tabs = ["🤖 Ask Questions", "📊 Visualizations", "📜 Commit History"]
    
    # Add semantic analysis tabs if enabled
    if st.session_state.code_changes:
        tabs.append("🧠 Semantic Analysis")
    if st.session_state.business_contexts:
        tabs.append("💼 Business Context")
    
    tab_objects = st.tabs(tabs)
    tab1, tab2, tab3 = tab_objects[0], tab_objects[1], tab_objects[2]
    tab4 = tab_objects[3] if len(tab_objects) > 3 else None
    tab5 = tab_objects[4] if len(tab_objects) > 4 else None
    
    with tab1:
        st.header("Ask Questions About Code Evolution")
        
        # Pre-built questions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Why was this pattern introduced?"):
                st.session_state.question = "What patterns or architectural decisions were introduced and why?"
        with col2:
            if st.button("How did authentication evolve?"):
                st.session_state.question = "How did the authentication system evolve over time?"
        
        # Custom question input
        question = st.text_area(
            "Ask a question about the codebase evolution:",
            placeholder="e.g., 'What major refactoring happened?' or 'How did the API design change?'",
            height=100
        )
        
        if st.button("🔍 Search", type="primary") and question and api_key:
            with st.spinner("Analyzing codebase history..."):
                try:
                    # Initialize LLM
                    llm = ChatAnthropic(
                        api_key=api_key,
                        model="claude-3-haiku-20240307",
                        temperature=0
                    )
                    
                    # Create prompt template
                    prompt_template = """You are analyzing a git repository's history to understand code evolution.
                    Use the following commit information to answer the question.
                    
                    Context from relevant commits:
                    {context}
                    
                    Question: {question}
                    
                    Provide a detailed answer explaining the evolution, patterns, and decisions based on the commit history.
                    Include specific commits and dates when relevant.
                    """
                    
                    prompt = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )
                    
                    # Create QA chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vector_store.as_retriever(
                            search_kwargs={"k": 10}
                        ),
                        chain_type_kwargs={"prompt": prompt}
                    )
                    
                    # Get answer
                    result = qa_chain.run(question)
                    
                    st.markdown("### Answer:")
                    st.markdown(result)
                    
                except Exception as e:
                    st.error(f"Error querying: {str(e)}")
    
    with tab2:
        st.header("Repository Visualizations")
        
        df = st.session_state.commits_df
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Commit frequency over time
            fig = px.histogram(
                df, x='date', 
                title='Commit Frequency Over Time',
                labels={'date': 'Date', 'count': 'Number of Commits'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top contributors
            contributor_stats = df.groupby('author').agg({
                'hash': 'count',
                'insertions': 'sum',
                'deletions': 'sum'
            }).sort_values('hash', ascending=False).head(10)
            
            fig = px.bar(
                contributor_stats.reset_index(), 
                x='author', y='hash',
                title='Top Contributors by Commit Count',
                labels={'hash': 'Commits', 'author': 'Author'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Code churn over time
            df['code_churn'] = df['insertions'] + df['deletions']
            fig = px.line(
                df, x='date', y='code_churn',
                title='Code Churn Over Time',
                labels={'code_churn': 'Lines Changed', 'date': 'Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Files changed distribution
            fig = px.box(
                df, y='files_changed',
                title='Files Changed per Commit Distribution',
                labels={'files_changed': 'Number of Files'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Commit History")
        
        # Search/filter commits
        search_term = st.text_input("Search commits", placeholder="Search by message, author...")
        
        df_display = st.session_state.commits_df.copy()
        if search_term:
            mask = (df_display['message'].str.contains(search_term, case=False, na=False) | 
                   df_display['author'].str.contains(search_term, case=False, na=False))
            df_display = df_display[mask]
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True
        )
        
        st.metric("Total Commits Analyzed", len(st.session_state.commits_df))
    
    # Semantic Analysis Tab
    if tab4 and st.session_state.code_changes:
        with tab4:
            st.header("🧠 Semantic Code Analysis")
            
            if st.session_state.evolution_summary:
                summary = st.session_state.evolution_summary
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Functions Added", summary['entities_evolution']['functions_added'])
                    st.metric("Classes Added", summary['entities_evolution']['classes_added'])
                
                with col2:
                    st.metric("Functions Removed", summary['entities_evolution']['functions_removed'])
                    st.metric("Classes Removed", summary['entities_evolution']['classes_removed'])
                
                with col3:
                    total_complexity = sum([c['complexity'] for c in summary['complexity_trend']])
                    st.metric("Total Complexity Change", f"{'+' if total_complexity >= 0 else ''}{total_complexity}")
                
                # Change type distribution
                if summary['change_types']:
                    st.subheader("Change Type Distribution")
                    fig = px.pie(
                        values=list(summary['change_types'].values()),
                        names=list(summary['change_types'].keys()),
                        title="Types of Changes"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Patterns detected
                if summary['patterns_detected']:
                    st.subheader("Design Patterns Detected")
                    pattern_df = pd.DataFrame([
                        {'Pattern': pattern, 'Count': count}
                        for pattern, count in summary['patterns_detected'].items()
                    ])
                    st.dataframe(pattern_df, use_container_width=True)
                
                # Complexity trend
                if summary['complexity_trend']:
                    st.subheader("Code Complexity Evolution")
                    complexity_df = pd.DataFrame(summary['complexity_trend'])
                    if not complexity_df.empty:
                        complexity_df['date'] = pd.to_datetime(complexity_df['date'])
                        fig = px.line(
                            complexity_df, 
                            x='date', 
                            y='complexity',
                            title='Cumulative Complexity Over Time',
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Top refactored files
                if summary['top_refactored_files']:
                    st.subheader("Most Frequently Modified Files")
                    files_df = pd.DataFrame([
                        {'File': file, 'Modifications': count}
                        for file, count in summary['top_refactored_files'].items()
                    ])
                    st.dataframe(files_df, use_container_width=True)
                
                # Dependency changes
                if summary['dependency_changes']['added'] or summary['dependency_changes']['removed']:
                    st.subheader("Dependency Changes")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Dependencies Added:**")
                        for dep in summary['dependency_changes']['added'][:10]:
                            st.write(f"• {dep}")
                    with col2:
                        st.write("**Dependencies Removed:**")
                        for dep in summary['dependency_changes']['removed'][:10]:
                            st.write(f"• {dep}")
    
    # Business Context Tab
    if tab5 and st.session_state.business_contexts:
        with tab5:
            st.header("💼 Business Context & Decision Tracking")
            
            # Extract business insights
            contexts_with_content = [
                (hash, ctx) for hash, ctx in st.session_state.business_contexts.items()
                if ctx.business_impact or ctx.decision_rationale or ctx.issues or ctx.pull_requests
            ]
            
            if contexts_with_content:
                st.subheader("Key Business Decisions")
                
                for commit_hash, context in contexts_with_content[:20]:  # Show top 20
                    with st.expander(f"Commit {commit_hash[:8]}"):
                        if context.decision_rationale:
                            st.write(f"**Decision:** {context.decision_rationale}")
                        if context.business_impact:
                            st.write(f"**Impact:** {context.business_impact}")
                        if context.feature_tags:
                            st.write(f"**Features:** {', '.join(context.feature_tags)}")
                        if context.issues:
                            issues_str = ', '.join([f"#{issue.number}" for issue in context.issues])
                            st.write(f"**Issues Resolved:** {issues_str}")
                        if context.pull_requests:
                            prs_str = ', '.join([f"#{pr.number}" for pr in context.pull_requests])
                            st.write(f"**Pull Requests:** {prs_str}")
                
                # Feature impact summary
                st.subheader("Feature Impact Summary")
                feature_counts = {}
                for _, context in st.session_state.business_contexts.items():
                    for tag in context.feature_tags:
                        feature_counts[tag] = feature_counts.get(tag, 0) + 1
                
                if feature_counts:
                    feature_df = pd.DataFrame([
                        {'Feature Area': feature, 'Commits': count}
                        for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
                    ])
                    
                    fig = px.bar(
                        feature_df,
                        x='Feature Area',
                        y='Commits',
                        title='Commits by Feature Area'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No business context found. Consider adding a GitHub token to enable PR/issue integration.")

else:
    st.info("👈 Enter a repository URL and API key in the sidebar to get started")
    
    st.markdown("""
    ### Features:
    - 🔍 **Semantic Search**: Ask natural language questions about code evolution
    - 📊 **Visualizations**: See commit trends, contributor stats, and code churn
    - 🕐 **Time Navigation**: Explore how your codebase changed over time
    - 🤖 **AI-Powered Analysis**: Understand the "why" behind code changes
    
    ### Example Questions:
    - "What major refactoring happened in the last 6 months?"
    - "How did the authentication system evolve?"
    - "What patterns were introduced for error handling?"
    - "Show me the evolution of the API design"
    """)