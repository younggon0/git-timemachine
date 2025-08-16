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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv('ANTHROPIC_API_KEY')

st.set_page_config(
    page_title="Codebase Time Machine",
    page_icon="⏰",
    layout="wide"
)

st.title("⏰ Codebase Time Machine")
st.markdown("Navigate any codebase through time with semantic understanding")

# Initialize session state
if 'repo' not in st.session_state:
    st.session_state.repo = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'commits_df' not in st.session_state:
    st.session_state.commits_df = None

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    repo_url = st.text_input("Repository URL", placeholder="https://github.com/user/repo.git")
    
    # Show API key status
    if api_key:
        st.success("✅ API key loaded from .env")
    else:
        st.warning("⚠️ No API key found. Please set ANTHROPIC_API_KEY in .env file")
    
    max_commits = st.slider("Max commits to analyze", 10, 500, 100)
    
    if st.button("🔄 Analyze Repository", type="primary"):
        if not repo_url:
            st.error("Please enter a repository URL")
        elif not api_key:
            st.error("Please set ANTHROPIC_API_KEY in .env file")
        else:
            with st.spinner("Cloning and analyzing repository..."):
                # Clone repository
                temp_dir = tempfile.mkdtemp()
                try:
                    repo = git.Repo.clone_from(repo_url, temp_dir)
                    st.session_state.repo = repo
                    st.success(f"✅ Cloned repository: {repo_url}")
                    
                    # Process commits
                    commits_data = []
                    documents = []
                    
                    for i, commit in enumerate(repo.iter_commits(max_count=max_commits)):
                        # Collect commit data
                        commits_data.append({
                            'hash': commit.hexsha[:8],
                            'author': commit.author.name,
                            'date': datetime.fromtimestamp(commit.committed_date),
                            'message': commit.message.strip(),
                            'files_changed': len(commit.stats.files),
                            'insertions': commit.stats.total['insertions'],
                            'deletions': commit.stats.total['deletions']
                        })
                        
                        # Create document for vector store
                        doc_content = f"""
                        Commit: {commit.hexsha}
                        Author: {commit.author.name}
                        Date: {datetime.fromtimestamp(commit.committed_date)}
                        Message: {commit.message}
                        
                        Files changed:
                        """
                        
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
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
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
    tab1, tab2, tab3 = st.tabs(["🤖 Ask Questions", "📊 Visualizations", "📜 Commit History"])
    
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