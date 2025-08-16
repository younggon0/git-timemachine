import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from github import Github, GithubException
import git

@dataclass
class PullRequest:
    """Represents a pull request with metadata"""
    number: int
    title: str
    description: str
    author: str
    created_at: datetime
    merged_at: Optional[datetime]
    labels: List[str]
    linked_issues: List[int]
    review_comments: List[str]
    files_changed: List[str]

@dataclass
class Issue:
    """Represents an issue/ticket with metadata"""
    number: int
    title: str
    description: str
    author: str
    created_at: datetime
    closed_at: Optional[datetime]
    labels: List[str]
    milestone: Optional[str]
    
@dataclass
class BusinessContext:
    """Business context for a code change"""
    commit_hash: str
    pull_requests: List[PullRequest] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    feature_tags: List[str] = field(default_factory=list)
    business_impact: str = ""
    decision_rationale: str = ""
    
class ContextEnricher:
    """Enriches code changes with business context from GitHub/GitLab"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token
        self.github_client = Github(github_token) if github_token else None
        
        # Patterns for extracting references
        self.issue_patterns = [
            r'#(\d+)',                    # GitHub style: #123
            r'(?:fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)\s+#(\d+)',
            r'(?:issue|bug|feature)\s+#?(\d+)',
            r'(?:JIRA|TICKET|TASK)-(\d+)',  # JIRA style
        ]
        
        self.pr_patterns = [
            r'PR\s+#?(\d+)',
            r'pull request\s+#?(\d+)',
            r'merge\s+#?(\d+)',
        ]
        
        self.feature_keywords = {
            'authentication': ['auth', 'login', 'oauth', 'jwt', 'session', 'password'],
            'api': ['endpoint', 'rest', 'graphql', 'route', 'controller'],
            'database': ['db', 'sql', 'query', 'migration', 'schema', 'model'],
            'ui': ['component', 'view', 'template', 'style', 'css', 'frontend'],
            'performance': ['optimize', 'cache', 'speed', 'performance', 'lazy'],
            'security': ['security', 'vulnerability', 'xss', 'csrf', 'encryption'],
            'testing': ['test', 'spec', 'coverage', 'unit', 'integration', 'e2e'],
            'deployment': ['deploy', 'ci/cd', 'docker', 'kubernetes', 'pipeline'],
            'monitoring': ['log', 'metric', 'monitor', 'alert', 'telemetry'],
            'documentation': ['doc', 'readme', 'comment', 'guide', 'tutorial']
        }
        
        self.decision_indicators = [
            'because', 'in order to', 'to support', 'to enable',
            'due to', 'as per', 'requirement', 'requested by',
            'decision:', 'rationale:', 'reason:'
        ]
    
    def enrich_repository(self, repo_path: str, repo_url: str, commits: List[git.Commit], max_api_calls: int = 20) -> Dict[str, BusinessContext]:
        """Enrich commits with business context"""
        
        contexts = {}
        repo_info = self._parse_repo_url(repo_url)
        
        if not repo_info or not self.github_client:
            # Fallback to local analysis only
            print("No GitHub client configured, using local analysis only")
            for commit in commits:
                contexts[commit.hexsha] = self._extract_local_context(commit)
        else:
            # Limited GitHub integration to avoid rate limiting
            try:
                github_repo = self.github_client.get_repo(f"{repo_info['owner']}/{repo_info['repo']}")
                
                # First pass: Extract all local context
                for commit in commits:
                    contexts[commit.hexsha] = self._extract_local_context(commit)
                
                # Second pass: Enrich only recent commits with GitHub data (limit API calls)
                api_calls_made = 0
                for i, commit in enumerate(commits[:max_api_calls]):  # Only enrich first N commits
                    if api_calls_made >= max_api_calls:
                        print(f"Reached API call limit ({max_api_calls}), skipping remaining commits")
                        break
                    
                    print(f"Enriching commit {i+1}/{min(len(commits), max_api_calls)}: {commit.hexsha[:8]}")
                    context = contexts[commit.hexsha]
                    
                    # Enrich with GitHub data (simplified)
                    self._enrich_with_github_data_limited(context, commit, github_repo)
                    api_calls_made += 1
                    
            except GithubException as e:
                print(f"GitHub API error: {e}")
                # Already have local analysis from first pass
        
        return contexts
    
    def _extract_local_context(self, commit: git.Commit) -> BusinessContext:
        """Extract context from commit message and local information"""
        
        context = BusinessContext(commit_hash=commit.hexsha)
        message = commit.message
        
        # Extract issue references
        for pattern in self.issue_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                issue_num = int(match) if match.isdigit() else 0
                if issue_num:
                    # Create placeholder issue
                    issue = Issue(
                        number=issue_num,
                        title=f"Issue #{issue_num}",
                        description="",
                        author="",
                        created_at=datetime.fromtimestamp(commit.committed_date),
                        closed_at=None,
                        labels=[],
                        milestone=None
                    )
                    context.issues.append(issue)
        
        # Extract PR references
        for pattern in self.pr_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                pr_num = int(match) if match.isdigit() else 0
                if pr_num:
                    # Create placeholder PR
                    pr = PullRequest(
                        number=pr_num,
                        title=f"PR #{pr_num}",
                        description="",
                        author=commit.author.name,
                        created_at=datetime.fromtimestamp(commit.committed_date),
                        merged_at=datetime.fromtimestamp(commit.committed_date),
                        labels=[],
                        linked_issues=[],
                        review_comments=[],
                        files_changed=[]
                    )
                    context.pull_requests.append(pr)
        
        # Extract feature tags
        context.feature_tags = self._extract_feature_tags(message)
        
        # Extract business impact
        context.business_impact = self._extract_business_impact(message)
        
        # Extract decision rationale
        context.decision_rationale = self._extract_decision_rationale(message)
        
        return context
    
    def _enrich_with_github_data_limited(self, context: BusinessContext, commit: git.Commit, github_repo):
        """Enrich context with LIMITED GitHub API calls for performance"""
        
        try:
            # Get commit from GitHub (1 API call)
            github_commit = github_repo.get_commit(commit.hexsha)
            
            # Find associated PRs (1 API call) - but don't fetch extra data
            prs = list(github_commit.get_pulls())
            
            # Only process first PR to limit API calls
            if prs:
                pr = prs[0]  # Just get the first PR
                pull_request = PullRequest(
                    number=pr.number,
                    title=pr.title,
                    description=pr.body or "",
                    author=pr.user.login if pr.user else "unknown",
                    created_at=pr.created_at,
                    merged_at=pr.merged_at,
                    labels=[label.name for label in pr.labels],
                    linked_issues=self._extract_linked_issues(pr.body or ""),
                    review_comments=[],  # Skip fetching comments to save API calls
                    files_changed=[]  # Skip fetching files to save API calls
                )
                context.pull_requests.append(pull_request)
                
                # Extract more context from PR description
                if pr.body:
                    context.feature_tags.extend(self._extract_feature_tags(pr.body))
                    
                    if not context.business_impact:
                        context.business_impact = self._extract_business_impact(pr.body)
                    
                    if not context.decision_rationale:
                        context.decision_rationale = self._extract_decision_rationale(pr.body)
            
            # Skip fetching issue details to save API calls
            # Issues already have basic info from commit message parsing
                    
        except GithubException as e:
            print(f"Could not enrich commit {commit.hexsha[:8]}: {e}")
    
    def _enrich_with_github_data(self, context: BusinessContext, commit: git.Commit, github_repo):
        """Full enrichment with all GitHub data (SLOW - many API calls)"""
        
        try:
            # Get commit from GitHub
            github_commit = github_repo.get_commit(commit.hexsha)
            
            # Find associated PRs
            prs = github_commit.get_pulls()
            for pr in prs:
                pull_request = PullRequest(
                    number=pr.number,
                    title=pr.title,
                    description=pr.body or "",
                    author=pr.user.login if pr.user else "unknown",
                    created_at=pr.created_at,
                    merged_at=pr.merged_at,
                    labels=[label.name for label in pr.labels],
                    linked_issues=self._extract_linked_issues(pr.body or ""),
                    review_comments=self._get_review_comments(pr),
                    files_changed=[f.filename for f in pr.get_files()]
                )
                context.pull_requests.append(pull_request)
                
                # Extract more context from PR description
                if pr.body:
                    context.feature_tags.extend(self._extract_feature_tags(pr.body))
                    
                    if not context.business_impact:
                        context.business_impact = self._extract_business_impact(pr.body)
                    
                    if not context.decision_rationale:
                        context.decision_rationale = self._extract_decision_rationale(pr.body)
            
            # Get related issues
            for issue_num in context.issues:
                try:
                    issue = github_repo.get_issue(issue_num.number)
                    
                    # Update with real data
                    issue_num.title = issue.title
                    issue_num.description = issue.body or ""
                    issue_num.author = issue.user.login if issue.user else "unknown"
                    issue_num.created_at = issue.created_at
                    issue_num.closed_at = issue.closed_at
                    issue_num.labels = [label.name for label in issue.labels]
                    issue_num.milestone = issue.milestone.title if issue.milestone else None
                    
                except GithubException:
                    pass  # Issue not found or no access
                    
        except GithubException as e:
            print(f"Could not enrich commit {commit.hexsha[:8]}: {e}")
    
    def _parse_repo_url(self, url: str) -> Optional[Dict[str, str]]:
        """Parse repository URL to extract owner and repo name"""
        
        # Handle various URL formats
        patterns = [
            r'github\.com[:/]([^/]+)/([^/\.]+)',
            r'gitlab\.com[:/]([^/]+)/([^/\.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return {
                    'owner': match.group(1),
                    'repo': match.group(2).replace('.git', '')
                }
        
        return None
    
    def _extract_linked_issues(self, text: str) -> List[int]:
        """Extract linked issue numbers from text"""
        
        issues = []
        for pattern in self.issue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.isdigit():
                    issues.append(int(match))
        
        return list(set(issues))
    
    def _get_review_comments(self, pr) -> List[str]:
        """Get review comments from a PR"""
        
        comments = []
        try:
            for comment in pr.get_review_comments():
                comments.append(comment.body)
        except GithubException:
            pass
        
        return comments[:10]  # Limit to 10 comments
    
    def _extract_feature_tags(self, text: str) -> List[str]:
        """Extract feature tags from text"""
        
        tags = []
        text_lower = text.lower()
        
        for feature, keywords in self.feature_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(feature)
        
        return tags
    
    def _extract_business_impact(self, text: str) -> str:
        """Extract business impact description from text"""
        
        # Look for impact statements
        impact_patterns = [
            r'impact:\s*(.+?)(?:\n|$)',
            r'affects?:\s*(.+?)(?:\n|$)',
            r'benefits?:\s*(.+?)(?:\n|$)',
            r'improves?\s+(.+?)(?:\n|$)',
            r'enables?\s+(.+?)(?:\n|$)',
        ]
        
        for pattern in impact_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_decision_rationale(self, text: str) -> str:
        """Extract decision rationale from text"""
        
        # Look for decision indicators
        for indicator in self.decision_indicators:
            index = text.lower().find(indicator)
            if index != -1:
                # Extract sentence containing the indicator
                start = text.rfind('.', 0, index) + 1
                end = text.find('.', index)
                if end == -1:
                    end = text.find('\n', index)
                if end == -1:
                    end = len(text)
                
                rationale = text[start:end].strip()
                if rationale:
                    return rationale
        
        return ""
    
    def generate_why_explanation(self, context: BusinessContext, code_changes: Dict[str, Any]) -> str:
        """Generate a comprehensive 'why' explanation for code changes"""
        
        explanation = []
        
        # Start with commit message
        if context.decision_rationale:
            explanation.append(f"Decision: {context.decision_rationale}")
        
        # Add business impact
        if context.business_impact:
            explanation.append(f"Business Impact: {context.business_impact}")
        
        # Add issue context
        if context.issues:
            issue_titles = [issue.title for issue in context.issues[:3]]
            explanation.append(f"Addresses issues: {', '.join(issue_titles)}")
        
        # Add PR context
        if context.pull_requests:
            pr_titles = [pr.title for pr in context.pull_requests[:2]]
            explanation.append(f"Part of: {', '.join(pr_titles)}")
        
        # Add feature context
        if context.feature_tags:
            explanation.append(f"Features affected: {', '.join(context.feature_tags)}")
        
        # Add code-level insights
        if code_changes:
            if 'patterns_detected' in code_changes and code_changes['patterns_detected']:
                explanation.append(f"Patterns applied: {', '.join(code_changes['patterns_detected'])}")
            
            if 'complexity_delta' in code_changes:
                delta = code_changes['complexity_delta']
                if delta > 0:
                    explanation.append(f"Increased complexity by {delta} (new functionality)")
                elif delta < 0:
                    explanation.append(f"Reduced complexity by {abs(delta)} (refactoring/simplification)")
        
        if not explanation:
            return "No explicit business context found. Likely technical maintenance or minor update."
        
        return " | ".join(explanation)
    
    def link_to_roadmap(self, contexts: Dict[str, BusinessContext], roadmap_items: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Link commits to roadmap/feature items"""
        
        commit_to_features = {}
        
        for commit_hash, context in contexts.items():
            linked_features = []
            
            # Check each roadmap item
            for item in roadmap_items:
                item_keywords = item.get('keywords', [])
                item_issues = item.get('issues', [])
                
                # Check if commit addresses any of the item's issues
                commit_issues = [issue.number for issue in context.issues]
                if any(issue in item_issues for issue in commit_issues):
                    linked_features.append(item['name'])
                    continue
                
                # Check if commit mentions any keywords
                all_text = " ".join([
                    context.decision_rationale,
                    context.business_impact,
                    " ".join(context.feature_tags)
                ]).lower()
                
                if any(keyword.lower() in all_text for keyword in item_keywords):
                    linked_features.append(item['name'])
            
            commit_to_features[commit_hash] = linked_features
        
        return commit_to_features
    
    def generate_decision_timeline(self, contexts: Dict[str, BusinessContext]) -> List[Dict[str, Any]]:
        """Generate a timeline of architectural/business decisions"""
        
        timeline = []
        
        for commit_hash, context in contexts.items():
            if context.decision_rationale or context.business_impact:
                timeline_entry = {
                    'commit': commit_hash[:8],
                    'date': None,  # Will be filled from commit data
                    'decision': context.decision_rationale,
                    'impact': context.business_impact,
                    'features': context.feature_tags,
                    'issues_resolved': [f"#{issue.number}" for issue in context.issues],
                    'pull_requests': [f"#{pr.number}" for pr in context.pull_requests]
                }
                timeline.append(timeline_entry)
        
        return timeline