import ast
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import git
from datetime import datetime

@dataclass
class CodeEntity:
    """Represents a code entity (function, class, method) with metadata"""
    name: str
    type: str  # 'function', 'class', 'method', 'import'
    file_path: str
    start_line: int
    end_line: int
    complexity: int = 0
    dependencies: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    signature: Optional[str] = None
    decorators: List[str] = field(default_factory=list)

@dataclass
class CodeChange:
    """Represents a change to code with semantic understanding"""
    commit_hash: str
    timestamp: datetime
    author: str
    message: str
    change_type: str  # 'refactor', 'feature', 'bugfix', 'docs', 'test'
    entities_added: List[CodeEntity] = field(default_factory=list)
    entities_modified: List[CodeEntity] = field(default_factory=list)
    entities_removed: List[CodeEntity] = field(default_factory=list)
    patterns_detected: List[str] = field(default_factory=list)
    complexity_delta: int = 0
    imports_added: List[str] = field(default_factory=list)
    imports_removed: List[str] = field(default_factory=list)
    
class CodeAnalyzer:
    """Analyzes code changes with semantic understanding"""
    
    def __init__(self):
        self.language_parsers = {
            '.py': self._parse_python,
            '.js': self._parse_javascript,
            '.ts': self._parse_typescript,
            '.jsx': self._parse_javascript,
            '.tsx': self._parse_typescript
        }
        
        self.change_patterns = {
            'refactor': ['refactor', 'clean', 'restructure', 'reorganize', 'simplify'],
            'feature': ['add', 'implement', 'feature', 'new', 'create'],
            'bugfix': ['fix', 'bug', 'resolve', 'patch', 'correct', 'repair'],
            'docs': ['doc', 'comment', 'readme', 'documentation'],
            'test': ['test', 'spec', 'coverage', 'unit', 'integration']
        }
        
        self.design_patterns = {
            'singleton': self._detect_singleton_pattern,
            'factory': self._detect_factory_pattern,
            'observer': self._detect_observer_pattern,
            'decorator': self._detect_decorator_pattern,
            'strategy': self._detect_strategy_pattern
        }
    
    def analyze_repository(self, repo: git.Repo, max_commits: int = 100, max_files_per_commit: int = 10) -> List[CodeChange]:
        """Analyze repository history with deep code understanding"""
        changes = []
        
        commits = list(repo.iter_commits(max_count=max_commits))
        for i, commit in enumerate(commits):
            print(f"Analyzing commit {i+1}/{len(commits)}: {commit.hexsha[:8]}")
            change = self._analyze_commit(commit, max_files_per_commit)
            if change:
                changes.append(change)
        
        return changes
    
    def _analyze_commit(self, commit: git.Commit, max_files: int = 10) -> Optional[CodeChange]:
        """Analyze a single commit for code changes"""
        change = CodeChange(
            commit_hash=commit.hexsha,
            timestamp=datetime.fromtimestamp(commit.committed_date),
            author=commit.author.name,
            message=commit.message.strip(),
            change_type=self._classify_change_type(commit.message)
        )
        
        if not commit.parents:
            return change
        
        parent = commit.parents[0]
        diffs = commit.diff(parent)
        
        # Limit files analyzed per commit for performance
        files_analyzed = 0
        for diff in diffs:
            if files_analyzed >= max_files:
                break
            if diff.a_blob or diff.b_blob:
                # Skip large files and non-code files
                if self._should_analyze_file(diff):
                    self._analyze_diff(diff, change)
                    files_analyzed += 1
        
        # Detect patterns in the overall change
        change.patterns_detected = self._detect_patterns_in_change(change)
        
        return change
    
    def _should_analyze_file(self, diff) -> bool:
        """Check if a file should be analyzed"""
        file_path = diff.b_path or diff.a_path
        if not file_path:
            return False
        
        # Skip non-code files
        suffix = Path(file_path).suffix
        if suffix not in self.language_parsers:
            return False
        
        # Skip large files (>100KB)
        if diff.b_blob and diff.b_blob.size > 100000:
            return False
        if diff.a_blob and diff.a_blob.size > 100000:
            return False
        
        # Skip vendor/node_modules/build directories
        excluded_dirs = ['node_modules', 'vendor', 'build', 'dist', '.git', '__pycache__']
        if any(excluded in file_path for excluded in excluded_dirs):
            return False
        
        return True
    
    def _analyze_diff(self, diff, change: CodeChange):
        """Analyze a file diff for semantic changes"""
        file_path = diff.b_path or diff.a_path
        if not file_path:
            return
        
        suffix = Path(file_path).suffix
        if suffix not in self.language_parsers:
            return
        
        # Get file content before and after
        old_content = diff.a_blob.data_stream.read().decode('utf-8', errors='ignore') if diff.a_blob else ""
        new_content = diff.b_blob.data_stream.read().decode('utf-8', errors='ignore') if diff.b_blob else ""
        
        # Parse old and new content
        old_entities = self._parse_code(old_content, file_path, suffix) if old_content else []
        new_entities = self._parse_code(new_content, file_path, suffix) if new_content else []
        
        # Determine what changed
        old_entity_map = {e.name: e for e in old_entities}
        new_entity_map = {e.name: e for e in new_entities}
        
        # Find added entities
        for name, entity in new_entity_map.items():
            if name not in old_entity_map:
                change.entities_added.append(entity)
            else:
                # Check if modified
                old_entity = old_entity_map[name]
                if self._entity_modified(old_entity, entity):
                    change.entities_modified.append(entity)
        
        # Find removed entities
        for name, entity in old_entity_map.items():
            if name not in new_entity_map:
                change.entities_removed.append(entity)
        
        # Calculate complexity change
        old_complexity = sum(e.complexity for e in old_entities)
        new_complexity = sum(e.complexity for e in new_entities)
        change.complexity_delta += (new_complexity - old_complexity)
        
        # Track import changes
        old_imports = self._extract_imports(old_content, suffix)
        new_imports = self._extract_imports(new_content, suffix)
        change.imports_added.extend(set(new_imports) - set(old_imports))
        change.imports_removed.extend(set(old_imports) - set(new_imports))
    
    def _parse_code(self, content: str, file_path: str, suffix: str) -> List[CodeEntity]:
        """Parse code content and extract entities"""
        parser = self.language_parsers.get(suffix)
        if parser:
            return parser(content, file_path)
        return []
    
    def _parse_python(self, content: str, file_path: str) -> List[CodeEntity]:
        """Parse Python code and extract entities"""
        entities = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    entity = CodeEntity(
                        name=node.name,
                        type='function',
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        complexity=self._calculate_complexity(node),
                        docstring=ast.get_docstring(node),
                        signature=self._get_function_signature(node),
                        decorators=[d.id for d in node.decorator_list if hasattr(d, 'id')]
                    )
                    entities.append(entity)
                elif isinstance(node, ast.ClassDef):
                    entity = CodeEntity(
                        name=node.name,
                        type='class',
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        complexity=self._calculate_class_complexity(node),
                        docstring=ast.get_docstring(node),
                        decorators=[d.id for d in node.decorator_list if hasattr(d, 'id')]
                    )
                    entities.append(entity)
                    
                    # Add methods as separate entities
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method = CodeEntity(
                                name=f"{node.name}.{item.name}",
                                type='method',
                                file_path=file_path,
                                start_line=item.lineno,
                                end_line=item.end_lineno or item.lineno,
                                complexity=self._calculate_complexity(item),
                                docstring=ast.get_docstring(item),
                                signature=self._get_function_signature(item)
                            )
                            entities.append(method)
        except SyntaxError:
            pass  # Invalid Python code
        
        return entities
    
    def _parse_javascript(self, content: str, file_path: str) -> List[CodeEntity]:
        """Parse JavaScript code - basic regex-based parsing"""
        entities = []
        
        # Function patterns
        function_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))'
        class_pattern = r'class\s+(\w+)'
        
        for match in re.finditer(function_pattern, content):
            name = match.group(1) or match.group(2)
            if name:
                entities.append(CodeEntity(
                    name=name,
                    type='function',
                    file_path=file_path,
                    start_line=content[:match.start()].count('\n') + 1,
                    end_line=content[:match.end()].count('\n') + 1
                ))
        
        for match in re.finditer(class_pattern, content):
            entities.append(CodeEntity(
                name=match.group(1),
                type='class',
                file_path=file_path,
                start_line=content[:match.start()].count('\n') + 1,
                end_line=content[:match.end()].count('\n') + 1
            ))
        
        return entities
    
    def _parse_typescript(self, content: str, file_path: str) -> List[CodeEntity]:
        """Parse TypeScript code - extends JavaScript parsing with types"""
        entities = self._parse_javascript(content, file_path)
        
        # Add interface detection
        interface_pattern = r'interface\s+(\w+)'
        type_pattern = r'type\s+(\w+)\s*='
        
        for match in re.finditer(interface_pattern, content):
            entities.append(CodeEntity(
                name=match.group(1),
                type='interface',
                file_path=file_path,
                start_line=content[:match.start()].count('\n') + 1,
                end_line=content[:match.end()].count('\n') + 1
            ))
        
        for match in re.finditer(type_pattern, content):
            entities.append(CodeEntity(
                name=match.group(1),
                type='type',
                file_path=file_path,
                start_line=content[:match.start()].count('\n') + 1,
                end_line=content[:match.end()].count('\n') + 1
            ))
        
        return entities
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function/method"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _calculate_class_complexity(self, node: ast.ClassDef) -> int:
        """Calculate complexity for a class"""
        complexity = 0
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                complexity += self._calculate_complexity(item)
        return complexity
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature"""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        return f"({', '.join(args)})"
    
    def _extract_imports(self, content: str, suffix: str) -> List[str]:
        """Extract import statements from code"""
        imports = []
        
        if suffix == '.py':
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
            except SyntaxError:
                pass
        elif suffix in ['.js', '.jsx', '.ts', '.tsx']:
            # Regex-based import extraction for JS/TS
            import_pattern = r'import\s+.*?\s+from\s+[\'"]([^\'\"]+)[\'"]'
            require_pattern = r'require\([\'"]([^\'\"]+)[\'"]\)'
            
            imports.extend(re.findall(import_pattern, content))
            imports.extend(re.findall(require_pattern, content))
        
        return imports
    
    def _entity_modified(self, old: CodeEntity, new: CodeEntity) -> bool:
        """Check if an entity was modified"""
        return (old.complexity != new.complexity or
                old.start_line != new.start_line or
                old.end_line != new.end_line or
                old.docstring != new.docstring or
                old.signature != new.signature)
    
    def _classify_change_type(self, message: str) -> str:
        """Classify the type of change based on commit message"""
        message_lower = message.lower()
        
        for change_type, keywords in self.change_patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                return change_type
        
        return 'other'
    
    def _detect_patterns_in_change(self, change: CodeChange) -> List[str]:
        """Detect design patterns and refactoring patterns in the change"""
        patterns = []
        
        # Check for common refactoring patterns
        if change.entities_removed and change.entities_added:
            # Possible rename or move
            if len(change.entities_removed) == len(change.entities_added):
                patterns.append('rename_refactoring')
        
        # Check for extract method pattern
        if change.entities_added and any(e.type in ['function', 'method'] for e in change.entities_added):
            if change.entities_modified:
                patterns.append('extract_method')
        
        # Check for design patterns in added code
        for entity in change.entities_added:
            if entity.type == 'class':
                for pattern_name, detector in self.design_patterns.items():
                    if detector(entity):
                        patterns.append(f'{pattern_name}_pattern')
        
        return patterns
    
    def _detect_singleton_pattern(self, entity: CodeEntity) -> bool:
        """Detect if a class implements singleton pattern"""
        # Simplified detection - check for instance variable and getInstance method
        return ('instance' in entity.name.lower() or 
                'singleton' in entity.name.lower())
    
    def _detect_factory_pattern(self, entity: CodeEntity) -> bool:
        """Detect if a class implements factory pattern"""
        return 'factory' in entity.name.lower()
    
    def _detect_observer_pattern(self, entity: CodeEntity) -> bool:
        """Detect if a class implements observer pattern"""
        return ('observer' in entity.name.lower() or 
                'listener' in entity.name.lower())
    
    def _detect_decorator_pattern(self, entity: CodeEntity) -> bool:
        """Detect if a class implements decorator pattern"""
        return ('decorator' in entity.name.lower() or 
                bool(entity.decorators))
    
    def _detect_strategy_pattern(self, entity: CodeEntity) -> bool:
        """Detect if a class implements strategy pattern"""
        return 'strategy' in entity.name.lower()
    
    def get_evolution_summary(self, changes: List[CodeChange]) -> Dict[str, Any]:
        """Generate a summary of code evolution"""
        summary = {
            'total_commits': len(changes),
            'change_types': {},
            'patterns_detected': {},
            'complexity_trend': [],
            'entities_evolution': {
                'functions_added': 0,
                'classes_added': 0,
                'functions_removed': 0,
                'classes_removed': 0
            },
            'top_refactored_files': {},
            'dependency_changes': {
                'added': set(),
                'removed': set()
            }
        }
        
        total_complexity = 0
        
        for change in changes:
            # Count change types
            change_type = change.change_type
            summary['change_types'][change_type] = summary['change_types'].get(change_type, 0) + 1
            
            # Count patterns
            for pattern in change.patterns_detected:
                summary['patterns_detected'][pattern] = summary['patterns_detected'].get(pattern, 0) + 1
            
            # Track complexity
            total_complexity += change.complexity_delta
            summary['complexity_trend'].append({
                'date': change.timestamp.isoformat(),
                'complexity': total_complexity
            })
            
            # Count entity changes
            for entity in change.entities_added:
                if entity.type == 'function':
                    summary['entities_evolution']['functions_added'] += 1
                elif entity.type == 'class':
                    summary['entities_evolution']['classes_added'] += 1
            
            for entity in change.entities_removed:
                if entity.type == 'function':
                    summary['entities_evolution']['functions_removed'] += 1
                elif entity.type == 'class':
                    summary['entities_evolution']['classes_removed'] += 1
            
            # Track file modifications
            for entity in change.entities_modified:
                file_path = entity.file_path
                summary['top_refactored_files'][file_path] = summary['top_refactored_files'].get(file_path, 0) + 1
            
            # Track dependency changes
            summary['dependency_changes']['added'].update(change.imports_added)
            summary['dependency_changes']['removed'].update(change.imports_removed)
        
        # Convert sets to lists for JSON serialization
        summary['dependency_changes']['added'] = list(summary['dependency_changes']['added'])
        summary['dependency_changes']['removed'] = list(summary['dependency_changes']['removed'])
        
        # Sort top refactored files
        summary['top_refactored_files'] = dict(
            sorted(summary['top_refactored_files'].items(), 
                   key=lambda x: x[1], reverse=True)[:10]
        )
        
        return summary