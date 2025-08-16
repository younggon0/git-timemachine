from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from code_analyzer import CodeChange, CodeEntity
import pickle
from pathlib import Path

@dataclass
class SemanticDocument:
    """Enhanced document with semantic metadata"""
    content: str
    metadata: Dict[str, Any]
    entity_type: str  # 'code', 'pattern', 'architecture', 'business'
    semantic_tags: List[str]
    relationships: List[str]  # Related entities
    
class SemanticIndexer:
    """Advanced semantic indexing for code understanding"""
    
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.indices = {
            'code_structure': None,     # AST-based code entities
            'business_logic': None,      # Comments, docstrings, commit messages
            'patterns': None,            # Design patterns and architectures
            'dependencies': None,        # Import/dependency relationships
            'evolution': None           # Code evolution timeline
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.entity_graph = {}  # Track relationships between entities
        
    def index_code_changes(self, changes: List[CodeChange]) -> Dict[str, FAISS]:
        """Index code changes into multiple specialized indices"""
        
        # Prepare documents for each index
        code_docs = []
        business_docs = []
        pattern_docs = []
        dependency_docs = []
        evolution_docs = []
        
        for change in changes:
            # Index code structure
            code_docs.extend(self._create_code_structure_docs(change))
            
            # Index business logic
            business_docs.extend(self._create_business_logic_docs(change))
            
            # Index patterns
            if change.patterns_detected:
                pattern_docs.extend(self._create_pattern_docs(change))
            
            # Index dependencies
            if change.imports_added or change.imports_removed:
                dependency_docs.extend(self._create_dependency_docs(change))
            
            # Index evolution
            evolution_docs.append(self._create_evolution_doc(change))
        
        # Create FAISS indices
        if code_docs:
            self.indices['code_structure'] = FAISS.from_documents(code_docs, self.embeddings)
        
        if business_docs:
            self.indices['business_logic'] = FAISS.from_documents(business_docs, self.embeddings)
        
        if pattern_docs:
            self.indices['patterns'] = FAISS.from_documents(pattern_docs, self.embeddings)
        
        if dependency_docs:
            self.indices['dependencies'] = FAISS.from_documents(dependency_docs, self.embeddings)
        
        if evolution_docs:
            self.indices['evolution'] = FAISS.from_documents(evolution_docs, self.embeddings)
        
        return self.indices
    
    def _create_code_structure_docs(self, change: CodeChange) -> List[Document]:
        """Create documents for code structure entities"""
        docs = []
        
        for entity in change.entities_added + change.entities_modified:
            content = self._generate_entity_description(entity, 'added/modified', change)
            
            doc = Document(
                page_content=content,
                metadata={
                    'entity_name': entity.name,
                    'entity_type': entity.type,
                    'file_path': entity.file_path,
                    'commit_hash': change.commit_hash,
                    'timestamp': change.timestamp.isoformat(),
                    'author': change.author,
                    'complexity': entity.complexity,
                    'change_type': change.change_type,
                    'semantic_type': 'code_structure'
                }
            )
            docs.append(doc)
            
            # Track entity relationships
            self._update_entity_graph(entity.name, entity)
        
        for entity in change.entities_removed:
            content = self._generate_entity_description(entity, 'removed', change)
            
            doc = Document(
                page_content=content,
                metadata={
                    'entity_name': entity.name,
                    'entity_type': entity.type,
                    'file_path': entity.file_path,
                    'commit_hash': change.commit_hash,
                    'timestamp': change.timestamp.isoformat(),
                    'author': change.author,
                    'change_type': 'removal',
                    'semantic_type': 'code_structure'
                }
            )
            docs.append(doc)
        
        return docs
    
    def _create_business_logic_docs(self, change: CodeChange) -> List[Document]:
        """Create documents for business logic understanding"""
        docs = []
        
        # Index commit message as business intent
        if change.message:
            doc = Document(
                page_content=f"Business Intent: {change.message}\n"
                           f"Change Type: {change.change_type}\n"
                           f"Author: {change.author}\n"
                           f"Impact: {len(change.entities_added)} additions, "
                           f"{len(change.entities_modified)} modifications, "
                           f"{len(change.entities_removed)} removals",
                metadata={
                    'commit_hash': change.commit_hash,
                    'timestamp': change.timestamp.isoformat(),
                    'author': change.author,
                    'change_type': change.change_type,
                    'semantic_type': 'business_logic'
                }
            )
            docs.append(doc)
        
        # Index docstrings as business documentation
        for entity in change.entities_added + change.entities_modified:
            if entity.docstring:
                doc = Document(
                    page_content=f"Documentation for {entity.name}:\n{entity.docstring}\n"
                               f"Type: {entity.type}\n"
                               f"File: {entity.file_path}",
                    metadata={
                        'entity_name': entity.name,
                        'entity_type': entity.type,
                        'file_path': entity.file_path,
                        'commit_hash': change.commit_hash,
                        'timestamp': change.timestamp.isoformat(),
                        'semantic_type': 'business_documentation'
                    }
                )
                docs.append(doc)
        
        return docs
    
    def _create_pattern_docs(self, change: CodeChange) -> List[Document]:
        """Create documents for design patterns"""
        docs = []
        
        for pattern in change.patterns_detected:
            content = (f"Pattern Detected: {pattern}\n"
                      f"In commit: {change.commit_hash[:8]}\n"
                      f"Message: {change.message}\n"
                      f"Entities involved: {', '.join([e.name for e in change.entities_added + change.entities_modified])}")
            
            doc = Document(
                page_content=content,
                metadata={
                    'pattern': pattern,
                    'commit_hash': change.commit_hash,
                    'timestamp': change.timestamp.isoformat(),
                    'author': change.author,
                    'semantic_type': 'pattern'
                }
            )
            docs.append(doc)
        
        return docs
    
    def _create_dependency_docs(self, change: CodeChange) -> List[Document]:
        """Create documents for dependency changes"""
        docs = []
        
        if change.imports_added:
            content = (f"Dependencies Added:\n"
                      f"{', '.join(change.imports_added)}\n"
                      f"In commit: {change.commit_hash[:8]}\n"
                      f"Reason: {change.message}")
            
            doc = Document(
                page_content=content,
                metadata={
                    'dependencies': change.imports_added,
                    'action': 'added',
                    'commit_hash': change.commit_hash,
                    'timestamp': change.timestamp.isoformat(),
                    'semantic_type': 'dependency'
                }
            )
            docs.append(doc)
        
        if change.imports_removed:
            content = (f"Dependencies Removed:\n"
                      f"{', '.join(change.imports_removed)}\n"
                      f"In commit: {change.commit_hash[:8]}\n"
                      f"Reason: {change.message}")
            
            doc = Document(
                page_content=content,
                metadata={
                    'dependencies': change.imports_removed,
                    'action': 'removed',
                    'commit_hash': change.commit_hash,
                    'timestamp': change.timestamp.isoformat(),
                    'semantic_type': 'dependency'
                }
            )
            docs.append(doc)
        
        return docs
    
    def _create_evolution_doc(self, change: CodeChange) -> Document:
        """Create document for code evolution timeline"""
        
        evolution_summary = (
            f"Evolution Point: {change.timestamp.isoformat()}\n"
            f"Commit: {change.commit_hash[:8]}\n"
            f"Author: {change.author}\n"
            f"Type: {change.change_type}\n"
            f"Message: {change.message}\n"
            f"Complexity Change: {'+' if change.complexity_delta >= 0 else ''}{change.complexity_delta}\n"
            f"Entities Added: {[e.name for e in change.entities_added]}\n"
            f"Entities Modified: {[e.name for e in change.entities_modified]}\n"
            f"Entities Removed: {[e.name for e in change.entities_removed]}\n"
            f"Patterns: {change.patterns_detected}\n"
        )
        
        return Document(
            page_content=evolution_summary,
            metadata={
                'commit_hash': change.commit_hash,
                'timestamp': change.timestamp.isoformat(),
                'author': change.author,
                'change_type': change.change_type,
                'complexity_delta': change.complexity_delta,
                'num_entities_changed': len(change.entities_added) + len(change.entities_modified) + len(change.entities_removed),
                'semantic_type': 'evolution'
            }
        )
    
    def _generate_entity_description(self, entity: CodeEntity, action: str, change: CodeChange) -> str:
        """Generate a rich description of a code entity"""
        
        description = (
            f"Code Entity: {entity.name}\n"
            f"Type: {entity.type}\n"
            f"Action: {action}\n"
            f"File: {entity.file_path}\n"
            f"Lines: {entity.start_line}-{entity.end_line}\n"
            f"Complexity: {entity.complexity}\n"
        )
        
        if entity.signature:
            description += f"Signature: {entity.signature}\n"
        
        if entity.docstring:
            description += f"Documentation: {entity.docstring}\n"
        
        if entity.decorators:
            description += f"Decorators: {', '.join(entity.decorators)}\n"
        
        if entity.dependencies:
            description += f"Dependencies: {', '.join(entity.dependencies)}\n"
        
        description += (
            f"Changed in commit: {change.commit_hash[:8]}\n"
            f"Change reason: {change.message}\n"
            f"Author: {change.author}"
        )
        
        return description
    
    def _update_entity_graph(self, entity_name: str, entity: CodeEntity):
        """Update the entity relationship graph"""
        if entity_name not in self.entity_graph:
            self.entity_graph[entity_name] = {
                'type': entity.type,
                'file': entity.file_path,
                'related_entities': set(),
                'commits': []
            }
        
        # Track relationships based on file proximity
        for other_name, other_data in self.entity_graph.items():
            if other_name != entity_name and other_data['file'] == entity.file_path:
                self.entity_graph[entity_name]['related_entities'].add(other_name)
                other_data['related_entities'].add(entity_name)
    
    def semantic_search(self, query: str, index_type: str = 'all', k: int = 10) -> List[Tuple[Document, float]]:
        """Perform semantic search across indices"""
        
        if index_type == 'all':
            # Search across all indices and merge results
            all_results = []
            for idx_name, index in self.indices.items():
                if index:
                    results = index.similarity_search_with_score(query, k=k)
                    # Add index type to metadata
                    for doc, score in results:
                        doc.metadata['index_type'] = idx_name
                    all_results.extend(results)
            
            # Sort by score and return top k
            all_results.sort(key=lambda x: x[1])
            return all_results[:k]
        
        elif index_type in self.indices and self.indices[index_type]:
            return self.indices[index_type].similarity_search_with_score(query, k=k)
        
        return []
    
    def query_code_evolution(self, entity_name: str) -> Dict[str, Any]:
        """Query the evolution of a specific code entity"""
        
        if entity_name in self.entity_graph:
            entity_data = self.entity_graph[entity_name]
            
            # Search for all mentions of this entity in evolution index
            query = f"Entity {entity_name} evolution history changes"
            evolution_results = self.semantic_search(query, index_type='evolution', k=20)
            
            return {
                'entity': entity_name,
                'type': entity_data['type'],
                'file': entity_data['file'],
                'related_entities': list(entity_data['related_entities']),
                'evolution_timeline': [
                    {
                        'content': doc.page_content,
                        'commit': doc.metadata.get('commit_hash'),
                        'timestamp': doc.metadata.get('timestamp'),
                        'author': doc.metadata.get('author')
                    }
                    for doc, _ in evolution_results
                    if entity_name in doc.page_content
                ]
            }
        
        return {}
    
    def find_pattern_usage(self, pattern_name: str) -> List[Dict[str, Any]]:
        """Find all usages of a specific pattern"""
        
        query = f"Pattern {pattern_name} implementation usage"
        results = self.semantic_search(query, index_type='patterns', k=20)
        
        pattern_usages = []
        for doc, score in results:
            if pattern_name.lower() in doc.metadata.get('pattern', '').lower():
                pattern_usages.append({
                    'pattern': doc.metadata.get('pattern'),
                    'commit': doc.metadata.get('commit_hash'),
                    'timestamp': doc.metadata.get('timestamp'),
                    'author': doc.metadata.get('author'),
                    'description': doc.page_content,
                    'relevance_score': float(score)
                })
        
        return pattern_usages
    
    def analyze_complexity_impact(self, file_path: str = None) -> Dict[str, Any]:
        """Analyze complexity changes and their impact"""
        
        if file_path:
            query = f"Complexity changes in {file_path}"
        else:
            query = "Code complexity evolution changes impact"
        
        results = self.semantic_search(query, index_type='code_structure', k=30)
        
        complexity_timeline = []
        total_complexity_change = 0
        
        for doc, _ in results:
            if not file_path or doc.metadata.get('file_path') == file_path:
                complexity = doc.metadata.get('complexity', 0)
                complexity_timeline.append({
                    'entity': doc.metadata.get('entity_name'),
                    'type': doc.metadata.get('entity_type'),
                    'file': doc.metadata.get('file_path'),
                    'complexity': complexity,
                    'timestamp': doc.metadata.get('timestamp'),
                    'commit': doc.metadata.get('commit_hash')
                })
                total_complexity_change += complexity
        
        # Sort by timestamp
        complexity_timeline.sort(key=lambda x: x['timestamp'])
        
        return {
            'file_path': file_path,
            'total_complexity_change': total_complexity_change,
            'timeline': complexity_timeline,
            'high_complexity_entities': [
                item for item in complexity_timeline 
                if item['complexity'] > 10
            ]
        }
    
    def save_indices(self, directory: str):
        """Save all indices to disk"""
        path = Path(directory)
        path.mkdir(exist_ok=True)
        
        for name, index in self.indices.items():
            if index:
                index.save_local(str(path / name))
        
        # Save entity graph
        with open(path / 'entity_graph.pkl', 'wb') as f:
            pickle.dump(self.entity_graph, f)
    
    def load_indices(self, directory: str):
        """Load indices from disk"""
        path = Path(directory)
        
        for name in self.indices.keys():
            index_path = path / name
            if index_path.exists():
                self.indices[name] = FAISS.load_local(
                    str(index_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
        
        # Load entity graph
        graph_path = path / 'entity_graph.pkl'
        if graph_path.exists():
            with open(graph_path, 'rb') as f:
                self.entity_graph = pickle.load(f)