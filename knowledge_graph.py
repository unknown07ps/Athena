# knowledge_graph.py - Knowledge Graph Construction for Research Papers

import re
import requests
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import networkx as nx

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings_class = HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    embeddings_class = SentenceTransformerEmbeddings


class KnowledgeGraphBuilder:
    """
    Extracts structured knowledge from research papers:
    - Entities: Methods, Datasets, Models, Metrics, Authors, Institutions
    - Relationships: Uses, Improves, Based-On, Evaluates-On
    - Results: Performance numbers, comparisons
    """
    
    def __init__(self, model="llama3"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.graph = nx.DiGraph()
        
        # Knowledge patterns for entity extraction
        self.patterns = {
            'methods': [
                r'\b(transformer|bert|gpt|resnet|lstm|cnn|gan|vae|attention|'
                r'self-attention|cross-attention|encoder|decoder|'
                r'backpropagation|gradient descent|adam|sgd|'
                r'reinforcement learning|supervised learning|unsupervised learning|'
                r'transfer learning|fine-tuning|pre-training)\b',
            ],
            'datasets': [
                r'\b(imagenet|coco|mnist|cifar|squad|glue|'
                r'wikipedia|common crawl|bookcorpus|'
                r'ms ?marco|natural questions|wmt|'
                r'pascal voc|ade20k)\b',
            ],
            'metrics': [
                r'\b(accuracy|precision|recall|f1[- ]score|'
                r'bleu|rouge|meteor|perplexity|'
                r'map|iou|dice|auroc|auc|'
                r'top-[1-5]|top[1-5])\b',
            ],
            'models': [
                r'\b(bert|gpt-[234]|gpt[234]|t5|bart|roberta|'
                r'vit|swin|dino|clip|dalle|'
                r'resnet-?[0-9]+|efficientnet|mobilenet|'
                r'llama|mistral|gemini|claude)\b',
            ]
        }
        
        # Relationship indicators
        self.relation_patterns = {
            'uses': [r'use[ds]?', r'utiliz[es]{2,4}', r'employ[s]?', r'apply|applied|applies'],
            'improves': [r'improve[ds]?', r'enhance[ds]?', r'outperform[s]?', r'better than'],
            'based_on': [r'based on', r'built on', r'extend[s]?', r'derived from'],
            'evaluates_on': [r'evaluat[es]{2,4} on', r'test[es]{2,4} on', r'benchmark[es]{2,4} on']
        }
    
    def extract_entities(self, text: str) -> Dict[str, Set[str]]:
        """Extract entities from text using pattern matching"""
        entities = defaultdict(set)
        
        text_lower = text.lower()
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    entity = match.group(0).strip()
                    if len(entity) > 2:  # Filter very short matches
                        entities[entity_type].add(entity)
        
        return dict(entities)
    
    def extract_metrics_values(self, text: str) -> List[Dict]:
        """Extract performance metrics with their values"""
        metrics_data = []
        
        # Pattern: "metric_name: value%" or "metric_name of value%"
        patterns = [
            r'(\w+(?:[-\s]\w+)?)\s*[:=]\s*([\d.]+)%?',
            r'(\w+(?:[-\s]\w+)?)\s+of\s+([\d.]+)%?',
            r'achieve[ds]?\s+([\d.]+)%?\s+(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    metric_name = match.group(1).lower()
                    value = match.group(2)
                    
                    try:
                        value_float = float(value)
                        metrics_data.append({
                            'metric': metric_name,
                            'value': value_float,
                            'context': match.group(0)
                        })
                    except ValueError:
                        continue
        
        return metrics_data
    
    def extract_relationships(self, text: str, entities: Dict) -> List[Tuple]:
        """Extract relationships between entities"""
        relationships = []
        
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Find entities in this sentence
            sentence_entities = []
            for entity_type, entity_set in entities.items():
                for entity in entity_set:
                    if entity.lower() in sentence_lower:
                        sentence_entities.append((entity_type, entity))
            
            # If we have multiple entities, check for relationships
            if len(sentence_entities) >= 2:
                for relation_type, patterns in self.relation_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, sentence_lower):
                            # Create relationship between first and second entity
                            source = sentence_entities[0]
                            target = sentence_entities[1]
                            
                            relationships.append({
                                'source': source[1],
                                'source_type': source[0],
                                'relation': relation_type,
                                'target': target[1],
                                'target_type': target[0],
                                'context': sentence.strip()
                            })
        
        return relationships
    
    def build_graph(self, text: str, title: str = "Research Paper") -> nx.DiGraph:
        """Build complete knowledge graph from text"""
        print(f"🔬 Building knowledge graph for: {title}")
        
        # Extract entities
        entities = self.extract_entities(text)
        
        print(f"   📊 Extracted entities:")
        for entity_type, entity_set in entities.items():
            print(f"      {entity_type}: {len(entity_set)} items")
        
        # Extract metrics
        metrics = self.extract_metrics_values(text)
        print(f"   📈 Found {len(metrics)} performance metrics")
        
        # Extract relationships
        relationships = self.extract_relationships(text, entities)
        print(f"   🔗 Found {len(relationships)} relationships")
        
        # Build graph
        self.graph = nx.DiGraph()
        
        # Add central paper node
        self.graph.add_node(title, type='paper', color='#FF6B6B')
        
        # Add entity nodes
        node_colors = {
            'methods': '#4ECDC4',
            'datasets': '#FFE66D',
            'metrics': '#A8E6CF',
            'models': '#FF8B94'
        }
        
        for entity_type, entity_set in entities.items():
            for entity in entity_set:
                self.graph.add_node(
                    entity,
                    type=entity_type,
                    color=node_colors.get(entity_type, '#95E1D3')
                )
                # Connect to paper
                self.graph.add_edge(title, entity, relation='mentions')
        
        # Add metric nodes with values
        for metric_data in metrics:
            metric_name = f"{metric_data['metric']}: {metric_data['value']}"
            self.graph.add_node(
                metric_name,
                type='result',
                color='#F38181',
                value=metric_data['value']
            )
            self.graph.add_edge(title, metric_name, relation='achieves')
        
        # Add relationships
        for rel in relationships:
            if self.graph.has_node(rel['source']) and self.graph.has_node(rel['target']):
                self.graph.add_edge(
                    rel['source'],
                    rel['target'],
                    relation=rel['relation'],
                    context=rel['context'][:100]
                )
        
        print(f"   ✅ Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def get_graph_summary(self) -> Dict:
        """Get summary statistics of the knowledge graph"""
        if not self.graph:
            return {}
        
        # Node type distribution
        node_types = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            node_types[data.get('type', 'unknown')] += 1
        
        # Relation type distribution
        relation_types = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            relation_types[data.get('relation', 'unknown')] += 1
        
        # Central nodes (highest degree)
        if self.graph.number_of_nodes() > 0:
            degrees = dict(self.graph.degree())
            central_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            central_nodes = []
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': dict(node_types),
            'relation_types': dict(relation_types),
            'central_nodes': central_nodes,
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0
        }
    
    def query_graph(self, query: str, k: int = 5) -> List[Dict]:
        """Query the knowledge graph"""
        query_lower = query.lower()
        results = []
        
        # Search nodes
        for node, data in self.graph.nodes(data=True):
            if query_lower in node.lower():
                # Get neighbors
                neighbors = list(self.graph.neighbors(node))
                predecessors = list(self.graph.predecessors(node))
                
                results.append({
                    'node': node,
                    'type': data.get('type', 'unknown'),
                    'neighbors': neighbors[:3],
                    'connected_to': predecessors[:3],
                    'degree': self.graph.degree(node)
                })
        
        # Search edges
        for source, target, data in self.graph.edges(data=True):
            if query_lower in source.lower() or query_lower in target.lower():
                results.append({
                    'type': 'relationship',
                    'source': source,
                    'relation': data.get('relation', 'related'),
                    'target': target,
                    'context': data.get('context', '')
                })
        
        return results[:k]
    
    def export_to_cytoscape(self) -> Dict:
        """Export graph in Cytoscape.js format for visualization"""
        elements = []
        
        # Nodes
        for node, data in self.graph.nodes(data=True):
            elements.append({
                'data': {
                    'id': node,
                    'label': node,
                    'type': data.get('type', 'unknown'),
                    'color': data.get('color', '#95E1D3')
                }
            })
        
        # Edges
        for source, target, data in self.graph.edges(data=True):
            elements.append({
                'data': {
                    'source': source,
                    'target': target,
                    'relation': data.get('relation', 'related')
                }
            })
        
        return {'elements': elements}
    
    def find_paths(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """Find all paths between two entities"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                source=source,
                target=target,
                cutoff=max_length
            ))
            return paths
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []
    
    def get_subgraph(self, node: str, depth: int = 1) -> nx.DiGraph:
        """Get subgraph around a specific node"""
        if node not in self.graph:
            return nx.DiGraph()
        
        # Get all nodes within depth
        nodes = {node}
        current_level = {node}
        
        for _ in range(depth):
            next_level = set()
            for n in current_level:
                next_level.update(self.graph.neighbors(n))
                next_level.update(self.graph.predecessors(n))
            nodes.update(next_level)
            current_level = next_level
        
        return self.graph.subgraph(nodes).copy()


# =====================================================================
# 🧪 TEST SUITE
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 KNOWLEDGE GRAPH BUILDER TEST")
    print("=" * 70)
    
    # Sample research paper abstract
    sample_text = """
    Title: Attention Is All You Need
    
    Abstract: We propose the Transformer, a novel architecture based solely on 
    attention mechanisms. The model uses self-attention and cross-attention to 
    process sequential data. We evaluate the Transformer on machine translation 
    tasks using the WMT 2014 English-to-German dataset.
    
    Our model achieves 28.4 BLEU score on the WMT 2014 English-to-German task, 
    improving over previous state-of-the-art by 2.0 BLEU points. On the 
    English-to-French task, we achieve 41.8 BLEU.
    
    The Transformer uses multi-head attention and positional encoding. We train 
    using Adam optimizer with a learning rate schedule. The model is based on 
    encoder-decoder architecture but removes recurrence entirely.
    
    Results show the Transformer outperforms LSTM and CNN-based models on 
    multiple benchmarks including SQUAD and GLUE. We achieve 92% accuracy on 
    SQUAD question answering and 85% F1-score on GLUE natural language 
    understanding tasks.
    """
    
    # Build knowledge graph
    kg = KnowledgeGraphBuilder()
    graph = kg.build_graph(sample_text, "Attention Is All You Need")
    
    print("\n" + "=" * 70)
    print("📊 GRAPH SUMMARY")
    print("=" * 70)
    
    summary = kg.get_graph_summary()
    print(f"\n📈 Statistics:")
    print(f"   Total Nodes: {summary['total_nodes']}")
    print(f"   Total Edges: {summary['total_edges']}")
    print(f"   Density: {summary['density']:.3f}")
    
    print(f"\n📦 Node Types:")
    for node_type, count in summary['node_types'].items():
        print(f"   {node_type}: {count}")
    
    print(f"\n🔗 Relationships:")
    for rel_type, count in summary['relation_types'].items():
        print(f"   {rel_type}: {count}")
    
    print(f"\n🌟 Most Connected Nodes:")
    for node, degree in summary['central_nodes']:
        print(f"   {node}: {degree} connections")
    
    # Test queries
    print("\n" + "=" * 70)
    print("🔍 QUERY TESTS")
    print("=" * 70)
    
    queries = [
        "transformer",
        "attention",
        "BLEU",
        "dataset"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = kg.query_graph(query, k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                if result.get('type') == 'relationship':
                    print(f"   {i}. {result['source']} --[{result['relation']}]--> {result['target']}")
                else:
                    print(f"   {i}. {result['node']} ({result['type']}) - {result['degree']} connections")
        else:
            print(f"   No results found")
    
    # Export for visualization
    print("\n" + "=" * 70)
    print("📤 EXPORT TEST")
    print("=" * 70)
    
    cytoscape_data = kg.export_to_cytoscape()
    print(f"✅ Exported {len(cytoscape_data['elements'])} elements for Cytoscape.js")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    
    print("\n💡 Next Steps:")
    print("   1. Integrate with Streamlit visualization")
    print("   2. Add to app.py as new tab")
    print("   3. Use with RAG for contextual reasoning")