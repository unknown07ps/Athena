import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class ResearchPaper:
    """Research paper metadata"""
    title: str
    authors: List[str]
    abstract: str
    year: int
    url: str
    pdf_url: Optional[str]
    source: str  # arxiv, semantic_scholar, pubmed
    citations: int = 0
    venue: str = ""
    
    def __str__(self):
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"{self.title}\n{authors_str} ({self.year})\n{self.abstract[:200]}..."


class PaperFetcher:
    """Fetch research papers from multiple academic sources"""
    
    def __init__(self):
        self.arxiv_base = "http://export.arxiv.org/api/query"
        self.semantic_scholar_base = "https://api.semanticscholar.org/graph/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Athena-Research-Assistant/1.0'
        })
    
    def search_papers(self, query: str, max_results: int = 10, 
                     sources: List[str] = None) -> List[ResearchPaper]:
        """
        Search for papers across multiple sources
        
        Args:
            query: Search query
            max_results: Maximum number of papers to fetch
            sources: List of sources to search ['arxiv', 'semantic_scholar']
                    If None, searches all available sources
        
        Returns:
            List of ResearchPaper objects
        """
        if sources is None:
            sources = ['arxiv', 'semantic_scholar']
        
        all_papers = []
        papers_per_source = max(5, max_results // len(sources))
        
        print(f"\n Searching for: '{query}'")
        print(f" Target: {max_results} papers from {len(sources)} sources")
        
        # Search arXiv
        if 'arxiv' in sources:
            try:
                arxiv_papers = self._search_arxiv(query, papers_per_source)
                all_papers.extend(arxiv_papers)
                print(f"    arXiv: {len(arxiv_papers)} papers")
            except Exception as e:
                print(f"    arXiv error: {e}")
        
        # Search Semantic Scholar
        if 'semantic_scholar' in sources:
            try:
                ss_papers = self._search_semantic_scholar(query, papers_per_source)
                all_papers.extend(ss_papers)
                print(f"    Semantic Scholar: {len(ss_papers)} papers")
            except Exception as e:
                print(f"    Semantic Scholar error: {e}")
        
        # Remove duplicates based on title similarity
        all_papers = self._deduplicate_papers(all_papers)
        
        # Sort by citations (if available) and year
        all_papers.sort(key=lambda p: (p.citations, p.year), reverse=True)
        
        # Limit to max_results
        all_papers = all_papers[:max_results]
        
        print(f" Total unique papers: {len(all_papers)}")
        
        return all_papers
    
    def _search_arxiv(self, query: str, max_results: int = 10) -> List[ResearchPaper]:
        """Search arXiv API"""
        import xml.etree.ElementTree as ET
        
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        response = self.session.get(self.arxiv_base, params=params, timeout=15)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        
        for entry in root.findall('atom:entry', namespace):
            # Extract data
            title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ')
            
            # Authors
            authors = [
                author.find('atom:name', namespace).text 
                for author in entry.findall('atom:author', namespace)
            ]
            
            # Abstract
            summary = entry.find('atom:summary', namespace)
            abstract = summary.text.strip().replace('\n', ' ') if summary is not None else ""
            
            # Published date
            published = entry.find('atom:published', namespace).text
            year = int(published[:4])
            
            # Links
            pdf_url = None
            html_url = None
            
            for link in entry.findall('atom:link', namespace):
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href')
                elif link.get('type') == 'text/html':
                    html_url = link.get('href')
            
            paper = ResearchPaper(
                title=title,
                authors=authors,
                abstract=abstract,
                year=year,
                url=html_url or pdf_url,
                pdf_url=pdf_url,
                source='arXiv',
                citations=0,  # arXiv doesn't provide citation counts
                venue='arXiv'
            )
            
            papers.append(paper)
            
            # Rate limiting
            time.sleep(0.1)
        
        return papers
    
    def _search_semantic_scholar(self, query: str, max_results: int = 10) -> List[ResearchPaper]:
        """Search Semantic Scholar API"""
        
        endpoint = f"{self.semantic_scholar_base}/paper/search"
        
        params = {
            'query': query,
            'limit': max_results,
            'fields': 'title,authors,abstract,year,url,citationCount,venue,openAccessPdf'
        }
        
        response = self.session.get(endpoint, params=params, timeout=15)
        
        if response.status_code == 429:
            print("    Rate limited, waiting...")
            time.sleep(2)
            response = self.session.get(endpoint, params=params, timeout=15)
        
        response.raise_for_status()
        data = response.json()
        
        papers = []
        
        for item in data.get('data', []):
            # Extract authors
            authors = [
                author.get('name', 'Unknown')
                for author in item.get('authors', [])
            ]
            
            # PDF URL
            pdf_info = item.get('openAccessPdf')
            pdf_url = pdf_info.get('url') if pdf_info else None
            
            paper = ResearchPaper(
                title=item.get('title', 'Untitled'),
                authors=authors,
                abstract=item.get('abstract', 'No abstract available'),
                year=item.get('year', 0),
                url=item.get('url', ''),
                pdf_url=pdf_url,
                source='Semantic Scholar',
                citations=item.get('citationCount', 0),
                venue=item.get('venue', '')
            )
            
            papers.append(paper)
            
            # Rate limiting
            time.sleep(0.1)
        
        return papers
    
    def _deduplicate_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Remove duplicate papers based on title similarity"""
        from difflib import SequenceMatcher
        
        unique_papers = []
        seen_titles = []
        
        for paper in papers:
            # Check if similar title already exists
            is_duplicate = False
            
            for seen_title in seen_titles:
                similarity = SequenceMatcher(None, paper.title.lower(), seen_title.lower()).ratio()
                if similarity > 0.85:  # 85% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.append(paper.title)
        
        return unique_papers
    
    def download_paper_pdf(self, paper: ResearchPaper, output_dir: str = ".") -> Optional[str]:
        """Download PDF of a paper"""
        import os
        
        if not paper.pdf_url:
            print(f"    No PDF available for: {paper.title[:50]}")
            return None
        
        try:
            # Create safe filename
            safe_title = "".join(c for c in paper.title[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_title}_{paper.year}.pdf"
            filepath = os.path.join(output_dir, filename)
            
            print(f"    Downloading: {paper.title[:50]}...")
            
            response = self.session.get(paper.pdf_url, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"   ✅ Saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            return None
    
    def format_papers_summary(self, papers: List[ResearchPaper]) -> str:
        """Format papers into a readable summary"""
        if not papers:
            return "No papers found."
        
        summary = f"# Research Papers on Your Topic\n\n"
        summary += f"Found {len(papers)} relevant papers:\n\n"
        summary += "---\n\n"
        
        for i, paper in enumerate(papers, 1):
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += f" et al. ({len(paper.authors)} authors)"
            
            summary += f"## {i}. {paper.title}\n\n"
            summary += f"**Authors:** {authors_str}\n\n"
            summary += f"**Year:** {paper.year} | **Source:** {paper.source}"
            
            if paper.citations > 0:
                summary += f" | **Citations:** {paper.citations}"
            
            if paper.venue:
                summary += f" | **Venue:** {paper.venue}"
            
            summary += "\n\n"
            summary += f"**Abstract:** {paper.abstract}\n\n"
            
            summary += f"**Links:**\n"
            summary += f"- [View Paper]({paper.url})\n"
            if paper.pdf_url:
                summary += f"- [Download PDF]({paper.pdf_url})\n"
            
            summary += "\n---\n\n"
        
        return summary



#  TEST SUITE


if __name__ == "__main__":
    print("=" * 70)
    print(" RESEARCH PAPER FETCHER TEST")
    print("=" * 70)
    
    fetcher = PaperFetcher()
    
    # Test queries
    test_queries = [
        "transformer attention mechanism",
        "computer vision deep learning",
        "reinforcement learning robotics"
    ]
    
    query = test_queries[0]
    print(f"\n Testing with query: '{query}'")
    print("=" * 70)
    
    # Fetch papers
    papers = fetcher.search_papers(query, max_results=5)
    
    if papers:
        print(f"\n Found {len(papers)} papers!\n")
        
        # Display papers
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper.title}")
            print(f"   Authors: {', '.join(paper.authors[:2])}{'...' if len(paper.authors) > 2 else ''}")
            print(f"   Year: {paper.year} | Citations: {paper.citations} | Source: {paper.source}")
            print(f"   Abstract: {paper.abstract[:150]}...")
            print(f"   URL: {paper.url}")
            if paper.pdf_url:
                print(f"   PDF: {paper.pdf_url}")
        
        # Test summary formatting
        print("\n" + "=" * 70)
        print(" FORMATTED SUMMARY")
        print("=" * 70)
        
        summary = fetcher.format_papers_summary(papers[:3])
        print(summary[:500] + "...\n")
        
        # Test download (optional)
        print("=" * 70)
        print(" To test PDF download:")
        print("   paper = papers[0]")
        print("   fetcher.download_paper_pdf(paper)")
        
    else:
        print("❌ No papers found")
    
    print("\n" + "=" * 70)
    print(" TEST COMPLETE!")
    print("=" * 70)
    
    print("\n Usage in your code:")
    print("""
    from paper_fetcher import PaperFetcher
    
    fetcher = PaperFetcher()
    papers = fetcher.search_papers("your topic", max_results=10)
    
    # Get formatted summary
    summary = fetcher.format_papers_summary(papers)
    
    # Download PDFs
    for paper in papers:
        if paper.pdf_url:
            fetcher.download_paper_pdf(paper, output_dir="papers/")
    """)