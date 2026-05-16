from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup

# Your existing A1 loaders and models
from vector_space_model.load_documents import extract_en_documents_from_file, extract_cs_documents_from_file
from vector_space_model.load_topics import parse_topics_file, build_topic_query_text
from vector_space_model.results import RetrievalResult

logger = logging.getLogger(__name__)

# =============================================================================
# WHOOSH INDEXING AND SEARCHING
# =============================================================================

def build_whoosh_index(documents_path: Path, index_dir: Path, analyzer: Any, lang: str) -> None:
    """Reads XML documents and builds a persistent Whoosh index on disk."""
    
    # Define the Schema using the provided analyzer
    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        content=TEXT(analyzer=analyzer, vector=True)
    )
    
    index_dir.mkdir(parents=True, exist_ok=True)
    ix = create_in(index_dir, schema)
    writer = ix.writer(limitmb=256)
    
    # Load the actual document files based on the .lst file
    documents_dir = documents_path.parent / documents_path.stem
    with open(documents_path, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    extract_fn = extract_cs_documents_from_file if lang == "cs" else extract_en_documents_from_file
    
    # Add documents to the index
    docs_processed = 0
    for filename in filenames:
        file_path = documents_dir / filename
        if not file_path.exists():
            continue
            
        extracted_docs = extract_fn(file_path)
        for doc in extracted_docs:
            # Concatenate title and text, letting Whoosh's analyzer handle tokenization
            full_text = f"{doc.raw_title} {doc.raw_text}"
            writer.add_document(doc_id=doc.doc_id, content=full_text)
            docs_processed += 1
            
    print(f"Committing {docs_processed} documents to index (this might take a minute)...")
    writer.commit()


def run_whoosh_search(index_dir: Path, topics_path: Path, config: Any, lang: str) -> list[RetrievalResult]:
    """Runs topics against the built index using the configured scorer."""
    
    ix = open_dir(index_dir)
    topics = parse_topics_file(topics_path)
    results_to_write = []
    
    # Use OrGroup so documents don't have to contain ALL query words to match
    parser = QueryParser("content", ix.schema, group=OrGroup.factory(0.9))
    
    with ix.searcher(weighting=config.scorer) as searcher:
        for topic in topics:
            # Build query dynamically based on config (title vs title_desc_narr)
            query_str = build_topic_query_text(topic, query_construction=config.query_construction)
            query = parser.parse(query_str)
            
            # Run the initial search
            hits = searcher.search(query, limit=config.top_k)
            
            # --- PSEUDO-RELEVANCE FEEDBACK (PRF) ---
            if config.use_prf and len(hits) > 0:
                # Extract the top 3 statistically significant words from the top 5 documents
                key_terms = hits.key_terms("content", docs=5, numterms=3)
                expansion_words = [term for term, score in key_terms]
                
                if expansion_words:
                    # Append new words to the original query and search again
                    expanded_query_str = f"{query_str} {' '.join(expansion_words)}"
                    expanded_query = parser.parse(expanded_query_str)
                    hits = searcher.search(expanded_query, limit=config.top_k)
            # ---------------------------------------
            
            # Format results for TREC
            for rank, hit in enumerate(hits):
                results_to_write.append(
                    RetrievalResult(
                        topic_id=topic.topic_id,
                        doc_id=hit["doc_id"],
                        rank=rank,
                        score=hit.score
                    )
                )
                
    return results_to_write