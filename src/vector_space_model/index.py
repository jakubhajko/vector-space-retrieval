from __future__ import annotations

"""Inverted index construction and collection statistics.

This module builds a lightweight inverted index over preprocessed documents.
It intentionally does *not* materialize full document vectors.

Stored information
------------------
- postings: term -> list of (doc_id, raw_term_frequency)
- document frequency per term
- collection frequency per term
- document lengths
- average document length
- canonical indexed token stream construction

The default document representation for retrieval is:
    title_tokens + text_tokens

Field weighting is intentionally omitted for now.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import fsum
from typing import Callable, Iterable, Protocol


class SupportsProcessedDocument(Protocol):
    """Structural protocol for processed documents expected by the indexer."""

    doc_id: str
    title_tokens: list[str]
    text_tokens: list[str]


TokenExtractorFn = Callable[[SupportsProcessedDocument], list[str]]


@dataclass(frozen=True)
class Posting:
    """One posting entry in an inverted index."""

    doc_id: str
    term_frequency: int


@dataclass(frozen=True)
class IndexedDocumentMetadata:
    """Index-time metadata for one document."""

    doc_id: str
    length: int


@dataclass
class InvertedIndex:
    """Inverted index with collection-level statistics."""

    postings: dict[str, list[Posting]]
    document_frequency: dict[str, int]
    collection_frequency: dict[str, int]
    document_lengths: dict[str, int]
    average_document_length: float
    num_documents: int
    vocabulary_size: int

    def get_postings(self, term: str) -> list[Posting]:
        """Return the postings list for a term, or an empty list if missing."""
        return self.postings.get(term, [])

    def get_document_frequency(self, term: str) -> int:
        """Return document frequency for a term."""
        return self.document_frequency.get(term, 0)

    def get_collection_frequency(self, term: str) -> int:
        """Return collection frequency for a term."""
        return self.collection_frequency.get(term, 0)

    def get_document_length(self, doc_id: str) -> int:
        """Return indexed document length."""
        return self.document_lengths[doc_id]


def default_document_token_extractor(document: SupportsProcessedDocument) -> list[str]:
    """Build the canonical document token stream used for indexing.

    Current policy
    --------------
    Concatenate:
        title_tokens + text_tokens

    This keeps retrieval behavior explicit and deterministic.
    """
    return list(document.title_tokens) + list(document.text_tokens)


def build_inverted_index(
    documents: Iterable[SupportsProcessedDocument],
    *,
    token_extractor: TokenExtractorFn = default_document_token_extractor,
) -> InvertedIndex:
    """Build an inverted index from processed documents.

    Parameters
    ----------
    documents:
        Iterable of processed document objects.
    token_extractor:
        Function that defines the canonical indexed token stream.

    Returns
    -------
    InvertedIndex
        Index containing postings and collection statistics.
    """
    postings_builder: dict[str, list[Posting]] = defaultdict(list)
    document_frequency: dict[str, int] = {}
    collection_frequency: dict[str, int] = {}
    document_lengths: dict[str, int] = {}

    num_documents = 0

    for document in documents:
        tokens = token_extractor(document)
        term_counts = Counter(tokens)
        doc_length = int(sum(term_counts.values()))

        document_lengths[document.doc_id] = doc_length
        num_documents += 1

        for term, tf in term_counts.items():
            postings_builder[term].append(Posting(doc_id=document.doc_id, term_frequency=tf))

        for term, tf in term_counts.items():
            collection_frequency[term] = collection_frequency.get(term, 0) + tf
            document_frequency[term] = document_frequency.get(term, 0) + 1

    average_document_length = (
        fsum(document_lengths.values()) / num_documents if num_documents > 0 else 0.0
    )

    return InvertedIndex(
        postings=dict(postings_builder),
        document_frequency=document_frequency,
        collection_frequency=collection_frequency,
        document_lengths=document_lengths,
        average_document_length=average_document_length,
        num_documents=num_documents,
        vocabulary_size=len(document_frequency),
    )


__all__ = [
    "IndexedDocumentMetadata",
    "InvertedIndex",
    "Posting",
    "SupportsProcessedDocument",
    "TokenExtractorFn",
    "build_inverted_index",
    "default_document_token_extractor",
]