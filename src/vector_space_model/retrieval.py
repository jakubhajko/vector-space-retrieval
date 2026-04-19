from __future__ import annotations

"""Retrieval over an inverted index with configurable scoring.

This module performs on-the-fly scoring using only:
- postings
- collection statistics
- optional cached document norms

It intentionally does not materialize full document vectors.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from heapq import nlargest
from typing import Mapping, Sequence

from .index import InvertedIndex
from .scoring import (
    ScoringConfig,
    build_weighted_query_terms,
    compute_cosine_norm,
    compute_pivot_denominator,
    compute_term_weight,
)


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked retrieval result."""

    topic_id: str
    doc_id: str
    rank: int
    score: float


@dataclass
class RetrievalSystem:
    """Retrieval system over a prebuilt inverted index."""

    index: InvertedIndex
    _cosine_doc_norm_cache: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)

    def _cosine_cache_key(self, scoring: ScoringConfig) -> tuple[str, str]:
        """Build a stable cache key for cosine document norms."""
        return (scoring.tf_weighting, scoring.df_weighting)

    def _get_or_compute_cosine_document_norms(
        self,
        scoring: ScoringConfig,
    ) -> dict[str, float]:
        """Get cached cosine document norms for the given weighting setup.

        The norm for document d is:
            sqrt(sum_t w(t, d)^2)

        with:
            w(t, d) = tf_weight(tf_td) * df_weight(df_t, N)
        """
        cache_key = self._cosine_cache_key(scoring)
        cached = self._cosine_doc_norm_cache.get(cache_key)
        if cached is not None:
            return cached

        squared_sums: dict[str, float] = defaultdict(float)

        for term, postings in self.index.postings.items():
            df = self.index.get_document_frequency(term)
            for posting in postings:
                weight = compute_term_weight(
                    tf=posting.term_frequency,
                    df=df,
                    num_documents=self.index.num_documents,
                    tf_weighting=scoring.tf_weighting,
                    df_weighting=scoring.df_weighting,
                )
                squared_sums[posting.doc_id] += weight * weight

        norms = {
            doc_id: (value ** 0.5 if value > 0.0 else 0.0)
            for doc_id, value in squared_sums.items()
        }

        # Ensure all indexed documents are present, even empty ones.
        for doc_id in self.index.document_lengths:
            norms.setdefault(doc_id, 0.0)

        self._cosine_doc_norm_cache[cache_key] = norms
        return norms

    def _score_cosine(
        self,
        query_tokens: Sequence[str],
        *,
        scoring: ScoringConfig,
        thesaurus: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, float]:
        """Score documents with cosine-style accumulation.

        Raw score:
            sum_t w(t, q) * w(t, d)

        Then optionally normalized by:
        - cosine document/query norm
        - pivoted document length normalization
        """
        query_term_weights = build_weighted_query_terms(
            query_tokens,
            num_documents=self.index.num_documents,
            document_frequency_lookup=self.index.document_frequency,
            scoring=scoring,
            thesaurus=thesaurus,
        )

        if not query_term_weights:
            return {}

        scores: dict[str, float] = defaultdict(float)

        for term, query_weight in query_term_weights.items():
            if query_weight == 0.0:
                continue

            df = self.index.get_document_frequency(term)
            if df <= 0:
                continue

            for posting in self.index.get_postings(term):
                doc_weight = compute_term_weight(
                    tf=posting.term_frequency,
                    df=df,
                    num_documents=self.index.num_documents,
                    tf_weighting=scoring.tf_weighting,
                    df_weighting=scoring.df_weighting,
                )
                if doc_weight != 0.0:
                    scores[posting.doc_id] += query_weight * doc_weight

        if not scores:
            return {}

        if scoring.normalization == "cosine":
            query_norm = compute_cosine_norm(query_term_weights)
            if query_norm > 0.0:
                doc_norms = self._get_or_compute_cosine_document_norms(scoring)
                normalized_scores: dict[str, float] = {}
                for doc_id, score in scores.items():
                    doc_norm = doc_norms.get(doc_id, 0.0)
                    denominator = query_norm * doc_norm
                    if denominator > 0.0:
                        normalized_scores[doc_id] = score / denominator
                return normalized_scores
            return {}

        if scoring.normalization == "pivoted":
            normalized_scores: dict[str, float] = {}
            for doc_id, score in scores.items():
                denominator = compute_pivot_denominator(
                    self.index.get_document_length(doc_id),
                    self.index.average_document_length,
                    slope=scoring.pivoted.slope,
                )
                if denominator > 0.0:
                    normalized_scores[doc_id] = score / denominator
            return normalized_scores

        raise ValueError(f"Unsupported normalization mode: {scoring.normalization}")

    def _score_bm25(
        self,
        query_tokens: Sequence[str],
        *,
        scoring: ScoringConfig,
        thesaurus: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, float]:
        """Score documents with BM25.

        Notes
        -----
        - BM25 uses its own built-in length normalization.
        - ``scoring.normalization`` is intentionally ignored here.
        - Query-side weights are used as multiplicative query term factors.
        """
        from scoring import bm25_term_score  # local import to keep namespace tidy

        query_term_weights = build_weighted_query_terms(
            query_tokens,
            num_documents=self.index.num_documents,
            document_frequency_lookup=self.index.document_frequency,
            scoring=scoring,
            thesaurus=thesaurus,
        )

        if not query_term_weights:
            return {}

        scores: dict[str, float] = defaultdict(float)

        for term, query_weight in query_term_weights.items():
            if query_weight == 0.0:
                continue

            df = self.index.get_document_frequency(term)
            if df <= 0:
                continue

            for posting in self.index.get_postings(term):
                scores[posting.doc_id] += bm25_term_score(
                    tf=posting.term_frequency,
                    df=df,
                    num_documents=self.index.num_documents,
                    document_length=self.index.get_document_length(posting.doc_id),
                    average_document_length=self.index.average_document_length,
                    query_term_weight=query_weight,
                    k1=scoring.bm25.k1,
                    b=scoring.bm25.b,
                )

        return dict(scores)

    def score(
        self,
        query_tokens: Sequence[str],
        *,
        scoring: ScoringConfig,
        thesaurus: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, float]:
        """Score all matching documents for one tokenized query."""
        if scoring.similarity == "cosine":
            return self._score_cosine(
                query_tokens,
                scoring=scoring,
                thesaurus=thesaurus,
            )

        if scoring.similarity == "bm25":
            return self._score_bm25(
                query_tokens,
                scoring=scoring,
                thesaurus=thesaurus,
            )

        raise ValueError(f"Unsupported similarity mode: {scoring.similarity}")

    def retrieve(
        self,
        topic_id: str,
        query_tokens: Sequence[str],
        *,
        scoring: ScoringConfig,
        thesaurus: Mapping[str, Sequence[str]] | None = None,
        top_k: int = 1000,
    ) -> list[RetrievalResult]:
        """Retrieve ranked results for one topic.

        Ranking tie-break
        -----------------
        Results are ordered by:
        1. descending score
        2. ascending doc_id

        This makes output deterministic.
        """
        scores = self.score(
            query_tokens,
            scoring=scoring,
            thesaurus=thesaurus,
        )

        if not scores:
            return []

        ranked_items = nlargest(
            top_k,
            scores.items(),
            key=lambda item: (item[1], item[0]),
        )

        ranked_items = sorted(
            ranked_items,
            key=lambda item: (-item[1], item[0]),
        )

        return [
            RetrievalResult(
                topic_id=topic_id,
                doc_id=doc_id,
                rank=rank,
                score=score,
            )
            for rank, (doc_id, score) in enumerate(ranked_items)
        ]

    def retrieve_many(
        self,
        topics: Mapping[str, Sequence[str]],
        *,
        scoring: ScoringConfig,
        thesaurus: Mapping[str, Sequence[str]] | None = None,
        top_k: int = 1000,
    ) -> dict[str, list[RetrievalResult]]:
        """Retrieve ranked results for multiple topics."""
        return {
            topic_id: self.retrieve(
                topic_id,
                query_tokens,
                scoring=scoring,
                thesaurus=thesaurus,
                top_k=top_k,
            )
            for topic_id, query_tokens in topics.items()
        }


__all__ = [
    "RetrievalResult",
    "RetrievalSystem",
]