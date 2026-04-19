from __future__ import annotations

"""Scoring formulas, weighting options, normalization, and query expansion.

Supported options
-----------------
Term weighting:
- natural
- logarithm

Document frequency weighting:
- none
- idf
- probabilistic_idf

Normalization:
- none
- cosine
- pivoted

Similarity:
- cosine
- bm25

Query expansion:
- none
- thesaurus_based

Formula notes
-------------
Term frequency weighting
~~~~~~~~~~~~~~~~~~~~~~~~
natural:
    tf_w(tf) = tf

logarithm:
    tf_w(tf) = 1 + log10(tf), for tf > 0

Document frequency weighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
none:
    df_w(df, N) = 1

idf:
    df_w(df, N) = log10(N / df), for df > 0

probabilistic_idf:
    df_w(df, N) = log10((N - df) / df), for 0 < df < N
    This can be negative for very frequent terms; that is allowed.

Normalization
~~~~~~~~~~~~~
cosine:
    ||d|| = sqrt(sum_t w(t, d)^2)
    normalized weight = w(t, d) / ||d||

pivoted:
    normalized weight = w(t, d) / ((1 - s) + s * (|d| / avgdl))
    where s is the pivot slope, typically around 0.2

BM25
~~~~
score(q, d) = sum_t idf_bm25(t) * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d|/avgdl))) * q_w(t)

where:
    idf_bm25(t) = log10((N - df + 0.5) / (df + 0.5))

Thesaurus expansion
~~~~~~~~~~~~~~~~~~~
Expanded query terms receive:
    added_weight = expansion_weight * original_query_term_weight

The thesaurus is assumed to already be in the same token space as the processed
queries (i.e. keys and synonyms are already normalized/stemmed if needed).
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log10, sqrt
from typing import Mapping, Sequence, Literal


TfWeighting = Literal["natural", "logarithm"]
DfWeighting = Literal["none", "idf", "probabilistic_idf"]
Normalization = Literal["none", "cosine", "pivoted"]
Similarity = Literal["cosine", "bm25"]
QueryExpansionMode = Literal["none", "thesaurus_based"]


@dataclass(frozen=True)
class BM25Parameters:
    """BM25 hyperparameters."""

    k1: float = 1.2
    b: float = 0.75


@dataclass(frozen=True)
class PivotedNormalizationParameters:
    """Pivoted normalization hyperparameters."""

    slope: float = 0.2


@dataclass(frozen=True)
class ThesaurusExpansionConfig:
    """Configuration for thesaurus-based query expansion."""

    expansion_weight: float = 0.3
    max_expansions_per_term: int | None = None


@dataclass(frozen=True)
class ScoringConfig:
    """Top-level retrieval scoring configuration."""

    tf_weighting: TfWeighting = "natural"
    df_weighting: DfWeighting = "idf"
    normalization: Normalization = "cosine"
    similarity: Similarity = "cosine"
    query_expansion: QueryExpansionMode = "none"
    bm25: BM25Parameters = BM25Parameters()
    pivoted: PivotedNormalizationParameters = PivotedNormalizationParameters()
    thesaurus: ThesaurusExpansionConfig = ThesaurusExpansionConfig()


def apply_tf_weight(tf: int | float, mode: TfWeighting) -> float:
    """Apply term-frequency weighting.

    Returns 0.0 for non-positive frequencies.
    """
    if tf <= 0:
        return 0.0
    if mode == "natural":
        return float(tf)
    if mode == "logarithm":
        return 1.0 + log10(float(tf))
    raise ValueError(f"Unsupported tf weighting mode: {mode}")


def apply_df_weight(df: int, num_documents: int, mode: DfWeighting) -> float:
    """Apply document-frequency weighting."""
    if df <= 0 or num_documents <= 0:
        return 0.0

    if mode == "none":
        return 1.0

    if mode == "idf":
        return log10(num_documents / df)

    if mode == "probabilistic_idf":
        if df >= num_documents:
            return 0.0
        return log10((num_documents - df) / df)

    raise ValueError(f"Unsupported df weighting mode: {mode}")


def compute_term_weight(
    *,
    tf: int | float,
    df: int,
    num_documents: int,
    tf_weighting: TfWeighting,
    df_weighting: DfWeighting,
) -> float:
    """Compute a combined weighted term value."""
    return apply_tf_weight(tf, tf_weighting) * apply_df_weight(df, num_documents, df_weighting)


def compute_cosine_norm(weights_by_term: Mapping[str, float]) -> float:
    """Compute Euclidean norm for cosine normalization."""
    squared_sum = sum(weight * weight for weight in weights_by_term.values())
    return sqrt(squared_sum) if squared_sum > 0.0 else 0.0


def compute_pivot_denominator(
    document_length: int,
    average_document_length: float,
    *,
    slope: float,
) -> float:
    """Compute pivoted normalization denominator.

    Formula:
        (1 - s) + s * (|d| / avgdl)
    """
    if average_document_length <= 0.0:
        return 1.0
    return (1.0 - slope) + slope * (document_length / average_document_length)


def bm25_idf(df: int, num_documents: int) -> float:
    """Compute BM25's IDF-like component.

    Formula:
        log10((N - df + 0.5) / (df + 0.5))
    """
    if df <= 0 or num_documents <= 0:
        return 0.0
    return log10((num_documents - df + 0.5) / (df + 0.5))


def bm25_term_score(
    *,
    tf: int,
    df: int,
    num_documents: int,
    document_length: int,
    average_document_length: float,
    query_term_weight: float,
    k1: float,
    b: float,
) -> float:
    """Compute the BM25 contribution of one term for one document."""
    if tf <= 0:
        return 0.0

    avgdl = average_document_length if average_document_length > 0.0 else 1.0
    length_factor = 1.0 - b + b * (document_length / avgdl)
    numerator = tf * (k1 + 1.0)
    denominator = tf + k1 * length_factor

    return bm25_idf(df, num_documents) * (numerator / denominator) * query_term_weight


def build_query_term_counts(tokens: Sequence[str]) -> Counter[str]:
    """Count query tokens."""
    return Counter(tokens)


def expand_query_term_weights(
    base_term_weights: Mapping[str, float],
    *,
    thesaurus: Mapping[str, Sequence[str]] | None,
    config: ThesaurusExpansionConfig,
) -> dict[str, float]:
    """Expand query weights using a thesaurus mapping.

    Parameters
    ----------
    base_term_weights:
        Original query term weights.
    thesaurus:
        Mapping from term -> sequence of expansion terms.
    config:
        Expansion hyperparameters.

    Returns
    -------
    dict[str, float]
        Expanded query term weights. Original terms are preserved.
    """
    expanded: dict[str, float] = defaultdict(float)
    for term, weight in base_term_weights.items():
        expanded[term] += weight

    if not thesaurus:
        return dict(expanded)

    for term, original_weight in base_term_weights.items():
        synonyms = list(thesaurus.get(term, ()))
        if config.max_expansions_per_term is not None:
            synonyms = synonyms[: config.max_expansions_per_term]

        added_weight = original_weight * config.expansion_weight
        if added_weight == 0.0:
            continue

        for synonym in synonyms:
            if not synonym:
                continue
            expanded[synonym] += added_weight

    return dict(expanded)


def build_weighted_query_terms(
    query_tokens: Sequence[str],
    *,
    num_documents: int,
    document_frequency_lookup: Mapping[str, int],
    scoring: ScoringConfig,
    thesaurus: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, float]:
    """Build weighted query term weights.

    For cosine-style retrieval, this applies tf and df weighting.
    For BM25 retrieval, the returned values act as query multipliers.
    """
    counts = build_query_term_counts(query_tokens)

    if scoring.similarity == "bm25":
        base_weights = {
            term: apply_tf_weight(tf, scoring.tf_weighting)
            for term, tf in counts.items()
        }
    else:
        base_weights = {
            term: compute_term_weight(
                tf=tf,
                df=document_frequency_lookup.get(term, 0),
                num_documents=num_documents,
                tf_weighting=scoring.tf_weighting,
                df_weighting=scoring.df_weighting,
            )
            for term, tf in counts.items()
            if document_frequency_lookup.get(term, 0) > 0
        }

    if scoring.query_expansion == "thesaurus_based":
        return expand_query_term_weights(
            base_weights,
            thesaurus=thesaurus,
            config=scoring.thesaurus,
        )

    return dict(base_weights)


__all__ = [
    "BM25Parameters",
    "DfWeighting",
    "Normalization",
    "PivotedNormalizationParameters",
    "QueryExpansionMode",
    "ScoringConfig",
    "Similarity",
    "TfWeighting",
    "ThesaurusExpansionConfig",
    "apply_df_weight",
    "apply_tf_weight",
    "bm25_idf",
    "bm25_term_score",
    "build_query_term_counts",
    "build_weighted_query_terms",
    "compute_cosine_norm",
    "compute_pivot_denominator",
    "compute_term_weight",
    "expand_query_term_weights",
]