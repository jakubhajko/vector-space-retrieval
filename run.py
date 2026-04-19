from __future__ import annotations

"""Simple experiment entry point for the IR system.

This file is meant to be the main place where you choose:
- which collection split to run on
- which predefined experiment configuration to use
- where to save the output run file

Available split choices
-----------------------
- "train_en"
- "test_en"
- "train_cs"
- "test_cs"

Available predefined runs
-------------------------
- RUN_0_CONFIG:
    Baseline run requested by the assignment.
- RUN_1_CONFIG:
    A stronger experimental run using only title-based queries as requested.

Notes on configurable choices
-----------------------------
Query construction:
- "title"
- "title_desc"
- "title_desc_narr"

Term weighting:
- "natural"
- "logarithm"

Document-frequency weighting:
- "none"
- "idf"
- "probabilistic_idf"

Normalization:
- "none"
- "cosine"
- "pivoted"

Similarity:
- "cosine"
- "bm25"

Query expansion:
- "none"
- "thesaurus_based"
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

from vector_space_model import (
    DATA_DIR,
    RetrievalSystem,
    ScoringConfig,
    build_inverted_index,
    flatten_results_by_topic,
    preprocess_cs_documents,
    preprocess_cs_topics,
    preprocess_en_documents,
    preprocess_en_topics,
    regex_word_tokenizer,
    write_trec_results,
)

CollectionSplit = Literal["train_en", "test_en", "train_cs", "test_cs"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level run configuration.

    The preprocessing callables are split for documents and topics so you can
    easily experiment with them independently if needed.
    """

    run_id: str
    query_construction: str = "title"

    # Document preprocessing
    document_tokenizer_fn: object | None = None
    document_equivalence_classing_fn: object | None = None
    document_stopword_removal_fn: object | None = None

    # Topic preprocessing
    topic_tokenizer_fn: object | None = None
    topic_equivalence_classing_fn: object | None = None
    topic_stopword_removal_fn: object | None = None

    # Retrieval/scoring
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    # Optional thesaurus for query expansion experiments
    thesaurus: Mapping[str, Sequence[str]] | None = None

    # Max retrieved documents per topic
    top_k: int = 1000


@dataclass(frozen=True)
class CollectionPaths:
    """Resolved paths and metadata for one collection split."""

    split: CollectionSplit
    language: Literal["en", "cs"]
    documents_lst_path: Path
    documents_dir: Path
    topics_xml_path: Path


def resolve_collection_paths(split: CollectionSplit) -> CollectionPaths:
    """Resolve all required input paths for a selected collection split."""
    if split == "train_en":
        return CollectionPaths(
            split=split,
            language="en",
            documents_lst_path=Path(DATA_DIR) / "documents_en.lst",
            documents_dir=Path(DATA_DIR) / "documents_en",
            topics_xml_path=Path(DATA_DIR) / "topics-train_en.xml",
        )

    if split == "test_en":
        return CollectionPaths(
            split=split,
            language="en",
            documents_lst_path=Path(DATA_DIR) / "documents_en.lst",
            documents_dir=Path(DATA_DIR) / "documents_en",
            topics_xml_path=Path(DATA_DIR) / "topics-test_en.xml",
        )

    if split == "train_cs":
        return CollectionPaths(
            split=split,
            language="cs",
            documents_lst_path=Path(DATA_DIR) / "documents_cs.lst",
            documents_dir=Path(DATA_DIR) / "documents_cs",
            topics_xml_path=Path(DATA_DIR) / "topics-train_cs.xml",
        )

    if split == "test_cs":
        return CollectionPaths(
            split=split,
            language="cs",
            documents_lst_path=Path(DATA_DIR) / "documents_cs.lst",
            documents_dir=Path(DATA_DIR) / "documents_cs",
            topics_xml_path=Path(DATA_DIR) / "topics-test_cs.xml",
        )

    raise ValueError(f"Unsupported collection split: {split}")


def load_documents_for_split(
    collection: CollectionPaths,
    config: ExperimentConfig,
) -> dict[str, object]:
    """Load and preprocess documents for the selected language/split."""
    if collection.language == "en":
        return preprocess_en_documents(
            lst_path=collection.documents_lst_path,
            documents_dir=collection.documents_dir,
            tokenizer_fn=(
                config.document_tokenizer_fn
                if config.document_tokenizer_fn is not None
                else preprocess_en_documents.__defaults__[2]  # type: ignore[index]
            ),
            equivalence_classing_fn=config.document_equivalence_classing_fn,
            stopword_removal_fn=config.document_stopword_removal_fn,
        )

    return preprocess_cs_documents(
        lst_path=collection.documents_lst_path,
        documents_dir=collection.documents_dir,
        tokenizer_fn=(
            config.document_tokenizer_fn
            if config.document_tokenizer_fn is not None
            else preprocess_cs_documents.__defaults__[2]  # type: ignore[index]
        ),
        equivalence_classing_fn=config.document_equivalence_classing_fn,
        stopword_removal_fn=config.document_stopword_removal_fn,
    )


def load_topics_for_split(
    collection: CollectionPaths,
    config: ExperimentConfig,
) -> dict[str, object]:
    """Load and preprocess topics for the selected language/split."""
    if collection.language == "en":
        return preprocess_en_topics(
            xml_path=collection.topics_xml_path,
            query_construction=config.query_construction,
            tokenizer_fn=(
                config.topic_tokenizer_fn
                if config.topic_tokenizer_fn is not None
                else preprocess_en_topics.__defaults__[1]  # type: ignore[index]
            ),
            equivalence_classing_fn=config.topic_equivalence_classing_fn,
            stopword_removal_fn=config.topic_stopword_removal_fn,
        )

    return preprocess_cs_topics(
        xml_path=collection.topics_xml_path,
        query_construction=config.query_construction,
        tokenizer_fn=(
            config.topic_tokenizer_fn
            if config.topic_tokenizer_fn is not None
            else preprocess_cs_topics.__defaults__[1]  # type: ignore[index]
        ),
        equivalence_classing_fn=config.topic_equivalence_classing_fn,
        stopword_removal_fn=config.topic_stopword_removal_fn,
    )


def build_output_filename(run_id: str, split: CollectionSplit) -> str:
    """Build output filename following the current naming style."""
    return f"{run_id}_{split}.res"


def execute_run(
    *,
    split: CollectionSplit,
    config: ExperimentConfig,
    output_dir: str | Path = ".",
) -> Path:
    """Run one full experiment and write the TREC result file."""
    collection = resolve_collection_paths(split)

    # 1) Load documents
    documents = load_documents_for_split(collection, config)

    # 2) Build index
    index = build_inverted_index(documents.values())

    # 3) Load topics
    topics = load_topics_for_split(collection, config)

    # 4) Retrieve
    retrieval_system = RetrievalSystem(index)
    results_by_topic = retrieval_system.retrieve_many(
        {topic_id: topic.query_tokens for topic_id, topic in topics.items()},
        scoring=config.scoring,
        thesaurus=config.thesaurus,
        top_k=config.top_k,
    )

    # 5) Write results
    output_path = Path(output_dir) / build_output_filename(config.run_id, split)
    write_trec_results(
        output_path,
        flatten_results_by_topic(results_by_topic),
        run_id=config.run_id,
    )

    return output_path


# ---------------------------------------------------------------------------
# Predefined experiment configurations
# ---------------------------------------------------------------------------

# Baseline run requested by the assignment:
# - token delimiters: any sequence of whitespace and punctuation marks
# - term equivalence classes: none
# - removing stopwords: no
# - query construction: title only
# - tf weighting: natural
# - df weighting: none
# - normalization: cosine
# - similarity: cosine
# - pseudo-relevance feedback: none
# - query expansion: none
RUN_0_CONFIG = ExperimentConfig(
    run_id="run-0",
    query_construction="title",
    document_tokenizer_fn=regex_word_tokenizer,
    document_equivalence_classing_fn=None,
    document_stopword_removal_fn=None,
    topic_tokenizer_fn=regex_word_tokenizer,
    topic_equivalence_classing_fn=None,
    topic_stopword_removal_fn=None,
    scoring=ScoringConfig(
        tf_weighting="natural",
        df_weighting="none",
        normalization="cosine",
        similarity="cosine",
        query_expansion="none",
    ),
    thesaurus=None,
    top_k=1000,
)

# Stronger title-only run:
# Chosen to be a sensible general improvement from the available options.
#
# Rationale:
# - keep title-only queries as required
# - use the default language-specific preprocessing from the loaders
#   (English: default tokenizer + default equivalence classing + stopword removal,
#    Czech:   default tokenizer + default equivalence classing + stopword removal)
# - use log tf + idf + cosine + cosine similarity
#
# This is a conservative "best overall" choice for a vector-space baseline/experiment.
RUN_1_CONFIG = ExperimentConfig(
    run_id="run-1",
    query_construction="title",
    # None here means: use the loader defaults for the given language.
    document_tokenizer_fn=None,
    document_equivalence_classing_fn=None,
    document_stopword_removal_fn=None,
    topic_tokenizer_fn=None,
    topic_equivalence_classing_fn=None,
    topic_stopword_removal_fn=None,
    scoring=ScoringConfig(
        tf_weighting="logarithm",
        df_weighting="idf",
        normalization="cosine",
        similarity="cosine",
        query_expansion="none",
    ),
    thesaurus=None,
    top_k=1000,
)


# ---------------------------------------------------------------------------
# Main selection
# ---------------------------------------------------------------------------

# Change only these two lines for most experiments:
SELECTED_SPLIT: CollectionSplit = "test_cs"
SELECTED_CONFIG: ExperimentConfig = RUN_0_CONFIG

# Optional output directory. "." means current working directory.
OUTPUT_DIR = "."


if __name__ == "__main__":
    output_path = execute_run(
        split=SELECTED_SPLIT,
        config=SELECTED_CONFIG,
        output_dir=OUTPUT_DIR,
    )
    print(f"Saved results to: {output_path}")