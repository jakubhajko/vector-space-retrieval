from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence, Any

# Assuming these are your local module imports
from vector_space_model.config import DATA_DIR
from vector_space_model.load_documents import preprocess_cs_documents, preprocess_en_documents
from vector_space_model.load_topics import preprocess_cs_topics, preprocess_en_topics
from vector_space_model.text_preprocessing import (
    regex_word_tokenizer, regex_tokenizer_with_connectors, 
    casefold_tokens, normalize_numbers, casefold_and_normalize_numbers, 
    english_casefold_and_stem, czech_casefold_and_stem, 
    english_stopword_removal, czech_stopword_removal
)
from vector_space_model.index import build_inverted_index
from vector_space_model.retrieval import RetrievalSystem
from vector_space_model.scoring import ScoringConfig
from vector_space_model.results import write_trec_results, flatten_results_by_topic


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level run configuration."""
    query_construction: str = "title"

    # Preprocessing Callables
    # Note: Using dicts for language-specific rules (e.g., {"en": func, "cs": func})
    tokenizer_fn: Any = regex_word_tokenizer
    equivalence_classing_fn: dict[str, Any] | None = None
    stopword_removal_fn: dict[str, Any] | None = None

    # Retrieval/scoring
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    thesaurus: Mapping[str, Sequence[str]] | None = None
    top_k: int = 1000


# --- Predefined Runs ---

# =============================================================================
# EXPERIMENT CONFIGURATION DOMAIN
# =============================================================================
# TEXT PREPROCESSING OPTIONS:
# ----------------------------------------------------------------
#  tokenizer_fn:             regex_word_tokenizer, regex_tokenizer_with_connectors 
#  equivalence_classing_fn:  None, english_casefold_and_stem, czech_casefold_and_stem, 
#                            casefold_and_normalize_numbers, casefold_tokens, normalize_numbers
#  stopword_removal_fn:      None ,english_stopword_removal, czech_stopword_removal
#
# RETRIEVAL/SCORING OPTIONS:
# ----------------------------------------------------------------
#  query_construction:       "title", "title_desc", "title_desc_narr"
#  tf_weighting:             "natural", "logarithm"
#  df_weighting:             "none", "idf", "probabilistic_idf"
#  normalization:            "none", "cosine", "pivoted"
#  similarity:               "cosine", "bm25"
#  query_expansion:          "none", "thesaurus_based"
# =============================================================================

RUN_0_CONFIG = ExperimentConfig(
    # Baseline run requested by the assignment
    query_construction="title",
    tokenizer_fn=regex_word_tokenizer,
    equivalence_classing_fn=None,
    stopword_removal_fn=None,
    scoring=ScoringConfig(
        tf_weighting="natural",
        df_weighting="none",
        normalization="cosine",
        similarity="cosine",
        query_expansion="none",
    )
)

RUN_1_CONFIG = ExperimentConfig(
    # Stronger title-only run
    query_construction="title",
    tokenizer_fn=regex_tokenizer_with_connectors,
    equivalence_classing_fn={
        "en": english_casefold_and_stem,
        "cs": czech_casefold_and_stem
    },
    stopword_removal_fn={
        "en": english_stopword_removal,
        "cs": czech_stopword_removal
    },
    scoring=ScoringConfig(
        tf_weighting="logarithm",
        df_weighting="idf",
        normalization="pivoted",
        similarity="cosine",
        query_expansion="none",
    )
)

RUN_2_CONFIG = ExperimentConfig(
    # Strongest no constraints run
    query_construction="title_desc_narr",
    tokenizer_fn=regex_tokenizer_with_connectors,
    equivalence_classing_fn={
        "en": english_casefold_and_stem,
        "cs": czech_casefold_and_stem
    },
    stopword_removal_fn={
        "en": english_stopword_removal,
        "cs": czech_stopword_removal
    },
    scoring=ScoringConfig(
        tf_weighting="logarithm",
        df_weighting="idf",
        normalization="pivoted",
        similarity="cosine",
        query_expansion="none",
    )
)

# Easily extendable mapping of run IDs to configurations
CONFIGS = {
    "run-0": RUN_0_CONFIG,
    "run-1": RUN_1_CONFIG,
    "run-2": RUN_2_CONFIG,  
}


# =============================================================================
# CORE EXECUTION LOGIC
# =============================================================================

def print_active_configuration(run_id: str, config: ExperimentConfig):
    """Prints the configuration clearly for easy inclusion in your report."""
    print(f"\n{'='*55}")
    print(f"EXECUTING EXPERIMENT: {run_id}")
    print(f"{'='*55}")
    print(f"CONFIGURATION:\n")
    print(f"Query Construction:   {config.query_construction}")
    print(f"Tokenizer:            {config.tokenizer_fn.__name__ if config.tokenizer_fn else 'None'}")
    print(f"Equivalence Classing: {config.equivalence_classing_fn[run_id[-2:]].__name__ if config.equivalence_classing_fn else 'None'}")
    print(f"Stopword Removal:     {config.stopword_removal_fn[run_id[-2:]].__name__ if config.stopword_removal_fn else 'None'}")
    print(f"TF Weighting:         {config.scoring.tf_weighting}")
    print(f"DF Weighting:         {config.scoring.df_weighting}")
    print(f"Normalization:        {config.scoring.normalization}")
    print(f"Similarity:           {config.scoring.similarity}")
    print(f"Query Expansion:      {config.scoring.query_expansion}")
    print(f"Top K Docs:           {config.top_k}")
    print(f"{'='*55}\n")


def execute_run(topics_path: Path, documents_path: Path, run_id: str, output_path: Path):
    """Run one full experiment and write the TREC result file."""
    
    # 1. Determine base config and language
    # Maps e.g. "run-0_cs" -> "run-0" base config
    base_run_id = next((k for k in CONFIGS.keys() if k in run_id), "run-0")
    config = CONFIGS[base_run_id]
    
    lang = "cs" if "_cs" in str(topics_path) else "en"
    print_active_configuration(run_id, config)

    # Resolve language-specific preprocessing functions
    eq_classing = config.equivalence_classing_fn[lang] if config.equivalence_classing_fn else None
    stopwords = config.stopword_removal_fn[lang] if config.stopword_removal_fn else None

    # Derive documents directory from the .lst file (e.g. documents_en.lst -> documents_en dir)
    documents_dir = documents_path.parent / documents_path.stem

    # 2. Load Documents
    print(f"Loading documents from {documents_path}...")
    if lang == "en":
        documents = preprocess_en_documents(
            lst_path=documents_path, documents_dir=documents_dir,
            tokenizer_fn=config.tokenizer_fn,
            equivalence_classing_fn=eq_classing,
            stopword_removal_fn=stopwords
        )
    else:
        documents = preprocess_cs_documents(
            lst_path=documents_path, documents_dir=documents_dir,
            tokenizer_fn=config.tokenizer_fn,
            equivalence_classing_fn=eq_classing,
            stopword_removal_fn=stopwords
        )

    # 3. Build Index
    print("Building inverted index...")
    index = build_inverted_index(documents.values())

    # 4. Load Topics
    print(f"Loading topics from {topics_path}...")
    if lang == "en":
        topics = preprocess_en_topics(
            xml_path=topics_path, query_construction=config.query_construction,
            tokenizer_fn=config.tokenizer_fn,
            equivalence_classing_fn=eq_classing,
            stopword_removal_fn=stopwords
        )
    else:
        topics = preprocess_cs_topics(
            xml_path=topics_path, query_construction=config.query_construction,
            tokenizer_fn=config.tokenizer_fn,
            equivalence_classing_fn=eq_classing,
            stopword_removal_fn=stopwords
        )

    # 5. Retrieve
    print("Retrieving documents...")
    retrieval_system = RetrievalSystem(index)
    results_by_topic = retrieval_system.retrieve_many(
        {topic_id: topic.query_tokens for topic_id, topic in topics.items()},
        scoring=config.scoring,
        thesaurus=config.thesaurus,
        top_k=config.top_k,
    )

    # 6. Write Results
    print(f"Writing results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trec_results(
        output_path,
        flatten_results_by_topic(results_by_topic),
        run_id=run_id,
    )
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPFL103 Information Retrieval System")
    parser.add_argument("-q", "--topics", type=Path, required=True, help="Topics XML file (e.g., topics-train_en.xml)")
    parser.add_argument("-d", "--documents", type=Path, required=True, help="Documents list file (e.g., documents_en.lst)")
    parser.add_argument("-r", "--run_id", type=str, required=True, help="Experiment run ID (e.g., run-0_cs, run-1_en)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output results file (e.g., run-0_train_cs.res)")
    
    args = parser.parse_args()
    
    execute_run(
        topics_path=args.topics,
        documents_path=args.documents,
        run_id=args.run_id,
        output_path=args.output
    )

    # Example usage:

    # Run 0
    # python run.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-0_train_en -o results/run-0_train_en.res
    # python run.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-0_train_cs -o results/run-0_train_cs.res
    # python run.py -q data/topics-test_en.xml -d data/documents_en.lst -r run-0_test_en -o results/run-0_test_en.res
    # python run.py -q data/topics-test_cs.xml -d data/documents_cs.lst -r run-0_test_cs -o results/run-0_test_cs.res

    # Run 1
    # python run.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-1_train_en -o results/run-1_train_en.res
    # python run.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-1_train_cs -o results/run-1_train_cs.res
    # python run.py -q data/topics-test_en.xml -d data/documents_en.lst -r run-1_test_en -o results/run-1_test_en.res
    # python run.py -q data/topics-test_cs.xml -d data/documents_cs.lst -r run-1_test_cs -o results/run-1_test_cs.res


    # For uv users:
    # Run 0 
    # uv run run.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-0_train_en -o results/run-0_train_en.res
    # uv run run.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-0_train_cs -o results/run-0_train_cs.res
    # uv run run.py -q data/topics-test_en.xml -d data/documents_en.lst -r run-0_test_en -o results/run-0_test_en.res
    # uv run run.py -q data/topics-test_cs.xml -d data/documents_cs.lst -r run-0_test_cs -o results/run-0_test_cs.res

    # Run 1
    # uv run run.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-1_train_en -o results/run-1_train_en.res
    # uv run run.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-1_train_cs -o results/run-1_train_cs.res
    # uv run run.py -q data/topics-test_en.xml -d data/documents_en.lst -r run-1_test_en -o results/run-1_test_en.res
    # uv run run.py -q data/topics-test_cs.xml -d data/documents_cs.lst -r run-1_test_cs -o results/run-1_test_cs.res


    #EN
    # uv run run.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-1_train_1_en -o experiments/en/run-1_train_1_en.res
    # uv run run.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-1_train_2_en -o experiments/en/run-1_train_2_en.res
    # uv run run.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-1_train_3_en -o experiments/en/run-1_train_3_en.res
    # uv run run.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-1_train_4_en -o experiments/en/run-1_train_4_en.res
    # uv run run.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-1_train_5_en -o experiments/en/run-1_train_5_en.res
    # uv run run.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-1_train_6_en -o experiments/en/run-1_train_6_en.res

    #CS
    # uv run run.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-1_train_1_cs -o experiments/cs/run-1_train_1_cs.res
    # uv run run.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-1_train_2_cs -o experiments/cs/run-1_train_2_cs.res
    # uv run run.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-1_train_3_cs -o experiments/cs/run-1_train_3_cs.res
    # uv run run.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-1_train_4_cs -o experiments/cs/run-1_train_4_cs.res
    # uv run run.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-1_train_5_cs -o experiments/cs/run-1_train_5_cs.res
    # uv run run.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-1_train_6_cs -o experiments/cs/run-1_train_6_cs.res

    # Run 2
    # uv run run.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-2_train_en -o experiments/en/run-2_train_en.res
    # uv run run.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-2_train_cs -o experiments/cs/run-2_train_cs.res