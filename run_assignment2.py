from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from whoosh.analysis import RegexTokenizer, LowercaseFilter, StopFilter, StemmingAnalyzer, Filter
from whoosh.scoring import BM25F, PL2, WeightingModel

# Your local imports
from vector_space_model.config import DATA_DIR
from vector_space_model.results import write_trec_results
from vector_space_model.text_preprocessing import CZECH_STOPWORDS, _CZECH_STEMMER
from vector_space_model.whoosh_engine import build_whoosh_index, run_whoosh_search


@dataclass(frozen=True)
class WhooshExperimentConfig:
    """Top-level Whoosh run configuration."""
    query_construction: str = "title"
    analyzer_en: Any = field(default_factory=RegexTokenizer)
    analyzer_cs: Any = field(default_factory=RegexTokenizer)
    scorer: WeightingModel = field(default_factory=BM25F)
    use_prf: bool = False
    top_k: int = 1000

# =============================================================================
# CUSTOM CZECH TEXT PROCESSING
# =============================================================================
class CzechStemFilter(Filter):
    """Wraps your A1 Czech stemmer into a Whoosh filter pipeline."""
    def __call__(self, tokens):
        for t in tokens:
            t.text = _CZECH_STEMMER.stem(t.text)
            yield t

# The ultimate Czech analyzer: Tokenize -> Lowercase -> Remove Stopwords -> Stem
czech_stemming_analyzer = RegexTokenizer() | LowercaseFilter() | StopFilter(stoplist=CZECH_STOPWORDS) | CzechStemFilter()

# =============================================================================
# CONFIGURATIONS
# =============================================================================
CONFIGS = {
    # Run 1: The Modern Standard (Title Only)
    "run-1_standard": WhooshExperimentConfig(
        query_construction="title",
        analyzer_en=StemmingAnalyzer(),
        analyzer_cs=czech_stemming_analyzer,
        scorer=BM25F(B=0.75, K1=1.2),
        use_prf=False
    ),
    # Run 1: The Vocabulary Expander (Title Only + PRF)
    "run-1_prf": WhooshExperimentConfig(
        query_construction="title",
        analyzer_en=StemmingAnalyzer(),
        analyzer_cs=czech_stemming_analyzer,
        scorer=BM25F(B=0.75, K1=1.2),
        use_prf=True 
    ),
    # Run 1: The Academic Wildcard (Title Only + PL2 + PRF)
    "run-1_pl2": WhooshExperimentConfig(
        query_construction="title",
        analyzer_en=StemmingAnalyzer(),
        analyzer_cs=czech_stemming_analyzer,
        scorer=PL2(c=10.0),
        use_prf=True
    ),
    # Run 2: Unconstrained Extra Credit (Title+Desc + BM25 + PRF)
    "run-2_experiment1": WhooshExperimentConfig(
        query_construction="title_desc", 
        analyzer_en=StemmingAnalyzer(),
        analyzer_cs=czech_stemming_analyzer,
        scorer=PL2(c=10.0),
        use_prf=True
    ),
    "run-2_experiment2": WhooshExperimentConfig(
        query_construction="title", 
        analyzer_en=StemmingAnalyzer(),
        analyzer_cs=czech_stemming_analyzer,
        scorer=PL2(c=10.0),
        use_prf=True
    ),
    # Run 1 Final (Title Only) - This is the config you should submit as your primary run for evaluation
    "run-1_final": WhooshExperimentConfig(
        query_construction="title", 
        analyzer_en=StemmingAnalyzer(),
        analyzer_cs=czech_stemming_analyzer,
        scorer=PL2(c=10.0),
        use_prf=True
    ),
    # Run 2 Final (Unconstrained Extra Credit) - This is the config you should submit as your secondary run for evaluation if you implemented any extra credit features
    "run-2_final": WhooshExperimentConfig(
        query_construction="title", 
        analyzer_en=StemmingAnalyzer(),
        analyzer_cs=czech_stemming_analyzer,
        scorer=PL2(c=10.0),
        use_prf=True
    ),
}

# =============================================================================
# CORE EXECUTION LOGIC
# =============================================================================
def print_active_configuration(run_id: str, config: WhooshExperimentConfig, index_dir: Path):
    preprocessing_description = "Tokenize -> Lowercase -> Remove Stopwords -> Stem" # This is the same for all configs in this assignment since it is implemented in whoosh.

    print(f"\n{'='*55}")
    print(f"EXECUTING WHOOSH EXPERIMENT: {run_id}")
    print(f"{'='*55}")
    print(f"WHOOSH CONFIGURATION:\n")
    print(f"Text Processing:    {preprocessing_description}")
    print(f"Query Construction: {config.query_construction}")
    print(f"Scorer:             {config.scorer.__class__.__name__}")
    print(f"Using PRF:          {config.use_prf}")
    print(f"Top K Docs:         {config.top_k}")
    print(f"{'='*55}\n")


def execute_run(topics_path: Path, documents_path: Path, run_id: str, output_path: Path):
    
    # 1. Map the command line run_id to our configs
    base_run_id = next((k for k in CONFIGS.keys() if k in run_id), "run-1_standard")
    config = CONFIGS.get(base_run_id)
    if not config:
        raise ValueError(f"Run ID '{run_id}' does not match any known configs in run_assignment2.py")
        
    lang = "cs" if "_cs" in str(topics_path) else "en"
    
    # 2. Determine where the index lives. 
    # Because all of these configs use the exact same Stemming pipelines, 
    # they all get to share the exact same physical index!
    index_dir = DATA_DIR / f"whoosh_index_{lang}_stemmed"
    
    print_active_configuration(run_id, config, index_dir)

    # 3. Build Index ONLY if it doesn't exist
    if not index_dir.exists():
        print(f"Index not found. Building persistent index from {documents_path}...")
        analyzer = config.analyzer_cs if lang == "cs" else config.analyzer_en
        build_whoosh_index(documents_path, index_dir, analyzer, lang)
    else:
        print(f"Found existing index at {index_dir}. Skipping build phase.")

    # 4. Search and Retrieve
    print(f"Running queries from {topics_path}...")
    results = run_whoosh_search(index_dir, topics_path, config, lang)

    # 5. Write Results
    print(f"Writing {len(results)} results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trec_results(output_path, results, run_id=run_id)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPFL103 Assignment 2 - Whoosh Engine")
    parser.add_argument("-q", "--topics", type=Path, required=True, help="Topics XML file")
    parser.add_argument("-d", "--documents", type=Path, required=True, help="Documents list file")
    parser.add_argument("-r", "--run_id", type=str, required=True, help="Experiment run ID")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output results file")
    
    args = parser.parse_args()
    
    execute_run(
        topics_path=args.topics,
        documents_path=args.documents,
        run_id=args.run_id,
        output_path=args.output
    )


# EXPERIMENT EXECUTION COMMANDS
# RUN 1 (Title Only) - EN
# python run_assignment2.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-1_standard_train_en -o experiments/assignment2/en/run-1_standard_train_en.res
# python run_assignment2.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-1_prf_train_en -o experiments/assignment2/en/run-1_prf_train_en.res
# python run_assignment2.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-1_pl2_train_en -o experiments/assignment2/en/run-1_pl2_train_en.res

# RUN 1 (Title Only) - CS
# python run_assignment2.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-1_standard_train_cs -o experiments/assignment2/cs/run-1_standard_train_cs.res
# python run_assignment2.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-1_prf_train_cs -o experiments/assignment2/cs/run-1_prf_train_cs.res
# python run_assignment2.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-1_pl2_train_cs -o experiments/assignment2/cs/run-1_pl2_train_cs.res

# RUN 2 (Unconstrained Extra Credit) - EN
# python run_assignment2.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-2_experiment1_train_en -o experiments/assignment2/en/run-2_experiment1_train_en.res
# python run_assignment2.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-2_experiment2_train_en  -o experiments/assignment2/en/run-2_experiment2_train_en.res

# RUN 2 (Unconstrained Extra Credit) - CS
# python run_assignment2.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-2_experiment1_train_cs -o experiments/assignment2/cs/run-2_experiment1_train_cs.res
# python run_assignment2.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-2_experiment2_train_cs  -o experiments/assignment2/cs/run-2_experiment2_train_cs.res


# SUBMISSION EXECUTION COMMANDS
# BEST RUN 1 (Title Only) - EN
# python run_assignment2.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-1_final_train_en -o results/assignment2/run-1_train_en.res
# python run_assignment2.py -q data/topics-test_en.xml -d data/documents_en.lst -r run-1_final_test_en -o results/assignment2/run-1_test_en.res

# BEST RUN 1 (Title Only) - CS
# python run_assignment2.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-1_final_train_cs -o results/assignment2/run-1_train_cs.res
# python run_assignment2.py -q data/topics-test_cs.xml -d data/documents_cs.lst -r run-1_final_test_cs -o results/assignment2/run-1_test_cs.res

# BEST RUN 2 (Unconstrained Extra Credit) - EN
# python run_assignment2.py -q data/topics-train_en.xml -d data/documents_en.lst -r run-2_final_train_en -o results/assignment2/run-2_train_en.res
# python run_assignment2.py -q data/topics-test_en.xml -d data/documents_en.lst -r run-2_final_test_en -o results/assignment2/run-2_test_en.res

# BEST RUN 2 (Unconstrained Extra Credit) - CS
# python run_assignment2.py -q data/topics-train_cs.xml -d data/documents_cs.lst -r run-2_final_train_cs -o results/assignment2/run-2_train_cs.res
# python run_assignment2.py -q data/topics-test_cs.xml -d data/documents_cs.lst -r run-2_final_test_cs -o results/assignment2/run-2_test_cs.res