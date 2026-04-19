"""Vector Space Model for Information Retrieval."""

from .config import DATA_DIR, PROJECT_ROOT
from .index import InvertedIndex, build_inverted_index
from .load_documents import preprocess_cs_documents, preprocess_en_documents
from .load_topics import preprocess_cs_topics, preprocess_en_topics
from .results import write_trec_results, flatten_results_by_topic
from .retrieval import RetrievalResult, RetrievalSystem
from .scoring import BM25Parameters, PivotedNormalizationParameters, ScoringConfig
from .text_preprocessing import PreprocessingPipeline, regex_word_tokenizer

__all__ = [
    "BM25Parameters",
    "DATA_DIR",
    "InvertedIndex",
    "PivotedNormalizationParameters",
    "PROJECT_ROOT",
    "PreprocessingPipeline",
    "RetrievalResult",
    "RetrievalSystem",
    "ScoringConfig",
    "build_inverted_index",
    "preprocess_cs_documents",
    "preprocess_cs_topics",
    "preprocess_en_documents",
    "preprocess_en_topics",
    "write_trec_results",
]