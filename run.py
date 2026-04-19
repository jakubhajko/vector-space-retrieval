from __future__ import annotations

# from load_documents import preprocess_en_documents
# from load_topics import preprocess_en_topics
# from index import build_inverted_index
# from retrieval import RetrievalSystem
# from results import flatten_results_by_topic, write_trec_results
# from scoring import ScoringConfig
from vector_space_model import (
    preprocess_en_documents,
    preprocess_en_topics,
    build_inverted_index,
    RetrievalSystem,
    write_trec_results,
    ScoringConfig,
    DATA_DIR,
    flatten_results_by_topic
)

# 1) Load and preprocess documents
documents = preprocess_en_documents(
    lst_path=f"{DATA_DIR}/documents_en.lst",
    documents_dir=f"{DATA_DIR}/documents_en",
)

# 2) Build index
index = build_inverted_index(documents.values())

# 3) Load and preprocess topics
topics = preprocess_en_topics(
    f"{DATA_DIR}/topics-test_en.xml",
    query_construction="title_desc",
)

# 4) Configure retrieval
scoring = ScoringConfig(
    tf_weighting="logarithm",
    df_weighting="idf",
    normalization="cosine",
    similarity="cosine",
    query_expansion="none",
)

# Optional thesaurus for future experiments
thesaurus = {
    # keys and values should already be in preprocessed token space
    # e.g. "car": ["automobil", "vehicle"]
}

# 5) Retrieve
retrieval_system = RetrievalSystem(index)
results_by_topic = retrieval_system.retrieve_many(
    {topic_id: topic.query_tokens for topic_id, topic in topics.items()},
    scoring=scoring,
    thesaurus=thesaurus,
    top_k=1000,
)

# 6) Write TREC run
write_trec_results(
    "run-0.txt",
    flatten_results_by_topic(results_by_topic),
    run_id="run-0",
)