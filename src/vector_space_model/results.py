from __future__ import annotations

"""Writing retrieval runs in TREC tab-separated format.

Expected columns
----------------
1. qid
2. iter
3. docno
4. rank
5. sim
6. run_id

Constraints
-----------
- at most 1000 results per topic are written by default
- rank starts at 0
- iter defaults to "0"
"""

from pathlib import Path
from typing import Iterable

from .retrieval import RetrievalResult


def format_trec_result_line(
    result: RetrievalResult,
    *,
    run_id: str,
    iteration: str = "0",
) -> str:
    """Format one retrieval result line in TREC tab-separated format."""
    return (
        f"{result.topic_id}\t"
        f"{iteration}\t"
        f"{result.doc_id}\t"
        f"{result.rank}\t"
        f"{float(result.score)}\t"
        f"{run_id}"
    )


def write_trec_results(
    output_path: str | Path,
    results: Iterable[RetrievalResult],
    *,
    run_id: str,
    iteration: str = "0",
    max_results_per_topic: int = 1000,
) -> None:
    """Write retrieval results to disk in TREC output format.

    Parameters
    ----------
    output_path:
        Target file path.
    results:
        Iterable of retrieval results. These are expected to already be ranked.
    run_id:
        Run identifier written in the last column.
    iteration:
        TREC ``iter`` field. A constant like ``"0"`` is typical.
    max_results_per_topic:
        Maximum number of results written per topic.
    """
    path = Path(output_path)
    counts_by_topic: dict[str, int] = {}

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            seen = counts_by_topic.get(result.topic_id, 0)
            if seen >= max_results_per_topic:
                continue

            handle.write(
                format_trec_result_line(
                    result,
                    run_id=run_id,
                    iteration=iteration,
                )
            )
            handle.write("\n")
            counts_by_topic[result.topic_id] = seen + 1


def flatten_results_by_topic(
    results_by_topic: dict[str, list[RetrievalResult]],
) -> list[RetrievalResult]:
    """Flatten a topic->results mapping into a stable list.

    Topics are emitted in ascending topic ID order.
    """
    flattened: list[RetrievalResult] = []
    for topic_id in sorted(results_by_topic):
        flattened.extend(results_by_topic[topic_id])
    return flattened


__all__ = [
    "flatten_results_by_topic",
    "format_trec_result_line",
    "write_trec_results",
]