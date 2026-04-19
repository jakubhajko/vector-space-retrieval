from __future__ import annotations

"""Topic loading and preprocessing for retrieval experiments.

This module mirrors the philosophy of ``load_documents.py``:
- parse raw XML topic files
- extract structured raw topic fields
- construct one merged query text from selected fields
- apply the reusable preprocessing pipeline from ``text_preprocessing.py``
- return processed topics keyed by topic ID

Expected topic XML structure
----------------------------
<topics>
    <top lang="[en/cs]">
        <num>...</num>
        <title>...</title>
        <desc>...</desc>
        <narr>...</narr>
    </top>
</topics>

Design notes
------------
- Missing fields are tolerated and normalized to empty strings.
- Selected query fields are merged *before* preprocessing.
- The final processed topic stores exactly one token list for retrieval.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable, Literal
import xml.etree.ElementTree as ET

from .text_preprocessing import (
    EquivalenceClassingFn,
    StopwordRemovalFn,
    TokenizerFn,
    apply_preprocessing_pipeline,
    default_cs_equivalence_classing,
    default_cs_stopword_removal,
    default_cs_tokenizer,
    default_en_equivalence_classing,
    default_en_stopword_removal,
    default_en_tokenizer,
)

logger = logging.getLogger(__name__)

QueryConstructionMode = Literal["title", "title_desc", "title_desc_narr"]


@dataclass(frozen=True)
class ExtractedTopic:
    """Raw topic fields extracted from XML."""

    topic_id: str
    raw_title: str
    raw_desc: str
    raw_narr: str
    source_file: str
    language: str


@dataclass(frozen=True)
class ProcessedTopic:
    """Processed topic representation ready for retrieval."""

    topic_id: str
    query_tokens: list[str]
    raw_title: str
    raw_desc: str
    raw_narr: str
    merged_query_text: str
    source_file: str
    language: str


def _normalize_whitespace(text: str | None) -> str:
    """Collapse repeated whitespace and handle missing values safely."""
    if not text:
        return ""
    return " ".join(text.split())


def _collect_text_content(element: ET.Element | None) -> str:
    """Collect textual content of an XML element including descendants."""
    if element is None:
        return ""
    return _normalize_whitespace("".join(element.itertext()))


def _join_text_parts(parts: Iterable[str]) -> str:
    """Join non-empty text fragments with spaces."""
    return " ".join(part for part in (_normalize_whitespace(p) for p in parts) if part)


def _resolve_topic_language(top_element: ET.Element, default_language: str | None) -> str:
    """Resolve topic language from XML attribute or fallback."""
    lang_attr = top_element.attrib.get("lang")
    if lang_attr:
        return lang_attr.strip().lower()
    return (default_language or "").strip().lower()


def _resolve_topic_id(top_element: ET.Element) -> str | None:
    """Resolve topic ID from the <num> element."""
    topic_id = _collect_text_content(top_element.find("num"))
    return topic_id or None


def parse_topics_file(
    xml_path: str | Path,
    *,
    default_language: str | None = None,
) -> list[ExtractedTopic]:
    """Parse a topic XML file and return extracted topics.

    Parameters
    ----------
    xml_path:
        Path to a topic XML file.
    default_language:
        Optional fallback language if a ``<top>`` element lacks the ``lang`` attribute.

    Returns
    -------
    list[ExtractedTopic]
        Extracted topics in XML order. Malformed topics are skipped individually.
    """
    path = Path(xml_path)
    extracted_topics: list[ExtractedTopic] = []

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except (ET.ParseError, OSError) as exc:
        logger.warning("Failed to parse topic XML file %s: %s", path, exc)
        return extracted_topics

    for top_element in root.findall(".//top"):
        try:
            topic_id = _resolve_topic_id(top_element)
            if not topic_id:
                logger.warning("Skipping topic without <num> in %s", path)
                continue

            language = _resolve_topic_language(top_element, default_language)
            raw_title = _collect_text_content(top_element.find("title"))
            raw_desc = _collect_text_content(top_element.find("desc"))
            raw_narr = _collect_text_content(top_element.find("narr"))

            extracted_topics.append(
                ExtractedTopic(
                    topic_id=topic_id,
                    raw_title=raw_title,
                    raw_desc=raw_desc,
                    raw_narr=raw_narr,
                    source_file=path.name,
                    language=language,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive per-topic guard
            logger.warning(
                "Skipping malformed topic in %s due to unexpected error: %s",
                path,
                exc,
            )

    return extracted_topics


def build_topic_query_text(
    topic: ExtractedTopic,
    *,
    query_construction: QueryConstructionMode = "title",
) -> str:
    """Construct one merged query string from the selected raw topic fields."""
    if query_construction == "title":
        parts = [topic.raw_title]
    elif query_construction == "title_desc":
        parts = [topic.raw_title, topic.raw_desc]
    elif query_construction == "title_desc_narr":
        parts = [topic.raw_title, topic.raw_desc, topic.raw_narr]
    else:  # pragma: no cover - defensive branch for invalid caller input
        raise ValueError(f"Unsupported query construction mode: {query_construction}")

    return _join_text_parts(parts)


def preprocess_extracted_topic(
    topic: ExtractedTopic,
    *,
    query_construction: QueryConstructionMode,
    tokenizer_fn: TokenizerFn,
    equivalence_classing_fn: EquivalenceClassingFn,
    stopword_removal_fn: StopwordRemovalFn,
) -> ProcessedTopic:
    """Construct and preprocess a topic into a single tokenized query representation."""
    merged_query_text = build_topic_query_text(topic, query_construction=query_construction)
    query_tokens = apply_preprocessing_pipeline(
        merged_query_text,
        tokenizer_fn=tokenizer_fn,
        equivalence_classing_fn=equivalence_classing_fn,
        stopword_removal_fn=stopword_removal_fn,
    )

    return ProcessedTopic(
        topic_id=topic.topic_id,
        query_tokens=query_tokens,
        raw_title=topic.raw_title,
        raw_desc=topic.raw_desc,
        raw_narr=topic.raw_narr,
        merged_query_text=merged_query_text,
        source_file=topic.source_file,
        language=topic.language,
    )


def _merge_topics_with_duplicate_handling(
    topics: Iterable[ProcessedTopic],
) -> dict[str, ProcessedTopic]:
    """Collect topics into a dictionary, overwriting duplicates deterministically."""
    merged: dict[str, ProcessedTopic] = {}
    for topic in topics:
        if topic.topic_id in merged:
            logger.warning(
                "Duplicate topic ID %s encountered; overwriting previous entry from %s",
                topic.topic_id,
                topic.source_file,
            )
        merged[topic.topic_id] = topic
    return merged


def preprocess_topics(
    xml_path: str | Path,
    *,
    query_construction: QueryConstructionMode,
    tokenizer_fn: TokenizerFn,
    equivalence_classing_fn: EquivalenceClassingFn,
    stopword_removal_fn: StopwordRemovalFn,
    default_language: str | None = None,
) -> dict[str, ProcessedTopic]:
    """Parse and preprocess topics from one XML file."""
    extracted_topics = parse_topics_file(xml_path, default_language=default_language)
    processed_topics = (
        preprocess_extracted_topic(
            topic,
            query_construction=query_construction,
            tokenizer_fn=tokenizer_fn,
            equivalence_classing_fn=equivalence_classing_fn,
            stopword_removal_fn=stopword_removal_fn,
        )
        for topic in extracted_topics
    )
    return _merge_topics_with_duplicate_handling(processed_topics)


def preprocess_en_topics(
    xml_path: str | Path,
    *,
    query_construction: QueryConstructionMode = "title",
    tokenizer_fn: TokenizerFn = default_en_tokenizer,
    equivalence_classing_fn: EquivalenceClassingFn = default_en_equivalence_classing,
    stopword_removal_fn: StopwordRemovalFn = default_en_stopword_removal,
) -> dict[str, ProcessedTopic]:
    """Parse and preprocess English topics."""
    return preprocess_topics(
        xml_path,
        query_construction=query_construction,
        tokenizer_fn=tokenizer_fn,
        equivalence_classing_fn=equivalence_classing_fn,
        stopword_removal_fn=stopword_removal_fn,
        default_language="en",
    )


def preprocess_cs_topics(
    xml_path: str | Path,
    *,
    query_construction: QueryConstructionMode = "title",
    tokenizer_fn: TokenizerFn = default_cs_tokenizer,
    equivalence_classing_fn: EquivalenceClassingFn = default_cs_equivalence_classing,
    stopword_removal_fn: StopwordRemovalFn = default_cs_stopword_removal,
) -> dict[str, ProcessedTopic]:
    """Parse and preprocess Czech topics."""
    return preprocess_topics(
        xml_path,
        query_construction=query_construction,
        tokenizer_fn=tokenizer_fn,
        equivalence_classing_fn=equivalence_classing_fn,
        stopword_removal_fn=stopword_removal_fn,
        default_language="cs",
    )


__all__ = [
    "ExtractedTopic",
    "ProcessedTopic",
    "QueryConstructionMode",
    "build_topic_query_text",
    "parse_topics_file",
    "preprocess_cs_topics",
    "preprocess_en_topics",
    "preprocess_extracted_topic",
    "preprocess_topics",
]