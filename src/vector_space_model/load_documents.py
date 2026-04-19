from __future__ import annotations

"""Document loading and field-level preprocessing for vector-space retrieval.

This module parses the Czech and English XML corpora, extracts document title and
body text, preprocesses them, and returns a mapping from document ID to a clean
structured representation.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence
import xml.etree.ElementTree as ET

from .config import DATA_DIR
from .text_preprocessing import (
    EquivalenceClassingFn,
    StopwordRemovalFn,
    TokenizerFn,
    default_cs_equivalence_classing,
    default_cs_stopword_removal,
    default_cs_tokenizer,
    default_en_equivalence_classing,
    default_en_stopword_removal,
    default_en_tokenizer,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedDocument:
    """Raw text fields extracted from an XML document."""

    doc_id: str
    raw_title: str
    raw_text: str
    source_file: str
    language: str


@dataclass(frozen=True)
class ProcessedDocument:
    """Document representation ready for indexing / weighting stages."""

    doc_id: str
    title_tokens: list[str]
    text_tokens: list[str]
    source_file: str
    raw_title: str
    raw_text: str
    language: str


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


def _normalize_whitespace(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())



def _collect_text_content(element: ET.Element | None) -> str:
    """Collect all textual content under an XML element, including descendants."""
    if element is None:
        return ""
    return _normalize_whitespace("".join(element.itertext()))



def _join_text_parts(parts: Iterable[str]) -> str:
    """Join field fragments with spaces while avoiding accidental word merges."""
    return " ".join(part for part in (_normalize_whitespace(p) for p in parts) if part)



def _apply_optional_preprocessing_pipeline(
    text: str,
    *,
    tokenizer_fn: TokenizerFn,
    equivalence_classing_fn: EquivalenceClassingFn | None,
    stopword_removal_fn: StopwordRemovalFn | None,
) -> list[str]:
    """Apply preprocessing while allowing optional later pipeline stages.

    The tokenizer is always required. Equivalence classing and stopword removal
    are only applied when the corresponding callable is provided.
    """
    tokens = tokenizer_fn(text)

    if equivalence_classing_fn is not None:
        tokens = equivalence_classing_fn(tokens)

    if stopword_removal_fn is not None:
        tokens = stopword_removal_fn(tokens)

    return tokens


# ---------------------------------------------------------------------------
# File list loading
# ---------------------------------------------------------------------------


def _read_lst_file(lst_path: str | Path) -> list[str]:
    """Read XML file names from a .lst file, ignoring blanks and comments."""
    path = Path(lst_path)
    filenames: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            filenames.append(entry)
    return filenames


# ---------------------------------------------------------------------------
# XML extraction helpers
# ---------------------------------------------------------------------------


def _resolve_document_id(doc_element: ET.Element) -> str | None:
    """Use DOCID as primary key and DOCNO as fallback."""
    docid = _collect_text_content(doc_element.find("DOCID"))
    if docid:
        return docid
    docno = _collect_text_content(doc_element.find("DOCNO"))
    return docno or None



def extract_cs_documents_from_file(xml_path: str | Path) -> list[ExtractedDocument]:
    """Parse a Czech XML file and extract raw title/body fields.

    Extraction rules
    ----------------
    - title: TITLE if present
    - body: concatenation of HEADING and TEXT fields in document order
    - primary identifier: DOCID, falling back to DOCNO
    """
    path = Path(xml_path)
    extracted: list[ExtractedDocument] = []

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except (ET.ParseError, OSError) as exc:
        logger.warning("Failed to parse Czech XML file %s: %s", path, exc)
        return extracted

    for doc_element in root.findall(".//DOC"):
        try:
            doc_id = _resolve_document_id(doc_element)
            if not doc_id:
                logger.warning("Skipping Czech DOC without DOCID/DOCNO in %s", path)
                continue

            raw_title = _collect_text_content(doc_element.find("TITLE"))

            body_parts: list[str] = []
            for child in doc_element:
                tag = child.tag.upper() if isinstance(child.tag, str) else ""
                if tag in {"HEADING", "TEXT"}:
                    content = _collect_text_content(child)
                    if content:
                        body_parts.append(content)

            extracted.append(
                ExtractedDocument(
                    doc_id=doc_id,
                    raw_title=raw_title,
                    raw_text=_join_text_parts(body_parts),
                    source_file=path.name,
                    language="cs",
                )
            )
        except Exception as exc:  # pragma: no cover - defensive per-document guard
            logger.warning(
                "Skipping malformed Czech DOC in %s due to unexpected error: %s",
                path,
                exc,
            )

    return extracted



def extract_en_documents_from_file(xml_path: str | Path) -> list[ExtractedDocument]:
    """Parse an English XML file and extract raw title/body fields.

    Extraction rules
    ----------------
    - title: HD if present
    - body: concatenation of LD and TE fields in document order
    - primary identifier: DOCID, falling back to DOCNO
    """
    path = Path(xml_path)
    extracted: list[ExtractedDocument] = []

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except (ET.ParseError, OSError) as exc:
        logger.warning("Failed to parse English XML file %s: %s", path, exc)
        return extracted

    for doc_element in root.findall(".//DOC"):
        try:
            doc_id = _resolve_document_id(doc_element)
            if not doc_id:
                logger.warning("Skipping English DOC without DOCID/DOCNO in %s", path)
                continue

            raw_title = _collect_text_content(doc_element.find("HD"))

            body_parts: list[str] = []
            for child in doc_element:
                tag = child.tag.upper() if isinstance(child.tag, str) else ""
                if tag in {"LD", "TE"}:
                    content = _collect_text_content(child)
                    if content:
                        body_parts.append(content)

            extracted.append(
                ExtractedDocument(
                    doc_id=doc_id,
                    raw_title=raw_title,
                    raw_text=_join_text_parts(body_parts),
                    source_file=path.name,
                    language="en",
                )
            )
        except Exception as exc:  # pragma: no cover - defensive per-document guard
            logger.warning(
                "Skipping malformed English DOC in %s due to unexpected error: %s",
                path,
                exc,
            )

    return extracted


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def preprocess_extracted_document(
    document: ExtractedDocument,
    *,
    tokenizer_fn: TokenizerFn,
    equivalence_classing_fn: EquivalenceClassingFn | None,
    stopword_removal_fn: StopwordRemovalFn | None,
) -> ProcessedDocument:
    """Apply the configured preprocessing pipeline to title and body separately."""
    title_tokens = _apply_optional_preprocessing_pipeline(
        document.raw_title,
        tokenizer_fn=tokenizer_fn,
        equivalence_classing_fn=equivalence_classing_fn,
        stopword_removal_fn=stopword_removal_fn,
    )
    text_tokens = _apply_optional_preprocessing_pipeline(
        document.raw_text,
        tokenizer_fn=tokenizer_fn,
        equivalence_classing_fn=equivalence_classing_fn,
        stopword_removal_fn=stopword_removal_fn,
    )
    return ProcessedDocument(
        doc_id=document.doc_id,
        title_tokens=title_tokens,
        text_tokens=text_tokens,
        source_file=document.source_file,
        raw_title=document.raw_title,
        raw_text=document.raw_text,
        language=document.language,
    )



def _merge_documents_with_duplicate_handling(
    documents: Iterable[ProcessedDocument],
) -> dict[str, ProcessedDocument]:
    """Collect documents into a dictionary, overwriting duplicates deterministically.

    Behavior: later documents overwrite earlier ones and trigger a warning.
    This is deterministic because file order follows the .lst file order.
    """
    merged: dict[str, ProcessedDocument] = {}
    for document in documents:
        if document.doc_id in merged:
            logger.warning(
                "Duplicate document ID %s encountered; overwriting previous entry with source file %s",
                document.doc_id,
                document.source_file,
            )
        merged[document.doc_id] = document
    return merged


# ---------------------------------------------------------------------------
# Corpus processing internals
# ---------------------------------------------------------------------------


def _parse_files(
    filenames: Sequence[str],
    documents_dir: Path,
    parser_fn: Callable[[str | Path], list[ExtractedDocument]],
    *,
    num_workers: int = 1,
) -> Iterator[ExtractedDocument]:
    """Parse XML files sequentially or with optional thread-based parallelism."""
    paths = [documents_dir / filename for filename in filenames]

    if num_workers <= 1:
        for path in paths:
            if not path.exists():
                logger.warning("Listed XML file does not exist: %s", path)
                continue
            for document in parser_fn(path):
                yield document
        return

    def parse_one(path: Path) -> list[ExtractedDocument]:
        if not path.exists():
            logger.warning("Listed XML file does not exist: %s", path)
            return []
        return parser_fn(path)

    # Threads keep the implementation simple and avoid pickling user-provided
    # preprocessing callables. XML parsing is largely I/O-bound at file level.
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for parsed_docs in executor.map(parse_one, paths):
            for document in parsed_docs:
                yield document



def _preprocess_documents(
    extracted_documents: Iterable[ExtractedDocument],
    *,
    tokenizer_fn: TokenizerFn,
    equivalence_classing_fn: EquivalenceClassingFn | None,
    stopword_removal_fn: StopwordRemovalFn | None,
) -> dict[str, ProcessedDocument]:
    processed_documents = (
        preprocess_extracted_document(
            document,
            tokenizer_fn=tokenizer_fn,
            equivalence_classing_fn=equivalence_classing_fn,
            stopword_removal_fn=stopword_removal_fn,
        )
        for document in extracted_documents
    )
    return _merge_documents_with_duplicate_handling(processed_documents)


# ---------------------------------------------------------------------------
# Public corpus APIs
# ---------------------------------------------------------------------------


def preprocess_cs_documents(
    lst_path: str | Path,
    documents_dir: str | Path = DATA_DIR / "documents_cs",
    tokenizer_fn: TokenizerFn = default_cs_tokenizer,
    equivalence_classing_fn: EquivalenceClassingFn | None = default_cs_equivalence_classing,
    stopword_removal_fn: StopwordRemovalFn | None = default_cs_stopword_removal,
    num_workers: int = 1,
) -> dict[str, ProcessedDocument]:
    """Load, parse, and preprocess Czech XML documents.

    Parameters
    ----------
    lst_path:
        Path to the ``documents_cs.lst``-style file listing XML filenames.
    documents_dir:
        Directory containing the Czech XML files.
    tokenizer_fn, equivalence_classing_fn, stopword_removal_fn:
        Modular preprocessing stage callables. The tokenizer is required;
        equivalence classing and stopword removal may be set to ``None`` to skip
        those stages.
    num_workers:
        Optional file-level parallelism. ``1`` keeps processing sequential.
    """
    filenames = _read_lst_file(lst_path)
    extracted_documents = _parse_files(
        filenames,
        Path(documents_dir),
        extract_cs_documents_from_file,
        num_workers=num_workers,
    )
    return _preprocess_documents(
        extracted_documents,
        tokenizer_fn=tokenizer_fn,
        equivalence_classing_fn=equivalence_classing_fn,
        stopword_removal_fn=stopword_removal_fn,
    )



def preprocess_en_documents(
    lst_path: str | Path,
    documents_dir: str | Path = DATA_DIR / "documents_en",
    tokenizer_fn: TokenizerFn = default_en_tokenizer,
    equivalence_classing_fn: EquivalenceClassingFn | None = default_en_equivalence_classing,
    stopword_removal_fn: StopwordRemovalFn | None = default_en_stopword_removal,
    num_workers: int = 1,
) -> dict[str, ProcessedDocument]:
    """Load, parse, and preprocess English XML documents.

    Parameters
    ----------
    lst_path:
        Path to the ``documents_en.lst``-style file listing XML filenames.
    documents_dir:
        Directory containing the English XML files.
    tokenizer_fn, equivalence_classing_fn, stopword_removal_fn:
        Modular preprocessing stage callables. The tokenizer is required;
        equivalence classing and stopword removal may be set to ``None`` to skip
        those stages.
    num_workers:
        Optional file-level parallelism. ``1`` keeps processing sequential.
    """
    filenames = _read_lst_file(lst_path)
    extracted_documents = _parse_files(
        filenames,
        Path(documents_dir),
        extract_en_documents_from_file,
        num_workers=num_workers,
    )
    return _preprocess_documents(
        extracted_documents,
        tokenizer_fn=tokenizer_fn,
        equivalence_classing_fn=equivalence_classing_fn,
        stopword_removal_fn=stopword_removal_fn,
    )


__all__ = [
    "ExtractedDocument",
    "ProcessedDocument",
    "extract_cs_documents_from_file",
    "extract_en_documents_from_file",
    "preprocess_cs_documents",
    "preprocess_en_documents",
    "preprocess_extracted_document",
]