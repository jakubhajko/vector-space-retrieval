from __future__ import annotations

"""Reusable text preprocessing utilities for vector-space retrieval.

This module provides small, composable preprocessing stages:

- tokenizers: ``str -> list[str]``
- equivalence classing functions: ``list[str] -> list[str]``
- stopword removal functions: ``list[str] -> list[str]``

The defaults are designed to work well enough for IR experiments while keeping
third-party dependencies optional. Czech text keeps diacritics by default.
"""

from collections import Counter
import re
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, Sequence

TokenList = list[str]
TokenizerFn = Callable[[str], TokenList]
EquivalenceClassingFn = Callable[[TokenList], TokenList]
StopwordRemovalFn = Callable[[TokenList], TokenList]


# ---------------------------------------------------------------------------
# Built-in stopword lexicons
# ---------------------------------------------------------------------------

# Small but practical defaults, dependency-free and easy to override.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

CZECH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "aby",
        "aj",
        "ale",
        "ani",
        "ano",
        "asi",
        "bez",
        "bude",
        "budem",
        "budeme",
        "budes",
        "budete",
        "by",
        "byl",
        "byla",
        "byli",
        "bylo",
        "byt",
        "ci",
        "clanek",
        "co",
        "coz",
        "cz",
        "dalsi",
        "do",
        "ho",
        "i",
        "jak",
        "jako",
        "je",
        "jeho",
        "jej",
        "jeji",
        "jemu",
        "jen",
        "jenz",
        "jeste",
        "ji",
        "jich",
        "jim",
        "jine",
        "jiz",
        "jsem",
        "jses",
        "jsme",
        "jsou",
        "jste",
        "k",
        "kam",
        "kazdy",
        "kde",
        "kdo",
        "kdyz",
        "ke",
        "ktera",
        "ktere",
        "kterou",
        "ktery",
        "ma",
        "maji",
        "mate",
        "me",
        "mezi",
        "mi",
        "mit",
        "mne",
        "mnou",
        "muj",
        "muze",
        "na",
        "nad",
        "nam",
        "nas",
        "nasi",
        "ne",
        "nebo",
        "nebyl",
        "nebyla",
        "nebyli",
        "nebyt",
        "nechť",
        "nejsou",
        "neni",
        "nez",
        "nic",
        "nich",
        "nim",
        "nove",
        "novy",
        "o",
        "od",
        "on",
        "ona",
        "oni",
        "ono",
        "po",
        "pod",
        "podle",
        "pokud",
        "pozdě",
        "prave",
        "pred",
        "pres",
        "pri",
        "proc",
        "proto",
        "protoze",
        "prvni",
        "před",
        "při",
        "s",
        "se",
        "si",
        "sice",
        "strana",
        "svych",
        "svym",
        "ta",
        "take",
        "takze",
        "tato",
        "te",
        "ten",
        "tento",
        "teto",
        "tim",
        "timto",
        "to",
        "tohle",
        "toho",
        "tom",
        "tomto",
        "tomu",
        "toto",
        "tu",
        "tuto",
        "ty",
        "tyto",
        "u",
        "uz",
        "v",
        "vam",
        "vas",
        "ve",
        "vedle",
        "vice",
        "vsak",
        "vy",
        "z",
        "za",
        "zda",
        "zde",
        "ze",
        "zpet",
        "zpravy",
        "zprava",
        "zpráv",
    }
)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
_WORD_WITH_CONNECTORS_RE = re.compile(
    r"[0-9A-Za-zÀ-ÖØ-öø-ÿĀ-ž]+(?:[-'][0-9A-Za-zÀ-ÖØ-öø-ÿĀ-ž]+)*",
    flags=re.UNICODE,
)


def regex_word_tokenizer(text: str) -> TokenList:
    """Tokenize into word-like Unicode tokens.

    Keeps alphanumeric runs and letters with diacritics. Punctuation is dropped.
    Hyphenated and apostrophized terms are split.
    """
    if not text:
        return []
    return _WORD_RE.findall(text)



def regex_tokenizer_with_connectors(text: str) -> TokenList:
    """Tokenize while preserving internal hyphens and apostrophes.

    Useful for IR settings where terms like ``state-of-the-art`` or ``don't`` may
    reasonably be kept intact.
    """
    if not text:
        return []
    return _WORD_WITH_CONNECTORS_RE.findall(text)


# Friendly aliases for defaults.
default_en_tokenizer: TokenizerFn = regex_tokenizer_with_connectors
default_cs_tokenizer: TokenizerFn = regex_word_tokenizer


# ---------------------------------------------------------------------------
# Equivalence classing
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"^[+-]?(?:\d+[\d.,/:_-]*\d|\d)$")



def casefold_tokens(tokens: Sequence[str]) -> TokenList:
    """Lowercase tokens using Unicode-aware ``str.casefold``."""
    return [token.casefold() for token in tokens]



def normalize_numbers(tokens: Sequence[str], replacement: str = "<NUM>") -> TokenList:
    """Replace tokens that look numeric with a canonical marker."""
    normalized: TokenList = []
    for token in tokens:
        normalized.append(replacement if _NUMBER_RE.match(token) else token)
    return normalized



def casefold_and_normalize_numbers(tokens: Sequence[str]) -> TokenList:
    """Case-fold tokens and collapse numeric surface forms."""
    return normalize_numbers(casefold_tokens(tokens))


class OptionalStemmer:
    """Small wrapper that prefers NLTK if available and otherwise degrades cleanly."""

    def __init__(self, language: str) -> None:
        self.language = language.lower()
        self._stemmer = self._build_stemmer(self.language)

    @staticmethod
    def _build_stemmer(language: str):
        # Prefer NLTK Snowball stemmers where available.
        try:
            from nltk.stem import PorterStemmer, SnowballStemmer  # type: ignore

            if language == "english":
                return SnowballStemmer("english")
            try:
                return SnowballStemmer(language)
            except ValueError:
                if language == "english":
                    return PorterStemmer()
        except Exception:
            pass

        # Then try the standalone snowballstemmer package.
        try:
            import snowballstemmer  # type: ignore

            return snowballstemmer.stemmer(language)
        except Exception:
            return None

    def stem(self, token: str) -> str:
        if self._stemmer is None:
            return token

        # NLTK stemmers expose .stem(); the standalone package uses .stemWord().
        stem_method = getattr(self._stemmer, "stem", None)
        if callable(stem_method):
            return stem_method(token)

        stem_word_method = getattr(self._stemmer, "stemWord", None)
        if callable(stem_word_method):
            return stem_word_method(token)

        return token


_ENGLISH_STEMMER = OptionalStemmer("english")
_CZECH_STEMMER = OptionalStemmer("czech")



def english_casefold_and_stem(tokens: Sequence[str]) -> TokenList:
    """Case-fold, normalize numbers, then stem English tokens.

    If an English stemmer dependency is unavailable, this behaves like
    case-folding + number normalization.
    """
    normalized = casefold_and_normalize_numbers(tokens)
    return [_ENGLISH_STEMMER.stem(token) for token in normalized]



def czech_casefold_and_stem(tokens: Sequence[str]) -> TokenList:
    """Case-fold, normalize numbers, then stem Czech tokens when possible.

    If no Czech stemmer is installed, this gracefully falls back to case-folding
    + number normalization.
    """
    normalized = casefold_and_normalize_numbers(tokens)
    return [_CZECH_STEMMER.stem(token) for token in normalized]


# Friendly aliases for defaults.
default_en_equivalence_classing: EquivalenceClassingFn = english_casefold_and_stem
default_cs_equivalence_classing: EquivalenceClassingFn = casefold_and_normalize_numbers


# ---------------------------------------------------------------------------
# Stopword removal
# ---------------------------------------------------------------------------


def remove_stopwords(tokens: Sequence[str], stopwords: Iterable[str]) -> TokenList:
    """Remove tokens present in the provided stopword lexicon."""
    stopword_set = set(stopwords)
    return [token for token in tokens if token not in stopword_set]



def english_stopword_removal(
    tokens: Sequence[str],
    stopwords: Iterable[str] | None = None,
) -> TokenList:
    """Remove English stopwords using a provided or built-in lexicon."""
    lexicon = ENGLISH_STOPWORDS if stopwords is None else set(stopwords)
    return remove_stopwords(tokens, lexicon)



def czech_stopword_removal(
    tokens: Sequence[str],
    stopwords: Iterable[str] | None = None,
) -> TokenList:
    """Remove Czech stopwords using a provided or built-in lexicon."""
    lexicon = CZECH_STOPWORDS if stopwords is None else set(stopwords)
    return remove_stopwords(tokens, lexicon)



def filter_tokens_by_min_length(tokens: Sequence[str], min_length: int = 2) -> TokenList:
    """Drop very short tokens.

    This is a lightweight alternative/helper for stopword-like filtering.
    """
    if min_length <= 1:
        return list(tokens)
    return [token for token in tokens if len(token) >= min_length]



def build_frequency_stopword_set(
    documents: Sequence[Sequence[str]],
    *,
    max_document_frequency_ratio: float = 0.8,
    min_collection_frequency: int = 1,
) -> set[str]:
    """Build a frequency-based stopword set from a tokenized corpus.

    Parameters
    ----------
    documents:
        Iterable of token sequences, typically one sequence per document.
    max_document_frequency_ratio:
        Tokens occurring in more than this fraction of documents are selected.
    min_collection_frequency:
        Ignore very rare tokens when deriving the stopword set.
    """
    if not documents:
        return set()

    doc_freq: Counter[str] = Counter()
    collection_freq: Counter[str] = Counter()

    for doc_tokens in documents:
        token_list = list(doc_tokens)
        collection_freq.update(token_list)
        doc_freq.update(set(token_list))

    num_documents = len(documents)
    result: set[str] = set()
    for token, df in doc_freq.items():
        if collection_freq[token] < min_collection_frequency:
            continue
        if df / num_documents > max_document_frequency_ratio:
            result.add(token)
    return result


# Friendly aliases for defaults.
default_en_stopword_removal: StopwordRemovalFn = english_stopword_removal
default_cs_stopword_removal: StopwordRemovalFn = czech_stopword_removal


# ---------------------------------------------------------------------------
# Pipeline composition helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessingPipeline:
    """Composable preprocessing pipeline for IR text fields."""

    tokenizer_fn: TokenizerFn
    equivalence_classing_fn: EquivalenceClassingFn
    stopword_removal_fn: StopwordRemovalFn

    def __call__(self, text: str) -> TokenList:
        tokens = self.tokenizer_fn(text)
        tokens = self.equivalence_classing_fn(tokens)
        tokens = self.stopword_removal_fn(tokens)
        return tokens



def apply_preprocessing_pipeline(
    text: str,
    tokenizer_fn: TokenizerFn,
    equivalence_classing_fn: EquivalenceClassingFn,
    stopword_removal_fn: StopwordRemovalFn,
) -> TokenList:
    """Apply the standard preprocessing stages to a text field."""
    return PreprocessingPipeline(
        tokenizer_fn=tokenizer_fn,
        equivalence_classing_fn=equivalence_classing_fn,
        stopword_removal_fn=stopword_removal_fn,
    )(text)



def build_default_english_pipeline(
    *,
    preserve_hyphens_and_apostrophes: bool = True,
    stem: bool = True,
    stopwords: Iterable[str] | None = None,
) -> PreprocessingPipeline:
    """Create a practical default preprocessing pipeline for English news text."""
    tokenizer = (
        regex_tokenizer_with_connectors
        if preserve_hyphens_and_apostrophes
        else regex_word_tokenizer
    )
    equivalence = english_casefold_and_stem if stem else casefold_and_normalize_numbers
    stopword_fn = partial(english_stopword_removal, stopwords=stopwords)
    return PreprocessingPipeline(tokenizer, equivalence, stopword_fn)



def build_default_czech_pipeline(
    *,
    stem: bool = False,
    stopwords: Iterable[str] | None = None,
) -> PreprocessingPipeline:
    """Create a practical default preprocessing pipeline for Czech news text.

    By default this keeps processing conservative: case-folding, number
    normalization, and lexicon-based stopword removal. Czech stemming is made
    available when an optional dependency supports it.
    """
    equivalence = czech_casefold_and_stem if stem else casefold_and_normalize_numbers
    stopword_fn = partial(czech_stopword_removal, stopwords=stopwords)
    return PreprocessingPipeline(regex_word_tokenizer, equivalence, stopword_fn)


__all__ = [
    "CZECH_STOPWORDS",
    "ENGLISH_STOPWORDS",
    "EquivalenceClassingFn",
    "OptionalStemmer",
    "PreprocessingPipeline",
    "StopwordRemovalFn",
    "TokenList",
    "TokenizerFn",
    "apply_preprocessing_pipeline",
    "build_default_czech_pipeline",
    "build_default_english_pipeline",
    "build_frequency_stopword_set",
    "casefold_and_normalize_numbers",
    "casefold_tokens",
    "czech_casefold_and_stem",
    "czech_stopword_removal",
    "default_cs_equivalence_classing",
    "default_cs_stopword_removal",
    "default_cs_tokenizer",
    "default_en_equivalence_classing",
    "default_en_stopword_removal",
    "default_en_tokenizer",
    "english_casefold_and_stem",
    "english_stopword_removal",
    "filter_tokens_by_min_length",
    "normalize_numbers",
    "regex_tokenizer_with_connectors",
    "regex_word_tokenizer",
    "remove_stopwords",
]
