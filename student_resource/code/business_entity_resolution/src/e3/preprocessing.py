"""
Shared Preprocessing and Feature Extraction Utilities for E3 Blocking.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
Blocking (L2) requires aggressive, standardized tokenization, canonicalization,
deterministic transliteration, and structural component extraction to maximize true-match
recall while keeping candidate generation computationally tractable. Raw values and
conservative normalizations are preserved for downstream feature engineering (L3) and LightGBM (L4).
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

# Universal ISCII / Brahmic offset mapping to Latin ASCII
BRAHMIC_OFFSET_MAP: Dict[int, str] = {
    0x02: "m", 0x03: "h", 0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee",
    0x09: "u", 0x0a: "oo", 0x0b: "r", 0x0f: "e", 0x10: "ai", 0x13: "o", 0x14: "au",
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1a: "ch", 0x1b: "chh", 0x1c: "j", 0x1d: "jh", 0x1e: "ny",
    0x1f: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
    0x2a: "p", 0x2b: "ph", 0x2c: "b", 0x2d: "bh", 0x2e: "m",
    0x2f: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x3e: "aa", 0x3f: "i", 0x40: "ee", 0x41: "u", 0x42: "oo", 0x43: "r",
    0x47: "e", 0x48: "ai", 0x4b: "o", 0x4c: "au",
    0x4d: "",  # virama / halant (silent vowel suppressor)
}

# Legal suffixes to strip during blocking signature generation
LEGAL_SUFFIXES: Set[str] = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "co", "company", "pvt", "private", "llp", "pllc", "pc", "lp", "gmbh",
    "sarl", "sa", "eurl", "sas", "ag", "bv", "holding", "holdings",
}

# Generic business noise words (stripped in aggressive mode)
GENERIC_BUSINESS_WORDS: Set[str] = {
    "services", "enterprises", "industries", "solutions", "group", "holdings",
    "ventures", "associates", "international", "national", "global", "trading",
    "management", "consulting", "consultants", "technologies", "systems",
    "center", "centre", "store", "shop", "market", "marketing", "club",
}

# DBA / trade name patterns
RE_DBA_MARKERS = re.compile(
    r"\b(?:d/?b/?a|f/?k/?a|t/?a|trading\s+as|formerly\s+known\s+as|c/o|aka)\b",
    re.IGNORECASE,
)
RE_BRACKETS = re.compile(r"\[(.*?)\]|\((.*?)\)")
RE_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|org|net|in|co\.in|edu|gov|io|ai)\b", re.IGNORECASE)
RE_NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
RE_SPACES = re.compile(r"\s+")
RE_LEADING_ZEROS = re.compile(r"^0+(\d+)")

# Placeholder tokens discovered in E0
PLACEHOLDER_TOKENS: Set[str] = {
    "<null>", "null", "<na>", "na", "unknown", "none", "not applicable",
    "<blank>", "blank", "undefined",
}


def clean_text(text: str) -> str:
    """
    Lowercase, remove accents/diacritics (NFKD normalization),
    and replace punctuation with single spaces.
    """
    if not text:
        return ""
    # Normalize unicode (decompose accents like é -> e + accent)
    text_nfkd = unicodedata.normalize("NFKD", text)
    # Strip combining diacritical marks (e.g. accents)
    text_ascii = "".join(c for c in text_nfkd if not unicodedata.combining(c))
    cleaned = RE_NON_ALPHANUM.sub(" ", text_ascii.lower())
    return RE_SPACES.sub(" ", cleaned).strip()


def transliterate_indic(text: str) -> str:
    """
    Deterministic Brahmic-to-Latin transliteration across all 9 Indic scripts:
    Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam.
    Properly handles inherent schwa (/a/) between consonants and normalizes vowels.
    """
    if not text:
        return ""
    res: List[str] = []
    prev_was_consonant = False

    for ch in text:
        cp = ord(ch)
        matched = False
        # Indic scripts reside in contiguous 0x80 blocks between 0x0900 and 0x0D80
        for base in range(0x0900, 0x0D80, 0x80):
            if base <= cp < base + 0x80:
                offset = cp - base
                # Check virama / halant (0x4D): suppresses inherent vowel
                if offset == 0x4D:
                    prev_was_consonant = False
                    matched = True
                    break

                # Check if it's a vowel sign (matra 0x3E to 0x4C)
                if 0x3E <= offset <= 0x4C:
                    vowel = BRAHMIC_OFFSET_MAP.get(offset, "")
                    res.append(vowel)
                    prev_was_consonant = False
                    matched = True
                    break

                # Check consonant (0x15 to 0x39)
                if 0x15 <= offset <= 0x39:
                    if prev_was_consonant:
                        res.append("a")
                    consonant = BRAHMIC_OFFSET_MAP.get(offset, "")
                    res.append(consonant)
                    prev_was_consonant = True
                    matched = True
                    break

                # Independent vowel or other character
                if offset in BRAHMIC_OFFSET_MAP:
                    res.append(BRAHMIC_OFFSET_MAP[offset])
                    prev_was_consonant = False
                    matched = True
                    break

        if not matched:
            res.append(ch)
            prev_was_consonant = False

    raw_latin = clean_text("".join(res))
    simplified = (
        raw_latin.replace("aa", "a")
        .replace("ee", "i")
        .replace("oo", "u")
    )
    return simplified


def detect_script(text: str) -> str:
    """Detect dominant non-Latin script in text, or return 'LATIN'."""
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return "DEVANAGARI"
        elif 0x0980 <= cp <= 0x09FF:
            return "BENGALI"
        elif 0x0A00 <= cp <= 0x0A7F:
            return "GURMUKHI"
        elif 0x0A80 <= cp <= 0x0AFF:
            return "GUJARATI"
        elif 0x0B00 <= cp <= 0x0B7F:
            return "ORIYA"
        elif 0x0B80 <= cp <= 0x0BFF:
            return "TAMIL"
        elif 0x0C00 <= cp <= 0x0C7F:
            return "TELUGU"
        elif 0x0C80 <= cp <= 0x0CFF:
            return "KANNADA"
        elif 0x0D00 <= cp <= 0x0D7F:
            return "MALAYALAM"
    return "LATIN"


def extract_dba_components(name: str) -> Tuple[str, Optional[str]]:
    """
    Split name into (primary_name, optional_dba_alias).
    Handles brackets [...], parentheses (...), DBA markers (d/b/a, f/k/a), and domains.
    """
    if not name:
        return "", None

    # Check domain name format (e.g. maurewilliamscolombier.com)
    dom_match = RE_DOMAIN.match(name.strip())
    if dom_match:
        domain_stem = dom_match.group(1).replace("-", " ")
        return name, domain_stem

    # Check brackets/parentheses e.g. "Vertical International [School]"
    bracket_match = RE_BRACKETS.search(name)
    if bracket_match:
        content = bracket_match.group(1) or bracket_match.group(2)
        base = RE_BRACKETS.sub(" ", name)
        return clean_text(base), clean_text(content) if content else None

    # Check DBA / FKA markers e.g. "Lumvio F/K/A Suryava Al LLP"
    dba_parts = RE_DBA_MARKERS.split(name, maxsplit=1)
    if len(dba_parts) == 2:
        return clean_text(dba_parts[0]), clean_text(dba_parts[1])

    return clean_text(name), None


def canonicalize_number(num_str: str) -> str:
    """Canonicalize numeric string by stripping leading zeros (e.g. '0684' -> '684')."""
    s = num_str.lstrip("0")
    return s if s else "0"


def extract_address_features(address: str, country: str = "US") -> Dict[str, Any]:
    """
    Extract structural numeric and locality fragments from address:
    - building_numbers: list of numeric fragments (leading zeros stripped)
    - postal_code: 5-digit US ZIP or 6-digit India PIN
    - locality_tokens: normalized locality / city / state tokens
    """
    if not address:
        return {"building_numbers": [], "postal_code": None, "locality_tokens": []}

    cleaned = clean_text(address)
    # Strip placeholder tokens (e.g., 'null', '<null>')
    tokens = [t for t in cleaned.split() if t not in PLACEHOLDER_TOKENS]

    # Find numbers
    raw_numbers = re.findall(r"\b\d+\b", address)
    building_numbers: List[str] = []
    postal_code: Optional[str] = None

    for n in raw_numbers:
        # Check postal code format
        if country == "US" and len(n) == 5 and postal_code is None:
            postal_code = n
        elif country == "India" and len(n) == 6 and postal_code is None:
            postal_code = n
        else:
            canon = canonicalize_number(n)
            if len(canon) >= 1 and canon not in building_numbers:
                building_numbers.append(canon)

    # Locality tokens from address components (typically last 2 components in comma-separated address)
    parts = address.split(",")
    locality_tokens: List[str] = []
    if len(parts) >= 2:
        for p in parts[-2:]:
            p_clean = clean_text(p)
            for tok in p_clean.split():
                if tok not in PLACEHOLDER_TOKENS and len(tok) >= 3 and not tok.isdigit():
                    locality_tokens.append(tok)

    return {
        "building_numbers": building_numbers,
        "postal_code": postal_code,
        "locality_tokens": locality_tokens,
    }


def tokenize_name(name: str, aggressive: bool = False, min_len: int = 3) -> List[str]:
    """
    Extract informative name tokens:
    - Conservative: lowercased, accent-stripped, legal suffixes removed.
    - Aggressive: generic business noise words also removed.
    """
    cleaned = clean_text(name)
    if not cleaned:
        return []

    tokens: List[str] = []
    for t in cleaned.split():
        if t in PLACEHOLDER_TOKENS or t in LEGAL_SUFFIXES:
            continue
        if aggressive and t in GENERIC_BUSINESS_WORDS:
            continue
        if len(t) >= min_len:
            tokens.append(t)
    return tokens
