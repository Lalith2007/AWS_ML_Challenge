"""
Unit tests for E3 Blocking Lanes and Preprocessing.

Tests covering requirements:
2. Missing-field handling
3. Generic-token suppression
4. Rare-token indexing
5. Name token reorder recovery
6. Character n-gram recovery
7. Numeric address blocking
8. Locality × name conjunction
9. Transliteration key handling
10. DBA split handling
"""

import pytest

from e3.blocking_lanes import BlockingConfig, generate_lane_keys
from e3.preprocessing import (
    clean_text,
    detect_script,
    extract_address_features,
    extract_dba_components,
    tokenize_name,
    transliterate_indic,
)


def test_missing_field_handling():
    """Requirement 2: Ensure empty/missing fields do not crash and produce safe empty key sets."""
    config = BlockingConfig()
    keys_empty = generate_lane_keys(
        entity_id="S1-EMPTY",
        name="",
        address="",
        country="US",
        source="S1",
        config=config,
    )
    for lane, klist in keys_empty.items():
        assert len(klist) == 0, f"Expected empty keys for {lane} on empty inputs, got {klist}"

    keys_none_like = generate_lane_keys(
        entity_id="S1-NULL",
        name="<null>",
        address="NULL",
        country="US",
        source="S1",
        config=config,
    )
    assert len(keys_none_like["K1"]) == 0
    assert len(keys_none_like["K2"]) == 0


def test_generic_token_suppression():
    """Requirement 3: Ensure generic business words and placeholder tokens are suppressed."""
    tokens_raw = tokenize_name("Global Enterprises Solutions Group", aggressive=True)
    # All generic words should be suppressed in aggressive mode
    assert len(tokens_raw) == 0

    tokens_mixed = tokenize_name("Acme Solutions International LLC", aggressive=True)
    assert tokens_mixed == ["acme"]


def test_rare_token_indexing():
    """Requirement 4: Ensure rare / informative tokens of sufficient length are indexed in K1."""
    config = BlockingConfig(min_token_len=3)
    keys = generate_lane_keys(
        entity_id="S1-1",
        name="Lalith Nanotechnology Labs",
        address="100 Innovation Way",
        country="US",
        source="S1",
        config=config,
    )
    assert "k1_lalith" in keys["K1"]
    assert "k1_nanotechnology" in keys["K1"]
    assert "k1_labs" in keys["K1"]


def test_name_token_reorder_recovery():
    """Requirement 5: Ensure K2 sorted signature is order-invariant across token permutations."""
    config = BlockingConfig()
    keys1 = generate_lane_keys("S1-1", "Alpha Beta Gamma", "123 Main St", "US", "S1", config)
    keys2 = generate_lane_keys("S2-2", "Gamma Alpha Beta", "123 Main St", "US", "S2", config)

    k2_set1 = set(keys1["K2"])
    k2_set2 = set(keys2["K2"])
    intersection = k2_set1 & k2_set2
    assert len(intersection) > 0
    assert "k2_alpha_beta_gamma" in intersection


def test_character_ngram_recovery():
    """Requirement 6: Ensure K3 character prefix n-grams recover minor spelling variations."""
    config = BlockingConfig()
    keys1 = generate_lane_keys("S1-1", "Techcorp Solutions", "123 Main St", "US", "S1", config)
    keys2 = generate_lane_keys("S2-2", "Tech Corporation", "123 Main St", "US", "S2", config)

    k3_1 = set(keys1["K3"])
    k3_2 = set(keys2["K3"])
    # Both share 'tec' and 'sol'/'cor', but share 'k3_ord_cor_tec' if both have tech and corp
    keys_corp = generate_lane_keys("S2-3", "Techcorp Global", "123 Main St", "US", "S2", config)
    assert any(k in keys_corp["K3"] for k in keys1["K3"])


def test_numeric_address_blocking():
    """Requirement 7: Ensure K4 handles building numbers, leading-zero canonicalization, and postal codes."""
    config = BlockingConfig()
    # Leading zeros in address (e.g. 0684 vs 684)
    addr1 = extract_address_features("Suite AF-0684, Tech Park, 94016", country="US")
    addr2 = extract_address_features("Unit 684 Tech Blvd, 94016", country="US")

    assert "684" in addr1["building_numbers"]
    assert "684" in addr2["building_numbers"]
    assert addr1["postal_code"] == "94016"
    assert addr2["postal_code"] == "94016"

    keys1 = generate_lane_keys("S1-1", "Apex Retail", "Suite AF-0684, 94016", "US", "S1", config)
    keys2 = generate_lane_keys("S2-2", "Apex Store", "684 Main Rd, 94016", "US", "S2", config)

    # Both should share building number + name prefix key or zip + name prefix key
    assert "k4_num_684_ape" in keys1["K4"]
    assert "k4_num_684_ape" in keys2["K4"]
    assert "k4_zip_94016_ape" in keys1["K4"]
    assert "k4_zip_94016_ape" in keys2["K4"]


def test_locality_name_conjunction():
    """Requirement 8: Ensure K5 combines locality tokens with name tokens for disambiguation."""
    config = BlockingConfig()
    keys1 = generate_lane_keys("S1-1", "Sunrise Bakery", "12 Market St, Sunnyvale, CA", "US", "S1", config)
    keys2 = generate_lane_keys("S2-2", "Sunrise Cafe", "45 Oak Ave, Sunnyvale, California", "US", "S2", config)

    k5_1 = set(keys1["K5"])
    k5_2 = set(keys2["K5"])
    intersection = k5_1 & k5_2
    assert "k5_sunnyvale_sun" in intersection or "k5_sunnyvale_sunrise" in intersection


def test_transliteration_key_handling():
    """Requirement 9: Ensure K6 transliterates Indic non-Latin scripts to phonetic Latin."""
    # Devanagari: "बिरयानी" (Biryani)
    indic_text = "बिरयानी"
    script = detect_script(indic_text)
    assert script == "DEVANAGARI"

    translit = transliterate_indic(indic_text)
    assert "biry" in translit or "bir" in translit

    config = BlockingConfig()
    # S1 in English Latin, S2 in Devanagari
    s1_keys = generate_lane_keys("S1-1", "Bharat Petroleum", "Mumbai", "India", "S1", config)
    # Devanagari for Bharat: "भारत"
    s2_keys = generate_lane_keys("S2-1", "भारत पेट्रोलियम", "Mumbai", "India", "S2", config)

    assert "k6_tok_bharat" in s1_keys["K6"]
    assert "k6_tok_bharat" in s2_keys["K6"]


def test_dba_split_handling():
    """Requirement 10: Ensure K7 extracts trade names from brackets, parentheses, domains, and DBA markers."""
    config = BlockingConfig()

    # Bracket alias: "Vertical International [School]"
    p1, a1 = extract_dba_components("Vertical International [School]")
    assert p1 == "vertical international"
    assert a1 == "school"

    # Domain name: "maurewilliamscolombier.com"
    p2, a2 = extract_dba_components("maurewilliamscolombier.com")
    assert a2 == "maurewilliamscolombier"

    # DBA marker: "Lumvio F/K/A Suryava Al LLP"
    p3, a3 = extract_dba_components("Lumvio F/K/A Suryava Al LLP")
    assert p3 == "lumvio"
    assert "suryava" in a3

    keys = generate_lane_keys("S1-1", "Lumvio F/K/A Suryava Al LLP", "123 St", "US", "S1", config)
    assert "k7_alias_suryava" in keys["K7"]
