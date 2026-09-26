"""
Symbolic Blocking Lanes K1-K7 Implementation for E3.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
Architecture Version 2.0 defines a multi-lane symbolic blocking subsystem (K1-K7)
to generate candidate pairs with near-complete true-match recall while bounding candidate
volume before pairwise feature engineering (L3) and LightGBM classification (L4).
Each lane targets a distinct structural variation pattern:
- K1: Rare / selective name and address tokens
- K2: Sorted canonical-name signature (order-invariant)
- K3: Character n-gram conjunctions (spelling variations / typos)
- K4: Address structural / numeric keys (building numbers & postal codes)
- K5: Locality-token x name-token conjunctions
- K6: Transliterated-name keys (Indic-to-Latin script mismatches)
- K7: DBA / trade-name / domain alias keys
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .preprocessing import (
    clean_text,
    detect_script,
    extract_address_features,
    extract_dba_components,
    tokenize_name,
    transliterate_indic,
)


@dataclass
class BlockingConfig:
    """Configurable thresholds and hyperparameters for L2 blocking."""
    enabled_lanes: List[str] = field(
        default_factory=lambda: ["K1", "K2", "K3", "K4", "K5", "K6", "K7"]
    )
    max_block_size: int = 500  # Maximum posting list size before block suppression
    max_candidates_per_query: Optional[int] = None  # Configurable candidate cap (None = no query-level truncation)
    min_token_len: int = 3
    max_token_doc_freq: int = 2000  # Gating threshold to suppress ubiquitous tokens
    ngram_size: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled_lanes": self.enabled_lanes,
            "max_block_size": self.max_block_size,
            "max_candidates_per_query": self.max_candidates_per_query,
            "min_token_len": self.min_token_len,
            "max_token_doc_freq": self.max_token_doc_freq,
            "ngram_size": self.ngram_size,
        }


def generate_lane_keys(
    entity_id: str,
    name: str,
    address: str,
    country: str = "US",
    source: str = "S1",
    config: Optional[BlockingConfig] = None,
) -> Dict[str, List[str]]:
    """
    Generate typed blocking keys partitioned by lane for an entity record.
    Returns: {"K1": [...], "K2": [...], "K3": [...], "K4": [...], "K5": [...], "K6": [...], "K7": [...]}
    """
    if config is None:
        config = BlockingConfig()

    lane_keys: Dict[str, List[str]] = {lane: [] for lane in config.enabled_lanes}
    if not name and not address:
        return lane_keys

    # Preprocessing
    primary_name, dba_alias = extract_dba_components(name)
    tokens = tokenize_name(primary_name, aggressive=False, min_len=config.min_token_len)
    aggressive_tokens = tokenize_name(primary_name, aggressive=True, min_len=config.min_token_len)
    addr_features = extract_address_features(address, country=country)
    script = detect_script(name)

    name_prefix = tokens[0][:3] if tokens else (primary_name[:3] if primary_name else "")

    # ----------------------------------------------------
    # K1: Rare-Token Inverted Index
    # ----------------------------------------------------
    if "K1" in config.enabled_lanes:
        k1_keys = [f"k1_{t}" for t in tokens if len(t) >= 3]
        if len(tokens) >= 2:
            t1, t2 = sorted(tokens[:2])
            k1_keys.append(f"k1_pair_{t1}_{t2}")
        lane_keys["K1"] = k1_keys

    # ----------------------------------------------------
    # K2: Sorted Canonical-Name Signature (Order-invariant)
    # ----------------------------------------------------
    if "K2" in config.enabled_lanes:
        k2_keys: List[str] = []
        if tokens:
            sorted_tokens = sorted(tokens[:3])
            k2_keys.append(f"k2_{'_'.join(sorted_tokens)}")
        lane_keys["K2"] = k2_keys

    # ----------------------------------------------------
    # K3: Character N-Gram Conjunction Keys
    # ----------------------------------------------------
    if "K3" in config.enabled_lanes:
        k3_keys: List[str] = []
        if tokens and len(tokens[0]) >= 4:
            k3_keys.append(f"k3_pfx_{tokens[0][:4]}")
        if len(tokens) >= 2:
            p1 = tokens[0][:3]
            p2 = tokens[1][:3]
            k3_keys.append(f"k3_{p1}_{p2}")
            k3_keys.append(f"k3_ord_{'_'.join(sorted([p1, p2]))}")
        elif len(tokens) == 1 and len(tokens[0]) >= 4:
            k3_keys.append(f"k3_pfx_{tokens[0][:4]}")
        lane_keys["K3"] = k3_keys

    # ----------------------------------------------------
    # K4: Address Structural / Numeric Keys
    # ----------------------------------------------------
    if "K4" in config.enabled_lanes:
        k4_keys: List[str] = []
        building_nums = addr_features["building_numbers"]
        postal_code = addr_features["postal_code"]

        if name_prefix:
            if building_nums:
                k4_keys.append(f"k4_num_{building_nums[0]}_{name_prefix}")
            if postal_code:
                k4_keys.append(f"k4_zip_{postal_code}_{name_prefix}")
        # Precise structural address anchor: building number + postal code
        if postal_code and building_nums:
            k4_keys.append(f"k4_addr_{postal_code}_{building_nums[0]}")
        lane_keys["K4"] = k4_keys

    # ----------------------------------------------------
    # K5: Locality-Token x Name-Token Conjunction
    # ----------------------------------------------------
    if "K5" in config.enabled_lanes:
        k5_keys: List[str] = []
        locality_tokens = addr_features["locality_tokens"]
        if locality_tokens and tokens:
            primary_tok = aggressive_tokens[0] if aggressive_tokens else tokens[0]
            k5_keys.append(f"k5_{locality_tokens[0]}_{primary_tok}")
        lane_keys["K5"] = k5_keys

    # ----------------------------------------------------
    # K6: Transliterated-Name Keys (Active for Indic domain)
    # ----------------------------------------------------
    if "K6" in config.enabled_lanes and country == "India":
        k6_keys: List[str] = []
        if script != "LATIN":
            translit_name = transliterate_indic(name)
            translit_tokens = tokenize_name(translit_name, aggressive=False, min_len=3)
            for t in translit_tokens[:3]:
                if len(t) >= 3:
                    k6_keys.append(f"k6_tok_{t}")
                if len(t) >= 4:
                    k6_keys.append(f"k6_pfx_{t[:4]}")
            if len(translit_tokens) >= 2:
                k6_keys.append(f"k6_sig_{'_'.join(sorted(translit_tokens[:3]))}")
        else:
            for t in tokens[:3]:
                if len(t) >= 3:
                    k6_keys.append(f"k6_tok_{t}")
                if len(t) >= 4:
                    k6_keys.append(f"k6_pfx_{t[:4]}")
            if len(tokens) >= 2:
                k6_keys.append(f"k6_sig_{'_'.join(sorted(tokens[:3]))}")
        lane_keys["K6"] = k6_keys

    # ----------------------------------------------------
    # K7: DBA / Trade-Name / Domain Alias Keys
    # ----------------------------------------------------
    if "K7" in config.enabled_lanes and dba_alias:
        k7_keys: List[str] = []
        alias_tokens = tokenize_name(dba_alias, aggressive=False, min_len=3)
        for t in alias_tokens[:2]:
            k7_keys.append(f"k7_alias_{t}")
        if len(alias_tokens) >= 2:
            k7_keys.append(f"k7_asig_{'_'.join(sorted(alias_tokens[:3]))}")
        lane_keys["K7"] = k7_keys

    return lane_keys
