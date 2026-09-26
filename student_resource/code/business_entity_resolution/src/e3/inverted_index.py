"""
Inverted Index Implementation with Oversized Block Protection for E3.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
Inverted indexing allows O(1) candidate lookup per blocking key.
However, common keys (e.g. 'store', 'enterprises', street number '100') can cause
N x M candidate explosions. This inverted index enforces deterministic oversized-block
protection: keys exceeding max_block_size are automatically suppressed, bounded, or upgraded,
preventing pathological memory blowup while logging diagnostic telemetry.
"""

import array
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from .blocking_lanes import BlockingConfig, generate_lane_keys


class PartitionInvertedIndex:
    """
    Compact inverted index for a single (country, target_source) partition.
    Indexes target records by lane and key using 32-bit unsigned integer arrays.
    """

    def __init__(self, country: str, source: str, config: Optional[BlockingConfig] = None):
        self.country = country
        self.source = source
        self.config = config or BlockingConfig()

        # Target ID registry: int_id -> str_entity_id
        self.target_ids: List[str] = []

        # Lane-partitioned index: lane -> key -> array.array('I')
        self.index: Dict[str, Dict[str, array.array]] = {
            lane: defaultdict(lambda: array.array("I")) for lane in self.config.enabled_lanes
        }

        # Oversized block registry: lane -> set of suppressed keys
        self.oversized_keys: Dict[str, Set[str]] = {
            lane: set() for lane in self.config.enabled_lanes
        }

        self.is_frozen = False
        self.total_indexed_records = 0
        self.last_query_oversized_lanes: Set[str] = set()

    def add_record(self, entity_id: str, name: str, address: str) -> None:
        """Add a target record to the partition index."""
        if self.is_frozen:
            raise RuntimeError("Cannot add records to a frozen index.")

        target_idx = len(self.target_ids)
        self.target_ids.append(entity_id)

        # Generate lane keys for target record
        lane_keys = generate_lane_keys(
            entity_id=entity_id,
            name=name,
            address=address,
            country=self.country,
            source=self.source,
            config=self.config,
        )

        for lane, keys in lane_keys.items():
            if lane in self.index:
                # Deduplicate keys per record
                for k in set(keys):
                    self.index[lane][k].append(target_idx)

        self.total_indexed_records += 1

    def freeze_and_audit(self) -> Dict[str, Any]:
        """
        Finalize index, identify oversized blocks, and enforce protection rules.
        """
        self.is_frozen = True
        stats: Dict[str, Any] = {
            "total_target_records": len(self.target_ids),
            "lane_stats": {},
        }

        for lane in self.config.enabled_lanes:
            lane_dict = self.index[lane]
            oversized = set()
            total_keys = len(lane_dict)
            total_postings = sum(len(p) for p in lane_dict.values())

            for key, postings in lane_dict.items():
                posting_len = len(postings)
                # Check oversized threshold
                if posting_len > self.config.max_block_size:
                    oversized.add(key)
                # In K1: also check token document frequency
                elif lane == "K1" and posting_len > self.config.max_token_doc_freq:
                    oversized.add(key)

            self.oversized_keys[lane] = oversized
            stats["lane_stats"][lane] = {
                "unique_keys": total_keys,
                "total_postings": total_postings,
                "oversized_keys_count": len(oversized),
                "oversized_percentage": round((len(oversized) / total_keys * 100), 4) if total_keys > 0 else 0.0,
            }

        return stats

    def query(self, s1_lane_keys: Dict[str, List[str]]) -> Tuple[Dict[str, Set[str]], int]:
        """
        Query index with S1 lane keys.
        Returns:
            - candidates_by_lane: {lane: Set[target_entity_id]}
            - oversized_events_count: int
        """
        candidates_by_lane: Dict[str, Set[str]] = {lane: set() for lane in self.config.enabled_lanes}
        oversized_events = 0
        oversized_lanes: Set[str] = set()

        for lane, keys in s1_lane_keys.items():
            if lane not in self.index:
                continue

            lane_dict = self.index[lane]
            oversized_set = self.oversized_keys[lane]

            for key in keys:
                if key in oversized_set:
                    # Oversized block: suppress enumeration to prevent N x M explosion
                    oversized_events += 1
                    oversized_lanes.add(lane)
                    continue

                if key in lane_dict:
                    postings = lane_dict[key]
                    for idx in postings:
                        candidates_by_lane[lane].add(self.target_ids[idx])

        self.last_query_oversized_lanes = oversized_lanes
        return candidates_by_lane, oversized_events
