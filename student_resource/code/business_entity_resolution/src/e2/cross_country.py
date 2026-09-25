"""
Full Cross-Country Ground-Truth Edge Scan for E2.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
The full-GT cross-country scan determines whether country can safely be used
as a hard blocking partition in the locked L2 blocking layer (K1-K9).
If 100.0% of true matches occur within the same country, partitioning blocking
by country introduces 0% recall loss while cutting pairwise comparison complexity in half.
If true matches ever cross country lines, hard partitioning poses a measured recall risk.
"""

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CrossCountrySlice:
    """Metrics for a specific slice of ground-truth edges."""
    slice_type: str
    s1_country: str
    target_source: str
    total_edges: int = 0
    same_country_edges: int = 0
    cross_country_edges: int = 0

    @property
    def same_country_pct(self) -> float:
        if self.total_edges == 0:
            return 0.0
        return round((self.same_country_edges / self.total_edges) * 100.0, 6)

    @property
    def cross_country_pct(self) -> float:
        if self.total_edges == 0:
            return 0.0
        return round((self.cross_country_edges / self.total_edges) * 100.0, 6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_type": self.slice_type,
            "s1_country": self.s1_country,
            "target_source": self.target_source,
            "total_edges": self.total_edges,
            "same_country_edges": self.same_country_edges,
            "cross_country_edges": self.cross_country_edges,
            "same_country_pct": self.same_country_pct,
            "cross_country_pct": self.cross_country_pct,
        }


@dataclass
class CrossCountryScanResult:
    """Complete aggregated results of the full cross-country edge scan."""
    total_edges: int = 0
    same_country_edges: int = 0
    cross_country_edges: int = 0
    same_country_pct: float = 0.0
    cross_country_pct: float = 0.0

    # Source breakdown: S1 -> S2, S1 -> S3
    s2_slice: Optional[CrossCountrySlice] = None
    s3_slice: Optional[CrossCountrySlice] = None

    # Country breakdown: US, India
    country_slices: Dict[str, CrossCountrySlice] = field(default_factory=dict)

    # Country x Source breakdown: (US, S2), (US, S3), (India, S2), (India, S3)
    country_source_slices: Dict[Tuple[str, str], CrossCountrySlice] = field(default_factory=dict)

    # Cross-country examples if any exist
    cross_country_examples: List[Dict[str, str]] = field(default_factory=list)

    # Resolution tracking
    s1_entities_resolved: int = 0
    s2_targets_resolved: int = 0
    s3_targets_resolved: int = 0
    missing_s1_in_source: int = 0
    missing_s2_in_source: int = 0
    missing_s3_in_source: int = 0

    @property
    def is_safe_for_hard_partitioning(self) -> bool:
        """Determines whether country can safely be used as a hard blocking partition."""
        return self.cross_country_edges == 0 and self.total_edges > 0

    @property
    def architectural_status(self) -> str:
        if self.is_safe_for_hard_partitioning:
            return "SAFE FOR HARD PARTITIONING ON TRAINING GT"
        elif self.cross_country_edges > 0:
            return "HARD COUNTRY PARTITION WOULD CREATE A MEASURED RECALL RISK"
        else:
            return "UNRESOLVED"

    def get_summary_records(self) -> List[Dict[str, Any]]:
        """Return list of dicts representing all summary rows for CSV export."""
        records: List[Dict[str, Any]] = []

        # 1. Overall
        records.append({
            "slice_type": "OVERALL",
            "s1_country": "ALL",
            "target_source": "ALL",
            "total_edges": self.total_edges,
            "same_country_edges": self.same_country_edges,
            "cross_country_edges": self.cross_country_edges,
            "same_country_pct": self.same_country_pct,
            "cross_country_pct": self.cross_country_pct,
        })

        # 2. By target source
        if self.s2_slice:
            records.append(self.s2_slice.to_dict())
        if self.s3_slice:
            records.append(self.s3_slice.to_dict())

        # 3. By S1 country
        for c in sorted(self.country_slices.keys()):
            records.append(self.country_slices[c].to_dict())

        # 4. By S1 country x target source
        for key in sorted(self.country_source_slices.keys()):
            records.append(self.country_source_slices[key].to_dict())

        return records


def scan_cross_country_edges(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    gt_path: Path,
    max_examples: int = 100,
) -> CrossCountryScanResult:
    """
    Perform an authoritative, memory-efficient full cross-country edge scan.

    Streams files sequentially to avoid holding multiple multi-gigabyte tables in RAM:
    1. Read S1 -> {s1_id: country} (~80-150MB).
    2. Stream GT for S2 edges -> {s2_id: (s1_id, s1_country)}.
    3. Stream S2 source -> check (s1_country == s2_country), record S2 stats, free S2 map.
    4. Stream GT for S3 edges -> {s3_id: (s1_id, s1_country)}.
    5. Stream S3 source -> check (s1_country == s3_country), record S3 stats, free S3 map.
    6. Aggregate all breakdowns and return CrossCountryScanResult.
    """
    # Step 1: Load S1 countries
    s1_country: Dict[str, str] = {}
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if row:
                s1_id = row[0].strip()
                country = sys.intern(row[3].strip()) if len(row) > 3 else "Unknown"
                s1_country[s1_id] = country

    cross_examples: List[Dict[str, str]] = []
    missing_s1_count = 0

    # Step 2: Stream GT for S2 targets
    s2_target_to_s1: Dict[str, Tuple[str, str]] = {}
    total_s2_edges = 0
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            s1_c = s1_country.get(s1_id)
            if s1_c is None:
                missing_s1_count += 1
                continue
            raw_targets = row[1].strip().split(",") if len(row) > 1 and row[1].strip() else []
            for t in raw_targets:
                t = t.strip()
                if t.startswith("S2-"):
                    total_s2_edges += 1
                    s2_target_to_s1[t] = (s1_id, s1_c)

    # Step 3: Stream S2 source table
    s2_slice = CrossCountrySlice(slice_type="BY_TARGET_SOURCE", s1_country="ALL", target_source="S2")
    s2_by_country: Dict[str, CrossCountrySlice] = {}
    s2_resolved = 0

    with open(s2_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row:
                continue
            s2_id = row[0].strip()
            if s2_id in s2_target_to_s1:
                s2_resolved += 1
                s1_id, s1_c = s2_target_to_s1[s2_id]
                s2_c = sys.intern(row[3].strip()) if len(row) > 3 else "Unknown"

                s2_slice.total_edges += 1
                if s1_c not in s2_by_country:
                    s2_by_country[s1_c] = CrossCountrySlice(
                        slice_type="BY_COUNTRY_X_SOURCE",
                        s1_country=s1_c,
                        target_source="S2",
                    )
                s2_by_country[s1_c].total_edges += 1

                if s1_c == s2_c:
                    s2_slice.same_country_edges += 1
                    s2_by_country[s1_c].same_country_edges += 1
                else:
                    s2_slice.cross_country_edges += 1
                    s2_by_country[s1_c].cross_country_edges += 1
                    if len(cross_examples) < max_examples:
                        cross_examples.append({
                            "source1_entity_id": s1_id,
                            "source1_country": s1_c,
                            "target_entity_id": s2_id,
                            "target_country": s2_c,
                            "target_source": "S2",
                        })

    missing_s2 = len(s2_target_to_s1) - s2_resolved
    del s2_target_to_s1  # Free memory immediately

    # Step 4: Stream GT for S3 targets
    s3_target_to_s1: Dict[str, Tuple[str, str]] = {}
    total_s3_edges = 0
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            s1_c = s1_country.get(s1_id)
            if s1_c is None:
                continue
            raw_targets = row[1].strip().split(",") if len(row) > 1 and row[1].strip() else []
            for t in raw_targets:
                t = t.strip()
                if t.startswith("S3-"):
                    total_s3_edges += 1
                    s3_target_to_s1[t] = (s1_id, s1_c)

    # Step 5: Stream S3 source table
    s3_slice = CrossCountrySlice(slice_type="BY_TARGET_SOURCE", s1_country="ALL", target_source="S3")
    s3_by_country: Dict[str, CrossCountrySlice] = {}
    s3_resolved = 0

    with open(s3_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row:
                continue
            s3_id = row[0].strip()
            if s3_id in s3_target_to_s1:
                s3_resolved += 1
                s1_id, s1_c = s3_target_to_s1[s3_id]
                s3_c = sys.intern(row[3].strip()) if len(row) > 3 else "Unknown"

                s3_slice.total_edges += 1
                if s1_c not in s3_by_country:
                    s3_by_country[s1_c] = CrossCountrySlice(
                        slice_type="BY_COUNTRY_X_SOURCE",
                        s1_country=s1_c,
                        target_source="S3",
                    )
                s3_by_country[s1_c].total_edges += 1

                if s1_c == s3_c:
                    s3_slice.same_country_edges += 1
                    s3_by_country[s1_c].same_country_edges += 1
                else:
                    s3_slice.cross_country_edges += 1
                    s3_by_country[s1_c].cross_country_edges += 1
                    if len(cross_examples) < max_examples:
                        cross_examples.append({
                            "source1_entity_id": s1_id,
                            "source1_country": s1_c,
                            "target_entity_id": s3_id,
                            "target_country": s3_c,
                            "target_source": "S3",
                        })

    missing_s3 = len(s3_target_to_s1) - s3_resolved
    del s3_target_to_s1  # Free memory immediately

    # Step 6: Consolidate aggregations
    total_edges = s2_slice.total_edges + s3_slice.total_edges
    same_country_edges = s2_slice.same_country_edges + s3_slice.same_country_edges
    cross_country_edges = s2_slice.cross_country_edges + s3_slice.cross_country_edges

    same_country_pct = round((same_country_edges / total_edges) * 100.0, 6) if total_edges > 0 else 0.0
    cross_country_pct = round((cross_country_edges / total_edges) * 100.0, 6) if total_edges > 0 else 0.0

    # Build Country slices (US, India, etc.)
    all_countries = set(s2_by_country.keys()) | set(s3_by_country.keys())
    country_slices: Dict[str, CrossCountrySlice] = {}
    country_source_slices: Dict[Tuple[str, str], CrossCountrySlice] = {}

    for c in all_countries:
        s2_c_slice = s2_by_country.get(c, CrossCountrySlice("BY_COUNTRY_X_SOURCE", c, "S2"))
        s3_c_slice = s3_by_country.get(c, CrossCountrySlice("BY_COUNTRY_X_SOURCE", c, "S3"))

        country_source_slices[(c, "S2")] = s2_c_slice
        country_source_slices[(c, "S3")] = s3_c_slice

        country_slices[c] = CrossCountrySlice(
            slice_type="BY_S1_COUNTRY",
            s1_country=c,
            target_source="ALL",
            total_edges=s2_c_slice.total_edges + s3_c_slice.total_edges,
            same_country_edges=s2_c_slice.same_country_edges + s3_c_slice.same_country_edges,
            cross_country_edges=s2_c_slice.cross_country_edges + s3_c_slice.cross_country_edges,
        )

    return CrossCountryScanResult(
        total_edges=total_edges,
        same_country_edges=same_country_edges,
        cross_country_edges=cross_country_edges,
        same_country_pct=same_country_pct,
        cross_country_pct=cross_country_pct,
        s2_slice=s2_slice,
        s3_slice=s3_slice,
        country_slices=country_slices,
        country_source_slices=country_source_slices,
        cross_country_examples=cross_examples,
        s1_entities_resolved=len(s1_country),
        s2_targets_resolved=s2_resolved,
        s3_targets_resolved=s3_resolved,
        missing_s1_in_source=missing_s1_count,
        missing_s2_in_source=missing_s2,
        missing_s3_in_source=missing_s3,
    )


def save_cross_country_summary(result: CrossCountryScanResult, output_path: Path) -> None:
    """Save cross-country summary CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = result.get_summary_records()
    fieldnames = [
        "slice_type",
        "s1_country",
        "target_source",
        "total_edges",
        "same_country_edges",
        "cross_country_edges",
        "same_country_pct",
        "cross_country_pct",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_cross_country_examples(result: CrossCountryScanResult, output_path: Path) -> None:
    """Save cross-country examples TSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source1_entity_id",
        "source1_country",
        "target_entity_id",
        "target_country",
        "target_source",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(result.cross_country_examples)
