"""Pattern-collapse / bridging. Upstream: aayushraina21/drift-sense-synthetic-data."""
import numpy as np


def maybe_collapse_gap(gap_nm, threshold_nm, rng, collapse_prob=0.7):
    if gap_nm >= threshold_nm:
        return False
    return bool(rng.random() < collapse_prob)
