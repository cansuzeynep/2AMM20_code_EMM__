"""
self-contained beam search implementation over conjuctive descriptions of numeric attributes 
(later maybe we can add pysubgroup for a run since its much faster and has more features)
quality measures are in quality_measures.py

numeric split points are generated same way as Branderhorst's thesis section 2.1.3 
describes over the covered values and take evenly spaced quantile cut points, producing 'attr <=s' / 'attr > s' condition pairsz
"""

from dataclasses import dataclass, field
from typing import Callable, List
import numpy as np
import pandas as pd

@dataclass
class Description:
    """ conjuction of (attribute, operator, threshold)"""
    conditions: List[tuple] = field(default_factory=list)

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(df), dtype=bool)
        for attr, op, threshold in self.conditions:
            if op == '<=':
                m &= (df[attr].values <= threshold)
            elif op == '>':
                m &= (df[attr].values > threshold)
            else:
                raise ValueError(f"Unknown operator: {op}")
        return m

    def __repr__(self):
        if not self.conditions:
            return "(empty description)"
        return " AND ".join([f"{attr} {op} {threshold}" for attr, op, threshold in self.conditions])

    @staticmethod
    def candidate_split_points(df: pd.DataFrame, attr: str, covered_mask: np.ndarray, n_bins: int = 10) -> List[float]:
        """quantile based split points among the records currently covered by a description"""
        values = np.sort(df.loc[covered_mask, attr].values)
        n = len(values)
        if n < 2:
            return []
        idx = [int(np.floor(i * n / n_bins)) for i in range(1, n_bins)]
        idx = sorted(set(i for i in idx if 0 <= i < n))
        return [values[i] for i in idx]    

def refine(df: pd.DataFrame, description: Description, attrs: List[str], n_bins: int) -> List[Description]:
    """generate all one-condition deeper refinements of a description"""
    covered = description.mask(df)
    children = []
    for attr in attrs:
        for s in Description.candidate_split_points(df, attr, covered, n_bins):
            for op in ['<=', '>']:
                child = Description(description.conditions + [(attr, op, s)])
                children.append(child)
    return children


def beam_search(
    df: pd.DataFrame,
    quality_fn: Callable[[np.ndarray], float],
    attrs: List[str],
    depth: int = 2,
    beam_width: int = 20,
    top_k: int = 5,
    min_subgroup_size: int = 30,
    n_bins: int = 10,
):
    """
    quality_fn: function(subgroup_mask) -> float. Bind y/p_hat with a
                        lambda or functools.partial before passing in, for example
                        quality_fn = lambda mask: q_calib(y, p_hat, mask)
    attrs: list of column names in df usable as descriptors
    depth: max number of conjoined conditions (search levels)
    beam_width: how many candidates survive to the next level
    top_k: how many final results to return
    min_subgroup_size: subgroups smaller than this are discarded (they
                          exist to prevent small-sample noise masquerading
                          as "exceptional")
 
    returns a list of (Description, quality, subgroup_size) tuples, sorted by quality descending, deduplicated by covered record set.
    """
    beam = [Description([])]
    all_results = []
 
    for level in range(depth):
        candidates = []
        for d in beam:
            candidates.extend(refine(df, d, attrs, n_bins))
 
        scored = []
        for cand in candidates:
            mask = cand.mask(df)
            if mask.sum() < min_subgroup_size:
                continue
            q = quality_fn(mask)
            scored.append((cand, q, mask.sum()))
 
        if not scored:
            break
 
        scored.sort(key=lambda t: t[1], reverse=True)
        beam = [c for c, q, n in scored[:beam_width]]
        all_results.extend(scored)
 
    #deduplicate by the actual set of covered records (two descriptions
    # covering the same records are redundant, keep the simpler one?).
    seen_masks = []
    deduped = []
    for cand, q, n in sorted(all_results, key=lambda t: (-t[1], len(t[0].conditions))):
        mask = cand.mask(df)
        is_dup = any(np.array_equal(mask, seen) for seen in seen_masks)
        if not is_dup:
            seen_masks.append(mask)
            deduped.append((cand, q, n))
        if len(deduped) >= top_k:
            break
 
    return deduped