"""
self-contained beam search implementation over conjuctive descriptions of numeric attributes 
(later maybe we can add pysubgroup for a run since its much faster and has more features)
quality measures are in quality_measures.py

numeric split points are generated same way as Branderhorst's thesis section 2.1.3 
describes over the covered values and take evenly spaced quantile cut points,
producing 'attr <=s' / 'attr > s' condition pairs
"""

from dataclasses import dataclass, field
from typing import Callable, List
import numpy as np
import pandas as pd


@dataclass
class Description:
    """ Description Class: Manages the descriptions which are a conjunction of
    conditions.

    Attributes:
        conditions (List[tuple]): describes a condition
            as a tuple (attribute, operator, value). E.g. age > 50

    """
    conditions: List[tuple] = field(default_factory=list)

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(df), dtype=bool)
        for attr, op, value in self.conditions:
            if op == '<=':
                m &= (df[attr].values <= value)
            elif op == '>':
                m &= (df[attr].values > value)
            elif op == '!=':
                m &= (df[attr].values != value)
            elif op == '==':
                m &= (df[attr].values == value)
            else:
                raise ValueError(f"Unknown operator: {op}")
        return m

    def __repr__(self):
        if not self.conditions:
            return "(empty description)"
        return " AND ".join([f"{attr} {op} {threshold}" for attr, op, threshold in self.conditions])

    @staticmethod
    def candidate_split_points(
            df: pd.DataFrame,
            attr: str,
            covered_mask: np.ndarray,
            n_bins: int = 10
    ) -> List[float]:
        """ Return the list of quantile based split points of the given numerical
        description attribute

        Args:
            df (pd.DataFrame): The full dataset
            attr (str): The attribute to be partitioned
            covered_mask (np.ndarray): Mask array of subgroup. One (1) for record in
                subgroup ,zero (0) for record excluded from the subgroup.
            n_bins (int): The number of bins the attribute should be partitioned into

        Returns:
            List: The list of split points that divides the partitions.

        """
        values = np.sort(df.loc[covered_mask, attr].values)
        n = len(values)
        if n < 2:
            return []
        idx = [int(np.floor(i * n / n_bins)) for i in range(1, n_bins)]
        idx = sorted(set(i for i in idx if 0 <= i < n))
        return [values[i] for i in idx]

def refine(
        df: pd.DataFrame,
        description: Description,
        attrs: List[str],
        attrs_type: pd.Series,
        n_bins: int
) -> List[Description]:
    """ Generates all possible one-condition refinements of a description that
    move the description one layer deeper

    Args:
        df (pd.DataFrame): The full dataset
        description (Description): The current description. I.e. the current set of
            conditions
        attrs (List): The list of description attributes.
        attrs_type (pd.Series): The list of variable types that correspond to the
            description attributes in 'attrs'
        n_bins (Int): The number of bins a numerical variable should be partitioned
            into

    Returns:
        children (List): The list of descriptions that are one layer deeper. I.e
            descriptions that contains one condition more that the initial description.

    Raises:
        ValueError: When the attribute type is unknown.

    """
    covered = description.mask(df)
    children = []
    for i in range(0, attrs_type.shape[0]):
        # Case 1: Numerical Attributes
        if attrs_type[i] == "Integer" or attrs_type[i] == "Continuous":
            for s in Description.candidate_split_points(df, attrs[i], covered, n_bins):
                for op in ['<=', '>']:
                    child = Description(description.conditions + [(attrs[i], op, s)])
                    children.append(child)

        # Case 2: Categorical Attributes
        elif attrs_type[i] == "Categorical":
            unique_vars = df.loc[covered, attrs[i]].unique()
            # Check is variable is constant
            if unique_vars.size == 1:
                pass
            # Check is variable is binary
            elif unique_vars.size == 2:
                for var in unique_vars:
                    child = Description(description.conditions + [(attrs[i], '==', var)])
                    children.append(child)
            # Else handle like categorical
            else:
                for var in unique_vars:
                    for op in ['==', '!=']:
                        child = Description(description.conditions + [(attrs[i], op, var)])
                        children.append(child)
        else:
            raise ValueError(f"Unknown variable type {attrs_type[i]} for {attrs[i]}")

    return children


def beam_search(
    df: pd.DataFrame,
    quality_fn: Callable[[np.ndarray], float],
    attrs: List[str],
    attrs_type: pd.Series,
    depth: int = 2,
    beam_width: int = 20,
    top_k: int = 5,
    min_subgroup_size: int = 30,
    n_bins: int = 10,
) -> List:
    """ Beam search for exceptional subgroups over conjunctive descriptions

    Args:
        df (pd.DataFrame): The full dataset
        quality_fn (Callable[[np.ndarray], float]): Function that takes a boolean
            subgroup mask and returns its quality, where higher means more
            exceptional. E.g. the closure returned by 'setup_quality_fn'.
        attrs (List): The list of description attributes.
        attrs_type (pd.Series): The list of variable types that correspond to the
            description attributes in 'attrs'. E.g. attrs_type[i] corresponds to attrs[i].
        depth (Int): The maximum number of conditions in a description.
        beam_width (Int): The number of descriptions kept during beam search.
        top_k (Int): The number of highest quality subgroups that are at most returned
        min_subgroup_size (Int): The minimum number of records in a subgroup.
        n_bins (Int): The number of bins the attribute should be partitioned into

    Returns:
        Deduped (List): The 'top_k' tuples of (description, quality, subgroup size)
            sorted by quality

    """
    beam = [Description([])]
    all_results = []

    for level in range(depth):
        candidates = []
        for d in beam:
            candidates.extend(refine(df, d, attrs, attrs_type, n_bins))

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

    # deduplicate by the actual set of covered records (two descriptions
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
