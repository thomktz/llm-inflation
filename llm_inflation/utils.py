"""Utils for the LLM Inflation project."""

import numpy as np
import pandas as pd


def log_returns(self, n_shifts=1):
    """Log returns of a time series."""
    return np.log(self / self.shift(n_shifts))


pd.Series.log_returns = log_returns
pd.DataFrame.log_returns = log_returns


def lapeyres_aggregation(
    index_values_children, weights_df, hicp, country, parent_node="CP00"
):
    """
    Implements the Lapeyres Aggregation algorithm.

    Lapeyres Aggregation algorithm:
    1) Take the index values of the children of node 'c', and divide each value by the value of the last December of the previous year.
    2) For each children, this value is multiplied by the current-year weight of the children node.
    3) These values are summed among the children to obtain the evolution of the parent node.
    4) Finally, the value is multiplied by the last December value of the previous year of the parent node.

    The algorithm presented here takes a different but ultimately equivalent approach:
    1) Compute the log returns of the index values of the children of node 'c'.
    2) Compute their intra-year cumulative sums.
    3) For each children, this value is multiplied by the current-year weight of the children node.
    4) These values are summed among the children to obtain the evolution of the parent node.
    5) The resulting time-series is exponentiated to get levels.
    6) Finally, the value is scaled by the last December value of the previous year of the parent node.
    """
    # 1) Compute the log returns of the index values of the children of node 'c'
    mom_reconstructed_index_df = index_values_children.log_returns().iloc[1:]

    # 2) Compute their intra-year cumulative sums
    yearly_cumsum = mom_reconstructed_index_df.groupby(
        mom_reconstructed_index_df.index.year
    ).cumsum()

    # 3) For each children, multiply by the current-year weight of the children node.
    weighted_cumsum = yearly_cumsum.copy()
    for year in yearly_cumsum.index.year.unique():
        weighted_cumsum.loc[weighted_cumsum.index.year == year] *= (
            weights_df.loc[year].values / 1000
        )

    # 4 & 5) Sum the values among the children and exponentiate to get levels
    final_result = np.exp(weighted_cumsum.sum(axis=1))

    # 6) Scale by the last December value of the previous year of the parent node.
    for i, year in enumerate(final_result.index.year.unique()):
        # For the first year, we have to use known values from the previous year
        if i == 0:
            final_result.loc[final_result.index.year == year] *= hicp.prices[country][
                parent_node
            ].loc[f"{year-1}-12-01"]
        # After, we can use the values we just computed
        else:
            final_result.loc[final_result.index.year == year] *= final_result.loc[
                final_result.index.year == year - 1
            ].iloc[-1]

    return final_result
