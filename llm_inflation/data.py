"""Holds the path to the covariates file."""

import pkg_resources

covariates_path = pkg_resources.resource_filename(
    "llm_inflation", "data/covariates.csv"
)
