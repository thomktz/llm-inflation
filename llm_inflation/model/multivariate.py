"""Multivariate model for inflation forecasting."""

import pandas as pd
from typing import Union, Optional
from darts import TimeSeries
from darts.models import XGBModel, LinearRegressionModel, RandomForest
from darts.metrics import mae

multivariate_models = [
    {
        "name": "LinearRegression",
        "model": LinearRegressionModel,
        "kwargs": {},
    },
    {
        "name": "RandomForest",
        "model": RandomForest,
        "kwargs": {
            "n_estimators": 100,
            "max_depth": 10,
        },
    },
    {
        "name": "XGBoost",
        "model": XGBModel,
        "kwargs": {},
    },
]


class MultivariateForecaster:
    """Multivariate model for inflation forecasting. Wrapper around the 'darts' library."""

    def __init__(
        self,
        model_type: str,
        horizon: int,
        lags: Union[int, list[int]],
        prices: TimeSeries,
        covariates: Optional[TimeSeries] = None,
        lags_covariates: Optional[Union[int, list[int]]] = None,
    ):
        """
        Initialize the model.

        Parameters
        ----------
        model_type : str
            The type of model to use.
            ["LinearRegression", "RandomForest", "XGBoost"]
        horizon : int
            The number of periods to forecast.
        lags : int or list
            The lags to use for the model (positive int or negative list)
        prices : TimeSeries
            The prices to forecast.
        covariates : TimeSeries (optional)
            The covariates to use for the model.
        lags_covariates : list (optional)
            The lags to use for the covariates.
        """
        self.prices = prices
        self.covariates = covariates
        self.lags_covariates = lags_covariates
        self.horizon = horizon
        self.lags = lags
        self.model_type = model_type

        model_params = [
            model for model in multivariate_models if model["name"] == model_type
        ][0]
        self.model_class = model_params["model"]
        self.model_kwargs = model_params["kwargs"]

        self._init_model()

    def _init_model(self):
        """Initialize the 'darts' model."""
        self.model = self.model_class(
            **dict(
                **{"lags": self.lags, "output_chunk_length": self.horizon},
                **(self.model_kwargs or {}),
                **(
                    {"lags_past_covariates": self.lags_covariates}
                    if self.lags_covariates
                    else {}
                )
            )
        )

    def fit(
        self,
        new_prices: Optional[TimeSeries] = None,
        new_covariates: Optional[TimeSeries] = None,
    ):
        """
        Fit the model. Uses the initial prices and covariates by default.

        Parameters
        ----------
        new_prices : TimeSeries (optional)
            The new prices to use for fitting.
        new_covariates : TimeSeries (optional)
            The new covariates to use for fitting.
        """
        if new_prices is None:
            new_prices = self.prices
        if new_covariates is None:
            new_covariates = self.covariates
        self.model.fit(new_prices, past_covariates=new_covariates)

    def backtest(self, stride: int, metric: callable = mae):
        """
        Run a backtest on the model.

        Parameters
        ----------
        stride : int
            How many steps to move the test window at each iteration.
        metric : callable (optional)
            The metric to use for scoring. Defaults to Mean Absolute Error.
        """
        score = self.model.backtest(
            series=self.prices,
            past_covariates=self.covariates,
            stride=stride,
            metric=metric,
            verbose=True,
        )
        return score

    def make_training_data(self, forecast_start: str, prices_df: pd.DataFrame):
        """
        Make training data for the model.

        Parameters
        ----------
        forecast_start : str
            The date from which to start forecasting.
        prices_df : pd.DataFrame
            The DataFrame containing the prices, from cpilib.
        """
        training_data = self.prices.drop_after(pd.Timestamp(forecast_start))
        training_covariates = (
            self.covariates.drop_after(pd.Timestamp(forecast_start))
            if self.covariates
            else None
        )

        true_index = prices_df[prices_df["date"] < forecast_start].set_index("date")

        last_year = true_index.index.year[-1] - 1
        year_before_last = true_index.index.year.unique()[-2] - 1

        # Keep only last year and the december before
        true_index = true_index[
            (
                (true_index.index.year >= last_year)
                | (
                    (true_index.index.year == year_before_last)
                    & (true_index.index.month == 12)
                )
            )
        ]

        return training_data, training_covariates, true_index
