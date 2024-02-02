import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import XGBModel, LinearRegressionModel, RandomForest
from darts.metrics import mae
from dateutil.relativedelta import relativedelta

multivariate_models = [
    {
        "name": "LinearRegression",
        "model": LinearRegressionModel,
        "kwargs": {},
    },
    {
        "name": 'RandomForest',
        "model": RandomForest,
        "kwargs": {
            "n_estimators": 100,
            "max_depth": 10,
        },
    },
    {
        "name": 'XGBoost',
        "model": XGBModel,
        "kwargs": {},
    },
]


class MultivariateForecaster:
    def __init__(self, model_type, horizon, lags, prices, weights, covariates=None, lags_covariates=None):
        self.prices = prices
        self.weights = weights
        self.covariates = covariates
        self.lags_covariates = lags_covariates
        self.horizon = horizon
        self.lags = lags
        self.model_type = model_type
        
        model_params = [model for model in multivariate_models if model["name"] == model_type][0]
        self.model_class = model_params["model"]
        self.model_kwargs = model_params["kwargs"]
        
        self._init_model()
    
    def _init_model(self):
        self.model = self.model_class(
            **dict(
                **{
                    "lags": self.lags, 
                    "output_chunk_length": self.horizon
                },
                **(self.model_kwargs or {}),
                **({"lags_past_covariates": self.lags_covariates}  if self.lags_covariates else {})
            )
        )
    
    def fit(self, new_prices=None, new_covariates=None):
        if new_prices is None:
            new_prices = self.prices
        if new_covariates is None:
            new_covariates = self.covariates
        self.model.fit(new_prices, past_covariates=new_covariates)
    
    def backtest(self, stride, metric=mae):
        score = self.model.backtest(
            series=self.prices,
            past_covariates=self.covariates,
            stride=stride,
            metric=metric,
            verbose=True,
        )
        return score
    
    def aggregate(self, horizon: int, last_train_date: str, input_mode: str):
        """
        Aggregate the series to the full HICP.
        
        Parameters
        ----------
        horizon : int
            The number of periods to aggregate.
        last_train_date : str
            The last date to use for training.
        input_mode : str
            The input mode to use for aggregation.
            ["index", "mom", "yoy"]
        """
        
        # 1 - Make predictions and concatenate to known values
        training_data = self.prices.drop_after(pd.Timestamp(last_train_date))
        training_covariates = self.covariates.drop_after(pd.Timestamp(last_train_date)) if self.covariates else None
        self.model.fit(training_data, past_covariates=training_covariates)
        prediction = self.model.predict(horizon)
        full_series = self.prices.append(prediction).pd_dataframe()
        
        # 2 - Convert to MoM
        if input_mode == "index":
            series = full_series
            
    
    def show_predictions2(self, n=6, past_values=12):
        self.model.fit(self.prices, past_covariates=self.covariates)
        prediction = self.model.predict(n)
        prediction.plot(label="prediction")
        self.prices[-past_values:].plot(label="actual")
        plt.legend()
        plt.show()


    def show_predictions(self, n_predicted=6, past_values=12, end_date=None):
        # Ensure end_date is a pd.Timestamp for consistency
        if end_date:
            end_date = pd.Timestamp(end_date)

        # Fit the model up to end_date if specified, else use full data
        if end_date:
            prices_to_fit = self.prices.drop_after(end_date)
            covariates_to_fit = self.covariates.drop_after(end_date) if self.covariates else None
        else:
            prices_to_fit = self.prices
            covariates_to_fit = self.covariates

        # Fit the model
        self.model.fit(series=prices_to_fit, past_covariates=covariates_to_fit)

        # Generate predictions from the end of the training data
        prediction = self.model.predict(n=n_predicted)

        # Calculate the start date for the actual past values
        start_past = end_date - relativedelta(months=past_values) if end_date else self.prices.end_time() - relativedelta(months=past_values)

        # Calculate the end date for actual future values to be included
        end_future = end_date + relativedelta(months=n_predicted) if end_date else self.prices.end_time()

        # Select the actual values range to include both past and the available future
        actual = self.prices[start_past:end_future]

        # Plot actual values (both past and future when available)
        actual.plot(label="Actual")

        # Plot predictions
        prediction.plot(label="Predictions")

        plt.legend()
        plt.title("Predictions vs Actual Values")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()

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
        training_covariates = self.covariates.drop_after(pd.Timestamp(forecast_start)) if self.covariates else None

        true_index = prices_df[prices_df["date"] < forecast_start].set_index("date")

        last_year = true_index.index.year[-1]-1
        year_before_last = true_index.index.year.unique()[-2]-1

        # Keep only last year and the december before
        true_index = true_index[(
            (true_index.index.year >= last_year)
            | (
                (true_index.index.year == year_before_last)
                & (true_index.index.month == 12)
            )
        )]
        
        return training_data, training_covariates, true_index