#!/usr/bin/env python3
"""
Chapter 15: Time Series Analysis
Data Voyage: Analyzing and Forecasting Time-Dependent Data

This script covers essential time series analysis concepts and techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


def demonstrate_ts_overview():
    """Demonstrate time series overview and concepts."""
    print("Time Series Analysis Overview:")
    print("-" * 40)

    print("Time Series Analysis involves studying data points")
    print("collected over time to identify patterns, trends,")
    print("and make predictions about future values.")
    print()

    # 1. What is Time Series?
    print("1. WHAT IS TIME SERIES?")
    print("-" * 30)

    ts_concepts = {
        "Definition": "Sequence of data points measured over time",
        "Characteristics": "Ordered, time-dependent, potentially correlated",
        "Goal": "Understand patterns and forecast future values",
        "Applications": "Stock prices, weather, sales, sensor data",
    }

    for concept, description in ts_concepts.items():
        print(f"  {concept}: {description}")
    print()

    # 2. Time Series Types
    print("2. TIME SERIES TYPES:")
    print("-" * 25)

    ts_types = {
        "Continuous": ["Stock prices", "Temperature readings", "Heart rate"],
        "Discrete": ["Daily sales", "Monthly unemployment", "Quarterly GDP"],
        "Regular": ["Hourly measurements", "Daily records", "Monthly reports"],
        "Irregular": ["Event-based data", "Transaction timestamps", "Sensor failures"],
    }

    for ts_type, examples in ts_types.items():
        print(f"  {ts_type}:")
        for example in examples:
            print(f"    • {example}")
        print()

    # 3. Time Series Analysis Steps
    print("3. TIME SERIES ANALYSIS STEPS:")
    print("-" * 35)

    analysis_steps = [
        "1. Data Collection - Gather time-ordered observations",
        "2. Data Exploration - Visualize and understand patterns",
        "3. Component Analysis - Identify trend, seasonality, noise",
        "4. Stationarity Testing - Check for time-invariant properties",
        "5. Model Selection - Choose appropriate forecasting method",
        "6. Model Fitting - Train the selected model",
        "7. Validation - Assess model performance",
        "8. Forecasting - Make future predictions",
    ]

    for step in analysis_steps:
        print(f"  {step}")
    print()

    # 4. Applications
    print("4. TIME SERIES APPLICATIONS:")
    print("-" * 30)

    applications = {
        "Finance": "Stock price prediction, risk assessment, portfolio optimization",
        "Economics": "GDP forecasting, inflation analysis, unemployment trends",
        "Marketing": "Sales forecasting, demand planning, campaign effectiveness",
        "Healthcare": "Patient monitoring, disease progression, treatment outcomes",
        "Manufacturing": "Quality control, predictive maintenance, production planning",
        "Weather": "Climate prediction, storm forecasting, seasonal patterns",
    }

    for domain, examples in applications.items():
        print(f"  {domain}: {examples}")
    print()


def demonstrate_ts_components():
    """Demonstrate time series components and decomposition."""
    print("Time Series Components:")
    print("-" * 40)

    print("Time series can be decomposed into several")
    print("components that help understand the underlying patterns.")
    print()

    # 1. Generate Synthetic Time Series
    print("1. GENERATING SYNTHETIC TIME SERIES:")
    print("-" * 35)

    # Create time index
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
    n = len(dates)

    # Generate components
    np.random.seed(42)

    # Trend component (linear + polynomial)
    trend = 100 + 0.1 * np.arange(n) + 0.0001 * np.arange(n) ** 2

    # Seasonal component (annual + weekly)
    seasonal_annual = 20 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    seasonal_weekly = 5 * np.sin(2 * np.pi * np.arange(n) / 7)
    seasonal = seasonal_annual + seasonal_weekly

    # Cyclical component (business cycles)
    cycle = 10 * np.sin(2 * np.pi * np.arange(n) / 90)

    # Random noise
    noise = np.random.normal(0, 5, n)

    # Combine components
    time_series = trend + seasonal + cycle + noise

    # Create DataFrame
    ts_df = pd.DataFrame(
        {
            "date": dates,
            "value": time_series,
            "trend": trend,
            "seasonal": seasonal,
            "cycle": cycle,
            "noise": noise,
        }
    )

    print(f"✅ Created time series: {len(ts_df)} observations")
    print(f"   Date range: {ts_df['date'].min()} to {ts_df['date'].max()}")
    print(f"   Value range: {ts_df['value'].min():.2f} to {ts_df['value'].max():.2f}")
    print()

    # 2. Component Analysis
    print("2. COMPONENT ANALYSIS:")
    print("-" * 25)

    print("Component Statistics:")
    print(
        f"  Trend - Mean: {ts_df['trend'].mean():.2f}, Std: {ts_df['trend'].std():.2f}"
    )
    print(
        f"  Seasonal - Mean: {ts_df['seasonal'].mean():.2f}, Std: {ts_df['seasonal'].std():.2f}"
    )
    print(
        f"  Cycle - Mean: {ts_df['cycle'].mean():.2f}, Std: {ts_df['cycle'].std():.2f}"
    )
    print(
        f"  Noise - Mean: {ts_df['noise'].mean():.2f}, Std: {ts_df['noise'].std():.2f}"
    )
    print()

    # 3. Seasonal Decomposition
    print("3. SEASONAL DECOMPOSITION:")
    print("-" * 30)

    # Use statsmodels for decomposition
    try:
        # Resample to monthly for decomposition (daily data can be too noisy)
        monthly_data = ts_df.set_index("date")["value"].resample("M").mean()

        # Perform decomposition
        decomposition = seasonal_decompose(monthly_data, model="additive", period=12)

        print("✅ Decomposition completed:")
        print(f"   Trend component: {len(decomposition.trend)} observations")
        print(f"   Seasonal component: {len(decomposition.seasonal)} observations")
        print(f"   Residual component: {len(decomposition.resid)} observations")
        print()

        # 4. Visualization
        print("4. VISUALIZATION:")
        print("-" * 20)

        plt.figure(figsize=(15, 12))

        # Original time series
        plt.subplot(4, 1, 1)
        plt.plot(ts_df["date"], ts_df["value"], linewidth=0.8, alpha=0.8)
        plt.title("Original Time Series")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)

        # Trend component
        plt.subplot(4, 1, 2)
        plt.plot(ts_df["date"], ts_df["trend"], linewidth=1.5, color="red")
        plt.title("Trend Component")
        plt.ylabel("Trend")
        plt.grid(True, alpha=0.3)

        # Seasonal component
        plt.subplot(4, 1, 3)
        plt.plot(ts_df["date"], ts_df["seasonal"], linewidth=0.8, color="green")
        plt.title("Seasonal Component")
        plt.ylabel("Seasonal")
        plt.grid(True, alpha=0.3)

        # Noise component
        plt.subplot(4, 1, 4)
        plt.plot(
            ts_df["date"], ts_df["noise"], linewidth=0.5, color="purple", alpha=0.7
        )
        plt.title("Noise Component")
        plt.ylabel("Noise")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("time_series_components.png", dpi=300, bbox_inches="tight")
        print(
            "✅ Time series components visualization saved as 'time_series_components.png'"
        )
        plt.close()

        # Decomposition plots
        plt.figure(figsize=(15, 10))
        decomposition.plot()
        plt.tight_layout()
        plt.savefig("seasonal_decomposition.png", dpi=300, bbox_inches="tight")
        print(
            "✅ Seasonal decomposition visualization saved as 'seasonal_decomposition.png'"
        )
        plt.close()

    except Exception as e:
        print(f"⚠️  Decomposition failed: {e}")
        print("   Continuing with manual component analysis...")
        print()


def demonstrate_stationarity():
    """Demonstrate stationarity testing and analysis."""
    print("Stationarity and Testing:")
    print("-" * 40)

    print("Stationarity is a key concept in time series analysis.")
    print("A stationary series has constant statistical properties over time.")
    print()

    # 1. What is Stationarity?
    print("1. WHAT IS STATIONARITY?")
    print("-" * 30)

    stationarity_concepts = {
        "Definition": "Time series with constant statistical properties",
        "Properties": "Constant mean, variance, and autocorrelation",
        "Importance": "Required for many time series models",
        "Testing": "ADF test, KPSS test, visual inspection",
    }

    for concept, description in stationarity_concepts.items():
        print(f"  {concept}: {description}")
    print()

    # 2. Generate Different Types of Series
    print("2. GENERATING DIFFERENT SERIES TYPES:")
    print("-" * 40)

    np.random.seed(42)
    n = 1000

    # Stationary series (random walk)
    stationary = np.random.normal(0, 1, n)

    # Non-stationary series (trend)
    trend = np.cumsum(np.random.normal(0, 0.1, n))

    # Non-stationary series (changing variance)
    changing_var = np.random.normal(0, 1, n) * (1 + 0.01 * np.arange(n))

    # Non-stationary series (seasonal)
    seasonal_nonstat = 10 * np.sin(2 * np.pi * np.arange(n) / 50) + np.random.normal(
        0, 1, n
    )

    series_types = {
        "Stationary": stationary,
        "Trend": trend,
        "Changing Variance": changing_var,
        "Seasonal": seasonal_nonstat,
    }

    print("✅ Generated different series types:")
    for name, series in series_types.items():
        print(f"   {name}: {len(series)} observations")
    print()

    # 3. Stationarity Testing
    print("3. STATIONARITY TESTING:")
    print("-" * 30)

    def test_stationarity(series, name):
        """Test stationarity using ADF and KPSS tests."""
        print(f"Testing {name}:")

        # ADF Test
        try:
            adf_result = adfuller(series)
            adf_stat = adf_result[0]
            adf_pvalue = adf_result[1]
            adf_critical = adf_result[4]

            print(f"  ADF Test:")
            print(f"    Statistic: {adf_stat:.4f}")
            print(f"    p-value: {adf_pvalue:.4f}")
            print(f"    Critical values: {adf_critical}")

            if adf_pvalue < 0.05:
                print(f"    Result: Stationary (p < 0.05)")
            else:
                print(f"    Result: Non-stationary (p >= 0.05)")
        except Exception as e:
            print(f"    ADF test failed: {e}")

        # KPSS Test
        try:
            kpss_result = kpss(series)
            kpss_stat = kpss_result[0]
            kpss_pvalue = kpss_result[1]
            kpss_critical = kpss_result[3]

            print(f"  KPSS Test:")
            print(f"    Statistic: {kpss_stat:.4f}")
            print(f"    p-value: {kpss_pvalue:.4f}")
            print(f"    Critical values: {kpss_critical}")

            if kpss_pvalue > 0.05:
                print(f"    Result: Stationary (p > 0.05)")
            else:
                print(f"    Result: Non-stationary (p <= 0.05)")
        except Exception as e:
            print(f"    KPSS test failed: {e}")

        print()

    # Test each series
    for name, series in series_types.items():
        test_stationarity(series, name)

    # 4. Making Series Stationary
    print("4. MAKING SERIES STATIONARY:")
    print("-" * 35)

    # Difference the trend series
    trend_diff = np.diff(trend)

    # Log transform and difference the changing variance series
    changing_var_log = np.log(np.abs(changing_var) + 1e-10)
    changing_var_diff = np.diff(changing_var_log)

    # Remove seasonal component
    seasonal_detrended = seasonal_nonstat - 10 * np.sin(2 * np.pi * np.arange(n) / 50)

    print("✅ Applied transformations:")
    print(f"   Trend → Differenced: {len(trend_diff)} observations")
    print(
        f"   Changing Variance → Log + Differenced: {len(changing_var_diff)} observations"
    )
    print(f"   Seasonal → Detrended: {len(seasonal_detrended)} observations")
    print()

    # Test transformed series
    print("Testing transformed series:")
    test_stationarity(trend_diff, "Differenced Trend")
    test_stationarity(changing_var_diff, "Log + Differenced Variance")
    test_stationarity(seasonal_detrended, "Detrended Seasonal")

    # 5. Visualization
    print("5. VISUALIZATION:")
    print("-" * 20)

    plt.figure(figsize=(15, 10))

    # Original series
    plt.subplot(3, 2, 1)
    plt.plot(stationary)
    plt.title("Stationary Series")
    plt.ylabel("Value")

    plt.subplot(3, 2, 2)
    plt.plot(trend)
    plt.title("Trend Series")
    plt.ylabel("Value")

    plt.subplot(3, 2, 3)
    plt.plot(changing_var)
    plt.title("Changing Variance Series")
    plt.ylabel("Value")

    plt.subplot(3, 2, 4)
    plt.plot(seasonal_nonstat)
    plt.title("Seasonal Series")
    plt.ylabel("Value")

    # Transformed series
    plt.subplot(3, 2, 5)
    plt.plot(trend_diff)
    plt.title("Differenced Trend")
    plt.ylabel("Value")

    plt.subplot(3, 2, 6)
    plt.plot(changing_var_diff)
    plt.title("Log + Differenced Variance")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.savefig("stationarity_analysis.png", dpi=300, bbox_inches="tight")
    print("✅ Stationarity analysis visualization saved as 'stationarity_analysis.png'")
    plt.close()


def demonstrate_forecasting():
    """Demonstrate time series forecasting methods."""
    print("Time Series Forecasting:")
    print("-" * 40)

    print("Forecasting involves predicting future values")
    print("based on historical patterns and trends.")
    print()

    # 1. Forecasting Methods
    print("1. FORECASTING METHODS:")
    print("-" * 30)

    forecasting_methods = {
        "Moving Average": "Simple average of recent observations",
        "Exponential Smoothing": "Weighted average with decreasing weights",
        "ARIMA": "Autoregressive Integrated Moving Average",
        "SARIMA": "Seasonal ARIMA for seasonal data",
        "Prophet": "Facebook's forecasting tool",
        "Neural Networks": "Deep learning approaches (LSTM, GRU)",
    }

    for method, description in forecasting_methods.items():
        print(f"  {method}: {description}")
    print()

    # 2. Generate Forecasting Dataset
    print("2. GENERATING FORECASTING DATASET:")
    print("-" * 35)

    np.random.seed(42)

    # Create a more realistic time series for forecasting
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="M")
    n = len(dates)

    # Generate series with trend, seasonality, and noise
    trend = 100 + 2 * np.arange(n)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = np.random.normal(0, 5, n)

    # Combine components
    sales_data = trend + seasonal + noise

    # Create DataFrame
    sales_df = pd.DataFrame({"date": dates, "sales": sales_data}).set_index("date")

    print(f"✅ Created sales dataset: {len(sales_df)} monthly observations")
    print(f"   Date range: {sales_df.index.min()} to {sales_df.index.max()}")
    print(
        f"   Sales range: {sales_df['sales'].min():.2f} to {sales_df['sales'].max():.2f}"
    )
    print()

    # 3. Simple Forecasting Methods
    print("3. SIMPLE FORECASTING METHODS:")
    print("-" * 35)

    # Split data
    train_size = int(len(sales_df) * 0.8)
    train_data = sales_df[:train_size]
    test_data = sales_df[train_size:]

    print(f"Training data: {len(train_data)} observations")
    print(f"Test data: {len(test_data)} observations")
    print()

    # Moving Average
    ma_window = 12  # 12-month moving average
    ma_forecast = train_data["sales"].rolling(window=ma_window).mean().iloc[-1]

    # Simple Exponential Smoothing
    alpha = 0.3
    ses_forecast = train_data["sales"].ewm(alpha=alpha).mean().iloc[-1]

    # Naive forecast (last value)
    naive_forecast = train_data["sales"].iloc[-1]

    print("Simple Forecasts (next month):")
    print(f"  Moving Average ({ma_window} months): {ma_forecast:.2f}")
    print(f"  Exponential Smoothing (α={alpha}): {ses_forecast:.2f}")
    print(f"  Naive (last value): {naive_forecast:.2f}")
    print()

    # 4. ARIMA Modeling
    print("4. ARIMA MODELING:")
    print("-" * 25)

    try:
        # Fit ARIMA model
        model = ARIMA(train_data["sales"], order=(1, 1, 1))
        fitted_model = model.fit()

        print("✅ ARIMA(1,1,1) Model Fitted:")
        print(f"   AIC: {fitted_model.aic:.2f}")
        print(f"   BIC: {fitted_model.bic:.2f}")
        print(f"   Log Likelihood: {fitted_model.llf:.2f}")
        print()

        # Make forecast
        forecast_steps = len(test_data)
        arima_forecast = fitted_model.forecast(steps=forecast_steps)

        print(f"ARIMA Forecast (next {forecast_steps} months):")
        for i, (date, value) in enumerate(zip(test_data.index, arima_forecast)):
            print(f"  {date.strftime('%Y-%m')}: {value:.2f}")
        print()

        # 5. Model Evaluation
        print("5. MODEL EVALUATION:")
        print("-" * 25)

        # Calculate metrics for different methods
        def calculate_metrics(actual, predicted):
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, predicted)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape}

        # For simple methods, use constant forecasts
        ma_forecasts = [ma_forecast] * len(test_data)
        ses_forecasts = [ses_forecast] * len(test_data)
        naive_forecasts = [naive_forecast] * len(test_data)

        # Calculate metrics
        metrics = {
            "Moving Average": calculate_metrics(test_data["sales"], ma_forecasts),
            "Exponential Smoothing": calculate_metrics(
                test_data["sales"], ses_forecasts
            ),
            "Naive": calculate_metrics(test_data["sales"], naive_forecasts),
            "ARIMA": calculate_metrics(test_data["sales"], arima_forecast),
        }

        print("Forecast Accuracy Metrics:")
        print(f"{'Method':<20} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'MAPE':<10}")
        print("-" * 60)

        for method, metric in metrics.items():
            print(
                f"{method:<20} {metric['MSE']:<10.2f} {metric['RMSE']:<10.2f} "
                f"{metric['MAE']:<10.2f} {metric['MAPE']:<10.2f}"
            )
        print()

        # 6. Visualization
        print("6. VISUALIZATION:")
        print("-" * 20)

        plt.figure(figsize=(15, 8))

        # Plot training data
        plt.plot(
            train_data.index, train_data["sales"], label="Training Data", linewidth=2
        )

        # Plot test data
        plt.plot(
            test_data.index, test_data["sales"], label="Actual Test Data", linewidth=2
        )

        # Plot forecasts
        plt.plot(
            test_data.index,
            arima_forecast,
            label="ARIMA Forecast",
            linewidth=2,
            linestyle="--",
        )
        plt.axhline(
            y=ma_forecast,
            color="red",
            linestyle=":",
            label=f"MA Forecast ({ma_window} months)",
        )
        plt.axhline(
            y=ses_forecast,
            color="green",
            linestyle=":",
            label=f"SES Forecast (α={alpha})",
        )
        plt.axhline(
            y=naive_forecast, color="orange", linestyle=":", label="Naive Forecast"
        )

        plt.title("Time Series Forecasting Comparison")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig("time_series_forecasting.png", dpi=300, bbox_inches="tight")
        print(
            "✅ Time series forecasting visualization saved as 'time_series_forecasting.png'"
        )
        plt.close()

        # Residual analysis
        plt.figure(figsize=(15, 5))

        residuals = test_data["sales"] - arima_forecast

        plt.subplot(1, 3, 1)
        plt.plot(test_data.index, residuals)
        plt.title("ARIMA Residuals")
        plt.ylabel("Residuals")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.hist(residuals, bins=20, alpha=0.7, edgecolor="black")
        plt.title("Residuals Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")

        plt.subplot(1, 3, 3)
        plt.scatter(arima_forecast, residuals, alpha=0.6)
        plt.axhline(y=0, color="red", linestyle="--")
        plt.title("Residuals vs Forecasts")
        plt.xlabel("Forecasts")
        plt.ylabel("Residuals")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("forecast_residuals.png", dpi=300, bbox_inches="tight")
        print("✅ Forecast residuals visualization saved as 'forecast_residuals.png'")
        plt.close()

    except Exception as e:
        print(f"⚠️  ARIMA modeling failed: {e}")
        print("   Continuing with simple forecasting methods...")
        print()

    print("Forecasting Summary:")
    print("✅ Implemented multiple forecasting methods")
    print("✅ Evaluated model performance with metrics")
    print("✅ Visualized forecasts and residuals")
    print("✅ Demonstrated ARIMA modeling process")


def main():
    print("=" * 80)
    print("CHAPTER 15: TIME SERIES ANALYSIS")
    print("=" * 80)
    print()

    # Section 15.1: Time Series Overview
    print("15.1 TIME SERIES OVERVIEW")
    print("-" * 35)
    demonstrate_ts_overview()

    # Section 15.2: Time Series Components
    print("\n15.2 TIME SERIES COMPONENTS")
    print("-" * 35)
    demonstrate_ts_components()

    # Section 15.3: Stationarity and Testing
    print("\n15.3 STATIONARITY AND TESTING")
    print("-" * 35)
    demonstrate_stationarity()

    # Section 15.4: Time Series Forecasting
    print("\n15.4 TIME SERIES FORECASTING")
    print("-" * 35)
    demonstrate_forecasting()

    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Time series overview and components")
    print("✅ Stationarity testing and analysis")
    print("✅ Forecasting methods and evaluation")
    print("✅ Practical time series applications")
    print()
    print("Next: Chapter 16 - Big Data Processing")
    print("=" * 80)


if __name__ == "__main__":
    main()
