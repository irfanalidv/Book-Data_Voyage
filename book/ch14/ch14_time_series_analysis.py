#!/usr/bin/env python3
"""
Chapter 14: Time Series Analysis
Data Voyage: Analyzing and Forecasting Time-Dependent Data

This script covers essential time series analysis concepts and techniques using real datasets.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

_THIS_DIR = Path(__file__).resolve().parent
_FIGURES_DIR = _THIS_DIR / "reports" / "figures"
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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

    # 1. LOADING REAL TIME SERIES DATA:
    print("1. LOADING REAL TIME SERIES DATA:")
    print("-" * 35)

    def load_real_time_series():
        """Load real time series data from various sources."""
        datasets = {}

        try:
            # Try to load real COVID-19 data as an example of time series
            print("  Loading real COVID-19 data (example of time series)...")
            covid_url = "https://disease.sh/v3/covid-19/historical/all?lastdays=365"
            response = requests.get(covid_url, timeout=10)
            if response.status_code == 200:
                covid_data = response.json()

                # Convert to time series format
                dates = list(covid_data["cases"].keys())
                cases = list(covid_data["cases"].values())
                deaths = list(covid_data["deaths"].values())

                # Create DataFrame
                covid_df = pd.DataFrame(
                    {"date": pd.to_datetime(dates), "cases": cases, "deaths": deaths}
                )

                # Use cases as primary time series
                covid_ts = covid_df.set_index("date")["cases"]

                # Sample for demonstration (every 7 days to reduce noise)
                covid_ts_sampled = covid_ts[::7]

                datasets["covid"] = covid_ts_sampled
                datasets["covid_daily"] = covid_ts
                print(f"    ✅ COVID-19 data: {len(covid_ts_sampled)} observations")
                print(
                    f"     Date range: {covid_ts_sampled.index.min().date()} to {covid_ts_sampled.index.max().date()}"
                )
                print(
                    f"     Cases range: {covid_ts_sampled.min():,.0f} to {covid_ts_sampled.max():,.0f}"
                )
            else:
                raise Exception("Failed to fetch COVID data")

        except Exception as e:
            print(f"    ⚠️  Could not load COVID data: {e}")
            print("    📝 Creating realistic time series simulation...")

            # Create realistic time series simulation
            dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
            n = len(dates)

            # Realistic COVID-like patterns
            base_cases = 1000
            trend = base_cases + 50 * np.arange(n) + 0.1 * np.arange(n) ** 2
            seasonal = 200 * np.sin(2 * np.pi * np.arange(n) / 365.25)  # Annual cycle
            weekly = 50 * np.sin(2 * np.pi * np.arange(n) / 7)  # Weekly cycle
            noise = np.random.normal(0, 100, n)

            covid_simulation = trend + seasonal + weekly + noise
            covid_simulation = np.clip(covid_simulation, 0, None)  # No negative cases

            datasets["covid"] = pd.Series(covid_simulation, index=dates)
            print(f"    ✅ COVID-19 simulation: {len(dates)} observations")
            print(f"     Date range: {dates.min().date()} to {dates.max().date()}")
            print(
                f"     Cases range: {covid_simulation.min():,.0f} to {covid_simulation.max():,.0f}"
            )

        try:
            # Load weather data (simulated but realistic)
            print("Loading weather data...")
            dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
            n = len(dates)

            # Realistic temperature patterns
            base_temp = 15  # Average temperature
            annual_cycle = 20 * np.sin(2 * np.pi * np.arange(n) / 365.25)  # Seasonal variation
            weekly_cycle = 3 * np.sin(2 * np.pi * np.arange(n) / 7)  # Weekly patterns
            trend_change = 0.01 * np.arange(n)  # Climate change trend
            weather_noise = np.random.normal(0, 3, n)  # Daily variation

            temperatures = base_temp + annual_cycle + weekly_cycle + trend_change + weather_noise
            datasets["temperatures"] = pd.Series(temperatures, index=dates)
            print(f"  ✅ Temperature data: {len(dates)} observations")
            print(
                f"     Temperature range: {temperatures.min():.1f}°C to {temperatures.max():.1f}°C"
            )
        except Exception as e:
            print(f"  ⚠️  Error creating weather data: {e}")

        try:
            # Load economic data (simulated but realistic)
            print("Loading economic data...")
            dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="M")
            n = len(dates)

            # Realistic GDP growth patterns
            base_growth = 2.5  # Base annual growth rate
            business_cycle = 1.5 * np.sin(2 * np.pi * np.arange(n) / 48)  # 4-year business cycle
            seasonal_economic = 0.5 * np.sin(2 * np.pi * np.arange(n) / 12)  # Quarterly patterns
            trend_growth = 0.02 * np.arange(n)  # Long-term growth trend
            economic_noise = np.random.normal(0, 0.3, n)  # Economic uncertainty

            gdp_growth = (
                base_growth + business_cycle + seasonal_economic + trend_growth + economic_noise
            )
            datasets["gdp_growth"] = pd.Series(gdp_growth, index=dates)
            print(f"  ✅ GDP growth data: {len(dates)} observations")
            print(f"     Growth range: {gdp_growth.min():.1f}% to {gdp_growth.max():.1f}%")
        except Exception as e:
            print(f"  ⚠️  Error creating economic data: {e}")

        return datasets

    # Load real time series data
    real_datasets = load_real_time_series()
    series, source = _daily_series_for_decomposition(real_datasets)

    print(f"✅ Series for decomposition: {len(series)} daily observations ({source})")
    print(f"   Date range: {series.index.min().date()} to {series.index.max().date()}")
    print()

    # 2. Seasonal decomposition - estimated from the data, never drawn by hand.
    print("2. SEASONAL DECOMPOSITION (period = 7 days):")
    print("-" * 45)
    # Daily case reports carry a weekly cycle (weekend under-reporting), so the
    # period comes from domain knowledge: 7. Two full cycles is the minimum
    # seasonal_decompose accepts; we have dozens.
    decomposition = seasonal_decompose(series, model="additive", period=7)
    resid = decomposition.resid.dropna()
    seasonal_amp = decomposition.seasonal.max() - decomposition.seasonal.min()
    print(
        f"  Trend range:            {decomposition.trend.min():,.0f} to "
        f"{decomposition.trend.max():,.0f}"
    )
    print(f"  Weekly seasonal swing:  {seasonal_amp:,.0f} (peak-to-trough)")
    print(f"  Residual std:           {resid.std():,.0f}")
    print()

    # 3. Visualization
    print("3. VISUALIZATION:")
    print("-" * 20)
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    panels = [
        (series, "Observed (daily new cases)", "#2196F3"),
        (decomposition.trend, "Trend (7-day centred moving average)", "#F44336"),
        (decomposition.seasonal, "Weekly seasonal component", "#4CAF50"),
        (decomposition.resid, "Residual", "#9C27B0"),
    ]
    for ax, (data, title, colour) in zip(axes, panels):
        ax.plot(data.index, data.values, lw=0.8, color=colour)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(_FIGURES_DIR / "time_series_components.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Saved time_series_components.png")

    # Zoom on eight weeks so the weekly pattern is visible to the eye.
    window = series.iloc[-56:]
    seasonal_window = decomposition.seasonal.iloc[-56:]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(window.index, window.values, marker="o", ms=3, lw=1, color="#2196F3")
    axes[0].set_title("Last eight weeks — observed", fontsize=11, fontweight="bold")
    axes[1].bar(seasonal_window.index, seasonal_window.values, color="#4CAF50", width=0.8)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title(
        "Weekly seasonal effect (same seven values repeating)", fontsize=11, fontweight="bold"
    )
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_FIGURES_DIR / "seasonal_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Saved seasonal_decomposition.png")
    print()


def _daily_series_for_decomposition(real_datasets: dict) -> tuple[pd.Series, str]:
    """Return a daily series with a genuine weekly cycle, plus a label for its source.

    The COVID feed publishes cumulative totals; daily new cases are the first
    difference. Without network access, fall back to a seeded synthetic series
    with a known weekly pattern so the decomposition still has something real
    to recover.
    """
    covid = real_datasets.get("covid_daily")
    if covid is not None and len(covid) >= 28:
        daily = covid.diff().dropna().clip(lower=0)
        daily = daily.asfreq("D").interpolate()
        return daily, "COVID-19 daily new cases, disease.sh"

    rng = np.random.default_rng(14)
    dates = pd.date_range("2022-01-01", periods=365, freq="D")
    t = np.arange(len(dates))
    trend = 5_000 + 3_000 * np.exp(-(((t - 120) / 60) ** 2))
    weekly = np.array([400, 350, 300, 250, 100, -600, -800])[dates.dayofweek]
    values = trend + weekly + rng.normal(0, 150, len(dates))
    return pd.Series(values, index=dates), "synthetic fallback with a known weekly cycle"


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
    seasonal_nonstat = 10 * np.sin(2 * np.pi * np.arange(n) / 50) + np.random.normal(0, 1, n)

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

            print("  ADF Test:")
            print(f"    Statistic: {adf_stat:.4f}")
            print(f"    p-value: {adf_pvalue:.4f}")
            print(f"    Critical values: {adf_critical}")

            if adf_pvalue < 0.05:
                print("    Result: Stationary (p < 0.05)")
            else:
                print("    Result: Non-stationary (p >= 0.05)")
        except Exception as e:
            print(f"    ADF test failed: {e}")

        # KPSS Test
        try:
            kpss_result = kpss(series)
            kpss_stat = kpss_result[0]
            kpss_pvalue = kpss_result[1]
            kpss_critical = kpss_result[3]

            print("  KPSS Test:")
            print(f"    Statistic: {kpss_stat:.4f}")
            print(f"    p-value: {kpss_pvalue:.4f}")
            print(f"    Critical values: {kpss_critical}")

            if kpss_pvalue > 0.05:
                print("    Result: Stationary (p > 0.05)")
            else:
                print("    Result: Non-stationary (p <= 0.05)")
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
    print(f"   Changing Variance → Log + Differenced: {len(changing_var_diff)} observations")
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
    plt.savefig(_FIGURES_DIR / "stationarity_analysis.png", dpi=300, bbox_inches="tight")
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
    print(f"   Sales range: {sales_df['sales'].min():.2f} to {sales_df['sales'].max():.2f}")
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
            "Exponential Smoothing": calculate_metrics(test_data["sales"], ses_forecasts),
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
        plt.plot(train_data.index, train_data["sales"], label="Training Data", linewidth=2)

        # Plot test data
        plt.plot(test_data.index, test_data["sales"], label="Actual Test Data", linewidth=2)

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
        plt.axhline(y=naive_forecast, color="orange", linestyle=":", label="Naive Forecast")

        plt.title("Time Series Forecasting Comparison")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(_FIGURES_DIR / "time_series_forecasting.png", dpi=300, bbox_inches="tight")
        print("✅ Time series forecasting visualization saved as 'time_series_forecasting.png'")
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
        plt.savefig(_FIGURES_DIR / "forecast_residuals.png", dpi=300, bbox_inches="tight")
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
    print("CHAPTER 14: TIME SERIES ANALYSIS")
    print("=" * 80)
    print()

    # Section 14.1: Time Series Overview
    print("14.1 TIME SERIES OVERVIEW")
    print("-" * 35)
    demonstrate_ts_overview()

    # Section 14.2: Time Series Components
    print("\n14.2 TIME SERIES COMPONENTS")
    print("-" * 35)
    demonstrate_ts_components()

    # Section 14.3: Stationarity and Testing
    print("\n14.3 STATIONARITY AND TESTING")
    print("-" * 35)
    demonstrate_stationarity()

    # Section 14.4: Time Series Forecasting
    print("\n14.4 TIME SERIES FORECASTING")
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
    print("Next: Chapter 15 — Scaling Python")
    print("=" * 80)


if __name__ == "__main__":
    main()
