"""
Plotly Dash dashboard for visualizing volatility forecast results.

Panels:
    1. Predicted vs Actual realized volatility over time
    2. Model comparison metrics (MAE, RMSE, directional accuracy)
    3. Feature importance bar chart
    4. Residual distribution
"""

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from dash import Dash, html, dcc, Input, Output
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

from src.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def create_performance_figure(predictions_df: pd.DataFrame) -> go.Figure:
    """Create predicted vs actual time-series chart."""
    if predictions_df.empty:
        return go.Figure().update_layout(title="No predictions available")

    df = predictions_df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["actual"],
        mode="lines", name="Actual Vol",
        line=dict(color="#636EFA", width=1.5),
    ))

    if "pred_lightgbm" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["pred_lightgbm"],
            mode="lines", name="LightGBM Prediction",
            line=dict(color="#EF553B", width=1.5, dash="dash"),
        ))

    if "pred_naive" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["pred_naive"],
            mode="lines", name="Naive Baseline",
            line=dict(color="#00CC96", width=1, dash="dot"),
            opacity=0.6,
        ))

    fig.update_layout(
        title="1-Hour-Ahead Realized Volatility: Predicted vs Actual",
        xaxis_title="Time (UTC)",
        yaxis_title="Realized Volatility (|log return|)",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=450,
    )
    return fig


def create_metrics_figure(metrics: dict) -> go.Figure:
    """Create bar chart comparing model metrics."""
    if not metrics:
        return go.Figure().update_layout(title="No metrics available")

    models = list(metrics.keys())
    mae_values = [metrics[m]["mae"] for m in models]
    rmse_values = [metrics[m]["rmse"] for m in models]
    dir_acc = [metrics[m]["directional_accuracy"] * 100 for m in models]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("MAE (lower is better)", "RMSE (lower is better)", "Directional Accuracy %"),
    )

    colors = ["#636EFA", "#EF553B", "#00CC96"]

    fig.add_trace(go.Bar(
        x=models, y=mae_values,
        marker_color=colors[:len(models)], showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=models, y=rmse_values,
        marker_color=colors[:len(models)], showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=models, y=dir_acc,
        marker_color=colors[:len(models)], showlegend=False,
    ), row=1, col=3)

    fig.update_layout(
        title="Model Comparison",
        template="plotly_white",
        height=350,
    )
    return fig


def create_feature_importance_figure(importance_df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart of feature importances."""
    if importance_df.empty:
        return go.Figure().update_layout(title="No feature importance data")

    top_n = importance_df.head(15)

    fig = go.Figure(go.Bar(
        x=top_n["importance"],
        y=top_n["feature"],
        orientation="h",
        marker_color="#AB63FA",
    ))

    fig.update_layout(
        title="Top 15 Features by Gain",
        xaxis_title="Importance (Gain)",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=400,
    )
    return fig


def create_residual_figure(predictions_df: pd.DataFrame) -> go.Figure:
    """Create residual distribution histogram."""
    if predictions_df.empty or "pred_lightgbm" not in predictions_df.columns:
        return go.Figure().update_layout(title="No residual data")

    residuals = predictions_df["actual"] - predictions_df["pred_lightgbm"]

    fig = go.Figure(go.Histogram(
        x=residuals,
        nbinsx=50,
        marker_color="#FFA15A",
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.add_vline(x=residuals.mean(), line_dash="dot", line_color="red",
                  annotation_text=f"Mean: {residuals.mean():.6f}")

    fig.update_layout(
        title="Prediction Residuals (Actual - Predicted)",
        xaxis_title="Residual",
        yaxis_title="Count",
        template="plotly_white",
        height=350,
    )
    return fig


def build_dash_app(
    predictions_df: pd.DataFrame,
    metrics: dict,
    feature_importance: pd.DataFrame,
) -> "Dash":
    """
    Build and return a Dash application with all dashboard panels.

    Args:
        predictions_df: Backtest predictions (timestamp, actual, pred_*)
        metrics: Dict of model_name -> metric_dict
        feature_importance: DataFrame with feature/importance columns

    Returns:
        Configured Dash app (call app.run_server() to start)
    """
    if not HAS_DASH:
        raise ImportError("dash is required. Install with: pip install dash")

    app = Dash(__name__)

    app.layout = html.Div([
        html.H1(
            "Kalshi BTC Perps — Volatility Forecast Dashboard",
            style={"textAlign": "center", "fontFamily": "system-ui", "marginBottom": "10px"}
        ),
        html.P(
            "1-hour-ahead realized volatility forecasting using funding rates, "
            "volume, and market microstructure features",
            style={"textAlign": "center", "color": "#666", "marginBottom": "30px"}
        ),

        # Predicted vs Actual
        dcc.Graph(
            id="performance-chart",
            figure=create_performance_figure(predictions_df),
        ),

        # Metrics comparison
        dcc.Graph(
            id="metrics-chart",
            figure=create_metrics_figure(metrics),
        ),

        html.Div([
            # Feature importance
            html.Div([
                dcc.Graph(
                    id="importance-chart",
                    figure=create_feature_importance_figure(feature_importance),
                )
            ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),

            # Residuals
            html.Div([
                dcc.Graph(
                    id="residual-chart",
                    figure=create_residual_figure(predictions_df),
                )
            ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
        ]),

        # Summary stats
        html.Div([
            html.H3("Summary Statistics", style={"marginTop": "20px"}),
            html.Table([
                html.Tr([html.Th("Model"), html.Th("MAE"), html.Th("RMSE"),
                         html.Th("Dir. Accuracy"), html.Th("N Predictions")])
            ] + [
                html.Tr([
                    html.Td(name),
                    html.Td(f"{m['mae']:.6f}"),
                    html.Td(f"{m['rmse']:.6f}"),
                    html.Td(f"{m['directional_accuracy']:.1%}"),
                    html.Td(str(m["n_predictions"])),
                ])
                for name, m in metrics.items()
            ], style={"margin": "auto", "borderCollapse": "collapse"})
        ], style={"textAlign": "center", "marginBottom": "40px"}),

    ], style={"maxWidth": "1200px", "margin": "auto", "padding": "20px"})

    return app


# ─── CLI Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.database import Database
    from src.features import build_feature_matrix
    from src.model import backtest

    db = Database()
    candles = db.get_candlesticks_df(interval_minutes=60)
    funding = db.get_funding_rates_df()

    if candles.empty:
        print("No data in database. Run the pipeline first:")
        print("  python run_pipeline.py")
    else:
        features = build_feature_matrix(candles, funding)
        results = backtest(features)

        app = build_dash_app(
            predictions_df=results["predictions"],
            metrics=results["metrics"],
            feature_importance=results["feature_importance"],
        )
        print("\nDashboard starting at http://127.0.0.1:8050")
        app.run_server(debug=True, port=8050)
