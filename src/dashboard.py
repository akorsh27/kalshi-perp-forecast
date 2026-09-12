"""
Plotly Dash dashboard for visualizing volatility forecast results.

Tabs:
    1. Overview     — KPIs, price chart, and predicted vs actual vol
    2. Model Performance — Detailed metrics comparison, residuals, error over time
    3. Feature Analysis — Feature importance, correlations, distributions
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from dash import Dash, html, dcc, Input, Output, callback
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

from src.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ─── Color Palette ───────────────────────────────────────────────────────

COLORS = {
    "bg": "#0f1117",
    "card": "#1a1c23",
    "card_border": "#2d2f36",
    "text": "#e6e6e6",
    "text_muted": "#8b8d97",
    "accent": "#6366f1",  # indigo
    "success": "#22c55e",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "blue": "#3b82f6",
    "purple": "#a855f7",
    "cyan": "#06b6d4",
    "orange": "#f97316",
}

PLOT_TEMPLATE = "plotly_dark"
PLOT_BG = "rgba(26, 28, 35, 0)"
PLOT_PAPER_BG = "rgba(26, 28, 35, 0)"
PLOT_GRID = "rgba(255,255,255,0.06)"
PLOT_FONT = dict(family="Inter, system-ui, sans-serif", color=COLORS["text"])

# ─── Styles ──────────────────────────────────────────────────────────────

STYLE_PAGE = {
    "backgroundColor": COLORS["bg"],
    "minHeight": "100vh",
    "fontFamily": "Inter, system-ui, -apple-system, sans-serif",
    "color": COLORS["text"],
    "padding": "0",
    "margin": "0",
}

STYLE_CONTAINER = {
    "maxWidth": "1320px",
    "margin": "0 auto",
    "padding": "24px 32px",
}

STYLE_CARD = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "12px",
    "border": f"1px solid {COLORS['card_border']}",
    "padding": "24px",
    "marginBottom": "16px",
}

STYLE_KPI_CARD = {
    **STYLE_CARD,
    "textAlign": "center",
    "flex": "1",
    "minWidth": "180px",
}


def _plot_layout(**overrides):
    """Base layout options for all charts."""
    base = dict(
        template=PLOT_TEMPLATE,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_PAPER_BG,
        font=PLOT_FONT,
        xaxis=dict(gridcolor=PLOT_GRID, zeroline=False),
        yaxis=dict(gridcolor=PLOT_GRID, zeroline=False),
        margin=dict(l=48, r=24, t=48, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
    )
    base.update(overrides)
    return base


# ─── Figure Builders ─────────────────────────────────────────────────────

def create_performance_figure(predictions_df: pd.DataFrame) -> go.Figure:
    """Predicted vs Actual volatility time-series."""
    if predictions_df.empty:
        fig = go.Figure()
        fig.update_layout(**_plot_layout(height=420, title="No predictions available"))
        return fig

    df = predictions_df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["actual"],
        mode="lines", name="Actual",
        line=dict(color=COLORS["blue"], width=1.2),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
    ))

    if "pred_lightgbm" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["pred_lightgbm"],
            mode="lines", name="LightGBM",
            line=dict(color=COLORS["orange"], width=2, dash="dot"),
        ))

    if "pred_ma" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["pred_ma"],
            mode="lines", name="MA Baseline",
            line=dict(color=COLORS["text_muted"], width=1, dash="dash"),
            opacity=0.5,
            visible="legendonly",
        ))

    fig.update_layout(**_plot_layout(
        height=420,
        title=dict(text="Predicted vs Actual Realized Volatility", x=0, font=dict(size=15)),
        xaxis_title="",
        yaxis_title="Volatility (|log return|)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            gridcolor=PLOT_GRID, zeroline=False,
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor=COLORS["card"],
                activecolor=COLORS["accent"],
                font=dict(color=COLORS["text"]),
            ),
            rangeslider=dict(visible=False),
        ),
    ))
    return fig


def create_price_figure(candles_df: pd.DataFrame) -> go.Figure:
    """OHLC price candlestick chart."""
    if candles_df.empty:
        fig = go.Figure()
        fig.update_layout(**_plot_layout(height=320, title="No price data"))
        return fig

    df = candles_df.copy().tail(720)  # last 30 days of hourly
    df["datetime"] = pd.to_datetime(df["end_period_dt"])

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.02,
    )

    fig.add_trace(go.Candlestick(
        x=df["datetime"],
        open=df["price_open"], high=df["price_high"],
        low=df["price_low"], close=df["price_close"],
        increasing_line_color=COLORS["success"],
        decreasing_line_color=COLORS["danger"],
        name="Price",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df["datetime"], y=df["volume"],
        marker_color=COLORS["accent"],
        opacity=0.4,
        name="Volume",
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(**_plot_layout(
        height=380,
        title=dict(text="BTC Perps Price & Volume (Last 30 Days)", x=0, font=dict(size=15)),
        xaxis_rangeslider_visible=False,
        xaxis2=dict(gridcolor=PLOT_GRID),
        yaxis=dict(gridcolor=PLOT_GRID, title="Price ($)"),
        yaxis2=dict(gridcolor=PLOT_GRID, title="Volume"),
    ))
    return fig


def create_metrics_figure(metrics: dict) -> go.Figure:
    """Side-by-side model comparison bars."""
    if not metrics:
        fig = go.Figure()
        fig.update_layout(**_plot_layout(height=350, title="No metrics"))
        return fig

    model_names = list(metrics.keys())
    display_names = {"naive": "Naive", "ma": "Moving Avg", "lightgbm": "LightGBM"}
    labels = [display_names.get(m, m) for m in model_names]
    colors = [COLORS["text_muted"], COLORS["cyan"], COLORS["accent"]]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "<b>MAE</b> <span style='color:#8b8d97;font-size:11px'>(lower = better)</span>",
            "<b>RMSE</b> <span style='color:#8b8d97;font-size:11px'>(lower = better)</span>",
            "<b>Dir. Accuracy</b> <span style='color:#8b8d97;font-size:11px'>(higher = better)</span>",
        ],
        horizontal_spacing=0.08,
    )

    mae = [metrics[m]["mae"] for m in model_names]
    rmse = [metrics[m]["rmse"] for m in model_names]
    da = [metrics[m]["directional_accuracy"] * 100 for m in model_names]

    for col, (vals, fmt) in enumerate([(mae, ".5f"), (rmse, ".5f"), (da, ".1f")], 1):
        fig.add_trace(go.Bar(
            x=labels, y=vals,
            marker=dict(color=colors[:len(labels)], cornerradius=6),
            text=[f"{v:{fmt}}{'%' if col == 3 else ''}" for v in vals],
            textposition="outside",
            textfont=dict(size=12, color=COLORS["text"]),
            showlegend=False,
        ), row=1, col=col)

    fig.update_layout(**_plot_layout(
        height=340,
        showlegend=False,
    ))
    fig.update_annotations(font=dict(size=13, color=COLORS["text"]))
    return fig


def create_feature_importance_figure(importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of top features."""
    if importance_df.empty:
        fig = go.Figure()
        fig.update_layout(**_plot_layout(height=420, title="No feature importance"))
        return fig

    top = importance_df.head(12).iloc[::-1]  # reverse for bottom-up
    max_val = top["importance"].max()

    fig = go.Figure(go.Bar(
        x=top["importance"],
        y=top["feature"].str.replace("_", " ").str.title(),
        orientation="h",
        marker=dict(
            color=top["importance"],
            colorscale=[[0, COLORS["cyan"]], [1, COLORS["accent"]]],
            cornerradius=4,
        ),
        text=[f"{v:.4f}" for v in top["importance"]],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text_muted"]),
    ))

    fig.update_layout(**_plot_layout(
        height=420,
        title=dict(text="Top Features by Gain", x=0, font=dict(size=15)),
        xaxis_title="",
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        xaxis=dict(showticklabels=False, gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=160, r=60, t=48, b=24),
    ))
    return fig


def create_residual_figure(predictions_df: pd.DataFrame) -> go.Figure:
    """Residual distribution with KDE-style histogram."""
    if predictions_df.empty or "pred_lightgbm" not in predictions_df.columns:
        fig = go.Figure()
        fig.update_layout(**_plot_layout(height=380, title="No residual data"))
        return fig

    residuals = predictions_df["actual"] - predictions_df["pred_lightgbm"]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=60,
        marker=dict(
            color=COLORS["accent"],
            line=dict(width=0),
        ),
        opacity=0.7,
        name="Residuals",
    ))

    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["text_muted"], line_width=1)
    fig.add_vline(
        x=residuals.mean(), line_dash="dot", line_color=COLORS["warning"], line_width=1.5,
        annotation_text=f"Mean: {residuals.mean():.5f}",
        annotation_font=dict(color=COLORS["warning"], size=11),
        annotation_bgcolor=COLORS["card"],
    )

    fig.update_layout(**_plot_layout(
        height=380,
        title=dict(text="Prediction Residuals", x=0, font=dict(size=15)),
        xaxis_title="Residual (Actual − Predicted)",
        yaxis_title="Count",
        showlegend=False,
    ))
    return fig


def create_error_over_time(predictions_df: pd.DataFrame) -> go.Figure:
    """Rolling MAE over time to show if model degrades."""
    if predictions_df.empty or "pred_lightgbm" not in predictions_df.columns:
        fig = go.Figure()
        fig.update_layout(**_plot_layout(height=320, title="No data"))
        return fig

    df = predictions_df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["abs_error"] = (df["actual"] - df["pred_lightgbm"]).abs()
    df["rolling_mae"] = df["abs_error"].rolling(72, min_periods=12).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["abs_error"],
        mode="lines", name="Absolute Error",
        line=dict(color=COLORS["danger"], width=0.8),
        opacity=0.3,
    ))
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["rolling_mae"],
        mode="lines", name="Rolling MAE (72h)",
        line=dict(color=COLORS["warning"], width=2),
    ))

    fig.update_layout(**_plot_layout(
        height=320,
        title=dict(text="Prediction Error Over Time", x=0, font=dict(size=15)),
        xaxis_title="",
        yaxis_title="Absolute Error",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    ))
    return fig


# ─── KPI Cards ───────────────────────────────────────────────────────────

def _kpi_card(label: str, value: str, subtitle: str = "", color: str = COLORS["accent"]):
    """Render a single KPI stat card."""
    return html.Div([
        html.Div(label, style={
            "fontSize": "12px", "fontWeight": "500",
            "color": COLORS["text_muted"], "textTransform": "uppercase",
            "letterSpacing": "0.05em", "marginBottom": "8px",
        }),
        html.Div(value, style={
            "fontSize": "28px", "fontWeight": "700",
            "color": color, "lineHeight": "1.1",
        }),
        html.Div(subtitle, style={
            "fontSize": "12px", "color": COLORS["text_muted"],
            "marginTop": "6px",
        }) if subtitle else None,
    ], style=STYLE_KPI_CARD)


def build_kpi_row(metrics: dict, predictions_df: pd.DataFrame, n_candles: int):
    """Build the row of KPI cards for the overview tab."""
    lgbm = metrics.get("lightgbm", {})
    naive = metrics.get("naive", {})
    mae_improvement = (
        ((naive.get("mae", 0) - lgbm.get("mae", 0)) / naive.get("mae", 1)) * 100
        if naive.get("mae") else 0
    )

    date_range = ""
    if not predictions_df.empty:
        ts_min = pd.to_datetime(predictions_df["timestamp"].min(), unit="s", utc=True)
        ts_max = pd.to_datetime(predictions_df["timestamp"].max(), unit="s", utc=True)
        date_range = f"{ts_min.strftime('%b %d')} – {ts_max.strftime('%b %d, %Y')}"

    return html.Div([
        _kpi_card("LightGBM MAE", f"{lgbm.get('mae', 0):.5f}",
                   f"{mae_improvement:+.1f}% vs naive", COLORS["accent"]),
        _kpi_card("Dir. Accuracy", f"{lgbm.get('directional_accuracy', 0):.1%}",
                   "LightGBM", COLORS["success"] if lgbm.get("directional_accuracy", 0) > 0.5 else COLORS["warning"]),
        _kpi_card("Backtest Size", f"{lgbm.get('n_predictions', 0):,}",
                   "predictions", COLORS["cyan"]),
        _kpi_card("Training Data", f"{n_candles:,}",
                   "hourly candles", COLORS["purple"]),
        _kpi_card("Date Range", date_range.split(" – ")[0] if date_range else "—",
                   date_range.split(" – ")[1] if " – " in date_range else "", COLORS["text"]),
    ], style={
        "display": "flex", "gap": "12px", "marginBottom": "16px",
        "flexWrap": "wrap",
    })


# ─── Tab Builders ────────────────────────────────────────────────────────

def _tab_style(selected=False):
    base = {
        "padding": "12px 24px",
        "borderRadius": "8px 8px 0 0",
        "border": "none",
        "fontWeight": "600",
        "fontSize": "14px",
        "cursor": "pointer",
        "backgroundColor": COLORS["card"] if selected else "transparent",
        "color": COLORS["text"] if selected else COLORS["text_muted"],
        "borderBottom": f"2px solid {COLORS['accent']}" if selected else "2px solid transparent",
    }
    return base


def build_overview_tab(predictions_df, metrics, candles_df, n_candles):
    return html.Div([
        build_kpi_row(metrics, predictions_df, n_candles),
        html.Div([
            html.Div([
                dcc.Graph(id="perf-chart", figure=create_performance_figure(predictions_df),
                          config={"displayModeBar": False}),
            ], style={**STYLE_CARD, "flex": "1.2", "minWidth": "500px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
        html.Div([
            dcc.Graph(id="price-chart", figure=create_price_figure(candles_df),
                      config={"displayModeBar": False}),
        ], style=STYLE_CARD),
    ])


def build_model_tab(predictions_df, metrics):
    return html.Div([
        html.Div([
            dcc.Graph(id="metrics-chart", figure=create_metrics_figure(metrics),
                      config={"displayModeBar": False}),
        ], style=STYLE_CARD),
        html.Div([
            html.Div([
                dcc.Graph(id="error-time-chart", figure=create_error_over_time(predictions_df),
                          config={"displayModeBar": False}),
            ], style={**STYLE_CARD, "flex": "1"}),
            html.Div([
                dcc.Graph(id="residual-chart", figure=create_residual_figure(predictions_df),
                          config={"displayModeBar": False}),
            ], style={**STYLE_CARD, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
        # Summary table
        html.Div([
            html.Table([
                html.Thead(html.Tr([
                    html.Th(h, style={
                        "padding": "12px 20px", "textAlign": "left",
                        "borderBottom": f"1px solid {COLORS['card_border']}",
                        "color": COLORS["text_muted"], "fontSize": "12px",
                        "textTransform": "uppercase", "letterSpacing": "0.05em",
                    }) for h in ["Model", "MAE", "RMSE", "Dir. Accuracy", "Predictions"]
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td({"naive": "Naive (Last Value)", "ma": "Moving Average (12h)", "lightgbm": "LightGBM"}.get(name, name), style=_td_style(bold=True)),
                        html.Td(f"{m['mae']:.6f}", style=_td_style()),
                        html.Td(f"{m['rmse']:.6f}", style=_td_style()),
                        html.Td(f"{m['directional_accuracy']:.1%}", style=_td_style(
                            color=COLORS["success"] if m["directional_accuracy"] > 0.5 else COLORS["danger"]
                        )),
                        html.Td(f"{m['n_predictions']:,}", style=_td_style()),
                    ], style={"borderBottom": f"1px solid {COLORS['card_border']}"})
                    for name, m in metrics.items()
                ]),
            ], style={"width": "100%", "borderCollapse": "collapse"}),
        ], style=STYLE_CARD),
    ])


def build_features_tab(feature_importance, predictions_df):
    return html.Div([
        html.Div([
            dcc.Graph(id="importance-chart", figure=create_feature_importance_figure(feature_importance),
                      config={"displayModeBar": False}),
        ], style=STYLE_CARD),
        html.Div([
            html.H3("Feature Descriptions", style={
                "fontSize": "15px", "fontWeight": "600", "marginBottom": "16px",
            }),
            html.Div([
                _feature_row("Log Return 1h", "Hourly log return — primary momentum signal"),
                _feature_row("Realized Vol 24h", "Standard deviation of returns over 24 hours — captures vol regime"),
                _feature_row("Candle Range", "High-low range relative to close — intrabar volatility proxy"),
                _feature_row("Funding Rate", "Periodic funding rate from Kalshi — premium/discount signal"),
                _feature_row("OI Change", "Hour-over-hour open interest change — position buildup/unwind"),
                _feature_row("Volume Ratio", "Current volume vs 24h moving average — activity spike detection"),
                _feature_row("Spread", "Ask-bid spread relative to price — liquidity proxy"),
                _feature_row("Hour Sin/Cos", "Cyclical encoding of hour-of-day — time-of-day effects"),
            ]),
        ], style=STYLE_CARD),
    ])


def _feature_row(name, desc):
    return html.Div([
        html.Span(name, style={
            "fontWeight": "600", "fontSize": "13px",
            "color": COLORS["accent"], "minWidth": "160px", "display": "inline-block",
        }),
        html.Span(desc, style={
            "fontSize": "13px", "color": COLORS["text_muted"],
        }),
    ], style={"padding": "8px 0", "borderBottom": f"1px solid {COLORS['card_border']}"})


def _td_style(bold=False, color=None):
    return {
        "padding": "12px 20px",
        "fontSize": "14px",
        "fontWeight": "600" if bold else "400",
        "color": color or COLORS["text"],
    }


# ─── Main App Builder ────────────────────────────────────────────────────

def build_dash_app(
    predictions_df: pd.DataFrame,
    metrics: dict,
    feature_importance: pd.DataFrame,
    candles_df: pd.DataFrame = None,
) -> "Dash":
    """
    Build and return a tabbed Dash application.

    Args:
        predictions_df: Backtest predictions (timestamp, actual, pred_*)
        metrics: Dict of model_name -> metric_dict
        feature_importance: DataFrame with feature/importance columns
        candles_df: Raw candlestick data for price chart (optional)

    Returns:
        Configured Dash app
    """
    if not HAS_DASH:
        raise ImportError("dash is required. Install with: pip install dash")

    if candles_df is None:
        candles_df = pd.DataFrame()

    n_candles = len(candles_df)

    app = Dash(__name__)
    app.title = "Kalshi BTC Perps — Volatility Forecast"

    app.layout = html.Div([
        # Header
        html.Div([
            html.Div([
                html.H1("Kalshi BTC Perps", style={
                    "fontSize": "22px", "fontWeight": "700", "margin": "0",
                    "background": f"linear-gradient(135deg, {COLORS['accent']}, {COLORS['cyan']})",
                    "WebkitBackgroundClip": "text",
                    "WebkitTextFillColor": "transparent",
                }),
                html.Span("Volatility Forecast Dashboard", style={
                    "fontSize": "14px", "color": COLORS["text_muted"],
                    "marginLeft": "12px",
                }),
            ], style={"display": "flex", "alignItems": "baseline", "gap": "4px"}),
            html.Div(
                f"Last updated: {datetime.now(timezone.utc).strftime('%b %d, %Y %H:%M UTC')}",
                style={"fontSize": "12px", "color": COLORS["text_muted"]},
            ),
        ], style={
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            "padding": "16px 32px",
            "borderBottom": f"1px solid {COLORS['card_border']}",
        }),

        # Content
        html.Div([
            dcc.Tabs(
                id="main-tabs",
                value="overview",
                children=[
                    dcc.Tab(label="Overview", value="overview",
                            style=_tab_style(), selected_style=_tab_style(True)),
                    dcc.Tab(label="Model Performance", value="model",
                            style=_tab_style(), selected_style=_tab_style(True)),
                    dcc.Tab(label="Feature Analysis", value="features",
                            style=_tab_style(), selected_style=_tab_style(True)),
                ],
                style={"marginBottom": "20px", "borderBottom": f"1px solid {COLORS['card_border']}"},
                colors={
                    "border": "transparent",
                    "primary": COLORS["accent"],
                    "background": "transparent",
                },
            ),
            html.Div(id="tab-content"),
        ], style=STYLE_CONTAINER),
    ], style=STYLE_PAGE)

    # Tab switching callback
    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab(tab):
        if tab == "overview":
            return build_overview_tab(predictions_df, metrics, candles_df, n_candles)
        elif tab == "model":
            return build_model_tab(predictions_df, metrics)
        elif tab == "features":
            return build_features_tab(feature_importance, predictions_df)
        return html.Div("Select a tab")

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
            candles_df=candles,
        )
        print("\nDashboard starting at http://127.0.0.1:8050")
        app.run(debug=True, port=8050)
