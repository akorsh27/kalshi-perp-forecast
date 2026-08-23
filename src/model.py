"""
Model training, evaluation, and backtesting for 1-hour-ahead realized volatility.

Models:
    1. Naive baseline — predict last observed realized vol
    2. Moving-average baseline — predict rolling mean of recent realized vol
    3. LightGBM gradient-boosted regressor

Evaluation:
    - Time-based expanding-window cross-validation (no lookahead)
    - Metrics: MAE, RMSE, directional accuracy
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from src.config import LOG_LEVEL, PROJECT_ROOT
from src.features import build_feature_matrix, get_feature_names

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics for volatility predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Directional accuracy: did we correctly predict higher/lower vol vs previous?
    if len(y_true) > 1:
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        directional_acc = np.mean(true_direction == pred_direction)
    else:
        directional_acc = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_acc,
        "mean_prediction": float(np.mean(y_pred)),
        "mean_actual": float(np.mean(y_true)),
    }


class NaiveBaseline:
    """Predict realized vol = last observed realized vol (random walk)."""

    def __init__(self):
        self.name = "Naive (Last Value)"

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Use the most recent realized_vol_4h as the prediction."""
        if "realized_vol_4h" in features_df.columns:
            return features_df["realized_vol_4h"].fillna(0).values
        return features_df["log_return_1h"].abs().fillna(0).values


class MovingAverageBaseline:
    """Predict realized vol = rolling mean of recent abs returns."""

    def __init__(self, window: int = 12):
        self.window = window
        self.name = f"MA({window}h)"

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        col = "realized_vol_12h" if "realized_vol_12h" in features_df.columns else "log_return_1h"
        return features_df[col].fillna(0).values


class LightGBMForecaster:
    """LightGBM regressor for realized volatility."""

    def __init__(self, params: dict = None):
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")

        self.name = "LightGBM"
        self.params = params or {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 300,
            "early_stopping_rounds": 30,
        }
        self.model = None
        self.feature_names = get_feature_names()

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: pd.DataFrame = None, y_val: np.ndarray = None):
        """Train the LightGBM model."""
        feature_cols = [c for c in self.feature_names if c in X_train.columns]
        X = X_train[feature_cols].fillna(0)

        params = {k: v for k, v in self.params.items()
                  if k not in ("n_estimators", "early_stopping_rounds")}

        callbacks = []
        if self.params.get("early_stopping_rounds"):
            callbacks.append(lgb.early_stopping(self.params["early_stopping_rounds"]))
        callbacks.append(lgb.log_evaluation(period=50))

        train_set = lgb.Dataset(X, label=y_train)

        valid_sets = [train_set]
        if X_val is not None and y_val is not None:
            X_v = X_val[feature_cols].fillna(0)
            valid_sets.append(lgb.Dataset(X_v, label=y_val))

        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=self.params.get("n_estimators", 300),
            valid_sets=valid_sets,
            callbacks=callbacks,
        )

        logger.info("LightGBM trained: %d rounds", self.model.best_iteration or self.model.num_trees())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        feature_cols = [c for c in self.feature_names if c in X.columns]
        X_clean = X[feature_cols].fillna(0)
        return self.model.predict(X_clean)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted by gain."""
        if self.model is None:
            return pd.DataFrame()
        importance = self.model.feature_importance(importance_type="gain")
        names = self.model.feature_name()
        df = pd.DataFrame({"feature": names, "importance": importance})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: Path = None):
        """Serialize model to disk."""
        path = path or MODELS_DIR / "lightgbm_vol_forecast.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Model saved to %s", path)

    def load(self, path: Path = None):
        """Load model from disk."""
        path = path or MODELS_DIR / "lightgbm_vol_forecast.pkl"
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Model loaded from %s", path)
        return self


def backtest(
    features_df: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int = 168,  # 1 week of hourly data
) -> dict:
    """
    Run time-series cross-validation backtest.

    Uses expanding window: each fold adds more training data while
    the test set always moves forward in time.

    Args:
        features_df: Output of build_feature_matrix()
        n_splits: Number of time-series folds
        min_train_size: Minimum training samples before first evaluation

    Returns:
        Dict with per-model results and predictions DataFrame.
    """
    feature_cols = [c for c in get_feature_names() if c in features_df.columns]
    target_col = "target_realized_vol"

    X = features_df[feature_cols]
    y = features_df[target_col].values
    timestamps = features_df["end_period_ts"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = {
        "naive": {"predictions": [], "actuals": [], "timestamps": []},
        "ma": {"predictions": [], "actuals": [], "timestamps": []},
        "lightgbm": {"predictions": [], "actuals": [], "timestamps": []},
    }

    naive = NaiveBaseline()
    ma = MovingAverageBaseline(window=12)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < min_train_size:
            logger.debug("Fold %d: skipping (train size %d < %d)", fold, len(train_idx), min_train_size)
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        ts_test = timestamps[test_idx]

        # Naive baseline
        naive_preds = naive.predict(features_df.iloc[test_idx])
        results["naive"]["predictions"].extend(naive_preds)
        results["naive"]["actuals"].extend(y_test)
        results["naive"]["timestamps"].extend(ts_test)

        # MA baseline
        ma_preds = ma.predict(features_df.iloc[test_idx])
        results["ma"]["predictions"].extend(ma_preds)
        results["ma"]["actuals"].extend(y_test)
        results["ma"]["timestamps"].extend(ts_test)

        # LightGBM
        if HAS_LIGHTGBM:
            # Use last 20% of training as validation for early stopping
            val_size = max(int(len(train_idx) * 0.2), 24)
            val_idx = train_idx[-val_size:]
            pure_train_idx = train_idx[:-val_size]

            lgbm = LightGBMForecaster()
            lgbm.fit(
                X.iloc[pure_train_idx], y[pure_train_idx],
                X.iloc[val_idx], y[val_idx],
            )
            lgbm_preds = lgbm.predict(X_test)
            results["lightgbm"]["predictions"].extend(lgbm_preds)
            results["lightgbm"]["actuals"].extend(y_test)
            results["lightgbm"]["timestamps"].extend(ts_test)

        logger.info(
            "Fold %d: train=%d, test=%d", fold, len(train_idx), len(test_idx)
        )

    # Compute metrics
    summary = {}
    for model_name, data in results.items():
        if not data["predictions"]:
            continue
        metrics = evaluate_predictions(
            np.array(data["actuals"]),
            np.array(data["predictions"]),
        )
        metrics["model"] = model_name
        metrics["n_predictions"] = len(data["predictions"])
        summary[model_name] = metrics
        logger.info("%s — MAE: %.6f, RMSE: %.6f, Dir.Acc: %.2f%%",
                    model_name, metrics["mae"], metrics["rmse"],
                    metrics["directional_accuracy"] * 100)

    # Build predictions DataFrame for visualization
    predictions_df = pd.DataFrame()
    if results["lightgbm"]["predictions"]:
        predictions_df = pd.DataFrame({
            "timestamp": results["lightgbm"]["timestamps"],
            "actual": results["lightgbm"]["actuals"],
            "pred_naive": results["naive"]["predictions"][-len(results["lightgbm"]["predictions"]):],
            "pred_ma": results["ma"]["predictions"][-len(results["lightgbm"]["predictions"]):],
            "pred_lightgbm": results["lightgbm"]["predictions"],
        })

    return {
        "metrics": summary,
        "predictions": predictions_df,
        "feature_importance": lgbm.feature_importance() if HAS_LIGHTGBM and lgbm.model else pd.DataFrame(),
    }


def train_final_model(features_df: pd.DataFrame, save: bool = True) -> LightGBMForecaster:
    """
    Train a final LightGBM model on all available data (minus a holdout).

    Args:
        features_df: Full feature matrix
        save: Whether to persist the model to disk

    Returns:
        Trained LightGBMForecaster instance
    """
    feature_cols = [c for c in get_feature_names() if c in features_df.columns]
    target_col = "target_realized_vol"

    # 80/20 time-based split
    split_idx = int(len(features_df) * 0.8)
    train = features_df.iloc[:split_idx]
    test = features_df.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train[target_col].values
    X_test, y_test = test[feature_cols], test[target_col].values

    model = LightGBMForecaster()
    model.fit(X_train, y_train, X_test, y_test)

    # Evaluate on holdout
    preds = model.predict(test)
    metrics = evaluate_predictions(y_test, preds)
    logger.info("Final model holdout — MAE: %.6f, RMSE: %.6f, Dir.Acc: %.2f%%",
                metrics["mae"], metrics["rmse"], metrics["directional_accuracy"] * 100)

    if save:
        model.save()

    return model


# ─── CLI Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.database import Database

    db = Database()
    candles = db.get_candlesticks_df(interval_minutes=60)
    funding = db.get_funding_rates_df()

    if candles.empty:
        print("No data in database. Run ingestion first:")
        print("  python -m src.ingestion")
    else:
        features = build_feature_matrix(candles, funding)
        print(f"Feature matrix: {features.shape[0]} rows, {features.shape[1]} columns")

        if len(features) < 200:
            print(f"Only {len(features)} rows — need more data for meaningful backtest.")
            print("Running simple train/test split instead...")
            model = train_final_model(features)
        else:
            print("\nRunning backtest...")
            results = backtest(features)
            print("\n=== Backtest Results ===")
            for name, metrics in results["metrics"].items():
                print(f"  {name}: MAE={metrics['mae']:.6f}, "
                      f"RMSE={metrics['rmse']:.6f}, "
                      f"DirAcc={metrics['directional_accuracy']:.2%}")

            if not results["feature_importance"].empty:
                print("\n=== Top Features (by gain) ===")
                print(results["feature_importance"].head(10).to_string(index=False))
