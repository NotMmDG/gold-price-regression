"""
Gold Price Prediction Pipeline using Linear & Ridge Regression
Author: Mohammad Gharebaghi

This module implements an end-to-end machine learning pipeline to forecast
one-step-ahead closing prices for SPDR Gold Shares (GLD) using historical market data.
It evaluates Ordinary Least Squares (OLS) Linear Regression and L2-regularized
Ridge Regression, addressing multicollinearity in financial time series.
"""

import os
import sys
import argparse
from typing import Tuple, Dict, Any

# Ensure workspace packages or installed packages are accessible
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_PKGS = os.path.join(os.path.dirname(SCRIPT_DIR), "pkgs")
LOCAL_PKGS = os.path.join(SCRIPT_DIR, "pkgs")

for pkg_path in [LOCAL_PKGS, PARENT_PKGS]:
    if os.path.exists(pkg_path) and pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads historical market data from CSV, parses dates, sorts chronologically,
    and checks for missing values.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("Dataset must contain 'Date' and 'Close' columns.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"[!] Warning: Missing values detected:\n{missing[missing > 0]}")
        df = df.dropna().reset_index(drop=True)

    print(f"[+] Loaded dataset: {len(df)} rows ({df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')})")
    return df


def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the target variable: Next day's closing price (Close shifted by -1).
    Drops the last row which lacks future market data.
    """
    df_processed = df.copy()
    df_processed["Target_Next_Close"] = df_processed["Close"].shift(-1)
    df_processed = df_processed.dropna(subset=["Target_Next_Close"]).reset_index(drop=True)
    print(f"[+] Formulated target variable 'Target_Next_Close' (Length: {len(df_processed)} samples).")
    return df_processed


def split_data(
    df: pd.DataFrame, 
    feature_cols: list, 
    target_col: str = "Target_Next_Close", 
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Splits the dataset into training (80%) and testing (20%) chronologically without shuffling
    to preserve temporal causality and avoid look-ahead bias.
    StandardScaler is fitted strictly on the training set to prevent data leakage.
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"[+] Chronological Split (No Shuffling):")
    print(f"    - Training Set : {len(train_df)} samples ({train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"    - Testing Set  : {len(test_df)} samples ({test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')})")

    X_train_raw = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test_raw = test_df[feature_cols].values
    y_test = test_df[target_col].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return train_df, test_df, X_train_scaled, y_train, X_test_scaled, y_test, scaler


def train_models(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> Dict[str, Any]:
    """
    Trains:
      1) Ordinary Least Squares Linear Regression
      2) Ridge Regression (L2 penalty) with specified alpha
      3) Cross-Validated Ridge (RidgeCV) to discover optimal alpha
    """
    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # 2. Ridge Regression
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train, y_train)

    # 3. RidgeCV for optimal alpha search
    alphas = np.logspace(-3, 3, 50)
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    ridge_cv.fit(X_train, y_train)
    optimal_alpha = float(ridge_cv.alpha_)

    ridge_tuned = Ridge(alpha=optimal_alpha, random_state=42)
    ridge_tuned.fit(X_train, y_train)

    return {
        "Linear Regression": lr,
        f"Ridge Regression (alpha={alpha})": ridge,
        f"Ridge Regression (tuned alpha={optimal_alpha:.3f})": ridge_tuned,
        "optimal_alpha": optimal_alpha,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics: MAE, RMSE, R2, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def plot_actual_vs_predicted(test_dates: pd.Series, y_true: np.ndarray, lr_pred: np.ndarray, ridge_pred: np.ndarray, save_path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    plt.subplots_adjust(hspace=0.35)

    # Full test set
    ax1.plot(test_dates, y_true, label="Actual Next-Day Close", color="#1f77b4", linewidth=1.8)
    ax1.plot(test_dates, ridge_pred, label="Ridge Regression", linestyle="--", color="#2ca02c", linewidth=1.4)
    ax1.plot(test_dates, lr_pred, label="Linear Regression (OLS)", linestyle=":", color="#ff7f0e", linewidth=1.2, alpha=0.85)
    ax1.set_title("SPDR Gold Shares (GLD) Price Forecast: Out-of-Sample Period (2023 - 2024)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Price (USD)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Zoom window (last 60 days)
    zoom_n = min(60, len(test_dates))
    ax2.plot(test_dates.iloc[-zoom_n:], y_true[-zoom_n:], marker="o", markersize=3, label="Actual Next-Day Close", color="#1f77b4")
    ax2.plot(test_dates.iloc[-zoom_n:], ridge_pred[-zoom_n:], marker="x", markersize=3, linestyle="--", label="Ridge Regression", color="#2ca02c")
    ax2.plot(test_dates.iloc[-zoom_n:], lr_pred[-zoom_n:], marker=".", markersize=3, linestyle=":", label="Linear Regression", color="#ff7f0e")
    ax2.set_title(f"Detailed Dynamic Tracking: Final {zoom_n} Trading Sessions", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Price (USD)")
    ax2.set_xlabel("Date")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residuals(test_dates: pd.Series, y_true: np.ndarray, lr_pred: np.ndarray, ridge_pred: np.ndarray, save_path: str):
    lr_res = y_true - lr_pred
    ridge_res = y_true - ridge_pred

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    axes[0, 0].plot(test_dates, lr_res, color="#ff7f0e", alpha=0.7)
    axes[0, 0].axhline(0, color="black", linestyle="--")
    axes[0, 0].set_title("Linear Regression Residuals", fontweight="bold")
    axes[0, 0].set_ylabel("Error ($)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    axes[0, 1].plot(test_dates, ridge_res, color="#2ca02c", alpha=0.7)
    axes[0, 1].axhline(0, color="black", linestyle="--")
    axes[0, 1].set_title("Ridge Regression Residuals", fontweight="bold")
    axes[0, 1].set_ylabel("Error ($)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    axes[1, 0].hist(lr_res, bins=30, density=True, alpha=0.6, color="#ff7f0e", edgecolor="black", label=f"LR (Std={np.std(lr_res):.2f})")
    axes[1, 0].hist(ridge_res, bins=30, density=True, alpha=0.4, color="#2ca02c", edgecolor="black", label=f"Ridge (Std={np.std(ridge_res):.2f})")
    axes[1, 0].set_title("Residual Error Density Distribution", fontweight="bold")
    axes[1, 0].set_xlabel("Residual ($)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    axes[1, 1].scatter(y_true, lr_pred, alpha=0.4, color="#ff7f0e", s=15, label="Linear Regression")
    axes[1, 1].scatter(y_true, ridge_pred, alpha=0.4, color="#2ca02c", s=15, label="Ridge Regression")
    mn, mx = min(y_true.min(), lr_pred.min()), max(y_true.max(), lr_pred.max())
    axes[1, 1].plot([mn, mx], [mn, mx], "k--", label="1:1 Line")
    axes[1, 1].set_title("Actual vs. Predicted Scatter", fontweight="bold")
    axes[1, 1].set_xlabel("Actual Close ($)")
    axes[1, 1].set_ylabel("Predicted Close ($)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_features_and_tuning(df: pd.DataFrame, feature_cols: list, lr_model: LinearRegression, ridge_model: Ridge, X_train, y_train, X_test, y_test, feat_path: str, tune_path: str):
    # Correlation & Coefficients
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    corr = df[feature_cols + ["Target_Next_Close"]].corr()
    im = ax1.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cols = feature_cols + ["Target"]
    ax1.set_xticks(range(len(cols)))
    ax1.set_yticks(range(len(cols)))
    ax1.set_xticklabels(cols, rotation=45, ha="right")
    ax1.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax1.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.7 else "white", fontsize=8)
    ax1.set_title("Market Feature Correlation Matrix", fontweight="bold")

    x = np.arange(len(feature_cols))
    w = 0.35
    ax2.bar(x - w/2, lr_model.coef_, w, label="OLS", color="#ff7f0e", alpha=0.85)
    ax2.bar(x + w/2, ridge_model.coef_, w, label="Ridge", color="#2ca02c", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_cols)
    ax2.set_ylabel("Standardized Coefficient Weight")
    ax2.set_title("Standardized Model Coefficients", fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(feat_path, dpi=300)
    plt.close()

    # Ridge Tuning Curve
    alphas = np.logspace(-3, 4, 100)
    rmses, maes = [], []
    for a in alphas:
        m = Ridge(alpha=a).fit(X_train, y_train)
        p = m.predict(X_test)
        rmses.append(root_mean_squared_error(y_test, p))
        maes.append(mean_absolute_error(y_test, p))

    plt.figure(figsize=(8, 4.5))
    plt.semilogx(alphas, rmses, label="Test RMSE", color="#d62728", linewidth=2.0)
    plt.semilogx(alphas, maes, label="Test MAE", color="#1f77b4", linewidth=2.0)
    best_idx = np.argmin(rmses)
    plt.axvline(x=alphas[best_idx], color="black", linestyle="--", label=f"Optimal Alpha ({alphas[best_idx]:.3f})")
    plt.title("Bias-Variance Tradeoff across Regularization Strength (Alpha)", fontweight="bold")
    plt.xlabel("Regularization Penalty Alpha (log scale)")
    plt.ylabel("Test Error (USD)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(tune_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Gold Price Prediction with Linear & Ridge Regression")
    parser.add_argument("--data", default="GLD.csv", help="Path to historical GLD CSV dataset")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization parameter")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    args = parser.parse_args()

    print("=" * 70)
    print("  GOLD PRICE PREDICTION PIPELINE: LINEAR & RIDGE REGRESSION")
    print("=" * 70)

    # 1. Load data
    df = load_dataset(args.data)
    df = engineer_target(df)

    # 2. Split data
    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    train_df, test_df, X_train, y_train, X_test, y_test, scaler = split_data(
        df, feature_cols=feature_cols, target_col="Target_Next_Close", train_ratio=0.8
    )

    # 3. Train models
    models = train_models(X_train, y_train, alpha=args.alpha)

    # 4. Evaluation
    print("\n" + "=" * 70)
    print("                 MODEL EVALUATION METRICS (TEST SET)      ")
    print("=" * 70)
    header = f"{'Model':<35} | {'MAE':<8} | {'RMSE':<8} | {'R²':<8} | {'MAPE (%)':<8}"
    print(header)
    print("-" * len(header))

    predictions = {}
    for name, model in models.items():
        if name == "optimal_alpha" or not hasattr(model, "predict"):
            continue
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        predictions[name] = y_pred
        print(f"{name:<35} | {metrics['MAE']:<8.4f} | {metrics['RMSE']:<8.4f} | {metrics['R2']:<8.4f} | {metrics['MAPE']:<8.4f}%")

    print("=" * 70)

    # 5. Sample predictions
    print("\n" + "=" * 70)
    print("              SAMPLE PREDICTIONS: ACTUAL VS. PREDICTED    ")
    print("=" * 70)
    print(f"{'Date':<12} | {'Actual Close':<14} | {'Predicted LR':<14} | {'Predicted Ridge':<15}")
    print("-" * 65)

    lr_pred = predictions["Linear Regression"]
    ridge_key = [k for k in predictions if f"alpha={args.alpha}" in k or "tuned" in k][0]
    ridge_pred = predictions[ridge_key]

    sample_mask = (test_df["Date"] >= "2024-10-01") & (test_df["Date"] <= "2024-10-08")
    sample_indices = test_df[sample_mask].index.tolist()
    if not sample_indices:
        sample_indices = list(range(len(test_df) - 5, len(test_df)))

    for idx in sample_indices:
        d_str = test_df.loc[idx, "Date"].strftime("%Y-%m-%d")
        print(f"{d_str:<12} | {y_test[idx]:<14.2f} | {lr_pred[idx]:<14.2f} | {ridge_pred[idx]:<15.2f}")

    print("=" * 70)

    # 6. Generate plots
    if not args.no_plots:
        os.makedirs("figures", exist_ok=True)
        print("\n[+] Generating diagnostic figures...")
        plot_actual_vs_predicted(test_df["Date"], y_test, lr_pred, ridge_pred, "figures/actual_vs_predicted.png")
        plot_residuals(test_df["Date"], y_test, lr_pred, ridge_pred, "figures/residuals_analysis.png")
        plot_features_and_tuning(
            df, feature_cols, models["Linear Regression"], models[f"Ridge Regression (alpha={args.alpha})"],
            X_train, y_train, X_test, y_test,
            "figures/feature_analysis.png", "figures/ridge_tuning.png"
        )
        print("[+] Figures saved to figures/ directory.")

    print("\n[+] Pipeline execution completed successfully!\n")


if __name__ == "__main__":
    main()
