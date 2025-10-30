"""
Demo 1: Regression Model Comparison for AM

This script demonstrates different regression techniques for predicting
surface roughness from SLM process parameters.

Learning Goals:
- Compare multiple regression algorithms (9 models!)
- Understand trade-offs (accuracy, speed, interpretability)
- Evaluate model performance properly
- See when different algorithms excel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)


def load_and_explore_data(data_path):
    """Load and explore the dataset."""
    logger.info("="*70)
    logger.info("STEP 1: LOADING AND EXPLORING DATA")
    logger.info("="*70)
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded dataset: {len(df)} samples")
    logger.info(f"✓ Features: {list(df.columns[:-1])}")
    logger.info(f"✓ Target: {df.columns[-1]}")
    
    logger.info(f"\nTarget Statistics (Surface Roughness):")
    logger.info(f"  Mean: {df['surface_roughness_um'].mean():.2f} μm")
    logger.info(f"  Std: {df['surface_roughness_um'].std():.2f} μm")
    logger.info(f"  Range: [{df['surface_roughness_um'].min():.2f}, {df['surface_roughness_um'].max():.2f}] μm")
    
    # Check correlations
    correlations = df.corr()['surface_roughness_um'].sort_values(ascending=False)
    logger.info(f"\nTop Correlations with Target:")
    for feat, corr in correlations[1:4].items():
        logger.info(f"  {feat:30s}: {corr:+.3f}")
    
    return df


def prepare_data(df):
    """Split and scale data."""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: PREPARING DATA")
    logger.info("="*70)
    
    # Separate features and target
    X = df.drop('surface_roughness_um', axis=1)
    y = df['surface_roughness_um']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"✓ Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.0f}%)")
    logger.info(f"✓ Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.0f}%)")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"\n✓ Scaled features to zero mean and unit variance")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X.columns


def train_single_model(name, model, X_train, X_test, y_train, y_test, feature_names=None, needs_poly=False):
    """Train and evaluate a single model."""
    logger.info(f"\n{'─'*70}")
    logger.info(f"{name}")
    logger.info(f"{'─'*70}")
    
    start_time = time.time()
    
    # Handle polynomial features if needed
    if needs_poly:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_model = poly.fit_transform(X_train)
        X_test_model = poly.transform(X_test)
        logger.info(f"  Features expanded from {X_train.shape[1]} to {X_train_model.shape[1]}")
    else:
        X_train_model = X_train
        X_test_model = X_test
    
    model.fit(X_train_model, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test_model)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    logger.info(f"  Training time: {train_time:.4f} seconds")
    logger.info(f"  Test R²: {r2:.4f}")
    logger.info(f"  Test RMSE: {rmse:.4f} μm")
    logger.info(f"  Test MAE: {mae:.4f} μm")
    
    # Feature importance if available
    if feature_names is not None and hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        logger.info(f"\n  Top 3 Most Important Features:")
        top_idx = np.argsort(importance)[::-1][:3]
        for idx in top_idx:
            logger.info(f"    {feature_names[idx]:25s}: {importance[idx]:.3f}")
    elif feature_names is not None and hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
        importance_normalized = importance / importance.sum()
        logger.info(f"\n  Top 3 Most Important Features:")
        top_idx = np.argsort(importance_normalized)[::-1][:3]
        for idx in top_idx:
            logger.info(f"    {feature_names[idx]:25s}: {importance_normalized[idx]:.3f}")
    
    return {
        'model': model,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'train_time': train_time,
        'predictions': y_pred
    }


def train_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    """Train and evaluate multiple regression models."""
    logger.info("\n" + "="*70)
    logger.info("STEP 3: TRAINING AND COMPARING 9 REGRESSION MODELS")
    logger.info("="*70)
    
    results = {}
    
    # Linear Models
    results['Linear Regression'] = train_single_model(
        'Model 1: Linear Regression (OLS)',
        LinearRegression(),
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )
    
    results['Ridge (L2)'] = train_single_model(
        'Model 2: Ridge Regression (L2 Regularization)',
        Ridge(alpha=1.0),
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )
    
    results['Lasso (L1)'] = train_single_model(
        'Model 3: Lasso Regression (L1 Regularization - Feature Selection)',
        Lasso(alpha=0.1, max_iter=10000),
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )
    
    results['ElasticNet'] = train_single_model(
        'Model 4: ElasticNet (L1 + L2 Regularization)',
        ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )
    
    # Polynomial
    results['Polynomial (degree 2)'] = train_single_model(
        'Model 5: Polynomial Regression (degree 2)',
        Ridge(alpha=1.0),
        X_train_scaled, X_test_scaled, y_train, y_test, None, needs_poly=True
    )
    
    # Tree-based Models
    results['Decision Tree'] = train_single_model(
        'Model 6: Decision Tree',
        DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=42),
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )
    
    results['Random Forest'] = train_single_model(
        'Model 7: Random Forest',
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )
    
    results['Gradient Boosting'] = train_single_model(
        'Model 8: Gradient Boosting',
        GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )
    
    # Other Methods
    results['K-Nearest Neighbors'] = train_single_model(
        'Model 9: K-Nearest Neighbors (k=5)',
        KNeighborsRegressor(n_neighbors=5, weights='distance'),
        X_train_scaled, X_test_scaled, y_train, y_test, None
    )
    
    results['Support Vector Regressor'] = train_single_model(
        'Model 10: Support Vector Regression (RBF kernel)',
        SVR(kernel='rbf', C=10.0, epsilon=0.1),
        X_train_scaled, X_test_scaled, y_train, y_test, None
    )
    
    results['Neural Network'] = train_single_model(
        'Model 11: Neural Network (MLP - 2 hidden layers)',
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True),
        X_train_scaled, X_test_scaled, y_train, y_test, None
    )
    
    return results


def create_visualizations(results, y_test, output_dir):
    """Create comparison visualizations."""
    logger.info("\n" + "="*70)
    logger.info("STEP 4: CREATING VISUALIZATIONS")
    logger.info("="*70)
    
    model_names = list(results.keys())
    
    # Create larger figure to accommodate more models
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    fig.suptitle('Regression Model Comparison for Surface Roughness Prediction (11 Models)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: R² comparison (takes 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    r2_scores = [results[name]['r2'] for name in model_names]
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    
    bars = ax1.barh(model_names, r2_scores, color=colors)
    ax1.set_xlabel('R² Score', fontsize=11)
    ax1.set_title('Model Comparison: R² Score (Higher is Better)', fontsize=12, fontweight='bold')
    ax1.set_xlim([0, 1])
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Good threshold (0.8)')
    ax1.legend()
    
    for bar, score in zip(bars, r2_scores):
        ax1.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=9)
    
    # Plot 2: RMSE comparison (takes 1 column)
    ax2 = fig.add_subplot(gs[0, 2])
    rmse_scores = [results[name]['rmse'] for name in model_names]
    
    bars = ax2.barh(model_names, rmse_scores, color=colors)
    ax2.set_xlabel('RMSE (μm)', fontsize=11)
    ax2.set_title('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    for bar, score in zip(bars, rmse_scores):
        ax2.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}', va='center', fontsize=8)
    
    # Plot 3: Training time comparison (takes 1 column)
    ax3 = fig.add_subplot(gs[0, 3])
    train_times = [results[name]['train_time'] for name in model_names]
    
    bars = ax3.barh(model_names, train_times, color=colors)
    ax3.set_xlabel('Training Time (s)', fontsize=11)
    ax3.set_title('Training Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xscale('log')  # Log scale for better visualization
    
    for bar, time_val in zip(bars, train_times):
        ax3.text(time_val * 1.2, bar.get_y() + bar.get_height()/2, 
                f'{time_val:.3f}s', va='center', fontsize=8)
    
    # Plots 4-9: Predicted vs Actual for top 6 models
    top_6_models = sorted(model_names, key=lambda x: results[x]['r2'], reverse=True)[:6]
    
    for idx, model_name in enumerate(top_6_models):
        row = 1 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        y_pred = results[model_name]['predictions']
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7, label='Perfect')
        
        ax.set_xlabel('Actual (μm)', fontsize=10)
        ax.set_ylabel('Predicted (μm)', fontsize=10)
        ax.set_title(f'{model_name}\nR²={results[model_name]["r2"]:.4f}, RMSE={results[model_name]["rmse"]:.3f}', 
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot 10: Model Categories (bottom right)
    ax10 = fig.add_subplot(gs[2, 3])
    ax10.axis('off')
    
    category_text = """
MODEL CATEGORIES:

Linear Models:
  • Linear Regression (baseline)
  • Ridge, Lasso, ElasticNet (regularized)
  • Polynomial (non-linear features)

Tree-Based:
  • Decision Tree (interpretable)
  • Random Forest (ensemble)
  • Gradient Boosting (sequential)

Other:
  • K-NN (distance-based)
  • SVR (kernel methods)
  • Neural Network (deep learning)
    """
    
    ax10.text(0.05, 0.95, category_text, transform=ax10.transAxes, fontsize=9,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    viz_path = output_dir / '01_regression_comparison.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {viz_path}")
    plt.close()


def summarize_results(results):
    """Print final summary."""
    logger.info("\n" + "="*70)
    logger.info("SUMMARY AND RECOMMENDATIONS")
    logger.info("="*70)
    
    # Rank models by different criteria
    by_r2 = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    by_speed = sorted(results.items(), key=lambda x: x[1]['train_time'])
    by_rmse = sorted(results.items(), key=lambda x: x[1]['rmse'])
    
    logger.info(f"\n✓ TOP 3 BY ACCURACY (R²):")
    for i, (name, metrics) in enumerate(by_r2[:3], 1):
        logger.info(f"  {i}. {name:30s} R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.3f}μm")
    
    logger.info(f"\n✓ TOP 3 BY SPEED (Training Time):")
    for i, (name, metrics) in enumerate(by_speed[:3], 1):
        logger.info(f"  {i}. {name:30s} {metrics['train_time']:.4f}s, R²={metrics['r2']:.4f}")
    
    logger.info(f"\n✓ TOP 3 BY ERROR (RMSE):")
    for i, (name, metrics) in enumerate(by_rmse[:3], 1):
        logger.info(f"  {i}. {name:30s} RMSE={metrics['rmse']:.3f}μm, R²={metrics['r2']:.4f}")
    
    # Best balanced model (top 3 in R², reasonably fast)
    fast_and_accurate = [(name, m) for name, m in by_r2[:5] if m['train_time'] < 1.0]
    if fast_and_accurate:
        logger.info(f"\n✓ BEST BALANCED (Accurate + Fast):")
        name, metrics = fast_and_accurate[0]
        logger.info(f"  {name}")
        logger.info(f"    R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.3f}μm, Time = {metrics['train_time']:.4f}s")
    
    logger.info(f"\n✓ KEY INSIGHTS:")
    logger.info(f"  • Linear models: Fast but may underfit (best for simple relationships)")
    logger.info(f"  • Regularization (Ridge/Lasso/ElasticNet): Prevents overfitting")
    logger.info(f"  • Tree-based (RF, GB): Usually best performance, good for non-linear data")
    logger.info(f"  • KNN: Simple but slow on large datasets, sensitive to scale")
    logger.info(f"  • SVR: Good for small datasets, can be slow")
    logger.info(f"  • Neural Networks: Powerful but need more data, harder to tune")
    logger.info(f"  • For optimization: Use Gaussian Process (next demo!) for uncertainty")
    
    best_r2 = by_r2[0][1]['r2']
    logger.info(f"\n✓ DATASET ASSESSMENT:")
    if best_r2 > 0.85:
        logger.info(f"  Excellent fit (R²={best_r2:.3f})! Models capture the physics well")
    elif best_r2 > 0.7:
        logger.info(f"  Good fit (R²={best_r2:.3f}), ready for optimization")
    elif best_r2 > 0.5:
        logger.info(f"  Moderate fit (R²={best_r2:.3f}) - consider feature engineering")
    else:
        logger.info(f"  Weak fit (R²={best_r2:.3f}) - need more features or data")


def main():
    """Main execution function."""
    logger.info("\n" + "="*70)
    logger.info("REGRESSION MODEL COMPARISON DEMO")
    logger.info("="*70)
    logger.info("Comparing 11 regression algorithms for surface roughness prediction")
    logger.info("="*70 + "\n")
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_path = script_dir / 'data' / 'slm_surface_roughness.csv'
    output_dir = script_dir / 'outputs' / '01_regression_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute pipeline
    df = load_and_explore_data(data_path)
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_names = prepare_data(df)
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
    create_visualizations(results, y_test, output_dir)
    summarize_results(results)
    
    logger.info(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    logger.info("\n" + "="*70)
    logger.info("DEMO COMPLETE")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
